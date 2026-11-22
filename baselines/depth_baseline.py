"""WP1 depth + patch geometry baselines: depth from foundation models vs GT depth.

GT depth: dataset depth (RGB-D, e.g. TUM) or KITTI raw LiDAR (kitti_lidar.py, sparse)
- sparse GT: gt_sample_radius_px > 0 -> median of the valid GT depths in a small window
- all models on the same pixels / points (valid in every model, e.g. VGGT-Omega crop)

Depth (per frame, single image):
- dense: AbsRel, SqRel, RMSE, RMSE log, delta < 1.25^k on valid GT pixels
- at keypoints (ORB2 / SuperPoint = where matching happens), same metrics
- alignment 'metric' (as predicted, metric models only) / 'median' (per-frame median scale)
- scale = median(GT) / median(pred): metric check (~1 if metric) + stability over frames
Patch geometry (pairs (i, i + gap), GT pose):
- GT correspondences: keypoints of frame i -> GT depth + GT pose -> frame i + gap (visible only)
- patch geometry (patch_geometry.py: Open3D normals) from predicted vs GT depth:
  normal error (needs dense GT), cross-view normal consistency (normal rotated into the other view;
  '_all': all correspondences, no GT normal needed),
  reprojection with predicted depth + GT pose (px error to the GT correspondence),
  depth ratio between the views (only with a common scale: metric or multi-view model)
- multi-view models: both images in one forward pass

pySLAM parts: Config, dataset_factory, groundtruth_factory, camera_from_config,
feature_tracker_factory + FeatureTrackerConfigs, DA3 wrapper (via depth_models.py).

Usage (repo root, pySLAM env active):
    python baselines/depth_baseline.py baselines/configs/depth_tum_fr1_desk.yaml
"""

import argparse
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from pyslam.config import Config  # first: sets up pySLAM paths
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.slam import camera_from_config

from depth_models import FP32_ONLY, make_depth_model, pick_device
from matching_baseline import gt_pose, write_csv
from gt_depth import finite, gt_correspondences, gt_depth_source, project_or_nan, ray_at, sample_depth
from patch_geometry import compute_patch_geometry, normal_angle_deg, normal_map, pixel_rays, sample_nearest

DEPTH_METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]


def depth_metrics(pred, gt):
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": np.mean(np.abs(pred - gt) / gt),
        "sq_rel": np.mean((pred - gt) ** 2 / gt),
        "rmse": np.sqrt(np.mean((pred - gt) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)),
        "delta1": np.mean(ratio < 1.25),
        "delta2": np.mean(ratio < 1.25**2),
        "delta3": np.mean(ratio < 1.25**3),
    }


def median_scale(pred, gt):
    """median(GT) / median(pred) on pixels valid in both."""
    ok = np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    return float(np.median(gt[ok]) / np.median(pred[ok])) if ok.any() else np.nan


def depth_rows(frame, models, frame_id, radius):
    """Depth metrics of one frame: dense + at keypoints, per model and alignment, same pixels for all models."""
    rows, gt = [], frame["gt"]
    dense = np.isfinite(gt)
    for name in models:
        dense &= np.isfinite(frame["pred"][name]) & (frame["pred"][name] > 0)
    kp = {}
    for det, pts in frame["kps"].items():
        z_gt = sample_depth(gt, pts, radius)
        z_p = {name: sample_nearest(frame["pred"][name], pts) for name in models}
        ok = finite(z_gt, *z_p.values())
        kp[det] = (z_gt[ok], {name: z[ok] for name, z in z_p.items()})
    for name, model in models.items():
        pred = frame["pred"][name]
        scale = float(np.median(gt[dense]) / np.median(pred[dense])) if dense.any() else np.nan
        alignments = {"median": scale, **({"metric": 1.0} if model.metric else {})}
        regions = {"dense": (gt[dense], pred[dense])}
        regions.update({f"kp_{det}": (g, z_p[name]) for det, (g, z_p) in kp.items()})
        for align, s in alignments.items():
            for region, (g, p) in regions.items():
                if len(g) == 0:
                    continue
                rows.append({"model": name, "align": align, "region": region, "frame": frame_id,
                             "n": len(g), **depth_metrics(s * p, g), "scale": scale,
                             "time_ms": frame["time_ms"][name]})
    return rows


NORMAL_KEYS = ["normal_err", "xview_gt", "xview_pred", "xview_pred_all"]


def cached_normals(frame, key, depth, rays, knn):
    """Normal map of one depth map of a frame, computed once per neighbour count (GT, single-image predictions)."""
    cache = frame.setdefault("normals", {})
    if (key, knn) not in cache:
        cache[(key, knn)] = normal_map(depth, rays, knn)
    return cache[(key, knn)]


def pair_points(ref, cur, pred, model, pts, rays, camera, params, dense_gt, knns):
    """Per-point patch geometry errors of one pair (one model, one keypoint set).

    pred = (depth ref, depth cur, {knn: normal map ref}, {knn: normal map cur}) of the model;
    dense_gt: GT normals only from dense GT depth (RGB-D), not from sparse LiDAR.
    Normal errors per neighbour count: '<key>_k<knn>' (knns[0] = main, Open3D default)."""
    d_ref, d_cur, n_ref, n_cur = pred
    pts_ref, pts_cur, R, t = gt_correspondences(ref, cur, pts, rays, camera, params)
    keep = np.ones(len(pts_ref), dtype=bool)  # same points for all models (e.g. VGGT-Omega crop)
    for name in ref["pred"]:
        keep &= np.isfinite(sample_nearest(ref["pred"][name], pts_ref))
        keep &= np.isfinite(sample_nearest(cur["pred"][name], pts_cur))
    pts_ref, pts_cur = pts_ref[keep], pts_cur[keep]
    if len(pts_ref) == 0:
        return None

    out = {"n": len(pts_ref)}
    for knn in knns:
        p_ref = compute_patch_geometry(d_ref, rays, pts_ref, n_ref[knn])  # predicted depth
        p_cur = compute_patch_geometry(d_cur, rays, pts_cur, n_cur[knn])
        # cross view, predicted depth: normal of ref rotated into cur vs normal in cur (all correspondences)
        xa = p_ref["valid"] & p_cur["valid"]
        out[f"xview_pred_all_k{knn}"] = normal_angle_deg(p_ref["normal"][xa] @ R.T, p_cur["normal"][xa])
        if dense_gt:
            g_ref = compute_patch_geometry(ref["gt"], rays, pts_ref, cached_normals(ref, "gt", ref["gt"], rays, knn))
            g_cur = compute_patch_geometry(cur["gt"], rays, pts_cur, cached_normals(cur, "gt", cur["gt"], rays, knn))
            # same view: normal from predicted vs GT depth
            same = g_ref["valid"] & p_ref["valid"]
            out[f"normal_err_k{knn}"] = normal_angle_deg(p_ref["normal"][same], g_ref["normal"][same])
            # cross view on the points with GT normals in both views (GT: noise floor of the GT itself)
            xv = same & g_cur["valid"] & p_cur["valid"]
            out[f"xview_gt_k{knn}"] = normal_angle_deg(g_ref["normal"][xv] @ R.T, g_cur["normal"][xv])
            out[f"xview_pred_k{knn}"] = normal_angle_deg(p_ref["normal"][xv] @ R.T, p_cur["normal"][xv])
        else:
            out[f"normal_err_k{knn}"] = out[f"xview_gt_k{knn}"] = out[f"xview_pred_k{knn}"] = np.array([])
    p_ref = compute_patch_geometry(d_ref, rays, pts_ref, n_ref[knns[0]])  # depth part, independent of knn
    p_cur = compute_patch_geometry(d_cur, rays, pts_cur, n_cur[knns[0]])
    # reprojection with predicted depth (+ GT pose) -> error to the GT correspondence
    rays_ref = ray_at(rays, pts_ref)
    scale_ref = median_scale(d_ref, ref["gt"])
    for key, s in (("reproj", 1.0 if model.metric else np.nan), ("reproj_aligned", scale_ref)):
        uv = project_or_nan((s * p_ref["z"][:, None] * rays_ref) @ R.T + t, camera)
        err = np.linalg.norm(uv - pts_cur, axis=1)
        out[key] = err[np.isfinite(err)]
    # depth ratio between the views (needs a common scale)
    radius = params.get("gt_sample_radius_px", 0)
    zg_ref, zg_cur = sample_depth(ref["gt"], pts_ref, radius), sample_depth(cur["gt"], pts_cur, radius)
    ratio_ok = finite(zg_ref, zg_cur, p_ref["z"], p_cur["z"])
    if model.metric or model.multiview:
        out["depth_ratio_err"] = np.abs(np.log(p_cur["z"] / p_ref["z"]) - np.log(zg_cur / zg_ref))[ratio_ok]
    else:
        out["depth_ratio_err"] = np.array([])
    return out


def summarize_pairs(pooled, thr, knns):
    """patch_summary rows (normals with knns[0]) and normals_knn rows (normals for every neighbour count)."""
    med = lambda a: float(np.median(a)) if len(a) else np.nan
    frac = lambda a, x: float(np.mean(a < x)) if len(a) else np.nan
    summary, by_knn = [], []
    for (name, det, gap), d in pooled.items():
        cat = {k: np.concatenate(v) if v else np.array([]) for k, v in d.items() if k != "n"}
        normals = {knn: {f"{key}_med": med(cat[f"{key}_k{knn}"]) for key in NORMAL_KEYS} for knn in knns}
        summary.append({
            "model": name, "keypoints": det, "gap": gap, "pairs": len(d["n"]), "points": float(np.mean(d["n"])),
            **normals[knns[0]],
            "reproj_med": med(cat["reproj"]), "reproj_acc": frac(cat["reproj"], thr),
            "reproj_aligned_med": med(cat["reproj_aligned"]), "reproj_aligned_acc": frac(cat["reproj_aligned"], thr),
            "depth_ratio_err_med": med(cat["depth_ratio_err"]),
        })
        by_knn += [{"model": name, "keypoints": det, "gap": gap, "knn": knn, **normals[knn]} for knn in knns]
    return summary, by_knn


def summarize_depth(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["align"], r["region"])].append(r)
    summary = []
    for (name, align, region), sel in groups.items():
        scales = np.array([r["scale"] for r in sel])
        summary.append({"model": name, "align": align, "region": region, "frames": len(sel),
                        "n": float(np.mean([r["n"] for r in sel])),
                        **{m: float(np.mean([r[m] for r in sel])) for m in DEPTH_METRICS},
                        "scale_mean": float(np.mean(scales)), "scale_cv": float(np.std(scales) / np.mean(scales)),
                        "time_ms": float(np.median([r["time_ms"] for r in sel]))})  # median: first call = warm-up
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="pySLAM config.yaml + DEPTH_BASELINE section")
    args = parser.parse_args()

    config = Config(str(args.config))
    params = config.config["DEPTH_BASELINE"]
    gaps, step = params["frame_gaps"], params["frame_step"]
    lo, hi = params["gt_depth_range_m"]

    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings, cam_settings=config.cam_settings)
    camera = camera_from_config(config)
    rays = pixel_rays(camera)
    gt_depth = gt_depth_source(params, dataset, camera)
    dense_gt = (params.get("gt_depth") or {"type": "dataset"})["type"] == "dataset"  # RGB-D: GT normals
    knns = params.get("normal_knn", [30])  # Open3D estimate_normals neighbours; first = main (Open3D default 30)
    radius = params.get("gt_sample_radius_px", 0)

    detectors = {}
    for name in params["keypoints"]:
        tracker_config = dict(FeatureTrackerConfigs.get_config_from_name(name))
        if config.num_features_to_extract > 0:  # same override as main_slam.py
            tracker_config["num_features"] = config.num_features_to_extract
        detectors[name] = feature_tracker_factory(**tracker_config)
    models = {name: make_depth_model(name, camera) for name in params["models"]}

    out_dir = Path(f'{params["output_path"]}_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = lambda i: i % step == 0 or any(i - g >= 0 and (i - g) % step == 0 for g in gaps)

    # single sequential pass (works for video datasets too), buffer of the last max(gap) frames
    rows_depth, rows_pairs, pooled, buffer, frame_id = [], [], {}, {}, 0
    while (img := dataset.getImageColor(frame_id)) is not None:
        if needed(frame_id):
            gt = gt_depth(frame_id)
            gt[~((gt >= lo) & (gt <= hi))] = np.nan
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = {"gt": gt, "rgb": rgb, "T_w_c": gt_pose(groundtruth, frame_id),
                     "pred": {}, "time_ms": {}, "kps": {}}
            for name, model in models.items():
                t0 = time.perf_counter()
                frame["pred"][name] = model.predict([rgb])[0][0]
                frame["time_ms"][name] = 1000 * (time.perf_counter() - t0)
                if params.get("save_depth", False):
                    (out_dir / "depth" / name).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_dir / "depth" / name / f"{frame_id:06d}.npz",
                                        depth=frame["pred"][name].astype(np.float16))
            if frame_id % step == 0:
                for det, tracker in detectors.items():
                    kps, _ = tracker.detectAndCompute(img)
                    frame["kps"][det] = np.array([kp.pt for kp in kps], dtype=np.float64).reshape(-1, 2)
                rows_depth += depth_rows(frame, models, frame_id, radius)
            buffer[frame_id] = frame

            for gap in gaps:
                ref_id = frame_id - gap
                if ref_id < 0 or ref_id % step != 0 or ref_id not in buffer:
                    continue
                ref = buffer[ref_id]
                for name, model in models.items():
                    if model.multiview:  # both images in one forward pass -> common scale
                        (d_ref, _), (d_cur, _) = model.predict([ref["rgb"], frame["rgb"]])
                        pred = (d_ref, d_cur, {k: normal_map(d_ref, rays, k) for k in knns},
                                {k: normal_map(d_cur, rays, k) for k in knns})
                    else:
                        d_ref, d_cur = ref["pred"][name], frame["pred"][name]
                        pred = (d_ref, d_cur, {k: cached_normals(ref, name, d_ref, rays, k) for k in knns},
                                {k: cached_normals(frame, name, d_cur, rays, k) for k in knns})
                    for det, pts in ref["kps"].items():
                        res = pair_points(ref, frame, pred, model, pts, rays, camera, params, dense_gt, knns)
                        if res is None:
                            continue
                        acc = pooled.setdefault((name, det, gap), defaultdict(list))
                        for k, v in res.items():
                            acc[k].append(np.atleast_1d(v))
                        med = lambda k: float(np.median(res[k])) if len(res[k]) else np.nan
                        rows_pairs.append({"model": name, "keypoints": det, "gap": gap, "ref": ref_id,
                                           "cur": frame_id, "points": res["n"],
                                           "normal_err_med": med(f"normal_err_k{knns[0]}"),
                                           "xview_pred_med": med(f"xview_pred_k{knns[0]}"),
                                           "reproj_med": med("reproj"), "reproj_aligned_med": med("reproj_aligned")})
        for old in [k for k in buffer if k < frame_id - max(gaps)]:
            del buffer[old]
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"frame {frame_id}, depth frames: {len({r['frame'] for r in rows_depth})}, pairs: {len(rows_pairs)}")

    summary_depth = summarize_depth(rows_depth)
    for key in pooled:
        pooled[key]["n"] = [int(n) for a in pooled[key]["n"] for n in a]
    summary_pairs, summary_knn = summarize_pairs(pooled, params["reproj_threshold_px"], knns)

    write_csv(out_dir / "depth_frames.csv", rows_depth)
    write_csv(out_dir / "depth_summary.csv", summary_depth)
    write_csv(out_dir / "patch_pairs.csv", rows_pairs)
    write_csv(out_dir / "patch_summary.csv", summary_pairs)
    write_csv(out_dir / "normals_knn.csv", summary_knn)
    shutil.copy(args.config, out_dir / args.config.name)
    (out_dir / "info.txt").write_text(
        f"opencv: {cv2.__version__}\ntorch: {torch.__version__}\ndevice: {pick_device()}"
        f" (gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})\n"
        f"depth models fp32 only: {FP32_ONLY}\ntime_ms: median per frame (first call = warm-up)\n"
        f"frames read: {frame_id}\nmodels: {list(models)}\nkeypoints: {list(detectors)}\n")

    print(f"\n{'model':<17}{'align':<7}{'region':<14}{'frames':>6}{'AbsRel':>8}{'RMSE':>7}{'d1':>6}"
          f"{'scale':>7}{'cv':>6}{'ms':>6}")
    for s in summary_depth:
        print(f"{s['model']:<17}{s['align']:<7}{s['region']:<14}{s['frames']:>6}{s['abs_rel']:>8.3f}"
              f"{s['rmse']:>7.3f}{s['delta1']:>6.2f}{s['scale_mean']:>7.3f}{s['scale_cv']:>6.2f}{s['time_ms']:>6.0f}")
    print(f"\n{'model':<17}{'kps':<11}{'gap':>4}{'pts':>6}{'n_err':>7}"
          f"{'xv_gt':>7}{'xv_pr':>7}{'xv_all':>7}{'rp':>6}{'rp_al':>7}{'acc_al':>7}{'dr':>6}")
    for s in summary_pairs:
        print(f"{s['model']:<17}{s['keypoints']:<11}{s['gap']:>4}{s['points']:>6.0f}"
              f"{s['normal_err_med']:>7.1f}{s['xview_gt_med']:>7.1f}"
              f"{s['xview_pred_med']:>7.1f}{s['xview_pred_all_med']:>7.1f}{s['reproj_med']:>6.1f}"
              f"{s['reproj_aligned_med']:>7.1f}"
              f"{s['reproj_aligned_acc']:>7.2f}{s['depth_ratio_err_med']:>6.3f}")
    print(f"\nnormals per neighbour count (Open3D knn; table above: knn = {knns[0]})")
    print(f"{'model':<17}{'kps':<11}{'gap':>4}{'knn':>6}{'n_err':>7}{'xv_gt':>7}{'xv_pr':>7}{'xv_all':>7}")
    for s in summary_knn:
        print(f"{s['model']:<17}{s['keypoints']:<11}{s['gap']:>4}{s['knn']:>6}{s['normal_err_med']:>7.1f}"
              f"{s['xview_gt_med']:>7.1f}{s['xview_pred_med']:>7.1f}{s['xview_pred_all_med']:>7.1f}")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
