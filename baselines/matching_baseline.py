"""WP1 matching baselines: appearance-only matchers on image pairs with GT poses.

pySLAM parts (nothing re-implemented):
- data:     Config, dataset_factory, groundtruth_factory, camera_from_config
- matchers: feature_tracker_factory + FeatureTrackerConfigs (= SLAM presets)
- geometry: computeF12_numba, estimate_pose_ess_mat, so3_log_angle, xyzq2Tmat, inv_T
New here: pair loop, epipolar distances, translation-direction error, pose AUC, tables.
LIGHTGLUE_FULL: pySLAM's LIGHTGLUE preset with all 9 LightGlue layers (pySLAM builds it with 2).
Timing: median per pair; the first pair of each matcher runs once untimed (GPU / library warm-up).
info.txt records platform, GPU and the device of each network (pySLAM: CUDA if available, else CPU).

Usage (repo root, pySLAM env active):
    python baselines/matching_baseline.py baselines/configs/matching_kitti06_video.yaml
    python baselines/matching_baseline.py baselines/configs/matching_kitti06_video.yaml --matchers ORB2 LIGHTGLUE
"""

import argparse
import csv
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from pyslam.config import Config
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.ground_truth import groundtruth_factory
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.slam import camera_from_config
from pyslam.utilities.geom_2views import computeF12_numba, estimate_pose_ess_mat
from pyslam.utilities.geom_lie import so3_log_angle
from pyslam.utilities.geometry import add_ones, inv_T, xyzq2Tmat

MIN_GT_TRANSLATION_M = 0.01  # pairs without motion: translation direction undefined -> skipped
FULL_SUFFIX = "_FULL"  # e.g. LIGHTGLUE_FULL: LightGlue preset with all layers


def make_tracker(name, num_features):
    """pySLAM tracker of a FeatureTrackerConfigs preset; '<preset>_FULL': LightGlue with all 9 layers.

    pySLAM builds LightGlue with 2 of the 9 layers (faster). The weights contain all 9, so the full
    model is the same LightGlue class with its default depth (same weights, same device).
    """
    preset = name.removesuffix(FULL_SUFFIX)
    tracker_config = dict(FeatureTrackerConfigs.get_config_from_name(preset))
    if num_features > 0:  # same override as main_slam.py
        tracker_config["num_features"] = num_features
    tracker = feature_tracker_factory(**tracker_config)
    if name != preset:
        lg = getattr(tracker.matcher, "matcher", None)
        if type(lg).__name__ != "LightGlue":
            raise ValueError(f"{name}: '{FULL_SUFFIX}' only works for LightGlue presets")
        features = next(k for k, v in type(lg).features.items() if v["weights"] == lg.conf.weights)
        tracker.matcher.matcher = type(lg)(features=features).eval().to(tracker.matcher.torch_device)
    return tracker


def gt_pose(groundtruth, frame_id):
    """T_w_c of a frame (camera -> world)."""
    _, x, y, z, qx, qy, qz, qw, _ = groundtruth.getTimestampPoseAndAbsoluteScale(frame_id)
    return xyzq2Tmat(x, y, z, qx, qy, qz, qw)


def epipolar_distances(F12, kps1, kps2):
    """Pixel distance to the epipolar line, max over both images (x1^T F12 x2 = 0)."""
    x1, x2 = add_ones(kps1), add_ones(kps2)
    lines2 = x1 @ F12  # epipolar lines in image 2
    lines1 = x2 @ F12.T  # epipolar lines in image 1
    num = np.abs(np.sum(lines2 * x2, axis=1))
    d2 = num / np.maximum(np.linalg.norm(lines2[:, :2], axis=1), 1e-12)
    d1 = num / np.maximum(np.linalg.norm(lines1[:, :2], axis=1), 1e-12)
    return np.maximum(d1, d2)


def torch_devices(obj, depth=4):
    """Devices of the torch networks inside a pySLAM tracker (e.g. SuperPoint, LightGlue); empty if none."""
    if isinstance(obj, torch.nn.Module):
        return {str(p.device) for p in obj.parameters()}
    if depth == 0 or not hasattr(obj, "__dict__"):
        return set()
    return set().union(*(torch_devices(v, depth - 1) for v in vars(obj).values()))


def angle_deg(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_auc(errors, thresholds):
    """Area under the pose-error recall curve up to each threshold (as in SuperGlue/LightGlue)."""
    errors = np.sort(np.asarray(errors, dtype=float))
    recall = np.arange(1, len(errors) + 1) / len(errors)
    errors, recall = np.r_[0.0, errors], np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        e = np.r_[errors[:last], t]
        r = np.r_[recall[:last], recall[last - 1]]
        aucs.append(float(np.sum((e[1:] - e[:-1]) * (r[1:] + r[:-1]) / 2) / t))
    return aucs


def evaluate_pair(tracker, camera, img_ref, img_cur, T_w_ref, T_w_cur, params):
    t0 = time.perf_counter()
    kps, des = tracker.detectAndCompute(img_ref)
    kps_ref = np.array([kp.pt for kp in kps], dtype=np.float32)
    res = tracker.track(img_ref, img_cur, kps_ref, des)
    time_s = time.perf_counter() - t0

    u_ref = np.asarray(camera.undistort_points(res.kps_ref_matched), dtype=float).reshape(-1, 2)
    u_cur = np.asarray(camera.undistort_points(res.kps_cur_matched), dtype=float).reshape(-1, 2)
    n_matches = len(u_ref)

    # GT geometry
    T_ref_w, T_cur_w = inv_T(T_w_ref), inv_T(T_w_cur)
    K, Kinv = np.ascontiguousarray(camera.K, dtype=float), np.ascontiguousarray(camera.Kinv, dtype=float)
    F12, _ = computeF12_numba(
        np.ascontiguousarray(T_ref_w[:3, :3]), np.ascontiguousarray(T_ref_w[:3, 3]),
        np.ascontiguousarray(T_cur_w[:3, :3]), np.ascontiguousarray(T_cur_w[:3, 3]),
        Kinv, K, Kinv,
    )
    T_rc_gt = T_ref_w @ T_w_cur

    # match correctness
    n_correct = 0
    if n_matches > 0:
        n_correct = int(np.sum(epipolar_distances(F12, u_ref, u_cur) < params["epipolar_threshold_px"]))

    # relative pose from matches (essential matrix + RANSAC)
    rot_err = trans_err = np.inf
    inlier_ratio = 0.0
    if n_matches >= 5:
        n_ref = np.asarray(camera.unproject_points(u_ref)).reshape(-1, 2)
        n_cur = np.asarray(camera.unproject_points(u_cur)).reshape(-1, 2)
        try:
            T_rc_est, mask = estimate_pose_ess_mat(
                n_ref, n_cur, threshold=params["ransac_threshold_px"] / K[0, 0]
            )
            inlier_ratio = float(np.mean(mask > 0))
            rot_err = so3_log_angle(T_rc_est[:3, :3].T @ T_rc_gt[:3, :3], degrees=True)
            trans_err = angle_deg(T_rc_est[:3, 3], T_rc_gt[:3, 3])
        except (ValueError, cv2.error):
            pass  # no pose -> counts as failure (inf error)

    return {
        "n_kps_ref": len(kps_ref),
        "n_matches": n_matches,
        "n_correct": n_correct,
        "precision": n_correct / n_matches if n_matches > 0 else np.nan,
        "inlier_ratio": inlier_ratio,
        "rot_err_deg": rot_err,
        "trans_err_deg": trans_err,
        "pose_err_deg": max(rot_err, trans_err),
        "time_ms": 1000 * time_s,
    }


def summarize(rows, matchers, gaps, auc_thresholds):
    summary = []
    for name in matchers:
        for gap in gaps:
            sel = [r for r in rows if r["matcher"] == name and r["gap"] == gap]
            if not sel:
                continue
            aucs = pose_auc([r["pose_err_deg"] for r in sel], auc_thresholds)
            row = {
                "matcher": name,
                "gap": gap,
                "pairs": len(sel),
                "matches": np.mean([r["n_matches"] for r in sel]),
                "correct": np.mean([r["n_correct"] for r in sel]),
                "precision": np.nanmean([r["precision"] for r in sel]),
                "inlier_ratio": np.mean([r["inlier_ratio"] for r in sel]),
            }
            row.update({f"auc@{t}": a for t, a in zip(auc_thresholds, aucs)})
            row["time_ms"] = np.median([r["time_ms"] for r in sel])  # median: robust to throttling / spikes
            summary.append(row)
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="pySLAM config.yaml + MATCHING_BASELINE section")
    parser.add_argument("--matchers", nargs="+", help="override the config's matcher list")
    args = parser.parse_args()

    config = Config(str(args.config))
    params = config.config["MATCHING_BASELINE"]
    if args.matchers:
        params["matchers"] = args.matchers
    gaps, step = params["frame_gaps"], params["frame_step"]

    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings, cam_settings=config.cam_settings)
    camera = camera_from_config(config)

    trackers = {name: make_tracker(name, config.num_features_to_extract) for name in params["matchers"]}

    # single sequential pass (video datasets can't seek reliably), buffer of the last max(gap) frames
    rows, buffer, skipped, warmed_up, frame_id = [], {}, 0, set(), 0
    while (img := dataset.getImageColor(frame_id)) is not None:
        buffer[frame_id] = img
        buffer.pop(frame_id - max(gaps) - 1, None)
        for gap in gaps:
            ref_id = frame_id - gap
            if ref_id < 0 or ref_id % step != 0:
                continue
            T_w_ref, T_w_cur = gt_pose(groundtruth, ref_id), gt_pose(groundtruth, frame_id)
            if np.linalg.norm(T_w_cur[:3, 3] - T_w_ref[:3, 3]) < MIN_GT_TRANSLATION_M:
                skipped += 1
                continue
            for name, tracker in trackers.items():
                pair_args = (tracker, camera, buffer[ref_id], img, T_w_ref, T_w_cur, params)
                if name not in warmed_up:  # first call loads / compiles GPU kernels -> run once untimed
                    evaluate_pair(*pair_args)
                    warmed_up.add(name)
                result = evaluate_pair(*pair_args)
                rows.append({"matcher": name, "gap": gap, "ref": ref_id, "cur": frame_id, **result})
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"frame {frame_id}, pairs evaluated: {len(rows) // len(trackers)}")

    summary = summarize(rows, params["matchers"], gaps, params["auc_thresholds_deg"])

    out_dir = Path(f'{params["output_path"]}_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "pairs.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    shutil.copy(args.config, out_dir / args.config.name)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    devices = ", ".join(f"{name}: {'/'.join(sorted(torch_devices(tracker))) or 'cpu (no network)'}"
                        for name, tracker in trackers.items())
    (out_dir / "info.txt").write_text(
        f"opencv: {cv2.__version__}\ntorch: {torch.__version__}\nplatform: {platform.platform()}\n"
        f"cuda gpu: {gpu}\nnetworks on: {devices}\n"
        "time_ms: median per pair (detection in both images + matching), first pair per matcher untimed (warm-up)\n"
        f"skipped pairs (no motion): {skipped}\n")
    print(f"networks on: {devices} (cuda gpu: {gpu})")

    w = max(len(m) for m in ["matcher", *params["matchers"]]) + 1
    print(f"\n{'matcher':<{w}}{'gap':>4}{'pairs':>6}{'matches':>9}{'correct':>9}{'prec':>7}"
          + "".join(f"{'auc@' + str(t):>8}" for t in params["auc_thresholds_deg"]) + f"{'ms':>8}")
    for s in summary:
        print(f"{s['matcher']:<{w}}{s['gap']:>4}{s['pairs']:>6}{s['matches']:>9.0f}{s['correct']:>9.0f}"
              f"{s['precision']:>7.2f}" + "".join(f"{s['auc@' + str(t)]:>8.2f}" for t in params["auc_thresholds_deg"])
              + f"{s['time_ms']:>8.0f}")
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
