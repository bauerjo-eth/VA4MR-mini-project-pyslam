"""
Inference-time benchmark for the learned filters vs RANSAC-only, with ONLINE depth prediction.

Unlike evaluation_pipeline.py (which loads cached DA3 depth maps), this benchmark predicts
depth with DA3 on the fly and INCLUDES that time, so the numbers are the true per-frame cost of
the front end. Run from the repository root:

    python -m geomatch.bench_inference --pairs 30 --gap 80

Per image pair it times, separately:
  depth      DA3 metric depth for both frames        (GPU: cuda if available)
  describe   RootSIFT keypoints + descriptors        (CPU: OpenCV)
  match      nn_ratio(0.8)                           (CPU: OpenCV BFMatcher)
  filter     learned filter keep-mask (optional)     (GPU if available, else CPU)
  ransac     essential-matrix RANSAC on kept matches (CPU: OpenCV)
and grades the final matches (GT epipolar < 4 px = correct) for context.

Methods compared: `none` (RANSAC on all putative matches) vs learned_mlp / learned_cnn at
thresh 0.5 with given weights. Hardware + device placement is printed at the start.

NOTE: all times are ONLINE, batch-size-1 (one frame / one pair at a time, depth predicted on
the fly) — they are per-frame latency numbers as used by a real-time VO front end, NOT batched
throughput. Do not compare them to batched offline numbers.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from geomatch.depth import predict_depth
from geomatch.descriptors import compute_descriptors, depth_at
from geomatch.evaluation_metrics import epipolar_error
from geomatch.evaluation_pipeline import load_dataset, ransac_relative_pose
from geomatch.filtering import MatchContext, filter_matches
from geomatch.matchers import match_descriptors


def main():
    p = argparse.ArgumentParser(description="Online-depth inference-time benchmark.")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "kitti_test_config.yaml")
    p.add_argument("--pairs", type=int, default=30, help="number of TEST pairs to time")
    p.add_argument("--gap", type=int, default=80)
    p.add_argument("--mlp", type=str, default="geomatch/data/models/gaps/mlp_all_best.pt",
                   help="MLP weights (epipolar labels)")
    p.add_argument("--mlp-ransac", type=str, default="geomatch/data/models/gaps_ransac/mlp_all_best.pt",
                   help="MLP weights (ransac1px labels)")
    p.add_argument("--cnn", type=str, default="geomatch/data/models/gaps/cnn_all_best.pt")
    p.add_argument("--thresh", type=float, default=0.5)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"hardware: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU-only'} | "
          f"torch {torch.__version__} | devices: DA3 depth -> {dev}, learned filters -> {dev}, "
          f"SIFT/match/RANSAC -> CPU (OpenCV)")

    cfg = yaml.safe_load(open(args.config))
    cfg["dataset"]["pairs"]["gap"] = args.gap
    seq = load_dataset(cfg["dataset"])[0]  # first TEST sequence (09)
    pairs = seq.pairs[: args.pairs]
    print(f"benchmark: {len(pairs)} pairs of {seq.name}, gap={args.gap} (depth predicted ONLINE)")

    methods = [
        ("ransac_only", None, None),
        ("mlp_epi_all", "learned_mlp", args.mlp),
        ("mlp_ransac_all", "learned_mlp", args.mlp_ransac),
        ("cnn_epi_all", "learned_cnn", args.cnn),
    ]
    times = {m[0]: [] for m in methods}
    acc = {m[0]: [0, 0] for m in methods}  # correct, final

    for f1, f2 in pairs:
        img1, img2 = seq.frames[f1], seq.frames[f2]
        # depth is only needed by the learned filters (rootsift + RANSAC-only never touch it):
        # time it ONLY for methods that use it, so the RANSAC-only baseline is not charged for DA3
        t0 = time.perf_counter()
        depth1, depth2 = predict_depth(img1), predict_depth(img2)  # ONLINE, GPU
        t_depth = time.perf_counter() - t0

        t0 = time.perf_counter()
        pts1, des1 = compute_descriptors("rootsift", img1, depth1)
        pts2, des2 = compute_descriptors("rootsift", img2, depth2)
        t_desc = time.perf_counter() - t0

        t0 = time.perf_counter()
        i1, i2 = match_descriptors("nn_ratio", des1, des2, ratio=0.8)
        i1, i2 = np.asarray(i1, int), np.asarray(i2, int)
        t_match = time.perf_counter() - t0

        z1, z2 = depth_at(depth1, pts1), depth_at(depth2, pts2)
        ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2, des1=des1, des2=des2,
                           depth1=depth1, depth2=depth2, z1=z1, z2=z2, K=seq.K,
                           T1=seq.poses[f1], T2=seq.poses[f2])
        epi_all = epipolar_error(pts1[i1], pts2[i2], seq.poses[f1], seq.poses[f2], seq.K)

        for name, ftype, fpath in methods:
            t0 = time.perf_counter()
            if ftype is None:
                keep = np.ones(len(i1), bool)
                t_f = 0.0  # RANSAC-only needs no depth, no filter
            else:
                keep = filter_matches(ctx, method=ftype, model_path=fpath, thresh=args.thresh)
                t_f = t_depth  # charge the online depth computation to the filter methods only
            t_filter = time.perf_counter() - t0
            k1, k2 = i1[keep], i2[keep]

            t0 = time.perf_counter()
            R, t, inl = ransac_relative_pose(pts1[k1], pts2[k2], seq.K, threshold_px=1.0,
                                            confidence=0.99999)
            t_ransac = time.perf_counter() - t0

            times[name].append((t_f, t_desc, t_match, t_filter, t_ransac))
            epi = epi_all[keep][inl]
            acc[name][0] += int((epi < 4.0).sum())
            acc[name][1] += len(epi)

    print(f"\nper-pair mean times (ms) over {len(pairs)} pairs, gap={args.gap}, ONLINE depth "
          f"(charged only to methods that need it — RANSAC-only never computes depth):")
    hdr = f"{'method':16s} {'depth':>8s} {'descr':>8s} {'match':>8s} {'filter':>8s} {'ransac':>8s} {'TOTAL':>8s} {'prec_epi':>8s}"
    print(hdr)
    for name, _, _ in methods:
        arr = np.array(times[name]) * 1000
        tot = arr.sum(1).mean()
        print(f"{name:16s} " + " ".join(f"{v:8.1f}" for v in arr.mean(0)) +
              f" {tot:8.1f} {acc[name][0]/max(acc[name][1],1):8.3f}")


if __name__ == "__main__":
    main()
