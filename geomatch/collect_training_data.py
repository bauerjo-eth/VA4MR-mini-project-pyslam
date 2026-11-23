"""
Collect per-match training data for the learned filters from the TRAIN split only.

Run from the repository root:
    python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml

For every image pair of every sequence in the config (TRAIN split: KITTI 00-08), it computes
RootSIFT matches and stores, per putative match:
  - MLP features (22-d, see learned_filter.build_mlp_features)
  - CNN patches (2 x 32 x 32 uint8 log-depth, see learned_filter.build_cnn_patches)
  - label: 1 if the GT epipolar error < grading.correct_px (oracle label, training only)
  - pair_id: which pair the match came from (so train/val can be split by pair, not by sample)

Output: one .npz (default geomatch/data/training/kitti_train.npz) + a sidecar .json with
provenance (config, sequences, gaps, pair counts, seed, timing). The TEST split (KITTI 09-10)
is never touched by this script.

Gap sweep: `--gaps 1 5 10 20 50 80 --step 100` collects one pair set per gap in a single run
(frames are loaded once per sequence, depth cached as usual); every sample is tagged with its
pair's gap (`gap_ids` in the .npz), which train_filter.py uses for per-gap val metrics.

Timing is logged: depth-prediction time, per-pair time, total wall time.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from geomatch.cache_depth_pictures import cache_depth_maps, load_depth
from geomatch.descriptors import compute_descriptors, depth_at
from geomatch.evaluation_metrics import epipolar_error
from geomatch.evaluation_pipeline import load_dataset, split_type
from geomatch.filtering import MatchContext
from geomatch.learned_filter import PATCH_SIZE, build_cnn_patches, build_mlp_features
from geomatch.matchers import match_descriptors

DEFAULT_OUT = Path(__file__).parent / "data" / "training" / "kitti_train.npz"
log = logging.getLogger("geomatch")


def main():
    parser = argparse.ArgumentParser(description="Collect learned-filter training data (TRAIN split only).")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "kitti_train_config.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-matches-per-pair", type=int, default=200,
                        help="subsample cap per pair (seeded); keeps the .npz small")
    parser.add_argument("--gaps", type=int, nargs="+", default=None,
                        help="override the config gap; collect one pair set per gap (gap sweep). "
                             "Each pair is tagged with its gap (gap_ids in the .npz)")
    parser.add_argument("--step", type=int, default=None, help="override the config pair step")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    grading = cfg["grading"]
    correct_px = grading["correct_px"]
    gaps = args.gaps or [cfg["dataset"]["pairs"]["gap"]]
    step = args.step or cfg["dataset"]["pairs"]["step"]
    ds = cfg["dataset"]
    # metadata only — frames are loaded per sequence below (union over all gaps), so a gap
    # sweep does not re-read the same images once per gap
    sequences = load_dataset(ds, with_frames=False)
    # split guard: training data may only come from the TRAIN split (00-08), never from the
    # held-out TEST sequences (09-10) — otherwise the filters would be tested on train data
    from geomatch.learned_filter import assert_train_split_only
    assert_train_split_only([s.name for s in sequences])
    desc_type, desc_params = "rootsift", {}
    match_type, match_params = "nn_ratio", {"ratio": 0.8}

    t_start = time.perf_counter()
    all_feats, all_patches, all_labels, all_pair_ids, all_gap_ids = [], [], [], [], []
    pair_counter = 0
    depth_time = 0.0

    from geomatch.evaluation_pipeline import make_pairs, read_frames, read_image_folder
    import os
    for seq in sequences:
        # all (gap, f1, f2) pairs across the sweep, and the union of frames they need
        all_pairs = []
        for gap in gaps:
            p = dict(ds["pairs"], gap=gap, step=step)
            all_pairs += [(gap, f1, f2) for f1, f2 in make_pairs(p, len(seq.poses))]
        needed = {i for _, f1, f2 in all_pairs for i in (f1, f2)}
        if ds.get("type") == "kitti_odometry":
            root = Path(ds.get("base_path") or os.environ.get("KITTI_ODOMETRY_PATH", ""))
            img_dir = root / "sequences" / seq.name.replace("kitti", "") / "image_0"
            seq_frames = read_image_folder(img_dir, needed)
        else:
            seq_frames = read_frames(ds["video"], needed)
        t0 = time.perf_counter()
        depth_dir = cache_depth_maps(seq_frames, cfg["depth"]["model"], seq.name)
        depth_time += time.perf_counter() - t0
        log.info(f"[{seq.name}] {len(all_pairs)} pairs over gaps={gaps}, {len(needed)} frames, K fx={seq.K[0,0]:.2f}")

        for gap, f1, f2 in tqdm(all_pairs, desc=f"collect {seq.name}", unit="pair"):
            img1, img2 = seq_frames[f1], seq_frames[f2]
            depth1, depth2 = load_depth(depth_dir, f1), load_depth(depth_dir, f2)
            pts1, des1 = compute_descriptors(desc_type, img1, depth1, **desc_params)
            pts2, des2 = compute_descriptors(desc_type, img2, depth2, **desc_params)
            i1, i2 = match_descriptors(match_type, des1, des2, **match_params)
            i1, i2 = np.asarray(i1, int), np.asarray(i2, int)
            if len(i1) == 0:
                continue
            z1, z2 = depth_at(depth1, pts1), depth_at(depth2, pts2)
            ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2, des1=des1, des2=des2,
                               depth1=depth1, depth2=depth2, z1=z1, z2=z2, K=seq.K,
                               T1=seq.poses[f1], T2=seq.poses[f2])
            labels = (epipolar_error(pts1[i1], pts2[i2], seq.poses[f1], seq.poses[f2], seq.K)
                      < correct_px).astype(np.uint8)

            # seeded subsample so every run collects the same matches
            if len(i1) > args.max_matches_per_pair:
                sel = rng.choice(len(i1), args.max_matches_per_pair, replace=False)
                ctx = MatchContext(i1=i1[sel], i2=i2[sel], pts1=pts1, pts2=pts2,
                                   des1=des1, des2=des2, depth1=depth1, depth2=depth2,
                                   z1=z1, z2=z2, K=seq.K, T1=seq.poses[f1], T2=seq.poses[f2])
                labels = labels[sel]

            feats = build_mlp_features(ctx)
            patches = build_cnn_patches(ctx, PATCH_SIZE)
            all_feats.append(feats)
            all_patches.append((patches * 255).round().astype(np.uint8))  # store compact
            all_labels.append(labels)
            all_pair_ids.append(np.full(len(labels), pair_counter, np.int32))
            all_gap_ids.append(np.full(len(labels), gap, np.int32))
            pair_counter += 1

    feats = np.concatenate(all_feats)
    patches = np.concatenate(all_patches)
    labels = np.concatenate(all_labels)
    pair_ids = np.concatenate(all_pair_ids)
    gap_ids = np.concatenate(all_gap_ids)
    wall_s = time.perf_counter() - t_start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, mlp_features=feats, cnn_patches=patches,
                        labels=labels, pair_ids=pair_ids, gap_ids=gap_ids)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config), "sequences": [s.name for s in sequences],
        "descriptor": desc_type, "matcher": f"{match_type}({match_params})",
        "correct_px": correct_px, "seed": args.seed, "gaps": gaps, "step": step,
        "max_matches_per_pair": args.max_matches_per_pair,
        "n_pairs": pair_counter, "n_samples": int(len(labels)),
        "n_correct": int(labels.sum()), "patch_size": PATCH_SIZE,
        "depth_time_s": round(depth_time, 1), "wall_time_s": round(wall_s, 1),
    }
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"saved {len(labels)} samples ({labels.sum()} correct, {labels.mean():.1%}) "
             f"from {pair_counter} pairs in {wall_s:.1f} s (depth prediction {depth_time:.1f} s)")
    log.info(f"  features {feats.shape}, patches {patches.shape} -> {args.out}")


if __name__ == "__main__":
    main()
