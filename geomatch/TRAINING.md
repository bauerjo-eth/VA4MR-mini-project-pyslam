# Learned filter training & evaluation — provenance record

Single source of truth for **which data trained what, and what the numbers are**.
All learned-filter results below come from the TEST split only; training never touched it.

## Data split (KITTI odometry; only sequences 00–10 have GT poses)

| Split | Sequences | Frames | Used for |
|---|---|---|---|
| TRAIN | 00, 01, 02, 03, 04, 05, 06, 07, 08 | ~20,400 | training-data collection, MLP/CNN training, val split (by pair) |
| TEST | 09, 10 | ~2,800 | evaluation only — never seen during training |

- Configs: `kitti_train_config.yaml` (train), `kitti_test_config.yaml` (test). Default
  `evaluation_config.yaml` (bundled KITTI 06 video) is unchanged and needs no external data.
- **The split is enforced, not just documented:** `collect_training_data.py` and
  `train_filter.py` raise an error if the data covers TEST sequences 09–10
  (`learned_filter.assert_train_split_only`), and `evaluation_pipeline.py` raises an error if a
  `learned_*` filter is evaluated on any sequence its checkpoint was trained on
  (`evaluation_pipeline.guard_against_train_split_leak`, using the `train_sequences` stored in
  every checkpoint). Accidentally pointing the test config at 00–08 fails loudly.
- Pairs: `(s, s+80)`, `start=6`, `stop=-1` (sequence end); train `s` step 20, test step 10.
- Depth: DA3METRIC-LARGE, cached per sequence in `data/depth/<model>/kittiXX/`.
- Training-data descriptors: `rootsift` (n=2000), matcher `nn_ratio(ratio=0.8)`.
- Labels (training only): GT epipolar error < 4 px = correct. GT poses are never an input feature.
- Note: with gap=80 only ~12% of putative matches are correct at 4 px (much harder than the
  gap=10 demo) — this is the realistic SLAM regime where RANSAC has to remove many outliers.

## What is filter-specific vs reusable for matching (WP2)

The learned-filter work splits into a **shared infrastructure** part and a **filter-specific** part.
A matching/descriptor model (another group member) can reuse the shared part as-is:

| Component | Reusable for matching? | Notes |
|---|---|---|
| `kitti_odometry` dataset mode + TRAIN/TEST split (sequences 00–08 / 09–10) | ✅ yes | shared by any learned method; never evaluate on TEST |
| `collect_training_data.py` | ✅ yes, with tweaks | currently hardcodes `rootsift`+`nn_ratio`; parameterize `descriptor`/`matcher` from the config to collect data for other matchers. Labels = GT epipolar < 4 px work for scoring matches too |
| `build_mlp_features(ctx)` (22-d per-match features) | ✅ yes | per-match features are matcher-agnostic (depth stats, descriptor distance, positions) |
| `build_cnn_patches(ctx)` (depth patches) | ✅ yes | same patches describe a candidate match for a learned matcher |
| `train_filter.py` training loop | ✅ yes | seeding, pair-wise val split, GPU, class-balanced BCE, timing, runs-log JSON — model-agnostic; add your model class to `learned_filter.py` and a grid entry in `RUNS` |
| checkpoint format + `runs_*.json` log | ✅ yes | records hyperparams/seed/versions for replication |
| `FILTERS` registry hook in `filtering.py` (`@register("learned_*")` → keep-mask) | ❌ filter-only | a matcher plugs into the `MATCHERS` registry in `matchers.py` instead (returns `(i1, i2)` index pairs, not a mask) |
| `MatchContext` + `filter_matches()` call site in the pipeline | ⚠️ partly | `MatchContext` is shared and a matcher would receive the same per-pair data; but the pipeline calls filters **after** matching — a learned matcher replaces the `matcher` stage, not the filter stage |
| threshold sweep (`thresh` 0.3/0.5/0.7) | ❌ filter-only | matchers rank/score candidates instead of thresholding a mask |

Rule of thumb: **data, features, training loop, and evaluation split are shared; the registry hook and the keep-mask interface are filter-specific** — for matching, register in `matchers.py` and return scored candidate pairs.

## Training data (collected 2026-11-18)

| Field | Value |
|---|---|
| File | `geomatch/data/training/kitti_train.npz` (+ `.json` sidecar) |
| Samples / pairs / correct | 36,179 samples from 985 pairs, 4,343 correct (12.0%) |
| Subsample | max 200 matches/pair, seeded (`--seed 0`) |
| Collection wall time | **491.4 s** total, of which depth prediction (1,086 frames, GPU) **203.2 s** |

## Training runs (GPU: GTX 1660, torch 2.6.0+cu124, CUDA 12.4, seed 0)

Every run: seeds fixed (python/numpy/torch, cudnn deterministic), val = 15% of **pairs**
(grouped by `pair_id`, so no pair is in both train and val), class-balanced BCE, Adam.
Full per-run record (hyperparams, val metrics, per-epoch times, versions):
`geomatch/data/models/runs_2026-11-18_220508.json` (11 runs; earlier 6-run log:
`runs_2026-11-18_214801.json`). Checkpoints in `geomatch/data/models/`; best per model copied
to `<model>_best.pt` (what `kitti_test_config.yaml` uses).

| Model | Run | Hidden / channels | lr | ep | batch | val AUC | val acc | val P@0.5 | val R@0.5 | train time |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | logreg_lr1e-3_ep30 | logistic regression | 1e-3 | 30 | 4096 | 0.8433 | 0.808 | 0.368 | 0.741 | 10.9 s |
| MLP | h64x32_lr1e-3_ep30 | 64-32 | 1e-3 | 30 | 4096 | 0.8822 | 0.866 | 0.478 | 0.754 | 11.2 s |
| MLP | h128x64x32_lr1e-3_ep30 | 128-64-32 | 1e-3 | 30 | 4096 | 0.8930 | 0.867 | 0.481 | 0.800 | 11.3 s |
| MLP | h64x32_lr1e-4_ep50 | 64-32 | 1e-4 | 50 | 4096 | 0.8384 | 0.844 | 0.424 | 0.677 | 18.5 s |
| MLP | **h256x128x64_lr1e-3_ep30 (best)** | 256-128-64 | 1e-3 | 30 | 4096 | **0.8947** | 0.865 | 0.478 | 0.797 | 11.3 s |
| MLP | h64x32_do0.2_lr1e-3_ep40 | 64-32 + dropout 0.2 | 1e-3 | 40 | 4096 | 0.8837 | 0.871 | 0.490 | 0.754 | 15.0 s |
| CNN | c16x32x64_lr1e-3_ep15 | 16-32-64 | 1e-3 | 15 | 256 | 0.8623 | 0.867 | 0.479 | 0.709 | 17.3 s |
| CNN | c32x64x128_lr1e-3_ep15 | 32-64-128 | 1e-3 | 15 | 128 | 0.8554 | 0.837 | 0.415 | 0.736 | 24.2 s |
| CNN | **c16x32x64_lr5e-4_ep25 (best)** | 16-32-64 | 5e-4 | 25 | 256 | **0.8631** | 0.870 | 0.487 | 0.705 | 28.6 s |
| CNN | c8x16x32_lr1e-3_ep25 | 8-16-32 | 1e-3 | 25 | 256 | 0.8589 | 0.841 | 0.424 | 0.749 | 24.8 s |
| CNN | c16x32x64_p16_lr1e-3_ep25 | 16-32-64, patch 16 | 1e-3 | 25 | 256 | 0.8542 | 0.835 | 0.413 | 0.749 | 21.5 s |

Val ranking: **MLP 0.895 > CNN 0.863 > linear 0.843** — the 22 depth features carry most of
the signal; raw depth patches add less; nonlinearity beats plain logistic regression.

## Test-split evaluation (KITTI 09–10, 263 pairs)

Command: `python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_config.yaml`
Full 15-method comparison: `geomatch/results/kitti_test_2026-11-18_220846/`
(quick 5-method check: `kitti_test_2026-11-18_215038/`).

| Field | Value |
|---|---|
| Results dir (full comparison) | `geomatch/results/kitti_test_2026-11-18_220846/` |
| Depth maps (first run, GPU) | 279 frames predicted (~5 frames/s); cached afterwards |
| Eval wall time (depth cached) | **328.3 s** for 263 pairs × 15 methods (match+filter+RANSAC+grading) |

Final matches after RANSAC (correct = GT epipolar error < 4 px; putative = 8383 for all methods):

| method | filter_kept | filter_recall | final | precision_epi | recall | median_epi_px | auc@5° | auc@10° | auc@20° | ms/pair |
|---|---|---|---|---|---|---|---|---|---|---|
| sift (`none`) | 8383 | 1.00 | 1720 | 0.111 | 0.496 | 96.6 | 0.010 | 0.014 | 0.017 | 430 |
| depth_valid | 8383 | 1.00 | 1720 | 0.111 | 0.496 | 96.6 | 0.010 | 0.014 | 0.017 | 431 |
| desc_thresh_0.5 | 8351 | 1.00 | 1709 | 0.109 | 0.486 | 99.5 | 0.010 | 0.014 | 0.017 | 430 |
| desc_thresh_1.0 | 8383 | 1.00 | 1720 | 0.111 | 0.496 | 96.6 | 0.010 | 0.014 | 0.017 | 430 |
| reldepth_0.1 | 1741 | 0.32 | 848 | 0.111 | 0.244 | 90.8 | 0.008 | 0.010 | 0.014 | 272 |
| reldepth_0.2 | 3178 | 0.52 | 1264 | 0.112 | 0.366 | 84.5 | 0.011 | 0.013 | 0.022 | 289 |
| reldepth_0.4 | 5533 | 0.82 | 1638 | 0.120 | 0.509 | 96.0 | 0.017 | 0.025 | 0.035 | 350 |
| learned_linear_0.5 | 1521 | 0.60 | 740 | 0.231 | 0.444 | 44.7 | 0.014 | 0.019 | 0.028 | 289 |
| learned_mlp_0.3 | 1746 | 0.68 | 882 | 0.212 | 0.486 | 37.3 | **0.020** | 0.026 | 0.043 | 292 |
| **learned_mlp_0.5** | 917 | 0.55 | 429 | **0.359** | 0.400 | **7.1** | 0.018 | 0.026 | 0.038 | 290 |
| learned_mlp_0.7 | 493 | 0.43 | 220 | **0.514** | 0.294 | 3.9 | 0.013 | 0.022 | 0.030 | 290 |
| learned_cnn_0.3 | 1960 | 0.66 | 920 | 0.198 | 0.473 | 58.0 | 0.017 | 0.025 | 0.034 | 305 |
| learned_cnn_0.5 | 901 | 0.50 | 423 | 0.314 | 0.346 | 11.8 | 0.018 | 0.023 | 0.031 | 287 |
| learned_cnn_0.7 | 375 | 0.24 | 148 | 0.250 | 0.096 | 10.1 | 0.012 | 0.016 | 0.017 | 286 |
| sift_oracle (ref) | 385 | 1.00 | 195 | 1.000 | 0.507 | 1.9 | 0.022 | 0.027 | 0.032 | 268 |

### Comparison findings

- **Handcrafted filters barely help at gap=80**: `reldepth` peaks at precision 0.120 (reldepth_0.4)
  vs 0.111 for no filter; `desc_threshold` changes nothing (RootSIFT+ratio already bounds the
  distance); `depth_valid` is a no-op here. They do cut RANSAC input (hence lower ms/pair).
- **Learned filters dominate**: MLP@0.5 reaches precision 0.359 (3.2× baseline) with recall 0.40;
  MLP@0.7 reaches **0.514** precision. CNN trails MLP at equal threshold (0.314 vs 0.359).
  Linear sits between (0.231). Best pose AUC: MLP@0.3 (auc@5° 0.020 vs 0.010 baseline).
- **Threshold = precision/recall knob**: learned_mlp at 0.3/0.5/0.7 → precision 0.21/0.36/0.51,
  recall 0.49/0.40/0.29. Pick per SLAM need (more matches vs cleaner).
- **Inference time**: `time_ms_per_pair` counts descriptor+match+filter+RANSAC fairly per method.
  Learned filters run ~290 ms/pair vs 430 ms for plain sift — the filter itself is only a few ms
  (22-d features + tiny forward on CPU), and shrinking the RANSAC input from 8383 to ~900 matches
  more than pays for it. Learned filtering is **faster and more accurate** than no filter.
- Oracle upper bound: 1.0 precision with 385 correct putative matches of 8383 — headroom remains.
- Caveat: gap=80 is a hard regime (only 4.6% of putative matches are correct at 4 px); absolute
  numbers are lower than the gap=10 demo, but method ranking is what matters.

## Reproduce

```bash
export KITTI_ODOMETRY_PATH=/path/to/kitti_odometry/dataset   # contains sequences/ and poses/
python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml --seed 0
python -m geomatch.train_filter.py --data geomatch/data/training/kitti_train.npz --seed 0
python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_config.yaml
```

Seeds (0) + recorded hyperparams/versions make the runs replicable on other machines.
Models and data are gitignored (`geomatch/data/`) — retrain with the commands above;
trained weights are not committed.
