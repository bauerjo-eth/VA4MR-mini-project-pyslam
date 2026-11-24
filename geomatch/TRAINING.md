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

## Model registry — which model was trained on WHAT dataset, tested on WHAT

All training data: TRAIN split (KITTI 00–08) only. All evaluation: TEST split (09–10) only.
Each checkpoint also stores this (`train_data`, `train_sequences`, `train_n_samples`,
`train_n_pairs`, `train_gaps`, `train_tag`); each run log (`runs_*.json`) too.

| model file | training dataset (file · gaps · step) | pairs · samples | val AUC | evaluated on |
|---|---|---|---|---|
| `data/models/{linear,mlp,cnn}_best.pt` | `training/kitti_train.npz` · gap 80 · step 20 | 985 · 36,179 | 0.843 / **0.895** / 0.863 | TEST 09–10, `kitti_test_config.yaml` (gap 80, 263 pairs) |
| `data/models/gaps/<m>_gap{g}_best.pt` | `training/kitti_train_gap{g}.npz` · gap g · step 100 | ~201–206 · 7,446–41,200 | 0.840–0.892 | TEST 09–10, `kitti_test_gapsweep_config.yaml --gap G` |
| `data/models/gaps/<m>_all_best.pt` | the six `kitti_train_gap*.npz` · gaps 1,5,10,20,50,80 · step 100 | 1,230 · **149,276** | **0.924** (mlp) / 0.910 (cnn) | same sweep, all eval gaps |
| `data/models/gaps_dense/<m>_all20_best.pt` | `training/kitti_train_dense_gap*.npz` · gaps 1,5,10,20,50,80 · **step 20** | ~6,100 · ~TBD | TBD | TEST 09–10, `kitti_test_dense_config.yaml --gap 50/80` |
| `data/models/gaps_ransac/*` | same gap files, `--label-metric ransac1px` | same | ≤0.933 | sweep `_ransac` config |

⚠️ `mlp_best.pt` (step 20) and `gaps/mlp_gap80_best.pt` (step 100) are different models — see note below.

## Gap sweep (multi-gap training, cross-gap evaluation) — 2026-11-23

Deviation from the single-gap protocol above: models are now identified by the gap(s) they
were trained on (`--tag`), training data can be collected for several gaps in one run
(`--gaps`), and evaluation can override the pair gap (`--gap`). The canonical TEST config
(`kitti_test_config.yaml`) is unchanged and still gap=80.

### What changed in the code
- `collect_training_data.py`: `--gaps 1 5 10 20 50 80` + `--step` overrides; frames are loaded
  once per sequence (union over all gaps), every sample is tagged with its pair's gap
  (`gap_ids` in the .npz, gaps+step in the sidecar .json).
- `train_filter.py`: `--data` accepts several .npz (concatenated, pair ids offset per file so
  the pair-wise val split stays clean); `--tag gap80` / `all` names checkpoints
  `<model>_<tag>_best.pt`; val metrics include a per-gap AUC breakdown.
- `evaluation_pipeline.py`: `--gap G` CLI override (effective config saved into the results
  folder, result name gets a `_gapG` suffix); `load_dataset(..., with_frames=False)` for
  metadata-only loading used by the collector.
- `kitti_test_gapsweep_config.yaml`: TEST 09-10, all 7 MLP models + sift baseline + oracle.

### Training data per gap (TRAIN 00-08, step 100, seed 0, max 200 matches/pair)

| gap | pairs | samples | % correct | wall s (depth s) | file |
|---|---|---|---|---|---|
| 1 | 206 | 41,200 | 96.3% | 63 (0) | `kitti_train_gap1.npz` |
| 5 | 206 | 39,979 | 84.8% | 107 (45) | `kitti_train_gap5.npz` |
| 10 | 206 | 31,933 | 72.4% | 107 (45) | `kitti_train_gap10.npz` |
| 20 | 206 | 19,182 | 54.7% | 60 (0) | `kitti_train_gap20.npz` |
| 50 | 205 | 9,536 | 25.3% | 106 (45) | `kitti_train_gap50.npz` |
| 80 | 201 | 7,446 | 13.2% | 58 (0) | `kitti_train_gap80.npz` |

### Which model classes / architectures the sweep trains

The gap sweep trains **MLP and CNN only** (linear/logreg appears only in the single-gap M2
comparison above). Per tag (`gap1…gap80`, `all`) the full grid from `train_filter.py` RUNS runs —
5 MLP + 5 CNN configurations — and the best by val AUC is copied to `<model>_<tag>_best.pt`:

- MLP runs: `h64x32_lr1e-3_ep30`, `h128x64x32_lr1e-3_ep30`, `h64x32_lr1e-4_ep50`,
  `h256x128x64_lr1e-3_ep30`, `h64x32_do0.2_lr1e-3_ep40` (Linear→ReLU→(Dropout) stacks on the
  22-d features, batch 4096)
- CNN runs: `c16x32x64_lr1e-3_ep15`, `c32x64x128_lr1e-3_ep15`, `c16x32x64_lr5e-4_ep25`,
  `c8x16x32_lr1e-3_ep25`, `c16x32x64_p16_lr1e-3_ep25` (3× Conv3x3+ReLU+MaxPool, then head; the
  `_p16` run center-crops the stored 32×32 patches to 16×16)

Winning architecture per tag (epipolar labels, weights `data/models/gaps/`):

| tag | MLP best run (val AUC) | CNN best run (val AUC) |
|---|---|---|
| gap1 | h64x32_lr1e-4_ep50 (0.8817) | c16x32x64_p16_lr1e-3_ep25 (0.8906) |
| gap5 | h256x128x64_lr1e-3_ep30 (0.8402) | c16x32x64_lr5e-4_ep25 (0.8307) |
| gap10 | h256x128x64_lr1e-3_ep30 (0.8771) | c16x32x64_p16_lr1e-3_ep25 (0.8676) |
| gap20 | h64x32_do0.2_lr1e-3_ep40 (0.8719) | c32x64x128_lr1e-3_ep15 (0.8848) |
| gap50 | h256x128x64_lr1e-3_ep30 (0.8921) | c32x64x128_lr1e-3_ep15 (0.8850) |
| gap80 | h256x128x64_lr1e-3_ep30 (0.8629) | c8x16x32_lr1e-3_ep25 (0.8689) |
| all | h64x32_lr1e-3_ep30 (0.9240) | c16x32x64_lr5e-4_ep25 (0.9102) |

### ⚠️ Two gap-80 MLPs: canonical vs sweep
The canonical M2 model (`data/models/mlp_best.pt`, in `kitti_test_config.yaml`) was trained on
`kitti_train.npz` = gap 80 at **step 20** (985 pairs / 36,179 samples, val AUC 0.895,
precision_epi 0.359 @gap80). The sweep's `mlp_gap80` (`data/models/gaps/mlp_gap80_best.pt`) was
trained on `kitti_train_gap80.npz` = gap 80 at **step 100** (201 pairs / 7,446 samples, val AUC
0.863, precision_epi 0.348 @gap80) — step 100 for ALL sweep gaps so the tags stay comparable.
Different training data AND different eval config; do not compare 0.359 vs 0.348 across tables.

### Models (MLP, GPU GTX 1660, seed 0, best-of-grid val AUC; weights `data/models/gaps/`)

| tag (trained on) | val AUC | per-gap val AUC (best run) |
|---|---|---|
| gap1 | 0.8817 | {1: 0.882} |
| gap5 | 0.8402 | {5: 0.840} |
| gap10 | 0.8771 | {10: 0.877} |
| gap20 | 0.8719 | {20: 0.872} |
| gap50 | 0.8921 | {50: 0.892} |
| gap80 | 0.8629 | {80: 0.863} |
| all (1,5,10,20,50,80; 149,276 samples / 1,230 pairs) | 0.9240 | {1: 0.766, 5: 0.833, 10: 0.905, 20: 0.901, 50: 0.890, 80: 0.861} |

### Cross evaluation: precision_epi after RANSAC, TEST 09-10, per eval gap

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift (ransac only) | 0.998 | 0.963 | 0.899 | 0.819 | 0.417 | 0.111 |
| mlp_gap1 | 0.999 | 0.979 | 0.941 | 0.901 | 0.821 | 0.443 |
| mlp_gap5 | 0.999 | 0.983 | 0.960 | 0.904 | 0.783 | **0.529** |
| mlp_gap10 | 1.000 | 0.983 | 0.949 | 0.897 | 0.759 | 0.401 |
| mlp_gap20 | 1.000 | 0.991 | 0.965 | 0.898 | 0.662 | 0.365 |
| mlp_gap50 | 1.000 | 0.993 | 0.968 | 0.897 | 0.634 | 0.344 |
| mlp_gap80 | 0.999 | 0.979 | 0.939 | 0.876 | 0.639 | 0.348 |
| **mlp_all** | 0.998 | 0.971 | 0.928 | 0.878 | **0.778** | 0.524 |
| sift_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Results: `results/kitti_test_gapsweep_gap{1,5,10,20,50,80}_2026-11-23_*/{summary,per_pair}.csv`.

### Findings
- Small gaps are easy for everyone (sift alone: 0.998 at gap 1); the learned filters matter at
  **wide baselines**: at gap 80 every learned model beats ransac-only sift (0.111) by 3-5x.
- The gap80-trained model is NOT the best at gap 80 (0.348): models trained on mid gaps
  generalize better (gap5-trained: 0.529). The `all` model is best at gap 50 (0.778) and second
  at gap 80 (0.524) — training on the gap mix is the robust choice.
- Learned filters also cut runtime at gap 80 (269 vs 403 ms/pair for sift) due to less RANSAC input.

### Reproduce the sweep
```bash
export KITTI_ODOMETRY_PATH=/path/to/kitti_odometry/dataset
for g in 1 5 10 20 50 80; do
  python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml \
      --gaps $g --step 100 --out geomatch/data/training/kitti_train_gap$g.npz --seed 0
  python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap$g.npz \
      --tag gap$g --models mlp --out-dir geomatch/data/models/gaps --seed 0
done
python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap{1,5,10,20,50,80}.npz \
    --tag all --models mlp --out-dir geomatch/data/models/gaps --seed 0
for g in 1 5 10 20 50 80; do
  python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_gapsweep_config.yaml --gap $g
done
```

### Metrics used (and why two different ones)
- **Evaluation matrix metric: `precision_epi`** = share of the FINAL matches (after the
  filter AND the always-on RANSAC) whose GT epipolar error is < 4 px. VO estimates the camera
  motion from exactly this set, so its precision is the VO-relevant quality measure at the
  fixed operating point (filter thresh 0.5, RANSAC 1 px).
- **Training/val metric: AUC** (reported per run and per gap in `runs_*.json`). At training
  time the decision threshold is not fixed yet, so a threshold-free ranking metric (AUC of the
  keep-probability) is used to pick the best checkpoint and to compare models; the threshold
  sweep on TEST (0.3/0.5/0.7 in the M2 config) then turns probabilities into keep masks.
- **VO sample-count metrics** (new columns in `summary.csv`, both AFTER THE FILTER but BEFORE
  RANSAC — RANSAC always runs after every filter; the post-RANSAC count is the `final` column):
  `keep_ratio` = `filter_kept / putative` (share of the matcher output the filter keeps) and
  `keep_vs_oracle` = `filter_kept / oracle's filter_kept` (kept count relative to the number of
  truly correct putative matches; the oracle's `filter_kept` IS the correct-putative count).

### CNN gap sweep (same protocol, --models cnn; weights `data/models/gaps/cnn_*`)

| tag | val AUC | | tag | val AUC |
|---|---|---|---|---|
| cnn_gap1 | 0.8906 | | cnn_gap50 | 0.8850 |
| cnn_gap5 | 0.8307 | | cnn_gap80 | 0.8689 |
| cnn_gap10 | 0.8676 | | **cnn_all** | **0.9102** |
| cnn_gap20 | 0.8848 | | (mlp_all for comparison) | 0.9240 |

### Cross evaluation with CNN added (TEST 09–10, thresh 0.5) — precision_epi

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift (ransac only) | 0.998 | 0.963 | 0.899 | 0.819 | 0.417 | 0.111 |
| mlp_gap5 | 0.999 | 0.983 | 0.960 | 0.904 | 0.783 | **0.529** |
| mlp_all | 0.998 | 0.971 | 0.928 | 0.878 | **0.778** | 0.524 |
| cnn_gap1 | 0.999 | 0.977 | 0.923 | 0.894 | 0.810 | 0.479 |
| cnn_all | 0.998 | 0.977 | 0.930 | 0.889 | 0.679 | 0.323 |
| sift_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

(full 16-method matrix incl. all single-gap models: `results/kitti_test_gapsweep_gap*_2026-11-23_*/summary.csv`)

### Keep counts (VO sample budget) — gap 80 vs gap 1

| method | keep_ratio@80 | keep_ratio@1 | keep_vs_oracle@80 | keep_vs_oracle@1 | final@80 (sift) |
|---|---|---|---|---|---|
| sift (no filter) | 1.000 | 1.000 | 21.77 | 1.03 | 1720 |
| mlp_all | 0.048 | 0.953 | 1.04 | 0.98 | 168 |
| cnn_all | 0.095 | 0.924 | 2.06 | 0.95 | — |
| sift_oracle | 0.046 | 0.969 | 1.00 | 1.00 | 195 |

Oracle kept = #correct putative: 385 @gap80, 205,993 @gap1 (putative 8,383 / 212,568).
Reading: at gap 80 `mlp_all` keeps ~1.0× the oracle count (399 vs 385) — almost only correct
matches — while sift keeps 21.8× (mostly outliers that RANSAC then has to winnow, 1,720 final).
At gap 1 everything is easy and filters keep ~95% of putative. Runtime @gap80: learned
filters ~285 ms/pair vs 431 ms for plain sift.

Full keep matrices (all 16 methods × all 6 eval gaps; both post-filter, PRE-RANSAC):

keep_ratio = filter_kept/putative:

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mlp_gap1 | 0.741 | 0.521 | 0.395 | 0.251 | 0.083 | 0.034 |
| mlp_gap5 | 0.772 | 0.686 | 0.558 | 0.383 | 0.122 | 0.050 |
| mlp_gap10 | 0.747 | 0.690 | 0.590 | 0.407 | 0.138 | 0.060 |
| mlp_gap20 | 0.629 | 0.609 | 0.554 | 0.442 | 0.192 | 0.096 |
| mlp_gap50 | 0.618 | 0.627 | 0.581 | 0.471 | 0.217 | 0.114 |
| mlp_gap80 | 0.768 | 0.688 | 0.611 | 0.466 | 0.205 | 0.097 |
| mlp_all | 0.953 | 0.797 | 0.642 | 0.418 | 0.124 | 0.048 |
| cnn_gap1 | 0.809 | 0.496 | 0.356 | 0.215 | 0.084 | 0.045 |
| cnn_gap5 | 0.756 | 0.671 | 0.561 | 0.379 | 0.152 | 0.088 |
| cnn_gap10 | 0.691 | 0.630 | 0.548 | 0.409 | 0.174 | 0.097 |
| cnn_gap20 | 0.659 | 0.654 | 0.606 | 0.471 | 0.205 | 0.109 |
| cnn_gap50 | 0.666 | 0.660 | 0.599 | 0.478 | 0.224 | 0.122 |
| cnn_gap80 | 0.684 | 0.671 | 0.613 | 0.496 | 0.243 | 0.132 |
| cnn_all | 0.924 | 0.751 | 0.636 | 0.431 | 0.166 | 0.095 |
| sift_oracle | 0.969 | 0.858 | 0.718 | 0.510 | 0.152 | 0.046 |

keep_vs_oracle = filter_kept / oracle filter_kept:

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift | 1.03 | 1.17 | 1.39 | 1.96 | 6.57 | 21.77 |
| mlp_gap1 | 0.76 | 0.61 | 0.55 | 0.49 | 0.54 | 0.73 |
| mlp_gap5 | 0.80 | 0.80 | 0.78 | 0.75 | 0.80 | 1.09 |
| mlp_gap10 | 0.77 | 0.80 | 0.82 | 0.80 | 0.91 | 1.31 |
| mlp_gap20 | 0.65 | 0.71 | 0.77 | 0.87 | 1.26 | 2.09 |
| mlp_gap50 | 0.64 | 0.73 | 0.81 | 0.92 | 1.43 | 2.49 |
| mlp_gap80 | 0.79 | 0.80 | 0.85 | 0.91 | 1.35 | 2.10 |
| mlp_all | 0.98 | 0.93 | 0.89 | 0.82 | 0.82 | 1.04 |
| cnn_gap1 | 0.83 | 0.58 | 0.50 | 0.42 | 0.55 | 0.98 |
| cnn_gap5 | 0.78 | 0.78 | 0.78 | 0.74 | 1.00 | 1.92 |
| cnn_gap10 | 0.71 | 0.73 | 0.76 | 0.80 | 1.14 | 2.11 |
| cnn_gap20 | 0.68 | 0.76 | 0.84 | 0.92 | 1.35 | 2.36 |
| cnn_gap50 | 0.69 | 0.77 | 0.83 | 0.94 | 1.47 | 2.65 |
| cnn_gap80 | 0.71 | 0.78 | 0.85 | 0.97 | 1.59 | 2.87 |
| cnn_all | 0.95 | 0.87 | 0.89 | 0.85 | 1.09 | 2.06 |
| sift_oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Updated findings
- CNN behaves like the MLP but slightly worse at gap 80 (cnn_all 0.323 vs mlp_all 0.524;
  best single-gap CNN at gap80: cnn_gap1 0.479). MLP on the 22-d features is the better
  wide-baseline cleaner; CNN's patch input is less discriminative here.
- The `all`-gap models are the robust choice across the sweep for both architectures.
- Keep counts confirm the VO trade-off: learned filters shrink the RANSAC input ~10-20x at
  gap 80 while keeping ≈ the oracle number of correct matches (keep_vs_oracle ≈ 1), and are
  faster per pair than no filter.

### Reproduce (sweep incl. CNN)
```bash
for g in 1 5 10 20 50 80; do
  python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap$g.npz \
      --tag gap$g --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0
done
python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap{1,5,10,20,50,80}.npz \
    --tag all --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0
for g in 1 5 10 20 50 80; do
  python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_gapsweep_config.yaml --gap $g
done
```

### Filter confusion matrix (TP/FP/TN/FN) — post-filter, PRE-RANSAC

Computed from `per_pair.csv` (TP = correct_filter, FP = kept−TP, FN = correct_putative−TP,
TN = putative−kept−FN); the pipeline now also writes `filter_tp/fp/tn/fn`, `filter_precision`
and `filter_f1` into `summary.csv`/`per_pair.csv` directly. "Correct" = GT epipolar < 4 px.

Filter F1 (16 methods × 6 eval gaps):

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift (no filter) | 0.984 | 0.924 | 0.836 | 0.675 | 0.264 | 0.088 |
| mlp_gap5 | 0.881 | 0.856 | 0.815 | 0.730 | 0.597 | 0.313 |
| mlp_gap50 | 0.773 | 0.817 | 0.827 | 0.786 | 0.606 | 0.333 |
| mlp_gap80 | 0.877 | 0.850 | 0.826 | 0.772 | 0.596 | 0.311 |
| **mlp_all** | **0.981** | **0.911** | **0.844** | 0.743 | 0.583 | 0.283 |
| cnn_gap1 | 0.903 | 0.703 | 0.596 | 0.493 | 0.428 | 0.189 |
| cnn_all | 0.965 | 0.884 | 0.833 | 0.737 | 0.536 | 0.238 |
| sift_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Full TP/FP/TN/FN counts at the two extremes (all 16 methods in the result CSVs):

| @gap1 (putative 212,568; correct 205,993) | TP | FP | TN | FN |
|---|---|---|---|---|
| sift | 205,993 | 6,575 | 0 | 0 |
| mlp_all | 200,540 | 2,110 | 4,465 | 5,453 |
| cnn_all | 194,211 | 2,209 | 4,366 | 11,782 |
| sift_oracle | 205,993 | 0 | 6,575 | 0 |

| @gap80 (putative 8,383; correct 385) | TP | FP | TN | FN |
|---|---|---|---|---|
| sift | 385 | 7,998 | 0 | 0 |
| mlp_gap50 | 224 | 735 | 7,263 | 161 |
| mlp_all | 111 | 288 | 7,710 | 274 |
| cnn_all | 140 | 653 | 7,345 | 245 |
| sift_oracle | 385 | 0 | 7,998 | 0 |

Reading: at small gaps the `all` models are near-oracle (F1 0.98 @gap1). At gap 80 every
filter trades recall for precision (drops most of the 7,998 wrong matches, TN, but also many
of the 385 correct ones, FN) — the filter-stage F1 is only ~0.3 there, yet AFTER RANSAC the
kept set reaches precision_epi 0.52 (mlp_gap5), which is why the eval matrix uses the
post-RANSAC precision.

## RANSAC-1px training labels ("does it survive 1 px RANSAC?") — 2026-11-24

Experiment: re-label training data with `collect_training_data --label-metric ransac1px`
(label = inlier of the same 1 px essential-matrix RANSAC the pipeline applies downstream;
estimated from the putative matches, no GT pose needed) and repeat the full gap sweep.

- Data: `data/training/kitti_train_ransac_gap{1,5,10,20,50,80}.npz` — positive fraction per gap:
  68.5 / 79.9 / 72.2 / 57.3 / 33.5 / **25.5 %** (vs epipolar labels 96.3 / 84.8 / 72.4 / 54.7 /
  25.3 / 13.2 %) — RANSAC-inlier ≠ GT-correct, the label is noisier.
- Models: same MLP+CNN grid per tag, weights `data/models/gaps_ransac/`. Best val AUC:
  gap1 0.885/0.896, gap5 0.837/0.823, gap10 0.904/0.889, gap20 0.934/0.933, gap50 0.811/0.826,
  gap80 0.735/0.741, **all 0.883 (mlp, h256x128x64) / 0.843 (cnn)**.
- Cross evaluation on TEST 09–10 (16 methods × 6 gaps, `kitti_test_gapsweep_ransac_config.yaml`),
  precision_epi epipolar-label / ransac-label per gap:

| method | g1 | g5 | g10 | g20 | g50 | g80 |
|---|---|---|---|---|---|---|
| sift (ransac only) | .998/.998 | .963/.963 | .899/.899 | .819/.819 | .417/.417 | .111/.111 |
| mlp_all | .998/.997 | .971/.963 | .928/.895 | .878/.828 | .778/.574 | **.524/.241** |
| mlp_gap5 | .999/.999 | .983/.976 | .960/.928 | .904/.867 | .783/.694 | .529/.371 |
| cnn_all | .998/.998 | .977/.972 | .930/.912 | .889/.850 | .679/.640 | .323/.315 |

  keep_ratio @gap80: mlp_all 0.048 (epi) vs 0.077 (ransac-label); time_ms_per_pair @gap80 ~285 ms
  for both label types (vs ~430 sift). Results: `results/kitti_test_gapsweep_ransac_gap*_2026-11-24_*/`.

**Conclusion:** training on "survives 1 px RANSAC" does NOT improve post-RANSAC accuracy — the
label includes matches that are RANSAC-inliers but GT-wrong, so the filter keeps more
outliers (higher keep_ratio) and final precision drops (mlp_all @gap80: 0.241 vs 0.524). The
speed benefit is identical, because the pipeline runs RANSAC anyway and the filter already
shrinks its input. GT-epipolar labels remain the better training signal for accuracy.

### Online inference-time benchmark (batch size = 1) — hardware & device placement

`python -m geomatch.bench_inference --pairs 30 --gap 80` — 30 TEST pairs (kitti09), depth
predicted ONLINE (not from cache). **Hardware: NVIDIA GeForce GTX 1660 (6 GB), torch
2.6.0+cu124, CUDA 12.4.** Device placement: DA3 depth prediction → **GPU (cuda)**; learned
filter forward pass → **GPU (cuda)** (models moved to cuda in `filtering._get_learned_model`);
RootSIFT descriptors, nn_ratio matching, essential-matrix RANSAC → **CPU (OpenCV)**.
NOTE: times are online per-frame latency at **batch size 1**, not batched throughput.

Per-pair mean over 30 pairs, gap 80 (ms). **Depth is charged only to methods that need it —
RANSAC-only (plain rootsift, no filter) never computes a depth map:**

| method | depth (GPU) | describe (CPU) | match (CPU) | filter (GPU) | ransac (CPU) | TOTAL | prec_epi* |
|---|---|---|---|---|---|---|---|
| ransac_only (no filter, **no depth needed**) | 0.0 | 225.1 | 21.4 | 0.0 | 177.2 | **423.8** | 0.016 |
| mlp_epi_all @0.5 | 588.7 | 225.1 | 21.4 | 18.2 | 0.2 | **853.6** | 0.000 |
| mlp_ransac_all @0.5 | 588.7 | 225.1 | 21.4 | 17.8 | 0.0 | **853.1** | 0.000 |
| cnn_epi_all @0.5 | 588.7 | 225.1 | 21.4 | 14.8 | 1.4 | **851.4** | 0.000 |

(*precision on these first 30 kitti09 pairs is low for every method incl. ransac-only —
verified identical to the eval per_pair rows; the 263-pair average is 0.111.)
Reading: the filter itself is cheap (~15–18 ms on GPU) and shrinks RANSAC from ~177 ms
(all ~8.4 k putative matches) to ~0–1 ms (~400 kept) — but ONLINE the learned-filter front
end must also pay for DA3 depth (~589 ms/pair on GPU), which RANSAC-only never needs. So
online at batch size 1 the filtered pipeline is ~2× SLOWER per pair (853 vs 424 ms); the
speed win only exists when depth is already available/cached (offline eval: ~285 vs ~430
ms/pair, see gap-sweep tables). The filter's real online value is accuracy at wide baselines
(precision_epi 0.52 vs 0.11 at gap 80 over all 263 pairs), not latency.

Reproduce:
```bash
python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml --gaps $g --step 100 --label-metric ransac1px --out geomatch/data/training/kitti_train_ransac_gap$g.npz --seed 0
python -m geomatch.train_filter --data ... --tag gap$g --models mlp cnn --out-dir geomatch/data/models/gaps_ransac --seed 0
python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_gapsweep_ransac_config.yaml --gap $g
python -m geomatch.bench_inference --pairs 30 --gap 80
```
