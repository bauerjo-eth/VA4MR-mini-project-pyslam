# TRAINING_SUMMARY — learned match filters: the important numbers

Overview of what was trained, on what, and what it achieved. Full provenance, all 16-method
matrices, per-run logs: `TRAINING.md`. Raw outputs: `results/`. Everything below is TEST-split
(KITTI 09–10) unless stated otherwise; training used TRAIN (00–08) only.

## TL;DR

- **GT label/metric**: a match is "correct" if its **GT epipolar error < 4 px**
  (`grading.correct_px`). Training labels and the main eval metric (`precision_epi`) use it.
- **Winner**: at the canonical gap 80, the **MLP trained on gap 5** is best
  (**precision_epi 0.529** vs 0.111 RANSAC-only sift); the **MLP trained on all gaps
  (`mlp_all`) is the robust pick** (0.524 @gap80, best @gap50 = 0.778, best val AUC 0.924).
  CNN trails the MLP at wide baselines (cnn_all 0.323 @gap80).
- Learned filters matter only at **wide baselines** (gap ≥ 50); at gap 1 everything is easy.
- **Speed**: with cached depth, filters are faster than RANSAC-only (~285 vs ~430 ms/pair);
  online (depth computed per frame) the filter pipeline is ~2× slower because DA3 depth
  (~589 ms/pair on GPU) dominates and RANSAC-only never needs depth.

## Ground truth & metrics (definitions)

| term | meaning |
|---|---|
| GT label (training) | GT epipolar error < 4 px (oracle label, training only; GT poses never a model input) |
| `precision_epi` (eval matrix) | share of **final (post-filter, post-RANSAC)** matches with GT epipolar < 4 px — what VO consumes |
| val AUC (training) | threshold-free AUC of the keep-probability, used for checkpoint selection; per-gap AUC also logged |
| `filter_*` / `keep_*` columns | **pre-RANSAC** (post-filter); all other columns post-RANSAC (`final`, `precision_epi`, …) |
| "correct" | GT epipolar error < `correct_px` = 4.0 px |

RANSAC (essential matrix, 1 px, always on) runs after every filter for every method.

## Datasets (sequence-level split, enforced in code)

| split | sequences | use | gap | step | pairs | samples | % correct |
|---|---|---|---|---|---|---|---|
| TRAIN | 00–08 | training data | 80 | 20 | 985 | 36,179 | 12.0 % |
| TRAIN per-gap sweep files | 00–08 | gap sweep | 1/5/10/20/50/80 | 100 | ~201–206 each | 7,446–41,200 | 96.3→13.2 % |
| TEST | 09–10 | evaluation only | 80 (canonical) / any via `--gap` | 10 | 263 | — | 4.6 % |

Guards: collector/trainer refuse TEST sequences; evaluator refuses learned filters on
sequences their checkpoint was trained on. Depth: DA3METRIC-LARGE, cached per sequence in
`data/depth/<model>/kittiXX/`. Matcher everywhere: RootSIFT + `nn_ratio(0.8)`.

## Models and which won

Model zoo: linear/logreg, MLP (22-d features), DepthPatchCNN (2×32×32 log-depth patches +
descriptor distance). Grid per tag in `train_filter.py`; best by val AUC → `<model>_<tag>_best.pt`.

| model (tag = training gaps) | val AUC | winning architecture | weights |
|---|---|---|---|
| linear (gap80) | 0.843 | logreg | `data/models/linear_best.pt` |
| mlp (gap80, M2) | **0.895** | h256x128x64 | `data/models/mlp_best.pt` |
| cnn (gap80, M2) | 0.863 | c16x32x64_lr5e-4 | `data/models/cnn_best.pt` |
| mlp_gap5 | 0.840 | h256x128x64 | `data/models/gaps/` |
| mlp_gap80 (sweep: step 100, 7,446 samples — see note above) | 0.863 | h256x128x64 | `data/models/gaps/` |
| **mlp_all** (1,5,10,20,50,80) | **0.924** | h64x32_lr1e-3_ep30 | `data/models/gaps/mlp_all_best.pt` |
| cnn_all | 0.910 | c16x32x64_lr5e-4 | `data/models/gaps/cnn_all_best.pt` |

## Test-split results (TEST 09–10)

### ⚠️ Two different "gap-80 MLP" models — don't mix the tables

| | canonical M2 model | gap-sweep model |
|---|---|---|
| config / weights | `kitti_test_config.yaml` → `data/models/mlp_best.pt` | `kitti_test_gapsweep_config.yaml` → `data/models/gaps/mlp_gap80_best.pt` |
| **training data** | `kitti_train.npz`: gap 80, **step 20**, 985 pairs, **36,179 samples** | `kitti_train_gap80.npz`: gap 80, **step 100**, 201 pairs, **7,446 samples** |
| val AUC | 0.895 | 0.863 |
| precision_epi @gap80 | **0.359** (@0.5) | **0.348** |

So yes — **not only the evaluation config differs, also the training data** (pair step 20 vs 100
→ ~5× more samples for the canonical model, hence the higher val AUC and test precision). The
gap sweep uses step 100 for *all* gaps so the six tags are comparable to each other; the
canonical M1/M2 model used step 20. Within each table the comparison is apples-to-apples.

### Canonical gap-80 config (`kitti_test_config.yaml`, 263 pairs, 15 methods, thresh sweep)

| method | precision_epi | time ms/pair |
|---|---|---|
| sift (RANSAC only) | 0.111 | ~430 |
| reldepth_0.4 (best handcrafted) | 0.120 | ~350 |
| learned_linear @0.5 | 0.231 | ~289 |
| learned_mlp @**0.3** | 0.212 | ~290 |
| learned_mlp @**0.5** | **0.359** | ~290 |
| learned_mlp @**0.7** | **0.514** | ~290 |
| learned_cnn @0.5 | 0.314 | ~287 |
| sift_oracle (GT filter) | 1.000 | ~267 |

Threshold sweep: raising the decision threshold trades recall for precision (0.212/0.359/0.514
at 0.3/0.5/0.7).

### Gap sweep — precision_epi, model × eval gap (condensed; full 16×6 in TRAINING.md)

| model \ gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift | 0.998 | 0.963 | 0.899 | 0.819 | 0.417 | 0.111 |
| mlp_gap5 | 0.999 | 0.983 | 0.960 | 0.904 | 0.783 | **0.529** |
| mlp_gap80 | 0.999 | 0.979 | 0.939 | 0.876 | 0.639 | 0.348 |
| **mlp_all** | 0.998 | 0.971 | 0.928 | 0.878 | **0.778** | 0.524 |
| cnn_all | 0.998 | 0.977 | 0.930 | 0.889 | 0.679 | 0.323 |
| sift_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Keep counts, precision, recall, F1 (filter stage = pre-RANSAC; confusion vs GT label)

- `keep_ratio` = filter_kept / putative; `keep_vs_oracle` = filter_kept / oracle's kept
  (= #correct putative) — the natural oracle-relative keep measure.
- At gap 80: putative 8,383, correct (oracle-kept) **385**. sift keeps all (keep_vs_oracle
  **21.8**); **mlp_all keeps 399 → keep_ratio 0.048, keep_vs_oracle 1.04** (≈ only correct
  matches); RANSAC then outputs 1,720 (sift) vs 168 (mlp_all) final matches.
- Filter-stage confusion (TP/FP/TN/FN) and F1: near-oracle at gap 1 (mlp_all F1 **0.981**:
  TP 200,540 / FP 2,110 / TN 4,465 / FN 5,453), but at gap 80 F1 ≈ **0.28** (TP 111 / FP 288 /
  TN 7,710 / FN 274) — filters trade recall for precision; the accuracy win appears after
  RANSAC. `filter_precision`/`filter_recall`/`filter_f1` and counts are columns in
  `summary.csv`; oracle-relative recall = `filter_recall` (correct-kept / correct-putative).

## Inference time (hardware: NVIDIA GTX 1660 6 GB, torch 2.6.0+cu124, CUDA 12.4)

Device placement: DA3 depth → GPU; learned filter → GPU (~15–18 ms/pair); RootSIFT +
matching + RANSAC → CPU (OpenCV). **Times are online batch-size-1 latency, not batched
throughput** (`bench_inference.py`, 30 TEST pairs, gap 80, depth computed online):

| method | depth | describe | match | filter | RANSAC | total |
|---|---|---|---|---|---|---|
| RANSAC-only (no depth needed) | 0 | 225 | 21 | 0 | 177 | **424 ms/pair** |
| mlp_all @0.5 (online depth) | 589 | 225 | 21 | 18 | 0.2 | **854 ms/pair** |

With cached depth (offline pipeline): **~285 ms/pair filtered vs ~430 ms/pair RANSAC-only**
(filter shrinks RANSAC's input ~20×). So: filters are faster when depth is already available,
slower when depth must be computed online for filtering alone.

## Note: RANSAC-1px training labels (experiment)

Re-labeling with "survives 1 px RANSAC" (`--label-metric ransac1px`, full sweep in
`gaps_ransac/`) does **not** improve accuracy (mlp_all @gap80: 0.241 vs 0.524) — RANSAC-inlier
≠ GT-correct. GT-epipolar labels remain the training signal. Details: TRAINING.md.

## Reproduce

```bash
export KITTI_ODOMETRY_PATH=/path/to/kitti_odometry/dataset   # contains sequences/ and poses/

# 1. dataset generation (TRAIN only)
python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml --seed 0
# gap sweep (one file per gap):
for g in 1 5 10 20 50 80; do
  python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml \
      --gaps $g --step 100 --out geomatch/data/training/kitti_train_gap$g.npz --seed 0
done

# 2. training (GPU, seeded; tag = training gaps)
python -m geomatch.train_filter --data geomatch/data/training/kitti_train.npz --seed 0
for g in 1 5 10 20 50 80; do
  python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap$g.npz \
      --tag gap$g --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0
done
python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap{1,5,10,20,50,80}.npz \
    --tag all --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0

# 3. evaluation (TEST 09–10; canonical gap 80, or sweep with --gap)
python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_config.yaml
for g in 1 5 10 20 50 80; do
  python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_gapsweep_config.yaml --gap $g
done

# 4. online inference-time benchmark (batch size 1, hardware-dependent)
python -m geomatch.bench_inference --pairs 30 --gap 80
```
