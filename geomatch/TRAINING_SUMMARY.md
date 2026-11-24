# TRAINING_SUMMARY — learned match filters: overview

What was trained, on exactly which data, tested on what, and the key numbers.
Full detail: `TRAINING.md` (provenance, all matrices, timings). Raw outputs: `results/`.
Every model checkpoint also stores its own provenance (`train_data`, `train_sequences`,
`train_n_samples`, `train_n_pairs`, `train_gaps`, `train_tag`).

## TL;DR

- **Ground truth**: a match is "correct" if its GT epipolar error < 4 px. Used as training
  label and as the main eval metric (`precision_epi`, post-RANSAC). RANSAC (1 px essential
  matrix) always runs after every filter.
- **Winner**: MLP trained on **all gaps** (`mlp_all`, and denser `mlp_all20` below) — best
  val AUC and most robust across eval gaps. At the canonical gap-80 test, the best single
  model is the MLP trained on gap 5 (`mlp_gap5`, 0.529 vs 0.111 RANSAC-only).
- Learned filters only matter at **wide baselines** (gap ≥ 50).
- **Speed**: with cached depth ~285 vs ~430 ms/pair; online (DA3 depth per frame, GPU) the
  filtered pipeline is ~2× slower (854 vs 424 ms/pair) because depth (~589 ms) dominates and
  RANSAC-only never needs depth.

## Model registry — which model was trained on WHAT, tested on WHAT

All training data: KITTI odometry **TRAIN split = sequences 00–08 only** (never 09–10; enforced
in code). All evaluation: **TEST split = sequences 09–10 only** (263 pairs @gap80, step 10).
Matcher everywhere: RootSIFT + `nn_ratio(0.8)`. Label: GT epipolar < 4 px unless noted.

| model file | trained on (dataset file · gaps · step · pairs · samples) | val AUC | tested on | headline result |
|---|---|---|---|---|
| `data/models/linear_best.pt` | `kitti_train.npz` · gap 80 · step 20 · 985 pairs · 36,179 samples | 0.843 | TEST 09–10, `kitti_test_config.yaml`, gap 80 | precision_epi 0.231 @0.5 |
| `data/models/mlp_best.pt` | `kitti_train.npz` · gap 80 · step 20 · 985 pairs · 36,179 samples | 0.895 | same | **0.359 @0.5**, 0.514 @0.7 |
| `data/models/cnn_best.pt` | `kitti_train.npz` · gap 80 · step 20 · 985 pairs · 36,179 samples | 0.863 | same | 0.314 @0.5 |
| `data/models/gaps/mlp_gap{1,5,10,20,50,80}_best.pt` | `kitti_train_gap{g}.npz` · gap g · step 100 · ~201–206 pairs · 7,446–41,200 samples each | 0.840–0.892 | TEST 09–10, `kitti_test_gapsweep_config.yaml --gap G` for G in 1,5,10,20,50,80 | best @gap80: **mlp_gap5 0.529** |
| `data/models/gaps/mlp_all_best.pt` | the six `kitti_train_gap*.npz` (gaps 1,5,10,20,50,80 · step 100) · 1,230 pairs · **149,276 samples** | **0.924** | same sweep | 0.524 @gap80, **0.778 @gap50** (best) |
| `data/models/gaps/cnn_all_best.pt` | same six files · 149,276 samples | 0.910 | same sweep | 0.323 @gap80 |
| `data/models/gaps_dense/mlp_all20_best.pt` | `kitti_train_dense_gap*.npz` · gaps 1,5,10,20,50,80 · **step 20** · ~6,100 pairs · ~TBD samples | TBD | TEST 09–10, `kitti_test_dense_config.yaml`, gaps 50/80 | TBD |
| `data/models/gaps_ransac/*` | same gap files but `--label-metric ransac1px` labels | ≤0.933 | sweep config `_ransac` | worse than GT labels (mlp_all 0.241 @gap80) — see note |

⚠️ `mlp_best.pt` (step 20, 36,179 samples) and `gaps/mlp_gap80_best.pt` (step 100, 7,446
samples) are **different models** — 0.359 vs 0.348 @gap80 is mostly the 5× training data, not a
contradiction. Within each table the comparison is apples-to-apples.

## Results on TEST 09–10

### Canonical gap-80 test (`kitti_test_config.yaml`, 263 pairs, thresh sweep on MLP)

| method | precision_epi | ms/pair |
|---|---|---|
| sift (RANSAC only) | 0.111 | ~430 |
| best handcrafted (reldepth_0.4) | 0.120 | ~350 |
| learned_linear @0.5 | 0.231 | ~289 |
| learned_mlp @0.3 / **0.5** / 0.7 | 0.212 / **0.359** / 0.514 | ~290 |
| learned_cnn @0.5 | 0.314 | ~287 |
| sift_oracle (GT filter) | 1.000 | ~267 |

### Gap sweep matrix (precision_epi; 16 methods × 6 eval gaps, condensed)

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift | 0.998 | 0.963 | 0.899 | 0.819 | 0.417 | 0.111 |
| mlp_gap5 | 0.999 | 0.983 | 0.960 | 0.904 | 0.783 | **0.529** |
| mlp_gap80 | 0.999 | 0.979 | 0.939 | 0.876 | 0.639 | 0.348 |
| **mlp_all** | 0.998 | 0.971 | 0.928 | 0.878 | **0.778** | 0.524 |
| cnn_all | 0.998 | 0.977 | 0.930 | 0.889 | 0.679 | 0.323 |
| sift_oracle | 1.000 | … | … | … | 1.000 | 1.000 |

### Gap sweep — filter F1 (pre-RANSAC confusion vs GT label; condensed)

| model \ eval gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift (no filter) | 0.984 | 0.924 | 0.836 | 0.675 | 0.264 | 0.088 |
| mlp_gap5 | 0.881 | 0.856 | 0.815 | 0.730 | 0.597 | 0.313 |
| mlp_gap80 | 0.877 | 0.850 | 0.826 | 0.772 | 0.596 | 0.311 |
| **mlp_all** | **0.981** | **0.911** | **0.844** | 0.743 | 0.583 | 0.283 |
| cnn_all | 0.965 | 0.884 | 0.833 | 0.737 | 0.536 | 0.238 |
| sift_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

(reading: at small gaps the mixed model is near-oracle; at gap 80 every filter's F1 is ~0.3 —
precision is sacrificed for recall, and the accuracy win appears only after RANSAC.)

### Gap sweep — POST-RANSAC (what VO gets): precision_epi / recall / F1 per gap

| model \ gap | 1 | 5 | 10 | 20 | 50 | 80 |
|---|---|---|---|---|---|---|
| sift P / R / F1 | .998/.730/.843 | .963/.873/.916 | .899/.873/.886 | .818/.873/.845 | .417/.731/.531 | .111/.496/.181 |
| mlp_gap5 P / R / F1 | .999/.549/.709 | .983/.676/.801 | .960/.638/.767 | .904/.573/.702 | .783/.442/.565 | **.529/.257/.346** |
| mlp_all P / R / F1 | .998/.708/.829 | .971/.773/.861 | .928/.707/.802 | .878/.605/.716 | **.778/.437/.560** | .524/.229/.318 |
| cnn_all P / R / F1 | .998/.684/.811 | .977/.731/.836 | .930/.699/.798 | .889/.614/.727 | .679/.487/.567 | .323/.252/.283 |
| oracle P / R / F1 | 1/0.73/0.85 | 1/0.87/0.93 | 1/0.88/0.93 | 1/0.90/0.95 | 1/0.75/0.86 | 1/.506/.672 |

(reading: post-RANSAC recall is capped by RANSAC itself — even the GT oracle only recalls 0.51
of correct matches at gap 80. Filters raise precision at the cost of recall; mlp_gap5 wins on
post-RANSAC F1 at gap 80, mlp_all is most robust across gaps.)

## Keep counts & confusion (filter stage = pre-RANSAC)

- `keep_ratio` = kept/putative; `keep_vs_oracle` = kept / oracle-kept (= #correct putative).
- @gap80: putative 8,383, correct 385. sift keeps all (keep_vs_oracle **21.8**, 1,720 final
  after RANSAC); **mlp_all keeps 399 → keep_ratio 0.048, keep_vs_oracle 1.04**, 168 final.
- Filter F1 (vs GT label): mlp_all **0.981 @gap1** (TP 200,540 / FP 2,110 / TN 4,465 /
  FN 5,453) but **0.283 @gap80** (TP 111 / FP 288 / TN 7,710 / FN 274) — filters trade recall
  for precision; the win shows up after RANSAC. Columns `filter_tp/fp/tn/fn`,
  `filter_precision/recall/f1` in `summary.csv`.

## Inference time (hardware: GTX 1660 6 GB, torch 2.6.0+cu124, CUDA 12.4; batch size 1, online)

DA3 depth → GPU (~589 ms/pair), learned filter → GPU (~15–18 ms), SIFT/match/RANSAC → CPU:

| method (30 TEST pairs, gap 80) | depth | describe | match | filter | RANSAC | total |
|---|---|---|---|---|---|---|
| RANSAC-only (needs no depth) | 0 | 225 | 21 | 0 | 177 | **424 ms** |
| mlp_all @0.5 (online depth) | 589 | 225 | 21 | 18 | 0.2 | **854 ms** |

Cached-depth pipeline: ~285 ms (filtered) vs ~430 ms (RANSAC-only). Times are per-frame
latency at batch size 1, not batched throughput.

## Note: RANSAC-1px training labels

`--label-metric ransac1px` (full sweep in `gaps_ransac/`): does **not** improve accuracy —
RANSAC-inlier ≠ GT-correct (mlp_all @gap80: 0.241 vs 0.524 with GT labels). GT-epipolar labels
stay the training signal. Details: TRAINING.md.

## Reproduce

```bash
export KITTI_ODOMETRY_PATH=/path/to/kitti_odometry/dataset   # contains sequences/ and poses/

# dataset generation (TRAIN 00–08 only; TEST 09–10 is refused by the guard)
python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml --seed 0          # canonical gap80 step20
for g in 1 5 10 20 50 80; do                                                                            # gap sweep, step 100
  python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml \
      --gaps $g --step 100 --out geomatch/data/training/kitti_train_gap$g.npz --seed 0
  python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml \
      --gaps $g --step 20 --out geomatch/data/training/kitti_train_dense_gap$g.npz --seed 0             # dense sweep
 done

# training (GPU, seed 0; --tag names the training gaps; provenance stored in each checkpoint)
python -m geomatch.train_filter --data geomatch/data/training/kitti_train.npz --seed 0                 # canonical
for g in 1 5 10 20 50 80; do
  python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap$g.npz \
      --tag gap$g --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0
done
python -m geomatch.train_filter --data geomatch/data/training/kitti_train_gap{1,5,10,20,50,80}.npz \
    --tag all --models mlp cnn --out-dir geomatch/data/models/gaps --seed 0
python -m geomatch.train_filter --data geomatch/data/training/kitti_train_dense_gap{1,5,10,20,50,80}.npz \
    --tag all20 --models mlp cnn --out-dir geomatch/data/models/gaps_dense --seed 0

# evaluation (TEST 09–10 only; --gap picks the eval gap)
python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_config.yaml                     # canonical gap80
for g in 1 5 10 20 50 80; do
  python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_gapsweep_config.yaml --gap $g
done
python -m geomatch.evaluation_pipeline --config geomatch/kitti_test_dense_config.yaml --gap 80      # dense models

# online inference benchmark (batch size 1)
python -m geomatch.bench_inference --pairs 30 --gap 80
```
