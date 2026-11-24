# WP2: learned depth descriptor on top of frozen SuperPoint (first overnight run, 2026-11-24)

## Question
Is Depth Anything 3 (DA3) depth **complementary** to a learned visual descriptor (SuperPoint),
or **redundant** because both are learned from images?

## Method
    descriptor = [ SuperPoint 256-d (frozen) | w * f_theta(input) ]   (64-d learned branch)

- `f_theta`: small fully convolutional net (8 conv blocks, about 0.6M parameters) on scale-free
  channels of the input (x/y gradients and local contrast of log depth), read out at the
  SuperPoint keypoints. One network learns theta, the weight `w` and the temperature `tau`.
- Loss: symmetric InfoNCE on `s_ij = sp_i·sp_j + w²·d_i·d_j`. The true partner must be the
  nearest neighbour among all keypoints of the other image.
- **Control:** the same network and training, but the input is the grey image instead of DA3
  depth (`--mode image`). If depth > image, depth adds something that SuperPoint plus extra
  image capacity does not.
- Labels: KITTI ground-truth pose plus DA3 depth, scale-corrected per sequence (DA3 is 5–13%
  too small, which matches Robin's finding). SuperPoint keypoints are projected into the other
  frame and kept if: nearest keypoint within 3 px, occlusion check, mutual, epipolar error < 2 px.
- Data:
  - train on KITTI 00–07: 2926 pairs, gaps 5/10/20/40;
  - validate on 08: 400 pairs;
  - test on 09–10, which were never touched during training.

## Results

### Validation (seq 08): share of GT matches whose nearest neighbour is correct

| | all | gap 10 | gap 20 |
|---|---|---|---|
| SuperPoint alone | 0.8587 | 0.798 | 0.771 |
| + learned **depth** branch (best epoch 5) | 0.8756 | 0.824 | 0.797 |
| + learned **image** branch (control, best epoch 6) | **0.8775** | 0.824 | 0.798 |

### Test (seq 09–10): full pipeline, mutual NN, RANSAC, pose AUC

| method | gap10 AUC@10° | gap10 AUC@20° | gap20 AUC@10° | gap20 AUC@20° | gap10 epi precision |
|---|---|---|---|---|---|
| SIFT + ratio 0.8 | **0.859** | **0.917** | **0.595** | **0.662** | 0.898 |
| SuperPoint | 0.736 | 0.823 | 0.296 | 0.398 | 0.887 |
| SuperPoint + hand-made depth grid (w=0.3) | 0.749 | 0.828 | 0.303 | 0.389 | 0.884 |
| SuperPoint + learned depth | 0.785 | 0.861 | 0.348 | 0.434 | 0.885 |
| SuperPoint + learned image (control) | 0.777 | 0.854 | 0.349 | 0.438 | 0.888 |
| SuperPoint + GT-epipolar oracle filter | 0.893 | 0.945 | 0.880 | 0.929 | 1.000 |

At gap 80 every method fails (AUC@20° ≈ 0.02), so that row is not informative.

### Test again, same ratio test (0.8) for every method (`wp2_kitti_test_ratio_gap*`)
SuperPoint methods: mutual NN + ratio 0.8. SIFT methods: NN + ratio 0.8. Pose AUC (09–10):

| method | gap10 @10° | gap10 @20° | gap20 @10° | gap20 @20° |
|---|---|---|---|---|
| SIFT | 0.859 | 0.916 | 0.595 | 0.662 |
| 1: SIFT + log-depth grid (w=0.3) | 0.862 | 0.914 | 0.597 | 0.663 |
| 2: SIFT + SIFT on log depth (w=0.3, Jonas) | **0.874** | **0.924** | 0.581 | 0.651 |
| 2: same, w=0.6 | 0.844 | 0.900 | 0.464 | 0.542 |
| SuperPoint | 0.845 | 0.916 | 0.509 | 0.601 |
| SuperPoint + log-depth grid (w=0.3) | 0.846 | 0.919 | 0.464 | 0.564 |
| 3: SuperPoint + learned depth | 0.818 | 0.898 | 0.355 | 0.441 |
| SuperPoint + learned image (control) | 0.826 | 0.901 | 0.418 | 0.501 |
| SuperPoint + GT oracle filter | 0.860 | 0.928 | 0.617 | 0.708 |

With the same ratio test, plain SuperPoint gets as good as SIFT at gap 10. No depth method
gives a clear gain. The learned branches now **hurt**: the ratio test keeps fewer matches with
them (gap 20: 10k vs 14k). Those matches are more precise, but too few for RANSAC. The branch
was trained for nearest-neighbour accuracy, not for the ratio test, and ratio 0.8 is not
equally strict in different descriptor spaces. A fair next step: pick the ratio per method on
the validation sequence 08.

### Final: exactly the same matcher for every method
Configs: `wp2_kitti_test_nnratio_all.yaml` (NN + ratio 0.8) and
`wp2_kitti_test_mutualratio_all.yaml` (mutual NN + ratio 0.8). Pose AUC@20° on 09–10:

| method | ratio, gap10 | ratio, gap20 | mutual+ratio, gap10 | mutual+ratio, gap20 |
|---|---|---|---|---|
| SIFT | 0.916 | 0.662 | 0.927 | **0.689** |
| 1: SIFT + log-depth grid (w=0.3) | 0.914 | **0.663** | 0.932 | 0.674 |
| 2: SIFT + depth-SIFT (w=0.3) | **0.924** | 0.651 | **0.933** | 0.681 |
| 2: same, w=0.6 | 0.900 | 0.542 | 0.905 | 0.578 |
| SuperPoint | 0.915 | 0.585 | 0.916 | 0.601 |
| SuperPoint + log-depth grid | 0.915 | 0.546 | 0.919 | 0.564 |
| 3: SuperPoint + learned depth | 0.900 | 0.410 | 0.898 | 0.441 |
| SuperPoint + learned image (control) | 0.877 | 0.503 | 0.901 | 0.501 |
| SuperPoint + GT oracle filter | 0.920 | 0.716 | 0.928 | 0.708 |

- **Depth raises match precision every time** (for example, mutual+ratio at gap 20: SIFT 55% →
  58–59%, SuperPoint 66% → 67–70%), but it also removes correct matches, so the pose does not improve.
- Methods 1 and 2: at most +0.008 at gap 10 and −0.008 to −0.015 at gap 20. This is noise or a
  small loss, not a gain. A higher depth weight (w=0.6) always hurts.
- Method 3 is below plain SuperPoint under both matchers. With the ratio test it keeps 30% fewer matches.
- At gap 10 every method is close to the oracle (0.928), so the pose metric has little room to show a gain.
- **Conclusion:** on KITTI, DA3 depth in the descriptor (hand-designed or learned) does not improve
  relative pose. Whether it reduces the "right line, wrong object" errors needs the LiDAR
  true-point check (WP1).

## Interpretation (first run, mutual NN without ratio test)
1. Both learned branches clearly improve SuperPoint: about +2 pts of NN accuracy on val, and
   +4–5 pts of pose AUC on test.
2. **Depth and image control are equal** on val and test, within noise. The gain therefore
   comes from extra learned capacity adapted to KITTI, not from information that only DA3
   depth provides. This supports the **redundant** hypothesis: DA3 depth is itself a function
   of the image, and a small CNN on the image recovers what it adds.
3. The hand-designed log-depth grid adds almost nothing (+1 pt).
4. SuperPoint with plain mutual NN loses to SIFT with the ratio test. The ratio test is the
   reason: only 37% of SuperPoint putatives lie on the true epipolar line, vs 72% for SIFT.
   This is a matcher issue, not a descriptor issue. Before comparing against SIFT, add a ratio
   test or LightGlue for SuperPoint.

## Caveats
- Labels depend on DA3 depth (+ GT pose), which gives about 80 verified matches per pair. That
  may bias labels towards places where DA3 is right. A LiDAR or GT-depth check would be cleaner.
- One seed per model. The differences between depth and image (≤ 0.01) are below the
  likely seed noise.
- Both models overfit after epochs 5–6; the best checkpoint by val is used.

## Possible next steps
- `mutual_nn` + ratio test (or LightGlue) for all SuperPoint variants, then re-run the test.
- A **depth + image** branch: if it beats both, depth is partly complementary after all.
- Robin's "right epipolar line, wrong object" error metric: depth should help most there
  even if the average numbers do not move.
- Seeds 1–2 for error bars.

## Reproduce
    PY=~/.python/venvs/pyslam/bin/python
    $PY -m geomatch.collect_descriptor_data --sequences 00 01 02 03 04 05 06 07 08   # ~11 min
    $PY -m geomatch.train_depth_descriptor --mode depth                            # ~1.5 h on M-series GPU
    $PY -m geomatch.train_depth_descriptor --mode image                            # ~1.6 h
    $PY -m geomatch.evaluation_pipeline --config geomatch/wp2_kitti_test_config.yaml --gap 10
or everything at once: `PY=$PY bash geomatch/run_wp2.sh`.
Models: `geomatch/data/models/wp2_{depth,image}_best.pt`. Logs: `geomatch/data/wp2_run.log`.
