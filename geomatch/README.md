# geomatch

Does depth from a foundation model help local feature matching?

`geomatch` is a small, standalone prototype that tests this idea. It predicts depth with
**Depth Anything 3 (DA3)**, uses it to improve or filter SIFT matches, and checks the result
against ground-truth camera poses. It does not depend on pyslam.

Part of the VAMR mini-project *Foundation-Model-Guided Appearance and Local Geometry Matching
for Robust Visual SLAM*.

| Work package | Topic | Owner |
|---|---|---|
| WP1 | Appearance baselines, depth evaluation | Jonas |
| WP2 | Geometry-aware descriptors | Gregor |
| WP3 | Geometric match filtering | David |
| WP4 | SLAM integration, experiments | Jonas |

## What exists so far

### 1. Prototype (Gregor, WP2)
- **Appearance:** RootSIFT keypoints and descriptors, matched with a nearest-neighbour ratio test.
- **Depth:** DA3 predicts a depth map for each image.
- **Depth descriptor:** for every keypoint, sample the depth on a 5x5 grid around it and store
  `log(depth / center depth)`. This describes the local 3D shape and does not depend on the
  absolute depth scale.
- **Combined descriptor:** `[SIFT | w * depth descriptor]`. With `w = 0` it is plain SIFT.

### 2. Evaluation and filters (David, WP3)
- **Metrics:** each match is checked against the ground-truth camera motion. A match is
  *correct* if its error is below 4 px.
  - Using the ground-truth pose only: epipolar distance, symmetric epipolar distance, Sampson
    distance, triangulation reprojection error.
  - Also using the predicted depth: symmetric reprojection error.
- **Filters:** decide which matches to keep after matching. Available:

  | Filter | What it does |
  |---|---|
  | `none` | keep everything (baseline) |
  | `depth_valid` | drop matches without valid depth |
  | `desc_threshold` | drop matches whose descriptor distance is too large |
  | `relative_depth_threshold` | drop matches whose depths differ too much between the two images |
  | `gt_epipolar` | **reference only:** uses the ground-truth pose to show the best possible filter |

  Each filter can also be combined with RANSAC, run before or after it.

## Files

| File | Content |
|---|---|
| `core.py` | SIFT detection, matching, depth descriptor |
| `depth.py` | DA3 depth prediction |
| `filtering.py` | filter registry and built-in filters |
| `demo.py` | evaluation on KITTI 06 (prints all tables) |
| `requirements.txt` | Python environment |
| `data/` | generated data, e.g. cached depth maps (not in git) |
| `results/` | evaluation outputs (not in git) |

## Run it

From the repository root:

```bash
python -m geomatch.demo          # print the evaluation tables
python -m geomatch.demo --show   # also plot one image pair
```

The demo uses 10 image pairs from KITTI 06 (`data/videos/kitti06/` in the repo root), each
10 frames apart. It compares the depth weights `w = 0, 0.1, 0.3, 0.5` and all configured filters.

Setup instructions are at the top of `requirements.txt`.

## Add your own filter

Write a function in `filtering.py` that returns a boolean mask over the matches
(`True` = keep), register it, and add it to `FILTER_CONFIGS` in `demo.py`:

```python
@register("my_filter")
def my_filter(ctx, my_param=1.0, **params):
    ...
    return keep_mask
```

`ctx` (a `MatchContext`) holds the keypoints, descriptors, depth maps and camera intrinsics.
Do not use the ground-truth poses `ctx.T1` or `ctx.T2` in a real filter. They are only there
for the reference filter.
