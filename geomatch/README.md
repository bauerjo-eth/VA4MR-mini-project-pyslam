# geomatch

Does depth from a foundation model help local feature matching?

`geomatch` is a small, standalone prototype that tests this idea. It predicts depth with
**Depth Anything 3 (DA3)**, uses it to improve or filter SIFT matches, and checks the result
against ground-truth camera poses. It does not depend on pyslam.

Part of the VAMR mini-project *Foundation-Model-Guided Appearance and Local Geometry Matching
for Robust Visual SLAM*.

| Work package | Topic | Owner |
|---|---|---|
| WP1 | Appearance baselines, depth evaluation | Jonas & Robin |
| WP2 | Geometry-aware descriptors | Gregor |
| WP3 | Geometric match filtering | David |
| WP4 | SLAM integration, experiments | Jonas & Robin |

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

### 3. Evaluation pipeline (Jonas & Robin, WP1/WP4)
- A config file lists the **methods** to compare. Each method is a descriptor + a matcher + a
  filter, chosen by name.
- `evaluation_pipeline.py` runs every method on every image pair, grades the matches, and
  saves the results. Nothing in the code has to change to try a new combination.
- Descriptors, matchers and filters each live in their own file, together with a **registry**:
  a dictionary from name to function, which is how the config finds them.

## Files

| File | Content |
|---|---|
| `descriptors.py` | descriptor algorithms (RootSIFT, depth descriptor) and their registry |
| `matchers.py` | matcher algorithms (ratio test) and their registry |
| `filtering.py` | filter algorithms and their registry |
| `evaluation_metrics.py` | grading: error of each match w.r.t. the true camera motion |
| `evaluation_pipeline.py` | runs a config, saves the results |
| `evaluation_config.yaml` | default config: dataset, grading threshold, methods to compare |
| `depth.py` | DA3 depth prediction |
| `demo.py` | the original evaluation script (kept unchanged) |
| `core.py` | only re-imports the moved functions so `demo.py` keeps working |
| `requirements.txt` | Python environment (setup instructions at the top) |
| `data/` | generated data, e.g. cached depth maps (not in git) |
| `results/` | evaluation outputs (not in git) |

## Run it

From the repository root, with the environment activated:

```bash
python -m geomatch.evaluation_pipeline                          # uses evaluation_config.yaml
python -m geomatch.evaluation_pipeline --config my_config.yaml  # another experiment
```

Each run creates `results/<name>_<timestamp>/` with:

| File | Content |
|---|---|
| `config.yaml` | exact copy of the config used |
| `summary.csv` | one row per method: matches, share correct, median error, time per pair |
| `per_pair.csv` | the same numbers for every image pair |
| `log.txt` | everything printed during the run |

The original demo still works: `python -m geomatch.demo` (add `--show` to plot one pair).

## Add your own descriptor, matcher or filter

1. Write a function in the matching file and register it under a name.
2. Use that name as `type` in a config. All other keys in that block are passed to your
   function as parameters.
3. Run the pipeline and compare against the `sift` baseline in `summary.csv`.

**Descriptor** in `descriptors.py`. It gets the image and its depth map, and returns the
keypoints and their descriptors:

```python
@register_descriptor("my_descriptor")
def my_descriptor(img, depth, my_param=1.0):
    ...
    return pts, des            # Nx2 pixel coordinates, NxD float32 descriptors
```

**Matcher** in `matchers.py`. It gets the descriptors of both images and returns which ones
belong together:

```python
@register_matcher("my_matcher")
def my_matcher(des1, des2, my_param=1.0):
    ...
    return i1, i2              # des1[i1[k]] is matched to des2[i2[k]]
```

**Filter** in `filtering.py`. It gets all match data and returns which matches to keep:

```python
@register("my_filter")
def my_filter(ctx, my_param=1.0, **params):
    ...
    return keep_mask           # boolean, one entry per match, True = keep
```

`ctx` (a `MatchContext`) holds the keypoints, descriptors, depth maps and camera intrinsics.
Do not use the ground-truth poses `ctx.T1` or `ctx.T2` in a real filter. They are only there
for the reference filter.

Then, in the config:

```yaml
methods:
  - name: my_test
    descriptor: {type: my_descriptor, my_param: 2.0}
    matcher:    {type: my_matcher}
    filter:     {type: my_filter, ransac: after}
```

## To-do

- [ ] **Different image pair creation algorithm.** Pairs are currently fixed frame offsets
  (every 100th frame, partner 10 frames later). Pick pairs by their true camera motion instead,
  so the difficulty is controlled.
- [ ] **Flowchart in this README** explaining the approach: images → depth and keypoints →
  descriptor → matcher → filter → grading, and where WP1–WP3 plug in.
- [ ] **`cache_depth_pictures.py`**: run once per dataset to save all depth maps to `data/`,
  so research runs don't repeat the Depth Anything 3 inference.
