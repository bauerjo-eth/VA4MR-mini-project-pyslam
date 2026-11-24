# Phase 1 benchmark

Clean evaluation of every combination of

    backbone (keypoints + visual descriptor)  x  depth descriptor  x  match filter

on KITTI odometry, the same way for every method:

    descriptor = [visual | w * depth]  ->  matcher  ->  filter  ->  RANSAC  ->  grading

One command trains whatever is missing and writes a report:

    python -m geomatch.bench.run geomatch/experiments/phase1.yaml

Output: `geomatch/results/phase1_<timestamp>/report.md` (tables A1...D2 as in the Phase 1
diagram, plus a change-vs-baseline table), `pose_auc20.png`, `grid.csv` (all metrics per
method and gap), `per_pair.csv`, `log.txt`.

Letters: **A** = SIFT + geometric, **B** = SIFT + learned, **C** = SuperPoint + geometric,
**D** = SuperPoint + learned depth descriptor; **1** = with filter, **2** = without filter.

## Where to put your idea

You only edit `geomatch/components/` and the experiment file. Everything else (data,
ground truth, training loops, caching, train/test split, evaluation, report) is fixed.

| I want to try... | write | then in the experiment file |
|---|---|---|
| a hand-designed depth descriptor | `@geometric("name")` function in `components/depth_descriptors.py` | `my_desc: {type: name, w: 0.3}` under `depth_descriptors` |
| a new CNN for the learned depth descriptor | `@network("name")` class in `components/depth_descriptors.py` | `my_cnn: {learned: name, input: depth, dim: 64}` (+ the same with `input: image` as control) |
| different input channels for the CNN | `@input_channels("name")` in `components/depth_descriptors.py` | `input: name` |
| a hand-designed filter (WP3) | `@register("name")` in `geomatch/filtering.py` | `my_filter: {type: name, ...}` under `filters` |
| a learned filter (WP3) | `@learned_filter("name")` class (features + model) in `components/filters.py` | `my_filter: {learned: name, thresh: 0.5}` |
| another keypoint detector / visual descriptor | `@backbone("name")` in `components/backbones.py` | `name: {n: 2000}` under `backbones` |

Examples of each are already there (log-depth grid, depth-SIFT, `cnn_small`, `cnn_tiny`,
`depth`/`image`/`depth+image` inputs, David's MLP filter).

## What happens automatically

- **Learned depth descriptors** are trained per backbone on the TRAIN sequences (00-07),
  the best epoch is chosen on VAL (08). Ground truth: keypoints of image A, lifted with
  scale-corrected Depth Anything depth and moved with the true pose, must find their
  partner in image B as nearest neighbour (InfoNCE). The visual descriptor is frozen; the
  network, the weight w and the temperature are learned.
- **Learned filters** are trained per backbone x depth descriptor (descriptor distances
  differ between combinations), on the matches of that combination, label = GT epipolar
  error < 4 px, best epoch on VAL.
- **Models are cached** in `geomatch/data/bench/models/`, named by a hash of everything that
  defines them (component spec, training settings, sequences). Change a setting -> a new
  model is trained; nothing stale is reused. Changed the *code* of a network or filter?
  Use `--retrain <name>`.
- **Keypoints** of every frame are cached per backbone in `geomatch/data/bench/keypoints/`.
- **Test split guard**: models refuse to be evaluated on sequences they were trained or
  validated on; the experiment file refuses TEST sequences in TRAIN/VAL.
- **Same matcher for every method** (`matcher:` in the experiment file), so differences come
  from the descriptors and filters only.
- **Reference rows**: each backbone without depth descriptor (baseline), and a GT oracle
  filter (keeps exactly the correct matches) as upper bound.

## Useful options

    --quick 3                       smoke test: 3 test anchor frames per sequence
    --depth none log_grid           only some depth descriptors (the baseline is always kept)
    --filters none learned_mlp      only some filters
    --backbones superpoint          only some backbones
    --gaps 10 20                    other test gaps
    --retrain learned_depth         retrain after changing a network's code
    --no-train                      evaluate only what is already trained

For a quick development loop, copy the experiment file, shrink `training` (fewer epochs,
larger `step`) and the sequences, and run with `--quick`.

## Files

| file | role |
|---|---|
| `run.py` | entry point: train missing models, evaluate, report |
| `common.py` | experiment file, KITTI sequences, caches, model paths |
| `describe.py` | builds `[visual \| w * depth]`; wrapper turning a network into a learned descriptor |
| `train_descriptor.py` | GT correspondences + InfoNCE training of learned depth descriptors |
| `train_filter.py` | match collection + training of learned filters; filter `bench_learned` |
| `evaluate.py` | runs the grid through `evaluation_pipeline.run_method` (same grading as all other evaluations) |
| `report.py` | tables, figure, CSVs |

The older entry points (`evaluation_pipeline.py` with a config, `train_filter.py`,
`train_depth_descriptor.py`) still work unchanged.
