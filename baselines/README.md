# WP1 – Baselines: matching, SLAM, depth, local geometry

WP1 asks for three things: appearance-based matching and SLAM baselines (ORB, SuperPoint, LightGlue), an evaluation
of the depth from geometric foundation models (Depth Anything 3, VGGT-Ω), and local geometric information per image
patch. This folder does all three on two datasets with ground truth: KITTI 06 (car, outdoors) and TUM fr1/desk
(handheld camera, indoors). The other work packages compare against these numbers.

**Results: [`RESULTS.md`](RESULTS.md)** (short summary at the top).

## What is where

| Part | Question | Start (repo root) | Code |
|---|---|---|---|
| SLAM | How far off is the whole trajectory? | `python main_slam_evaluation.py -c baselines/configs/slam_kitti06_video.json` | pySLAM; RTE: `slam_rte.py` |
| Matching | Can the matcher find the same point in two images? | `python baselines/matching_baseline.py baselines/configs/matching_kitti06_video.yaml` | `matching_baseline.py` |
| Depth + local geometry | How good are depth and surface normals from one image? | `python baselines/depth_baseline.py baselines/configs/depth_tum_fr1_desk.yaml` | `depth_baseline.py` |

Each part has one config per dataset in `configs/` (`*kitti06*`, `*tum_fr1_desk*`); results go to `results/`.
`--matchers ORB2 LIGHTGLUE` runs only some matchers. Another dataset: copy a config (any pySLAM dataset with GT
poses; depth: also GT depth, `sensor_type: rgbd`).

Reusable for WP2–WP4:

| File | What it does |
|---|---|
| `depth_models.py` | one interface for DA3 and VGGT-Ω: image in, depth map out (meters for DA3 metric) |
| `patch_geometry.py` | per keypoint: 3D point, surface normal (Open3D), viewing angle |
| `gt_depth.py` | where a keypoint truly lands in another image (GT depth + GT pose) |
| `kitti_lidar.py` | KITTI LiDAR scan -> depth image of the camera |

```python
from baselines.depth_models import make_depth_model  # repo root on sys.path (python -m ...)
from baselines.patch_geometry import compute_patch_geometry, pixel_rays

model = make_depth_model("DA3METRIC-LARGE", camera)  # camera: pySLAM camera_from_config
depth, conf = model.predict([img_rgb])[0]            # HxW, meters if model.metric, NaN = invalid
geo = compute_patch_geometry(depth, pixel_rays(camera), kps_xy)
geo["z"], geo["X"], geo["normal"], geo["view_angle"], geo["valid"]  # per keypoint
```

## Setup

- pySLAM Python env active, run from the repo root; Linux, or Windows with WSL2
- KITTI 06 video: already in the repo (`data/videos/kitti06/`)
- TUM fr1/desk (~350 MB) and KITTI raw LiDAR (4.5 GB zip, ~2 GB extracted): download commands in the headers of
  `configs/depth_tum_fr1_desk.yaml` and `configs/depth_kitti06_lidar.yaml` (into `datasets/`, not tracked);
  matching on KITTI without the LiDAR: remove `depth_check` from its config (line check only)
- Depth Anything 3 (no CUDA): `pip install --no-deps "git+https://github.com/ByteDance-Seed/depth-anything-3@ed6989a23cd389e975ed9f7cbd7385396e6d867e"`
  (runtime deps: `geomatch/requirements.txt`); Linux + CUDA: `scripts/install_depth_anything_v3.sh`
- VGGT-Ω: access request on https://huggingface.co/facebook/VGGT-Omega, `hf auth login`, then
  `pip install --no-deps "git+https://github.com/facebookresearch/vggt-omega@48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"`
  (checkpoint 4.6 GB, downloaded on first use; without access remove `VGGT-OMEGA` from the depth configs)

## How it works

### SLAM

- pySLAM's own evaluation tool runs each matcher 3 times on KITTI 06 (SLAM differs from run to run), with and without
  loop closing. **Only the matcher differs** between the runs:

  | Preset | pySLAM name | Matcher |
  |---|---|---|
  | `orb2` | `ORB2` | ORB features, brute-force matching |
  | `superpoint` | `SUPERPOINT` | SuperPoint, brute-force + ratio test |
  | `lightglue` | `LIGHTGLUE` | SuperPoint + LightGlue |

- loop closing is the same for all (`DBOW3_INDEPENDENT`: own ORB features)
- error measures: ATE (whole trajectory, after aligning it to the ground truth) from pySLAM, and RTE (drift over
  100–800 m driven) with the rpg_trajectory_evaluation toolbox of Zhang & Scaramuzza (unmodified copy in
  `thirdparty/`): `python baselines/slam_rte.py with_loop=results/eval_<date>/wp1_slam_kitti06 no_loop=...`
- pySLAM changed in two places only: its evaluation tool (`slam_evaluation_manager.py`) and a 3-line LightGlue fix
  (see Pitfalls)

### Matching

- image pairs: every 10th frame with the frame 1, 5 or 10 later (TUM: 5, 10, 20); the same matchers as in SLAM
- each match is checked twice against the ground truth:
  1. **line check:** the true camera motion allows only one line in the other image (epipolar line); a correct match
     lies on it. Blind along the line: a match to the next of several identical trees passes.
  2. **true-point check:** with measured depth (LiDAR / Kinect) the exact point is known; the match must be there.
     Done in both directions (image 1 -> 2 and 2 -> 1): a wrong measured depth at the keypoint looks like a wrong
     match in one direction only, a wrong match fails in both.
- the camera motion estimated from the matches (RANSAC) is compared with the true one: pose AUC as in SuperGlue /
  LightGlue
- `LIGHTGLUE_FULL`: LightGlue with all 9 layers (pySLAM uses 2), for comparison only

### Depth and local geometry

- each model predicts a depth map from one image; compared with the measured depth (standard metrics: AbsRel,
  δ < 1.25), on the whole image and at keypoints
- local geometry: normals from the depth map with Open3D's standard method (plane through the 30 nearest 3D points,
  Open3D's default; also 100 / 300 for comparison); for a point seen in two images the normal must be the same
  after the camera rotation
- models (`models:` in the config):

  | Name | Model | Meters? |
  |---|---|---|
  | `DA3METRIC-LARGE` | Depth Anything 3, metric | yes |
  | `DA3-LARGE-1.1` | Depth Anything 3, any-view (two images at once) | no |
  | `VGGT-OMEGA` | VGGT-Ω 1B (two images at once) | no |
  | `DA3MONO-LARGE` | Depth Anything 3, relative (not used in the results) | no |

## Values chosen here

Methods are existing tools or published protocols. Values without a standard:

| Choice | Value | Why |
|---|---|---|
| image pairs | every 10th frame; gap 1 / 5 / 10 (KITTI, 0.1–1 s), 5 / 10 / 20 (TUM, 0.17–0.67 s) | motions as between SLAM frames / keyframes; 10th frame: neighbours are near duplicates |
| correct match, line check | < 5 px in both images | above the GT noise (KITTI gap 10: RANSAC inliers ~1.2 px from the GT line; 2 px too strict) |
| correct match, true-point check | < 10 px from the true point | GT noise of the true point 1–4.5 px (median), no second mode below 10 px |
| point projected with predicted depth | < 5 px | same as the line check |
| occlusion check of a GT correspondence | GT depths agree within 5 % | not tuned |
| sparse LiDAR at a point | median in 5 x 5 px | not tuned |
| normals | Open3D default k = 30 neighbours, also 100 / 300 | k = 30 is Open3D's default; results reported for all three |
| full SLAM run | ≥ 99 % of the frames with a pose | runs that lose tracking cover another part of the route |

<details>
<summary>Outputs (files and columns)</summary>

**SLAM** `results/eval_<date>/<output_path>/`: `table_rmse.csv`, `table_max.csv`, `table_percent_lost.csv`,
`report.pdf`; per run `<preset>/<dataset>/iteration_<i>/`: trajectory (`plot/trajectory_final.json`), EVO plots and
stats, used `config.yaml` + `settings.yaml`, `logs/log.txt`.

**RTE** `results/wp1_slam_rte_<date>/`: `summary.csv` (mean ± std over the full runs, partial runs listed),
`runs.csv`, `segments.csv` (every sub-trajectory), `rte_<name>.png` (box plots), `info.txt`, `rpg/` (toolbox input,
results and log per run).

**Matching** `results/wp1_matching_<dataset>_<date>/`: `pairs.csv`, `summary.csv`, config copy, `info.txt`
(versions, GPU, device of each network).

| Column | Meaning |
|---|---|
| `n_matches` | matches returned by the matcher |
| `n_correct` / `precision` | line check passed (epipolar distance < `epipolar_threshold_px` in both images) |
| `n_depth`, `n_depth_epi`, `n_depth_correct` | matches with a true point / ... passing the line check / ... also within `threshold_px` of the true point |
| `depth_err_med_px` | median distance to the true point of the matches on the line (mostly GT noise) |
| `n_both`, `n_both_wrong_fwd`, `n_both_wrong_both` | matches on the line with a true point in both directions / ... not at the true point in image 2 / ... and not at the true point in image 1 either |
| `inlier_ratio` | essential matrix RANSAC inliers / matches |
| `rot_err_deg`, `trans_err_deg`, `pose_err_deg` | estimated vs true camera motion (translation: direction only); pose error = max of both |
| `time_ms` | detection (both images) + matching; first pair per matcher untimed (warm-up) |
| `summary.csv` | per matcher and gap: means (time: median), pose AUC@5/10/20°; true-point check pooled: `depth_coverage`, `precision_epi_on_depth`, `precision_depth`; both directions: `both_coverage`, `hidden_wrong_fwd`, `hidden_wrong_both` (share of the matches on the line that are wrong: image 1 -> 2 only / confirmed in both directions) |

**Depth + local geometry** `results/wp1_depth_<dataset>_<date>/`: `depth_frames.csv`, `depth_summary.csv`,
`patch_pairs.csv`, `patch_summary.csv`, `normals_knn.csv`, config copy, `info.txt`.

| Column | Meaning |
|---|---|
| `region`, `align` | `dense` = all pixels with GT, `kp_<detector>` = at keypoints; `metric` = as predicted, `median` = scale matched per frame |
| `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `delta1..3` | standard monocular depth metrics |
| `scale_mean`, `scale_cv` | median(GT) / median(pred) per frame (~1 = right meters) and its variation over frames |
| `points` | GT correspondences per pair |
| `normal_err_med` | normal from predicted vs GT depth, same view (dense GT only) |
| `xview_gt_med`, `xview_pred_med` | normal of frame i rotated into frame i + gap vs the normal there (GT / predicted depth, same points; dense GT only) |
| `xview_pred_all_med` | same for predicted depth, all correspondences (KITTI) |
| `reproj_med`, `reproj_aligned_med` (+ `_acc`) | keypoint moved with predicted depth + GT pose: px to the true point (as predicted / scale matched; share < 5 px) |
| `depth_ratio_err_med` | error of `log(z_i+gap / z_i)` predicted vs GT (metric or two-image models) |
| `normals_knn.csv` | the normal columns for every k in `normal_knn` (`patch_summary.csv`: first k) |

`save_depth: True` in a depth config saves the predicted depth per frame (reuse in WP2 / WP3 without the model).

</details>

<details>
<summary>Pitfalls</summary>

**pySLAM and SLAM**
- pySLAM's C++ core loses tracking on our machine (WSL2 + AMD GPU) in most runs -> Python core:
  `PYSLAM_USE_CPP=false python main_slam_evaluation.py -c ...` (~6 min per run). It needs the 3-line LightGlue fix in
  `pyslam/slam/tracking.py`, `pyslam/slam/geometry_matchers.py`, `pyslam/local_features/feature_matcher.py`
  (in this branch; without it LightGlue runs crash or hang)
- evaluation tool: per-dataset `template_vars` (video path, file name, timestamps) and preset `"config"` sections
  added in `slam_evaluation_manager.py`; preset overrides go under `custom_parameters` (plain `settings` in a preset
  is silently ignored); keys already in the settings file can't be overridden by a preset (e.g. `nFeatures`)
- global parameters per preset / for all presets: `"config": {"GLOBAL_PARAMETERS": {...}}` in `custom_parameters` /
  `common_parameters`
- other dataset: new entry in `datasets`; video datasets need `template_vars`, standard datasets not, e.g.
  `"dataset_type": "KITTI_DATASET"`, `"datasets": [{"name": "06", "settings_path": "settings/KITTI04-12.yaml", "is_color": true}]`
- quick check: copy of the config with `number_of_runs_per_dataset: 1`
- check the number of poses per run (`sse / rmse²` in `plot/stats_final.json`): `percent_lost` counts only frames
  in state LOST and misses runs that lost their map (~0.1 %)
- without loop closing there is no relocalization: tracking loss is final, ATE only over the part before -> not
  comparable (KITTI 06: LightGlue lost at frame 393 / 512 in 2 of 3 runs)
- keyframe rate depends on machine speed (new keyframe only when local mapping is idle): a slow front-end
  (networks on the CPU) makes many keyframes, loop detection falls behind and can miss the loop -> report the machine
- KITTI 06 here is the H.264 video, not the original PNGs; results depend on the OpenCV version (4.13)
- `report.html` fails with pandas >= 3; CSV + PDF unaffected
- RTE toolbox needs `colorama` + `pyyaml` (in the pySLAM env); its `plot_utils.py` (LaTeX) is not used

**Matching**
- the line check is blind along the line -> its precision is an upper bound; the true-point check closes the gap
  where GT depth exists (KITTI: LiDAR only in the lower image part, ≤ 80 m -> not a random sample of the matches)
- a wrong GT depth at the keypoint (sparse LiDAR at object borders) moves the true point along the epipolar line,
  like a wrong match -> `precision_depth` alone counts such GT errors as wrong matches; `hidden_wrong_both` counts
  only matches that fail from both images (lower bound), `hidden_wrong_fwd` is the one-direction value on the same
  matches
- KITTI GT pose noise grows with the gap -> re-check the 5 px threshold for other datasets
- gap 1: small baseline -> translation direction poorly constrained -> low pose AUC is expected
- frames are read in one pass (video datasets can't seek reliably)
- times: networks on CUDA if available (AMD via ROCm also shows as `cuda`), else CPU -> compare only within one machine
- reproducibility: SuperPoint, LightGlue and RANSAC give identical results; pySLAM's ORB2 keypoints change slightly
  between calls (pose AUC up to ±0.07 at gap 1)

**Depth models**
- DA3 metric gives meters only after `* f / 300` (f = focal at DA3's processing size, long side 504 px; TUM check:
  GT / raw 1.45, `f / 300` 1.36). pySLAM's DA3 wrapper and `geomatch/depth.py` skip this factor (TUM x1.36 too
  small; KITTI x0.97)
- pySLAM's DA3 wrapper gets BGR images, DA3 expects RGB (TUM: AbsRel +2.5 %) -> `depth_models.py` calls DA3 directly
- VGGT-Ω center-crops images wider than 2:1 (KITTI: middle 740 of 1226 px, NaN outside) -> all models compared on
  the same pixels
- AMD GPU under WSL2: no fast fp16 / bf16 kernels -> DA3 and VGGT-Ω in fp32 (default on ROCm; `DEPTH_FP32=1` / `0`)
- TUM fr1: strong lens distortion -> per-pixel rays with distortion removed (`pixel_rays`)
- GT side: Kinect depth noisy at edges, depth and RGB up to 20 ms apart -> `xview_gt_med` = noise floor of the GT
- KITTI LiDAR: odometry 06 = raw `2011_09_30_drive_0020` (same frame index); sparse (~4 % of the pixels) -> median in
  5 x 5 px, no GT normals; LiDAR sits above / behind the camera (background points at object borders, as in the
  Eigen split); GT poses are camera 0, images camera 2 (6.6 cm apart) -> mm-level error, ignored

</details>
