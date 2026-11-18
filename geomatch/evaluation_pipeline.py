"""
Evaluation pipeline: run the methods from a config file and save the results.

Run from the repository root:
    python -m geomatch.evaluation_pipeline                              # evaluation_config.yaml
    python -m geomatch.evaluation_pipeline --config path/to/other.yaml

For every image pair, and every method in the config, the matches go through

    descriptor -> matcher -> filter (optional, the WP3 part) -> RANSAC (always) -> grade

RANSAC is the same for every method, as in a SLAM front end: it fits the camera motion
(essential matrix) to the matches and keeps only the inliers. Grading compares the final
matches and the estimated camera motion with the ground truth.

Results are saved to geomatch/results/<name>_<timestamp>/:
  config.yaml    exact copy of the config used
  per_pair.csv   one row per (pair, method)
  summary.csv    one row per method
  log.txt        everything printed during the run
"""

import argparse
import csv
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from geomatch.cache_depth_pictures import cache_depth_maps, load_depth
from geomatch.descriptors import compute_descriptors, depth_at
from geomatch.evaluation_metrics import (epipolar_error, pose_auc, relative_pose_error,
                                         reprojection_error)
from geomatch.filtering import MatchContext, filter_matches
from geomatch.matchers import match_descriptors

DEFAULT_CONFIG = Path(__file__).parent / "evaluation_config.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
log = logging.getLogger("geomatch")


# ---------- setup ----------

def setup_output(cfg, config_path):
    """Create the results folder and copy the config into it. Logging: log.info() goes to the
    console and log.txt, log.debug() (details) only to log.txt."""
    out_dir = RESULTS_DIR / f"{cfg['name']}_{datetime.now():%Y-%m-%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    log.setLevel(logging.DEBUG)
    console, logfile = logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt", encoding="utf-8")
    console.setLevel(logging.INFO)
    for handler in (console, logfile):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    return out_dir


def load_dataset(ds):
    """Returns the image pairs [(frame1, frame2)], the frames {index: RGB image},
    the 4x4 camera-to-world poses (one per frame) and the 3x3 intrinsics K."""
    p = ds["pairs"]
    pairs = [(s, s + p["gap"]) for s in range(p["start"], p["stop"], p["step"])]
    frames = read_frames(ds["video"], {i for pair in pairs for i in pair})

    poses_3x4 = np.loadtxt(ds["poses"]).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(poses_3x4), 1, 1))
    poses[:, :3] = poses_3x4

    k = ds["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    return pairs, frames, poses, K


def read_frames(video, idxs):
    """Read selected frames (RGB) by decoding the video sequentially (seeking is unreliable)."""
    cap, out, i = cv2.VideoCapture(video), {}, 0
    while len(out) < len(idxs):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"{video}: could not read frames {sorted(idxs - out.keys())}")
        if i in idxs:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    return out


def split_type(block):
    """Config block -> (registry name, parameters), e.g. {type: x, w: 1} -> ("x", {"w": 1})."""
    params = dict(block)
    return params.pop("type"), params


def describe(method):
    """Readable combination, e.g. 'rootsift_depth(w=0.3) + nn_ratio(ratio=0.8) + none'."""
    def part(block):
        name, params = split_type(block)
        args = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{name}({args})" if args else name
    return " + ".join(part(method[k]) for k in ("descriptor", "matcher", "filter"))


def cached(cache, key, compute):
    """Return (result, seconds it took to compute). The first call with a key runs compute()
    and stores the result; later calls with the same key reuse it and report the same time."""
    if key not in cache:
        t0 = time.perf_counter()
        result = compute()
        cache[key] = (result, time.perf_counter() - t0)
    return cache[key]


# ---------- RANSAC stage (same for every method) ----------

def ransac_relative_pose(p1, p2, K, threshold_px=1.0, confidence=0.99999):
    """Fit an essential matrix to the matches with RANSAC, keep its inliers, and recover the
    camera motion. Returns R, t (None if it failed) and a boolean inlier mask over the matches.
    Inliers must also lie in front of both cameras."""
    inliers = np.zeros(len(p1), dtype=bool)
    if len(p1) < 5:  # an essential matrix needs at least 5 matches
        return None, None, inliers
    cv2.setRNGSeed(0)  # same result regardless of what ran before
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=confidence, threshold=threshold_px)
    if E is None:
        return None, None, inliers

    best_n, best_R, best_t = 0, None, None
    for E_i in np.split(E, len(E) // 3):  # RANSAC can return up to 3 candidate solutions
        n, R, t, mask_i = cv2.recoverPose(E_i, p1, p2, K, mask=mask.copy())
        if n > best_n:
            best_n, best_R, best_t, inliers = n, R, t, mask_i.ravel() > 0
    return best_R, best_t, inliers


# ---------- one method on one pair ----------

def run_method(method, img1, img2, depth1, depth2, T1, T2, K, ransac_cfg, correct_px, cache):
    """descriptor -> matcher -> filter -> RANSAC, then grade. Returns the counts for this pair
    and the epipolar errors of the final matches.

    `cache` holds the descriptors and matches of the current pair: methods with the same
    descriptor config (and matcher config) reuse them instead of recomputing."""
    d_key = json.dumps(method["descriptor"], sort_keys=True)
    m_key = d_key + json.dumps(method["matcher"], sort_keys=True)

    d_type, d_params = split_type(method["descriptor"])
    (pts1, des1, pts2, des2), t_describe = cached(cache, d_key, lambda: (
        *compute_descriptors(d_type, img1, depth1, **d_params),
        *compute_descriptors(d_type, img2, depth2, **d_params)))

    m_type, m_params = split_type(method["matcher"])
    (i1, i2), t_match = cached(cache, m_key, lambda: tuple(
        np.asarray(i, dtype=int) for i in match_descriptors(m_type, des1, des2, **m_params)))

    t0 = time.perf_counter()
    f_type, f_params = split_type(method["filter"])
    if "ransac" in f_params:
        raise ValueError(f"method '{method['name']}': RANSAC is always applied by the pipeline, "
                         "remove 'ransac' from its filter config")
    z1, z2 = depth_at(depth1, pts1), depth_at(depth2, pts2)
    ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2, des1=des1, des2=des2,
                       depth1=depth1, depth2=depth2, z1=z1, z2=z2, K=K,
                       T1=T1, T2=T2)  # T1/T2 may only be used by the reference filter gt_epipolar
    keep = filter_matches(ctx, method=f_type, **f_params)
    k1, k2 = i1[keep], i2[keep]                     # matches after the filter

    R, t, inliers = ransac_relative_pose(pts1[k1], pts2[k2], K, **ransac_cfg)
    r1, r2 = k1[inliers], k2[inliers]               # final matches, after RANSAC
    t_filter_ransac = time.perf_counter() - t0

    # fair timing: shared work counts fully for every method that uses it
    time_ms = 1000 * (t_describe + t_match + t_filter_ransac)

    # grading (not timed): compare the matches and the camera motion with the ground truth
    epi_all = epipolar_error(pts1[i1], pts2[i2], T1, T2, K)  # every putative match
    epi_filter = epi_all[keep]                                # after the filter
    epi = epi_filter[inliers]                                 # final
    rep = reprojection_error(pts1[r1], pts2[r2], z1[r1], z2[r2], T1, T2, K)
    rot_err, trans_err = relative_pose_error(R, t, T1, T2)
    counts = dict(putative=len(i1), kept_by_filter=len(k1), final=len(r1),
                  correct_putative=int((epi_all < correct_px).sum()),
                  correct_filter=int((epi_filter < correct_px).sum()),
                  correct_final=int((epi < correct_px).sum()),
                  correct_reproj=int((rep < correct_px).sum()),
                  rot_err_deg=round(float(rot_err), 3),
                  trans_err_deg=round(float(trans_err), 3),
                  time_ms=round(time_ms, 1))
    return counts, epi


# ---------- results ----------

def summarize(rows, epi_errors, auc_thresholds):
    """One row per method, totalled over all pairs: match counts, precision, recall, median
    epipolar error, relative pose AUC and time. Unless named 'filter_...', columns describe
    the final matches (after RANSAC)."""
    def ratio(a, b):
        return round(a / b, 4) if b else 0.0

    summary = []
    for name, errors in epi_errors.items():
        r = [row for row in rows if row["method"] == name]
        final = sum(x["final"] for x in r)
        correct = sum(x["correct_final"] for x in r)
        correct_putative = sum(x["correct_putative"] for x in r)
        e = np.concatenate(errors)
        pose_errors = [max(x["rot_err_deg"], x["trans_err_deg"]) for x in r]
        aucs = pose_auc(pose_errors, auc_thresholds)
        summary.append({
            "method": name,
            "putative": sum(x["putative"] for x in r),
            "filter_kept": sum(x["kept_by_filter"] for x in r),
            "filter_recall": ratio(sum(x["correct_filter"] for x in r), correct_putative),
            "final": final,
            "correct": correct,
            "wrong": final - correct,
            "precision_epi": ratio(correct, final),
            "recall": ratio(correct, correct_putative),
            "precision_reproj": ratio(sum(x["correct_reproj"] for x in r), final),
            "median_epi_px": round(float(np.median(e)), 3) if len(e) else float("nan"),
            **{f"auc@{t}deg": round(a, 4) for t, a in zip(auc_thresholds, aucs)},
            "time_ms_per_pair": round(float(np.mean([x["time_ms"] for x in r])), 1),
        })
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def log_table(rows):
    cols = list(rows[0].keys())
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    log.info("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        log.info("  ".join(str(r[c]).rjust(w) for c, w in zip(cols, widths)))


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Run the geomatch evaluation from a config file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to a YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = setup_output(cfg, args.config)
    pairs, frames, poses, K = load_dataset(cfg["dataset"])
    grading = cfg["grading"]
    log.info(f"Running evaluation pipeline: {cfg['name']} ({len(pairs)} image pairs x {len(cfg['methods'])} methods)")
    log.info(f"Results: {out_dir}")
    depth_dir = cache_depth_maps(frames, cfg["depth"]["model"], cfg["dataset"]["name"])

    log.info("Methods (descriptor + matcher + filter, then RANSAC for all):")
    for method in cfg["methods"]:
        log.info(f"  {method['name']}: {describe(method)}")

    rows, epi_errors = [], {m["name"]: [] for m in cfg["methods"]}
    progress = tqdm(pairs, desc="Image pairs", unit="pair")
    for f1, f2 in progress:
        depth1, depth2 = load_depth(depth_dir, f1), load_depth(depth_dir, f2)
        log.debug(f"pair: frames {f1}-{f2}")

        cache = {}  # descriptors and matches of this pair, shared by the methods
        for method in cfg["methods"]:
            progress.set_postfix_str(f"running {method['name']}")
            counts, epi = run_method(method, frames[f1], frames[f2], depth1, depth2,
                                     poses[f1], poses[f2], K, cfg["ransac"], grading["correct_px"], cache)
            rows.append({"frame1": f1, "frame2": f2, "method": method["name"], **counts})
            epi_errors[method["name"]].append(epi)

    summary = summarize(rows, epi_errors, grading["pose_auc_deg"])
    write_csv(out_dir / "per_pair.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    log.info(f"\nSUMMARY (a match is correct if its epipolar error < {grading['correct_px']} px; "
             "columns describe the final matches after RANSAC unless named filter_...)")
    log_table(summary)
    log.info("recall = correct final matches / correct matches before filter and RANSAC | "
             "auc@Xdeg = relative pose accuracy (1.0 = every pair's camera motion recovered perfectly)")
    log.info(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
