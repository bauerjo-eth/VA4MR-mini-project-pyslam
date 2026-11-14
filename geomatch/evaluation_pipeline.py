"""
Evaluation pipeline: run the methods from a config file and save the results.

Run from the repository root:
    python -m geomatch.evaluation_pipeline                              # evaluation_config.yaml
    python -m geomatch.evaluation_pipeline --config path/to/other.yaml

For every image pair, and every method (descriptor + matcher + filter) in the config:
  1. compute descriptors in both images and match them
  2. filter the matches
  3. grade the kept matches against the ground-truth camera motion

Results are saved to geomatch/results/<name>_<timestamp>/:
  config.yaml    exact copy of the config used
  per_pair.csv   one row per (pair, method)
  summary.csv    one row per method
  log.txt        everything printed during the run
"""

import argparse
import csv
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from geomatch.depth import predict_depth
from geomatch.descriptors import compute_descriptors, depth_at
from geomatch.evaluation_metrics import epipolar_error, reprojection_error
from geomatch.filtering import MatchContext, filter_matches
from geomatch.matchers import match_descriptors

DEFAULT_CONFIG = Path(__file__).parent / "evaluation_config.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
log = logging.getLogger("geomatch")


# ---------- setup ----------

def setup_output(cfg, config_path):
    """Create the results folder, copy the config into it, and log to console + log.txt."""
    out_dir = RESULTS_DIR / f"{cfg['name']}_{datetime.now():%Y-%m-%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt", encoding="utf-8")):
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


# ---------- one method on one pair ----------

def run_method(method, img1, img2, depth1, depth2, T1, T2, K, correct_px):
    """Describe, match, filter and grade. Returns the counts for this pair and the
    epipolar errors of the kept matches."""
    t0 = time.perf_counter()

    d_type, d_params = split_type(method["descriptor"])
    pts1, des1 = compute_descriptors(d_type, img1, depth1, **d_params)
    pts2, des2 = compute_descriptors(d_type, img2, depth2, **d_params)

    m_type, m_params = split_type(method["matcher"])
    i1, i2 = match_descriptors(m_type, des1, des2, **m_params)
    i1, i2 = np.asarray(i1, dtype=int), np.asarray(i2, dtype=int)

    f_type, f_params = split_type(method["filter"])
    z1, z2 = depth_at(depth1, pts1), depth_at(depth2, pts2)
    ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2, des1=des1, des2=des2,
                       depth1=depth1, depth2=depth2, z1=z1, z2=z2, K=K,
                       T1=T1, T2=T2)  # T1/T2 may only be used by the reference filter gt_epipolar
    keep = filter_matches(ctx, method=f_type, **f_params)

    time_ms = 1000 * (time.perf_counter() - t0)

    # grading (not timed): compare the kept matches with the true camera motion
    k1, k2 = i1[keep], i2[keep]
    epi = epipolar_error(pts1[k1], pts2[k2], T1, T2, K)
    rep = reprojection_error(pts1[k1], pts2[k2], z1[k1], z2[k2], T1, T2, K)
    counts = dict(putative=len(i1), kept=len(k1),
                  correct_epi=int((epi < correct_px).sum()),
                  correct_reproj=int((rep < correct_px).sum()),
                  time_ms=round(time_ms, 1))
    return counts, epi


# ---------- results ----------

def summarize(rows, epi_errors):
    """One row per method: totals over all pairs, precision and median epipolar error."""
    summary = []
    for name, errors in epi_errors.items():
        r = [row for row in rows if row["method"] == name]
        kept = sum(x["kept"] for x in r)
        e = np.concatenate(errors)
        summary.append({
            "method": name,
            "putative": sum(x["putative"] for x in r),
            "kept": kept,
            "precision_epi": round(sum(x["correct_epi"] for x in r) / kept, 4) if kept else 0.0,
            "precision_reproj": round(sum(x["correct_reproj"] for x in r) / kept, 4) if kept else 0.0,
            "median_epi_px": round(float(np.median(e)), 3) if len(e) else float("nan"),
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
    correct_px = cfg["grading"]["correct_px"]
    log.info(f"{cfg['name']}: {len(pairs)} pairs x {len(cfg['methods'])} methods -> {out_dir}")

    rows, epi_errors = [], {m["name"]: [] for m in cfg["methods"]}
    for n, (f1, f2) in enumerate(pairs, 1):
        t0 = time.perf_counter()
        depth1 = predict_depth(frames[f1], cfg["depth"]["model"])
        depth2 = predict_depth(frames[f2], cfg["depth"]["model"])
        log.info(f"pair {n}/{len(pairs)}: frames {f1}-{f2} (depth {time.perf_counter() - t0:.1f} s)")

        for method in cfg["methods"]:
            counts, epi = run_method(method, frames[f1], frames[f2], depth1, depth2,
                                     poses[f1], poses[f2], K, correct_px)
            rows.append({"frame1": f1, "frame2": f2, "method": method["name"], **counts})
            epi_errors[method["name"]].append(epi)

    summary = summarize(rows, epi_errors)
    write_csv(out_dir / "per_pair.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    log.info(f"\nSUMMARY (a match is correct if its error < {correct_px} px)")
    log_table(summary)
    log.info(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
