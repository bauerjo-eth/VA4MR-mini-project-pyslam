"""WP1 SLAM baselines: relative trajectory error (RTE) of finished pySLAM evaluation runs.

Computed with the rpg_trajectory_evaluation toolbox (uzh-rpg; Zhang & Scaramuzza, IROS 2018), unmodified copy in
baselines/thirdparty/rpg_trajectory_evaluation:
- input: the trajectories pySLAM saved for its ATE (`plot/trajectory_final.json`: frame ids, estimated + GT poses,
  already associated) -> toolbox format (timestamp = frame id) in <output>/rpg/<name>/<preset>/<iteration>/
- toolbox: Sim(3) alignment over the whole run (mono, same as pySLAM's ATE), ATE, relative errors of
  sub-trajectories of 100, 200, ..., 800 m (KITTI lengths): a sub-trajectory starts at every pose and ends at the pose
  whose distance along the GT is closest to the length (within +-20 %); translation scaled with the global Sim(3)
  scale -> scale drift counts
- one number per run = mean over all sub-trajectories of all lengths: translation [% of the length],
  rotation [deg / 100 m]
New here: conversion to the toolbox format, mean per run, tables, box plots.
Runs that lose tracking: RTE only over their tracked part -> listed separately, not in the mean.

Usage (repo root, pySLAM env active), one eval output folder per argument, optional NAME=PATH:
    python baselines/slam_rte.py gpu_loop=results/eval_2026_11_21-20_47_12/wp1_slam_kitti06 ...
Output results/wp1_slam_rte_<date>/: segments.csv, runs.csv, summary.csv, rte_<name>.png, info.txt, rpg/ (toolbox
input + its saved results per run)
"""

import argparse
import contextlib
import csv
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOOLBOX = Path(__file__).resolve().parent / "thirdparty" / "rpg_trajectory_evaluation"
sys.path.insert(0, str(TOOLBOX))  # the toolbox imports its modules by plain name
warnings.filterwarnings("ignore", category=SyntaxWarning)  # toolbox: `suffix is not ''` (Python 2 code)
import transformations as rpg_tf  # noqa: E402  (toolbox copy: quaternion x y z w, as its file format)
from trajectory import Trajectory  # noqa: E402

SEGMENT_LENGTHS_M = [100, 200, 300, 400, 500, 600, 700, 800]  # KITTI lengths (toolbox default: 10-50 % of the run)
FULL_RUN_FRACTION = 0.99  # run counts as full if it has >= 99 % of the frames as poses


def write_stamped(path, stamps, T):
    """Toolbox file format: timestamp tx ty tz qx qy qz qw."""
    rows = [[s, *M[:3, 3], *rpg_tf.quaternion_from_matrix(M)] for s, M in zip(stamps, T)]
    np.savetxt(path, np.array(rows), fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")


def total_frames(iteration_dir):
    """num_total_frames from pySLAM's other_metrics_info.txt (None if missing)."""
    info = iteration_dir / "other_metrics_info.txt"
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("num_total_frames:"):
                return int(line.split(":")[1])
    return None


def evaluate_run(traj_json, run_dir):
    """Toolbox on one pySLAM run: Sim(3) alignment, ATE, relative errors. Returns (run info, segment rows)."""
    d = json.loads(traj_json.read_text())
    ids = np.array(d["trajectory_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_stamped(run_dir / "stamped_groundtruth.txt", ids, np.array(d["trajectory_gt"], dtype=float))
    write_stamped(run_dir / "stamped_traj_estimate.txt", ids, np.array(d["trajectory_est"], dtype=float))
    (run_dir / "eval_cfg.yaml").write_text("align_type: sim3\nalign_num_frames: -1\n")  # mono: all poses
    with open(run_dir / "toolbox_log.txt", "w") as log, contextlib.redirect_stdout(log):
        traj = Trajectory(str(run_dir), preset_boxplot_distances=SEGMENT_LENGTHS_M)
        traj.compute_absolute_error()
        traj.compute_relative_errors()

    segs = []
    for length in SEGMENT_LENGTHS_M:
        err = traj.rel_errors.get(length, {})
        for t_pct, r_deg_m in zip(err.get("rel_trans_perc", []), err.get("rel_rot_deg_per_m", [])):
            segs.append({"length_m": length, "t_err_pct": float(t_pct), "r_err_deg_100m": 100.0 * float(r_deg_m)})
    n_frames = total_frames(traj_json.parents[1])
    run = {
        "poses": len(ids),
        "total_frames": n_frames,
        "last_frame": int(ids[-1]),
        "full_run": n_frames is not None and len(ids) >= FULL_RUN_FRACTION * n_frames,
        "segments": len(segs),
        "t_err_pct": float(np.mean([s["t_err_pct"] for s in segs])) if segs else np.nan,
        "r_err_deg_100m": float(np.mean([s["r_err_deg_100m"] for s in segs])) if segs else np.nan,
        "ate_rmse_m": float(traj.abs_errors["abs_e_trans_stats"]["rmse"]),
        "ate_pyslam_m": float(d["ate"]),
        "sim3_scale": float(traj.scale),
    }
    return run, segs


def find_runs(eval_dir):
    """pySLAM layout: <eval_dir>/<preset>/<dataset>/iteration_<k>/plot/trajectory_final.json"""
    for p in sorted(eval_dir.glob("*/*/iteration_*/plot/trajectory_final.json")):
        yield p.parents[3].name, p.parents[2].name, p.parents[1].name, p


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def box_plot(path, name, segments, presets):
    """One box per (segment length, preset), segments of the full runs pooled (V10: box plots for RTE)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(presets)
    for k, preset in enumerate(presets):
        data = [[s["t_err_pct"] for s in segments if s["preset"] == preset and s["length_m"] == L]
                for L in SEGMENT_LENGTHS_M]
        pos = np.arange(len(SEGMENT_LENGTHS_M)) + (k - (len(presets) - 1) / 2) * width
        b = ax.boxplot(data, positions=pos, widths=0.9 * width, patch_artist=True, showfliers=False)
        for patch in b["boxes"]:
            patch.set_facecolor(f"C{k}")
        ax.plot([], [], color=f"C{k}", lw=8, label=preset)
    ax.set_xticks(np.arange(len(SEGMENT_LENGTHS_M)), [str(L) for L in SEGMENT_LENGTHS_M])
    ax.set_xlabel("segment length [m]")
    ax.set_ylabel("translation error [%]")
    ax.set_title(f"RTE {name} (full runs, outliers not shown)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dirs", nargs="+", help="pySLAM eval output folder, optionally NAME=PATH")
    args = ap.parse_args()

    out = Path("results") / f"wp1_slam_rte_{datetime.now():%Y_%m_%d-%H_%M_%S}"
    out.mkdir(parents=True)
    run_rows, seg_rows, summary = [], [], []

    for arg in args.eval_dirs:
        name, _, path = arg.rpartition("=")
        eval_dir = Path(path)
        name = name or f"{eval_dir.parent.name}/{eval_dir.name}"
        runs = list(find_runs(eval_dir))
        if not runs:
            raise SystemExit(f"no trajectory_final.json below {eval_dir}")
        for preset, dataset, iteration, traj_json in runs:
            run, segs = evaluate_run(traj_json, out / "rpg" / name.replace("/", "_") / preset / iteration)
            print(f"{name} {preset} {iteration}: t {run['t_err_pct']:.2f} %, r {run['r_err_deg_100m']:.2f} deg/100m")
            run_rows.append({"name": name, "preset": preset, "dataset": dataset, "iteration": iteration, **run})
            seg_rows += [{"name": name, "preset": preset, "iteration": iteration, "full_run": run["full_run"], **s}
                         for s in segs]

        presets = list(dict.fromkeys(r["preset"] for r in run_rows if r["name"] == name))
        for preset in presets:
            sel = [r for r in run_rows if r["name"] == name and r["preset"] == preset]
            full = [r for r in sel if r["full_run"]]
            row = {"name": name, "preset": preset, "full_runs": len(full), "runs": len(sel)}
            for key in ("t_err_pct", "r_err_deg_100m", "ate_rmse_m"):
                vals = [r[key] for r in full]
                row[f"{key}_mean"] = float(np.mean(vals)) if vals else np.nan
                row[f"{key}_std"] = float(np.std(vals)) if vals else np.nan  # population std, as pySLAM
            row["partial_runs"] = "; ".join(
                f"{r['iteration']}: {r['poses']} poses (last frame {r['last_frame']}), "
                f"t {r['t_err_pct']:.2f} %, r {r['r_err_deg_100m']:.2f} deg/100m"
                for r in sel if not r["full_run"])
            summary.append(row)
        full_segs = [s for s in seg_rows if s["name"] == name and s["full_run"]]
        if full_segs:
            box_plot(out / f"rte_{name.replace('/', '_')}.png", name, full_segs, presets)

    write_csv(out / "segments.csv", seg_rows)
    write_csv(out / "runs.csv", run_rows)
    write_csv(out / "summary.csv", summary)
    (out / "info.txt").write_text(
        f"date: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"inputs: {' '.join(args.eval_dirs)}\n"
        "toolbox: rpg_trajectory_evaluation (uzh-rpg), commit 8c8ceec, copy in baselines/thirdparty/\n"
        f"sub-trajectories: {SEGMENT_LENGTHS_M} m, start at every pose, end within +-20 % of the length (toolbox)\n"
        "alignment: Sim(3) over the whole run (toolbox, align_type sim3), as pySLAM's ATE\n"
        f"full run: >= {FULL_RUN_FRACTION:.0%} of the frames as poses; mean/std over full runs, std = population\n"
        f"python {platform.python_version()}, numpy {np.__version__}\n")

    w = max(len(r["name"]) for r in summary) + 2
    print(f"\n{'name':<{w}}{'preset':<12}{'full':>5}{'t_err %':>16}{'r_err deg/100m':>18}{'ATE m':>14}")
    for r in summary:
        print(f"{r['name']:<{w}}{r['preset']:<12}{r['full_runs']:>2}/{r['runs']:<2}"
              f"{r['t_err_pct_mean']:>9.2f} ± {r['t_err_pct_std']:<4.2f}"
              f"{r['r_err_deg_100m_mean']:>11.2f} ± {r['r_err_deg_100m_std']:<4.2f}"
              f"{r['ate_rmse_m_mean']:>8.1f} ± {r['ate_rmse_m_std']:<4.1f}")
        if r["partial_runs"]:
            print(f"{'':<{w + 12}}partial: {r['partial_runs']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
