"""
Report of a grid run: CSV files, a markdown report with the A1...D2 tables, and a figure.

Naming as in the Phase 1 diagram:
  A = SIFT + geometric depth descriptor      B = SIFT + learned depth descriptor
  C = SuperPoint + geometric depth descr.    D = SuperPoint + learned depth descriptor
  1 = with a depth-based filter              2 = without filter
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomatch.bench.common import is_learned  # noqa: E402

# validated categorical slots 1-3 (dataviz reference palette, light mode) + text/surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def letter(exp, backbone, depth, filt):
    spec = exp["depth_descriptors"][depth]
    if spec is None:
        return "baseline"
    row = {"sift": "AB", "superpoint": "CD"}.get(backbone, "??")[1 if is_learned(spec) else 0]
    if filt == "oracle":
        return ""
    return row + ("2" if exp["filters"][filt].get("type") == "none" else "1")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v, metric):
    return f"{int(v)}" if metric in ("final", "putative") else f"{v:.3f}"


def table(exp, summ, metric, gaps, delta=False):
    filters = list(exp["filters"])
    idx = {(s["gap"], s["backbone"], s["depth"], s["filter"]): s for s in summ}
    head = ["backbone", "depth descriptor"] + [f"gap {g} · {f}" for g in gaps for f in filters]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for b in exp["backbones"]:
        for d in exp["depth_descriptors"]:
            if delta and exp["depth_descriptors"][d] is None:
                continue
            cells = []
            for g in gaps:
                for f in filters:
                    s = idx.get((g, b, d, f))
                    if s is None:
                        cells.append("")
                        continue
                    v = s[metric]
                    tag = letter(exp, b, d, f)
                    if delta:
                        base = idx[(g, b, "none", f)][metric]
                        txt = f"{v - base:+.3f}"
                        txt = f"**{txt}**" if v - base >= 0.01 else txt
                    else:
                        txt = fmt(v, metric)
                    cells.append(f"{txt} {tag}" if tag not in ("", "baseline") else txt)
            lines.append(f"| {b} | {d} | " + " | ".join(cells) + " |")
        if not delta and exp.get("oracle", True):
            cells = []
            for g in gaps:
                s = idx.get((g, b, "none", "oracle"))
                cells += [fmt(s[metric], metric) if s else ""] + [""] * (len(filters) - 1)
            lines.append(f"| {b} | *GT oracle filter (upper bound)* | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(exp, summ, gaps, path, metric="auc@20deg"):
    backbones, depths, filters = list(exp["backbones"]), list(exp["depth_descriptors"]), list(exp["filters"])
    idx = {(s["gap"], s["backbone"], s["depth"], s["filter"]): s for s in summ}
    fig, axes = plt.subplots(len(backbones), len(gaps), figsize=(4.2 * len(gaps), 3.0 * len(backbones)),
                             squeeze=False, facecolor=SURFACE)
    width = 0.8 / len(filters)
    for r, b in enumerate(backbones):
        for c, g in enumerate(gaps):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            for k, f in enumerate(filters):
                xs = [i + (k - (len(filters) - 1) / 2) * width for i in range(len(depths))]
                ys = [idx[(g, b, d, f)][metric] if (g, b, d, f) in idx else 0 for d in depths]
                ax.bar(xs, ys, width, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                       label=f if (r, c) == (0, 0) else None, zorder=2)
            o = idx.get((g, b, "none", "oracle"))
            if o:
                ax.axhline(o[metric], color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=3)
                ax.text(len(depths) - 0.5, o[metric], " GT oracle", color=INK2, fontsize=8,
                        va="bottom", ha="right")
            ax.set_ylim(0, 1.05)  # AUC range, the same in every panel
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels(depths, rotation=25, ha="right", fontsize=8, color=INK2)
            ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, lw=1, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(INK2)
            ax.set_title(f"{b} · gap {g}", fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(metric.replace("deg", "°").replace("auc@", "pose AUC@"), fontsize=9, color=INK2)
    fig.legend(loc="upper center", ncol=len(filters), frameon=False, fontsize=9, title="filter",
               title_fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(exp, out_dir, rows, summ, gaps, info):
    out_dir = Path(out_dir)
    write_csv(out_dir / "per_pair.csv", rows)
    write_csv(out_dir / "grid.csv", summ)
    figure(exp, summ, gaps, out_dir / "pose_auc20.png")
    m = exp["matcher"]
    md = [f"# {exp['name']}: grid report", "",
          f"Test sequences {exp['data']['test']} (models trained on {exp['data']['train']}, "
          f"best epoch picked on {exp['data']['val']}). Matcher for every method: `{m}`. "
          f"RANSAC {exp['ransac']}. A match is correct if its GT epipolar error < "
          f"{exp['grading']['correct_px']} px.", "",
          "Letters: A = SIFT + geometric, B = SIFT + learned, C = SuperPoint + geometric, "
          "D = SuperPoint + learned; 1 = with filter, 2 = without. `learned_image` is the "
          "control (same network on the image instead of depth).", "",
          "![pose AUC@20](pose_auc20.png)", "",
          "## Pose AUC@20° (main metric)", "", table(exp, summ, "auc@20deg", gaps), "",
          "## Change vs. the same backbone without depth descriptor (pose AUC@20°, same filter and gap)",
          "", "Bold: at least +0.01.", "", table(exp, summ, "auc@20deg", gaps, delta=True), "",
          "## Pose AUC@10°", "", table(exp, summ, "auc@10deg", gaps), "",
          "## Precision after the filter, before RANSAC (share of kept matches that are correct)", "",
          table(exp, summ, "filter_precision", gaps), "",
          "## Final matches after RANSAC (total over all pairs)", "", table(exp, summ, "final", gaps), "",
          "## Precision of the final matches", "", table(exp, summ, "precision_epi", gaps), "",
          "## Trained models", ""]
    md += [f"- {k}: {v}" for k, v in info.items()]
    md += ["", "All columns per method and gap: `grid.csv`; per pair: `per_pair.csv`."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")
