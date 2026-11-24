"""
Evaluate the full grid on the TEST split: every backbone x depth descriptor x filter, at every
gap, plus a ground-truth oracle filter per backbone (upper bound).

Each grid cell goes through the existing pipeline code (evaluation_pipeline.run_method):
descriptor -> matcher -> filter -> RANSAC -> grading, so the numbers are computed exactly
like in the other evaluations. Descriptors are computed once per frame and shared by all
filters and gaps.
"""

import json
import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from geomatch.bench.common import desc_model_path, filter_model_path, is_learned, sequences
from geomatch.bench.describe import Describer, desc_device_free
from geomatch.evaluation_pipeline import run_method, summarize

log = logging.getLogger("geomatch")


def grid_methods(exp):
    """All methods of the grid. Each: name, backbone, depth, filter, and the run_method blocks."""
    out = []
    for b in exp["backbones"]:
        for d in exp["depth_descriptors"]:
            filters = list(exp["filters"].items())
            if exp.get("oracle", True) and d == "none":
                filters.append(("oracle", {"type": "gt_epipolar", "err_px": exp["grading"]["correct_px"]}))
            for f, spec in filters:
                if is_learned(spec):
                    fblock = {"type": "bench_learned", "thresh": spec.get("thresh", 0.5),
                              "model_path": str(filter_model_path(exp, b, d, f))}
                else:
                    fblock = dict(spec)
                out.append({"name": f"{b}+{d}+{f}", "backbone": b, "depth": d, "filter": f,
                            "descriptor": {"type": "bench", "backbone": b, "depth": d},
                            "matcher": exp["matcher"], "filter_block": fblock})
    return out


def check_no_leak(exp, methods):
    test = {s.name for s in sequences(exp, "test")}
    for m in methods:
        path = m["filter_block"].get("model_path")
        if path:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            leak = test & (set(ck["train_sequences"]) | set(ck["val_sequences"]))
            if leak:
                raise ValueError(f"{m['name']}: filter trained on TEST sequences {sorted(leak)}")


def evaluate(exp, gaps=None, max_pairs=None):
    """Returns rows (one per pair x method x gap) and per-gap summaries."""
    gaps = gaps or exp["data"]["test_pairs"]["gaps"]
    tp = exp["data"]["test_pairs"]
    methods = grid_methods(exp)
    check_no_leak(exp, methods)
    combos = sorted({(m["backbone"], m["depth"]) for m in methods})
    describers = {}
    for b, d in combos:
        spec = exp["depth_descriptors"][d]
        describers[(b, d)] = Describer(exp, b, d, desc_model_path(exp, b, d) if is_learned(spec) else None)
    log.info(f"[evaluate] {len(methods)} methods x gaps {gaps} on {exp['data']['test']}")

    rows = []
    epi = {(m["name"], g): [] for m in methods for g in gaps}
    t0 = time.time()
    for seq in sequences(exp, "test"):
        pairs = sorted({(f1, f2, g) for g in gaps for f1, f2 in seq.pairs(g, tp["step"], tp.get("start", 0))})
        if max_pairs:  # quick test: only the first max_pairs anchor frames
            anchors = set(sorted({p[0] for p in pairs})[:max_pairs])
            pairs = [p for p in pairs if p[0] in anchors]
        seq.ensure_depth({f for f1, f2, _ in pairs for f in (f1, f2)})
        window = {}
        for f1, f2, g in tqdm(pairs, desc=f"evaluate {seq.name}", unit="pair"):
            for f in [k for k in window if k < f1]:
                del window[f]
            for f in (f1, f2):
                if f not in window:
                    depth, img = seq.depth(f), seq.image(f)
                    entry = {"depth": depth, "img": img}
                    for key, dsc in describers.items():
                        t = time.perf_counter()
                        entry[key] = (dsc(seq, f, depth, img), time.perf_counter() - t)
                    window[f] = entry
            a, b = window[f1], window[f2]
            cache = {}
            for key in describers:
                (pts1, des1), t1 = a[key]
                (pts2, des2), t2 = b[key]
                d_key = json.dumps({"type": "bench", "backbone": key[0], "depth": key[1]}, sort_keys=True)
                cache[d_key] = ((pts1, des1, pts2, des2), t1 + t2)
            for m in methods:
                method = {"name": m["name"], "descriptor": m["descriptor"], "matcher": m["matcher"],
                          "filter": m["filter_block"]}
                counts, e = run_method(method, a["img"], b["img"], a["depth"], b["depth"],
                                       seq.poses[f1], seq.poses[f2], seq.K, exp["ransac"],
                                       exp["grading"]["correct_px"], cache)
                rows.append({"sequence": seq.name, "frame1": f1, "frame2": f2, "gap": g,
                             "method": m["name"], **counts})
                epi[(m["name"], g)].append(e)
            if len(rows) % 2000 < len(methods):
                desc_device_free()
    log.info(f"[evaluate] {len(rows)} runs in {time.time() - t0:.0f} s")

    summaries = []
    for g in gaps:
        rg = [r for r in rows if r["gap"] == g]
        s = summarize(rg, {m["name"]: epi[(m["name"], g)] for m in methods}, exp["grading"]["pose_auc_deg"])
        meta = {m["name"]: m for m in methods}
        for x in s:
            m = meta[x["method"]]
            summaries.append({"gap": g, "backbone": m["backbone"], "depth": m["depth"],
                              "filter": m["filter"], **x})
    return rows, summaries
