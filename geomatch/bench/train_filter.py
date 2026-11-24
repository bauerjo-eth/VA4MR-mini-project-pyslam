"""
Train learned match filters (WP3) for every descriptor combination of the experiment.

A filter sees the putative matches of one (backbone, depth descriptor) combination: their
descriptor distances, depths, positions. These differ between combinations, so each
combination gets its own filter, trained on its own matches (TRAIN split), with the best
epoch chosen on VAL. Label: GT epipolar error < grading.correct_px (as in David's training).

Also registers the filter `bench_learned` (geomatch.filtering) that evaluates such a model.
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from geomatch.bench.common import filter_model_path, sequences
from geomatch.bench.describe import Describer, desc_device_free
from geomatch.components.filters import LEARNED_FILTERS
from geomatch.descriptors import depth_at
from geomatch.evaluation_metrics import epipolar_error
from geomatch.filtering import MatchContext, register
from geomatch.matchers import match_descriptors

log = logging.getLogger("geomatch")


def match(exp, des1, des2):
    p = dict(exp["matcher"])
    i1, i2 = match_descriptors(p.pop("type"), des1, des2, **p)
    return np.asarray(i1, int), np.asarray(i2, int)


def collect(exp, jobs, split):
    """jobs: list of (backbone, depth_name, filter_name, desc_model_path). One pass over the
    frames of `split`; returns {job: (features, labels, pair_ids)}."""
    c = exp["training"]["filter"]
    rng = np.random.default_rng(0)
    combos = sorted({(b, d, p) for b, d, _, p in jobs}, key=str)
    describers = {(b, d): Describer(exp, b, d, p) for b, d, p in combos}
    feat_fns = {j: LEARNED_FILTERS[exp["filters"][j[2]]["learned"]]() for j in jobs}
    out = {j: ([], [], []) for j in jobs}
    pair_id = 0
    for seq in sequences(exp, split):
        pairs = sorted({p for g in c["gaps"] for p in seq.pairs(g, c["step"])})
        frames = sorted({f for p in pairs for f in p})
        seq.ensure_depth(frames)
        window = {}
        for f1, f2 in tqdm(pairs, desc=f"filter data {split} {seq.name}", unit="pair"):
            for f in [k for k in window if k < f1]:
                del window[f]
            for f in (f1, f2):
                if f not in window:
                    depth, img = seq.depth(f), seq.image(f)
                    window[f] = {"depth": depth,
                                 **{k: dsc(seq, f, depth, img, allow_trained=True)
                                    for k, dsc in describers.items()}}
            a, b = window[f1], window[f2]
            for (bb, dd), _ in describers.items():
                (pts1, des1), (pts2, des2) = a[(bb, dd)], b[(bb, dd)]
                i1, i2 = match(exp, des1, des2)
                if len(i1) < 5:
                    continue
                if len(i1) > c["max_matches_per_pair"]:
                    sel = np.sort(rng.choice(len(i1), c["max_matches_per_pair"], replace=False))
                    i1, i2 = i1[sel], i2[sel]
                ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2, des1=des1, des2=des2,
                                   depth1=a["depth"], depth2=b["depth"],
                                   z1=depth_at(a["depth"], pts1), z2=depth_at(b["depth"], pts2), K=seq.K)
                y = epipolar_error(pts1[i1], pts2[i2], seq.poses[f1], seq.poses[f2], seq.K) < exp["grading"]["correct_px"]
                for j in jobs:
                    if (j[0], j[1]) == (bb, dd):
                        out[j][0].append(feat_fns[j].features(ctx))
                        out[j][1].append(y.astype(np.float32))
                        out[j][2].append(np.full(len(y), pair_id))
            pair_id += 1
            if pair_id % 50 == 0:
                desc_device_free()
    return {j: tuple(np.concatenate(v) for v in vals) for j, vals in out.items()}


def auc(prob, y):
    order = np.argsort(prob)
    ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1))


def train_one(exp, job, train_data, val_data):
    backbone, depth_name, filter_name, _ = job
    spec = exp["filters"][filter_name]
    impl = LEARNED_FILTERS[spec["learned"]]()
    hp = impl.train
    (x, y, _), (xv, yv, _) = train_data, val_data
    torch.manual_seed(0)
    mean, std = x.mean(0), x.std(0) + 1e-8
    norm = lambda a: torch.from_numpy((a - mean) / std).float()
    X, Y, XV = norm(x), torch.from_numpy(y), norm(xv)
    model = impl.make_model(x.shape[1])
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])  # balance the classes
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    best, best_state, hist = -1, None, []
    for ep in range(hp["epochs"]):
        model.train()
        perm = torch.randperm(len(X))
        for k in range(0, len(X), hp["batch"]):
            idx = perm[k:k + hp["batch"]]
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(XV)).numpy()
        a = auc(pv, yv)
        hist.append(round(a, 4))
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    thresh = spec.get("thresh", 0.5)
    model.load_state_dict(best_state)
    with torch.no_grad():
        pv = torch.sigmoid(model(XV)).numpy()
    keep = pv > thresh
    val = {"auc": round(best, 4), "keep_ratio": round(float(keep.mean()), 4),
           "precision_kept": round(float(yv[keep].mean()) if keep.any() else 0.0, 4),
           "precision_all": round(float(yv.mean()), 4), "auc_per_epoch": hist}
    path = filter_model_path(exp, backbone, depth_name, filter_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"filter": spec["learned"], "in_dim": x.shape[1], "state_dict": best_state,
                "feat_mean": mean, "feat_std": std, "backbone": backbone, "depth_name": depth_name,
                "train_sequences": [s.name for s in sequences(exp, "train")],
                "val_sequences": [s.name for s in sequences(exp, "val")],
                "n_train": int(len(y)), "n_val": int(len(yv)), "val": val}, path)
    log.info(f"  filter {backbone}+{depth_name}+{filter_name}: {len(y)} train / {len(yv)} val matches, "
             f"val AUC {val['auc']:.3f}, keeps {val['keep_ratio']:.0%}, precision "
             f"{val['precision_all']:.3f} -> {val['precision_kept']:.3f}")
    return val


def train_filters(exp, jobs):
    t0 = time.time()
    log.info(f"[train filters] collecting matches for {len(jobs)} filter(s)")
    tr, va = collect(exp, jobs, "train"), collect(exp, jobs, "val")
    out = {}
    for j in jobs:
        out[j] = train_one(exp, j, tr[j], va[j])
    log.info(f"[train filters] done in {time.time() - t0:.0f} s")
    return out


# ======================= evaluation: generic learned filter =======================

_LOADED = {}


@register("bench_learned")
def _bench_learned(ctx, model_path, thresh=0.5, **params):
    """Learned filter trained by the benchmark (any @learned_filter in components/filters.py)."""
    if model_path not in _LOADED:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        impl = LEARNED_FILTERS[ck["filter"]]()
        model = impl.make_model(ck["in_dim"])
        model.load_state_dict(ck["state_dict"])
        _LOADED[model_path] = (impl, model.eval(), ck)
    impl, model, ck = _LOADED[model_path]
    if len(ctx.i1) == 0:
        return np.zeros(0, bool)
    x = (impl.features(ctx) - ck["feat_mean"]) / ck["feat_std"]
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).float())).numpy() > thresh
