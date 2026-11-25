"""
Train a learned depth descriptor for one backbone (TRAIN split; VAL picks the best epoch).

Ground truth: for image pairs of the TRAIN/VAL sequences, keypoints of A are lifted with
scale-corrected Depth Anything depth, moved with the true pose into B and matched to the
nearest B keypoint within 3 px (occlusion check, mutual, epipolar error < 2 px); see
geomatch/collect_descriptor_data.py. The true pose only makes labels, it is never an input.

Loss: symmetric InfoNCE on s_ij = visual_i . visual_j + w^2 * d_i . d_j over all keypoints,
i.e. the true partner must be the nearest neighbour of the combined descriptor. The visual
descriptor is frozen; trained: the network, w, tau.
"""

import json
import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geomatch.bench.common import BENCH_DATA, keypoints, sequences, short_hash
from geomatch.bench.describe import LearnedBranch, device, make_branch
from geomatch.collect_descriptor_data import correspondences, depth_scale

log = logging.getLogger("geomatch")


# ======================= ground-truth correspondences =======================

def correspondence_file(exp, backbone, seq):
    c = exp["training"]["descriptor"]
    key = [exp["backbones"][backbone], backbone, c["gaps"], c["step"], exp["data"]["depth_model"],
           "scale-from-superpoint"]
    return BENCH_DATA / "correspondences" / f"{backbone}_{short_hash(key)}" / f"{seq.name}.npz"


SCALE_BACKBONE = {"backbones": {"superpoint": {"n": 2000}}}


def sequence_scale(exp, seq):
    """One DA3 depth scale per sequence (DA3 metric depth is off by a nearly constant factor;
    Robin, WP1), cached. Estimated with SuperPoint keypoints for every backbone: the scale is a
    property of the depth model and the sequence, and dense SIFT keypoints (many at almost the
    same position) produce chance agreements at wrong scales (2.5 instead of ~1.1)."""
    c = exp["training"]["descriptor"]
    g = min(c["gaps"])
    path = (BENCH_DATA / "da3_scale" / exp["data"]["depth_model"].replace("/", "_")
            / f"{seq.name}_gap{g}_step{c['step']}.json")
    if path.exists():
        return json.loads(path.read_text())["scale"]
    small = seq.pairs(g, c["step"])[::3]
    seq.ensure_depth({f for p in small for f in p})
    shape = seq.depth(small[0][0]).shape
    est = []
    for a, b in tqdm(small, desc=f"DA3 scale {seq.name}", unit="pair"):
        pa, pb = (keypoints(SCALE_BACKBONE, "superpoint", seq, f).pts for f in (a, b))
        est.append(depth_scale(pa, pb, seq.depth(a), seq.depth(b), seq.poses[a], seq.poses[b], seq.K, shape))
    good = [s for s, n in est if n >= 80] or [s for s, n in est if n >= 30] or [1.0]
    s = float(np.median(good))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scale": s, "n_pairs": len(good), "of": len(est), "gap": g, "step": c["step"]}))
    log.info(f"{seq.name}: DA3 scale {s:.3f} (median of {len(good)} of {len(est)} pairs)")
    return s


def collect_correspondences(exp, backbone, seq):
    """GT keypoint correspondences of one sequence for one backbone, cached on disk."""
    path = correspondence_file(exp, backbone, seq)
    if path.exists():
        return dict(np.load(path))
    c = exp["training"]["descriptor"]
    gaps, step = c["gaps"], c["step"]
    pairs = [(a, a + g, g) for a in range(0, seq.n_frames - max(gaps), step) for g in gaps]
    frames = sorted({f for a, b, _ in pairs for f in (a, b)})
    log.info(f"[{backbone}] {seq.name}: correspondences for {len(pairs)} pairs ({len(frames)} frames)")
    seq.ensure_depth(frames)
    for i in tqdm(frames, desc=f"{backbone} keypoints {seq.name}", unit="frame"):
        keypoints(exp, backbone, seq, i)
    shape = seq.depth(frames[0]).shape
    feats = {}

    def feat(i):
        if i not in feats:
            feats[i] = {"pts": keypoints(exp, backbone, seq, i).pts, "depth": seq.depth(i)}
        return feats[i]

    # one DA3 scale per sequence (shared by all backbones), refined per pair within +-10 %
    s_seq = sequence_scale(exp, seq)
    rec = {"fa": [], "fb": [], "gap": [], "scale": [], "matches": []}
    for a, b, g in tqdm(pairs, desc=f"{backbone} correspondences {seq.name}", unit="pair"):
        fa, fb = feat(a), feat(b)
        s, n = depth_scale(fa["pts"], fb["pts"], fa["depth"], fb["depth"], seq.poses[a], seq.poses[b],
                           seq.K, shape, center=s_seq)
        m = correspondences(fa, fb, (s, s), seq.poses[a], seq.poses[b], seq.K, shape) if n >= 20 else []
        if len(m) >= 20:
            for k, v in zip(("fa", "fb", "gap", "scale", "matches"), (a, b, g, s, m)):
                rec[k].append(v)
        for old in [k for k in feats if k < a - max(gaps)]:
            del feats[old]
    out = {"fa": np.array(rec["fa"]), "fb": np.array(rec["fb"]), "gap": np.array(rec["gap"]),
           "scale": np.array(rec["scale"]), "scale_seq": np.array(s_seq),
           "offsets": np.cumsum([0] + [len(m) for m in rec["matches"]]),
           "matches": np.concatenate(rec["matches"]) if rec["matches"] else np.zeros((0, 2), np.int32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    log.info(f"[{backbone}] {seq.name}: DA3 scale {s_seq:.3f}, {len(rec['fa'])} of {len(pairs)} pairs kept, "
             f"median {np.median([len(m) for m in rec['matches']]) if rec['matches'] else 0:.0f} matches/pair")
    return out


class PairSet(torch.utils.data.Dataset):
    def __init__(self, exp, backbone, seqs, input_name):
        self.exp, self.backbone, self.input_name, self.items = exp, backbone, input_name, []
        for seq in seqs:
            d = collect_correspondences(exp, backbone, seq)
            for k in range(len(d["fa"])):
                m = d["matches"][d["offsets"][k]:d["offsets"][k + 1]]
                self.items.append((seq, int(d["fa"][k]), int(d["fb"][k]), int(d["gap"][k]), m))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        seq, fa, fb, gap, m = self.items[k]
        out = {"gap": gap, "matches": torch.from_numpy(m.astype(np.int64))}
        for tag, f in (("a", fa), ("b", fb)):
            kp = keypoints(self.exp, self.backbone, seq, f)
            depth = seq.depth(f)
            out[f"inp_{tag}"] = LearnedBranch.make_input(self.input_name, depth, seq.image(f))
            out[f"shape_{tag}"] = depth.shape
            out[f"pts_{tag}"] = torch.from_numpy(kp.pts)
            out[f"vis_{tag}"] = torch.from_numpy(kp.des)
        return out


# ======================= training =======================

def similarity(model, item, dev):
    vA, vB = item["vis_a"].to(dev), item["vis_b"].to(dev)
    s_vis = vA @ vB.T
    dA = model.describe(model(item["inp_a"][None].to(dev)), item["pts_a"].to(dev), *item["shape_a"])
    dB = model.describe(model(item["inp_b"][None].to(dev)), item["pts_b"].to(dev), *item["shape_b"])
    return s_vis + model.w ** 2 * (dA @ dB.T), s_vis


def info_nce(s, m, tau):
    i, j = m[:, 0], m[:, 1]
    logits = s / tau
    return 0.5 * (F.cross_entropy(logits[i], j) + F.cross_entropy(logits[:, j].T, i))


def nn_correct(s, m):
    i, j = m[:, 0], m[:, 1]
    return int((s[i].argmax(1) == j).sum() + (s[:, j].argmax(0) == i).sum()), 2 * len(m)


def free_gpu_cache():
    """The MPS allocator keeps buffers of every keypoint count it has seen; without emptying
    it, memory grows ~3.5 GB / 100 pairs and the Mac starts swapping."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def validate(model, data, dev):
    model.eval()
    tot = {}
    for k in range(len(data)):
        item = data[k]
        m = item["matches"].to(dev)
        s, s_vis = similarity(model, item, dev)
        c, n = nn_correct(s, m)
        c_vis, _ = nn_correct(s_vis, m)
        for key in ("all", f"gap{item['gap']}"):
            t = tot.setdefault(key, [0, 0, 0])
            t[0] += c; t[1] += c_vis; t[2] += n
        if k % 20 == 0:
            free_gpu_cache()
    model.train()
    return {k: {"acc": v[0] / v[2], "acc_visual": v[1] / v[2], "n": v[2]} for k, v in sorted(tot.items())}


def train_descriptor(exp, backbone, depth_name, out_path):
    spec = exp["depth_descriptors"][depth_name]
    c = exp["training"]["descriptor"]
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = device()
    train = PairSet(exp, backbone, sequences(exp, "train"), spec["input"])
    val = PairSet(exp, backbone, sequences(exp, "val"), spec["input"])
    val.items = val.items[::max(1, len(val.items) // c.get("max_val_pairs", 400))]
    model = make_branch(spec).to(dev)
    log.info(f"[train descriptor] {backbone} + {depth_name}: {len(train)} train / {len(val)} val pairs, "
             f"{sum(p.numel() for p in model.parameters())} parameters, device {dev}")
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    steps = max(1, c["epochs"] * len(train) // c["accum"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=c["lr"] * 0.02)
    loader = torch.utils.data.DataLoader(train, batch_size=None, shuffle=True, num_workers=0)

    v = validate(model, val, dev)
    history = [{"epoch": 0, "val": v}]
    log.info(f"  epoch 0: val acc {v['all']['acc']:.4f} (visual alone {v['all']['acc_visual']:.4f})")
    best, t0 = -1, time.time()
    partial = out_path.with_name(out_path.stem + ".partial.pt")
    extra = {"backbone": backbone, "depth_name": depth_name, "spec": spec, "training": c,
             "train_sequences": [s.name for s in sequences(exp, "train")],
             "val_sequences": [s.name for s in sequences(exp, "val")]}
    for epoch in range(1, c["epochs"] + 1):
        run_loss, n = 0.0, 0
        opt.zero_grad()
        for it, item in enumerate(tqdm(loader, desc=f"epoch {epoch}", unit="pair", leave=False), 1):
            s, _ = similarity(model, item, dev)
            loss = info_nce(s, item["matches"].to(dev), model.log_tau.exp()) / c["accum"]
            loss.backward()
            run_loss += loss.item() * c["accum"]; n += 1
            if it % 20 == 0:
                free_gpu_cache()
            if it % c["accum"] == 0:
                opt.step(); opt.zero_grad()
                if sched.last_epoch < steps:
                    sched.step()
        v = validate(model, val, dev)
        acc = v["all"]["acc"]
        history.append({"epoch": epoch, "train_loss": run_loss / max(n, 1), "w": model.w.item(),
                        "val": v, "seconds": time.time() - t0})
        log.info(f"  epoch {epoch}: loss {run_loss / max(n, 1):.4f} w {model.w.item():.3f} | val acc {acc:.4f} "
                 f"(visual alone {v['all']['acc_visual']:.4f}) {time.time() - t0:.0f} s")
        if acc > best:
            best = acc
            model.save(partial, {**extra, "epoch": epoch, "val": v})
    partial.rename(out_path)  # only a finished training counts as a model (an interrupted run retrains)
    out_path.with_suffix(".json").write_text(json.dumps({**extra, "history": history}, indent=2, default=str))
    log.info(f"  best val acc {best:.4f} (visual alone {history[0]['val']['all']['acc_visual']:.4f}) -> {out_path.name}")
