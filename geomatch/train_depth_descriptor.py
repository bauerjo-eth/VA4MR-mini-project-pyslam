"""
WP2: train the learned depth descriptor on top of frozen SuperPoint.

    descriptor = [ SuperPoint (256, frozen) | w * f_theta(input) ]

Run from the repository root after collect_descriptor_data.py:

    python -m geomatch.train_depth_descriptor --mode depth   # main model
    python -m geomatch.train_depth_descriptor --mode image   # control: same network on the image

Loss: symmetric InfoNCE over all keypoints. For every ground-truth match (i, j) the combined
similarity s_ij = sp_i . sp_j + w^2 * d_i . d_j must be the largest of its row (i against all
keypoints of image B) and of its column (j against all keypoints of image A):

    L = 1/2 [ CE(s_i. / tau, j) + CE(s_.j / tau, i) ]

This is exactly what nearest-neighbour matching needs at test time. Trainable: theta, w, tau.

Validation (sequence 08, never trained on): share of ground-truth matches whose nearest
neighbour is the true partner, with SuperPoint alone vs. SuperPoint + learned branch.
Output: geomatch/data/models/wp2_<mode>_best.pt (+ _last.pt) and wp2_<mode>_log.json.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from geomatch.cache_depth_pictures import depth_folder, load_depth
from geomatch.collect_descriptor_data import DEPTH_MODEL, OUT_DIR
from geomatch.depth_encoder import DepthEncoder, make_input, save_model
from geomatch.superpoint import device

MODEL_DIR = Path(__file__).parent / "data" / "models"
log = logging.getLogger("geomatch")


class PairData(torch.utils.data.Dataset):
    """One item = one image pair: inputs of both images, SuperPoint pts/des, matches."""

    def __init__(self, sequences, mode, base, gaps=None):
        self.mode, self.base, self.items = mode, Path(base), []
        for seq in sequences:
            name = f"kitti{seq}"
            d = np.load(OUT_DIR / f"pairs_{name}.npz")
            for k in range(len(d["fa"])):
                if gaps and int(d["gap"][k]) not in gaps:
                    continue
                m = d["matches"][d["offsets"][k]:d["offsets"][k + 1]]
                self.items.append((seq, name, int(d["fa"][k]), int(d["fb"][k]), int(d["gap"][k]), m))

    def __len__(self):
        return len(self.items)

    def load_input(self, seq, name, f):
        if self.mode == "depth":
            arr = load_depth(depth_folder(DEPTH_MODEL, name), f)
        else:
            path = self.base / "sequences" / seq / "image_0" / f"{f:06d}.png"
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return make_input(arr, self.mode), arr.shape

    def __getitem__(self, k):
        seq, name, fa, fb, gap, m = self.items[k]
        out = {"gap": gap, "matches": torch.from_numpy(m.astype(np.int64))}
        for tag, f in (("a", fa), ("b", fb)):
            sp = np.load(OUT_DIR / "sp" / name / f"frame_{f:06d}.npz")
            inp, shape = self.load_input(seq, name, f)
            out[f"inp_{tag}"] = inp
            out[f"shape_{tag}"] = shape
            out[f"pts_{tag}"] = torch.from_numpy(sp["pts"])
            out[f"sp_{tag}"] = torch.from_numpy(sp["des"].astype(np.float32))
        return out


def pair_similarity(model, item, dev):
    """Combined similarity matrix (Na x Nb) and the SuperPoint-only one."""
    spA, spB = item["sp_a"].to(dev), item["sp_b"].to(dev)
    s_sp = spA @ spB.T
    fA = model(item["inp_a"][None].to(dev))
    fB = model(item["inp_b"][None].to(dev))
    dA = model.describe(fA, item["pts_a"].to(dev), *item["shape_a"])
    dB = model.describe(fB, item["pts_b"].to(dev), *item["shape_b"])
    return s_sp + model.w ** 2 * (dA @ dB.T), s_sp


def free_gpu_cache():
    """Every pair has a different number of keypoints (and some sequences a different image
    size), so the Apple MPS allocator keeps caching new buffers: memory grew ~3.5 GB per 100
    pairs and training slowed ~4x as the Mac started swapping. Emptying the cache every 20
    pairs keeps memory flat (89 ms/pair instead of 280-400+)."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def info_nce(s, m, tau):
    i, j = m[:, 0], m[:, 1]
    logits = s / tau
    return 0.5 * (F.cross_entropy(logits[i], j) + F.cross_entropy(logits[:, j].T, i))


def nn_correct(s, m):
    """Number of GT matches whose nearest neighbour (row-wise and column-wise) is the partner."""
    i, j = m[:, 0], m[:, 1]
    return int((s[i].argmax(1) == j).sum() + (s[:, j].argmax(0) == i).sum()), 2 * len(m)


@torch.no_grad()
def validate(model, data, dev):
    model.eval()
    tot = {}
    for k in range(len(data)):
        item = data[k]
        m = item["matches"].to(dev)
        s, s_sp = pair_similarity(model, item, dev)
        c, n = nn_correct(s, m)
        c_sp, _ = nn_correct(s_sp, m)
        for key in ("all", f"gap{item['gap']}"):
            t = tot.setdefault(key, [0, 0, 0])
            t[0] += c; t[1] += c_sp; t[2] += n
        if k % 20 == 0:
            free_gpu_cache()
    model.train()
    return {k: {"acc": v[0] / v[2], "acc_superpoint": v[1] / v[2], "n": v[2]} for k, v in sorted(tot.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["depth", "image"], default="depth")
    ap.add_argument("--base", default="datasets/kitti_odometry/dataset")
    ap.add_argument("--train", nargs="+", default=[f"{i:02d}" for i in range(8)])
    ap.add_argument("--val", nargs="+", default=["08"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=4, help="pairs per optimizer step")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_val_pairs", type=int, default=400)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert not ({"09", "10"} & set(args.train + args.val)), "09/10 are the TEST split"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = f"wp2_{args.mode}{args.tag}"

    train = PairData(args.train, args.mode, args.base)
    val = PairData(args.val, args.mode, args.base)
    val.items = val.items[::max(1, len(val.items) // args.max_val_pairs)]
    log.info(f"{name}: {len(train)} train pairs, {len(val)} val pairs, device {dev}")

    model = DepthEncoder(dim=args.dim, mode=args.mode).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = max(1, args.epochs * len(train) // args.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=args.lr * 0.02)
    # num_workers=0: loading takes ~10 ms/pair; worker processes competed with the GPU process
    # for the CPU on the Mac and made training ~5x slower
    loader = torch.utils.data.DataLoader(train, batch_size=None, shuffle=True, num_workers=0)

    history = []
    v = validate(model, val, dev)
    log.info(f"epoch 0 val: {json.dumps({k: round(x['acc'], 4) for k, x in v.items()})} | "
             f"superpoint alone: {json.dumps({k: round(x['acc_superpoint'], 4) for k, x in v.items()})}")
    history.append({"epoch": 0, "val": v})
    best = -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        run_loss, n = 0.0, 0
        opt.zero_grad()
        for it, item in enumerate(loader, 1):
            m = item["matches"].to(dev)
            s, _ = pair_similarity(model, item, dev)
            loss = info_nce(s, m, model.log_tau.exp()) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum; n += 1
            if it % 20 == 0:
                free_gpu_cache()
            if it % args.accum == 0:
                opt.step(); opt.zero_grad()
                if sched.last_epoch < steps:
                    sched.step()
            if it % 500 == 0:
                log.info(f"  epoch {epoch} it {it}/{len(train)} loss {run_loss / n:.4f} "
                         f"w {model.w.item():.3f} tau {model.log_tau.exp().item():.4f} ({time.time() - t0:.0f} s)")
        v = validate(model, val, dev)
        acc = v["all"]["acc"]
        log.info(f"epoch {epoch} loss {run_loss / max(n, 1):.4f} w {model.w.item():.3f} | val acc "
                 f"{json.dumps({k: round(x['acc'], 4) for k, x in v.items()})} | superpoint alone "
                 f"{v['all']['acc_superpoint']:.4f} ({time.time() - t0:.0f} s)")
        history.append({"epoch": epoch, "train_loss": run_loss / max(n, 1), "w": model.w.item(),
                        "tau": model.log_tau.exp().item(), "val": v, "seconds": time.time() - t0})
        extra = {"train_sequences": [f"kitti{s}" for s in args.train],
                 "val_sequences": [f"kitti{s}" for s in args.val], "epoch": epoch, "val": v, "args": vars(args)}
        save_model(model, MODEL_DIR / f"{name}_last.pt", extra)
        if acc > best:
            best = acc
            save_model(model, MODEL_DIR / f"{name}_best.pt", extra)
        (MODEL_DIR / f"{name}_log.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    log.info(f"done: best val acc {best:.4f} (superpoint alone {history[0]['val']['all']['acc_superpoint']:.4f})")


if __name__ == "__main__":
    main()
