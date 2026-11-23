"""
Train the learned match filters (MLP and/or CNN) on the collected TRAIN-split data.

Run from the repository root:
    python -m geomatch.train_filter.py --data geomatch/data/training/kitti_train.npz

For each hyperparameter run (a small built-in grid per model, see RUNS below) it:
  - sets all seeds (python / numpy / torch) so runs are replicable on other machines
  - splits train/val BY PAIR (pair_id from the .npz), so no pair leaks into both
  - trains on the GPU (falls back to CPU with a warning) and times every run
  - reports val metrics (AUC, accuracy, precision/recall at thresh 0.5)
  - saves a checkpoint to geomatch/data/models/<model>_<run>.pt
  - appends the run to geomatch/data/models/runs_<timestamp>.json (hyperparams, seed,
    val metrics, train time, torch/CUDA versions) — the replicability record

The best run per model (by val AUC) is also copied to <model>_<tag>_best.pt, where --tag
names the training gap(s) (e.g. 'gap80', 'all') — so every model is identified by which GAP(s)
it was trained on. Test-split evaluation happens separately via evaluation_pipeline.py
--config geomatch/kitti_test_config.yaml (canonical single gap) or with --gap G for a sweep.

Multiple --data files are concatenated (pair ids are offset per file), so one model can be
trained on several gaps at once, e.g. all six gap files for the 'all' model. Val metrics
include a per-gap AUC breakdown (from gap_ids in the .npz).

Nothing here ever touches the TEST split (KITTI 09-10).
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from geomatch.learned_filter import (MODELS_DIR, PATCH_SIZE,
                                      DepthPatchCNN, MLP, make_model, save_checkpoint)

log = logging.getLogger("geomatch")

# Small hyperparameter grid per model — every run is logged with its full config.
RUNS = {
    "linear": [
        dict(name="logreg_lr1e-3_ep30", hidden=(), lr=1e-3, epochs=30, batch=4096),
    ],
    "mlp": [
        dict(name="h64x32_lr1e-3_ep30", hidden=(64, 32), lr=1e-3, epochs=30, batch=4096),
        dict(name="h128x64x32_lr1e-3_ep30", hidden=(128, 64, 32), lr=1e-3, epochs=30, batch=4096),
        dict(name="h64x32_lr1e-4_ep50", hidden=(64, 32), lr=1e-4, epochs=50, batch=4096),
        dict(name="h256x128x64_lr1e-3_ep30", hidden=(256, 128, 64), lr=1e-3, epochs=30, batch=4096),
        dict(name="h64x32_do0.2_lr1e-3_ep40", hidden=(64, 32), dropout=0.2, lr=1e-3, epochs=40, batch=4096),
    ],
    "cnn": [
        dict(name="c16x32x64_lr1e-3_ep15", channels=(16, 32, 64), lr=1e-3, epochs=15, batch=256),
        dict(name="c32x64x128_lr1e-3_ep15", channels=(32, 64, 128), lr=1e-3, epochs=15, batch=128),
        dict(name="c16x32x64_lr5e-4_ep25", channels=(16, 32, 64), lr=5e-4, epochs=25, batch=256),
        dict(name="c8x16x32_lr1e-3_ep25", channels=(8, 16, 32), lr=1e-3, epochs=25, batch=256),
        dict(name="c16x32x64_p16_lr1e-3_ep25", channels=(16, 32, 64), patch=16, lr=1e-3, epochs=25, batch=256),
    ],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(paths):
    """One or more .npz files -> concatenated (features, patches, labels, pair_ids, gap_ids).
    pair_ids are offset per file so the pair-wise val split never mixes pairs from different
    files; gap_ids (per-sample pair gap, 0 if the file predates the gap sweep) are concatenated."""
    F, P, L, PI, G = [], [], [], [], []
    offset = 0
    for path in paths:
        z = np.load(path)
        n = len(z["labels"])
        F.append(z["mlp_features"].astype(np.float32))
        P.append(z["cnn_patches"].astype(np.float32) / 255.0)
        L.append(z["labels"].astype(np.float32))
        PI.append(z["pair_ids"].astype(np.int64) + offset)
        G.append(z["gap_ids"].astype(np.int64) if "gap_ids" in z else np.zeros(n, np.int64))
        offset += int(z["pair_ids"].max()) + 1 if n else 0
    return (np.concatenate(F), np.concatenate(P), np.concatenate(L),
            np.concatenate(PI), np.concatenate(G))


def auc_of(prob, y):
    """AUC via rank statistic (Mann-Whitney), robust and dependency-free."""
    order = np.argsort(prob)
    ranks = np.empty(len(prob))
    ranks[order] = np.arange(1, len(prob) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def split_by_pair(pair_ids, val_frac, seed):
    """Train/val index split grouped by pair_id (no pair in both)."""
    rng = np.random.default_rng(seed)
    pairs = np.unique(pair_ids)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    val_pairs = set(pairs[:n_val].tolist())
    val_mask = np.isin(pair_ids, list(val_pairs))
    return ~val_mask, val_mask


def metrics_from_logits(logits, labels, thresh=0.5):
    """AUC + accuracy/precision/recall at a fixed threshold, from raw logits."""
    prob = torch.sigmoid(logits).numpy()
    y = labels.numpy() if torch.is_tensor(labels) else labels
    pred = prob > thresh
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    auc = auc_of(prob, y)
    return dict(auc=round(auc, 4),
                acc=round((tp + tn) / len(y), 4),
                precision=round(tp / max(tp + fp, 1), 4),
                recall=round(tp / max(tp + fn, 1), 4),
                thresh=thresh)


def train_one(model_type, run, feats, patches, labels, pair_ids, gap_ids, val_frac, seed, device):
    set_seed(seed)
    tr, va = split_by_pair(pair_ids, val_frac, seed)

    if model_type in ("mlp", "linear"):
        x = torch.from_numpy(feats)
        model = MLP(in_dim=feats.shape[1], hidden=run["hidden"],
                    dropout=run.get("dropout", 0.0)).to(device)
        ds_tr = TensorDataset(x[tr], torch.from_numpy(labels[tr]))
        ds_va = TensorDataset(x[va], torch.from_numpy(labels[va]))
        feat_mean, feat_std = feats[tr].mean(0), feats[tr].std(0) + 1e-8
        fm_t = torch.from_numpy(feat_mean).to(device)
        fs_t = torch.from_numpy(feat_std).to(device)
        norm = lambda xx: (xx - fm_t) / fs_t
        extra_mean = extra_std = 0.0, 1.0
    else:
        patch = run.get("patch", PATCH_SIZE)
        if patch != PATCH_SIZE:  # center-crop the stored patches (e.g. 32 -> 16)
            r = (PATCH_SIZE - patch) // 2
            patches = patches[:, :, r:r + patch, r:r + patch]
        x = torch.from_numpy(patches)
        ext = torch.from_numpy((feats[:, 3:4] - feats[tr, 3].mean()) / (feats[tr, 3].std() + 1e-8))
        model = DepthPatchCNN(patch=patch, channels=run["channels"]).to(device)
        ds_tr = TensorDataset(x[tr], ext[tr], torch.from_numpy(labels[tr]))
        ds_va = TensorDataset(x[va], ext[va], torch.from_numpy(labels[va]))
        feat_mean = feat_std = None
        extra_mean, extra_std = float(feats[tr, 3].mean()), float(feats[tr, 3].std() + 1e-8)

    # class balancing: matches are ~75% correct, so weight the minority (wrong) class
    pos_weight = torch.tensor([(labels[tr] == 0).sum() / max((labels[tr] == 1).sum(), 1)],
                              device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=run["lr"])

    def batches(ds, shuffle):
        return DataLoader(ds, batch_size=run["batch"], shuffle=shuffle)

    is_tabular = model_type in ("mlp", "linear")
    t0 = time.perf_counter()
    epoch_times = []
    for ep in range(run["epochs"]):
        model.train()
        te = time.perf_counter()
        for batch in batches(ds_tr, True):
            if is_tabular:
                xb, yb = norm(batch[0].to(device)), batch[1].to(device)
                loss = loss_fn(model(xb), yb)
            else:
                xb, eb, yb = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                loss = loss_fn(model(xb, eb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        epoch_times.append(round(time.perf_counter() - te, 2))
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        if is_tabular:
            logits = torch.cat([model(norm(b[0].to(device))) for b in batches(ds_va, False)])
        else:
            logits = torch.cat([model(b[0].to(device), b[1].to(device)) for b in batches(ds_va, False)])
    val = metrics_from_logits(logits.cpu(), torch.from_numpy(labels[va]))
    # per-gap val AUC (gap-sweep diagnostics): how well the model does on each training gap
    prob = torch.sigmoid(logits.cpu()).numpy()
    y = labels[va]
    per_gap = {}
    for g in np.unique(gap_ids[va]):
        m = gap_ids[va] == g
        if m.sum() > 20 and 0 < y[m].sum() < m.sum():
            per_gap[int(g)] = round(auc_of(prob[m], y[m]), 4)
    val["per_gap_auc"] = per_gap
    return model, val, train_s, epoch_times, feat_mean, feat_std, extra_mean, extra_std


def main():
    parser = argparse.ArgumentParser(description="Train learned match filters (TRAIN split only).")
    parser.add_argument("--data", type=Path, nargs="+",
                        default=[Path(__file__).parent / "data" / "training" / "kitti_train.npz"],
                        help="one or more .npz training files (concatenated; use several to "
                             "train one model on several gaps)")
    parser.add_argument("--tag", type=str, default=None,
                        help="name tag for this training set, e.g. 'gap80' or 'all' — checkpoints "
                             "are written as <model>_<tag>_<run>.pt and <model>_<tag>_best.pt, so "
                             "the model is identified by the gap(s) it was trained on")
    parser.add_argument("--models", nargs="+", default=["linear", "mlp", "cnn"],
                        choices=["linear", "mlp", "cnn"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()
    if args.tag is None:  # default: derive from the data file name(s)
        args.tag = "+".join(p.stem.replace("kitti_train", "").strip("_") or "base" for p in args.data)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available — training on CPU (slow).")
    log.info(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), "
             f"torch {torch.__version__}, cuda {torch.version.cuda}")

    # split guard: the training npz must come from the TRAIN split (00-08) only — its sidecar
    # .json records the sequences it was collected from; refuse TEST sequences (09-10)
    from geomatch.learned_filter import assert_train_split_only
    train_sequences = []
    for data_path in args.data:
        sidecar = Path(data_path).with_suffix(".json")
        if sidecar.exists():
            meta = json.load(open(sidecar))
            assert_train_split_only(meta["sequences"])
            train_sequences = meta["sequences"]

    feats, patches, labels, pair_ids, gap_ids = load_data(args.data)
    log.info(f"data: {[str(p) for p in args.data]}  {len(labels)} samples, {labels.mean():.1%} correct, "
             f"{len(np.unique(pair_ids))} pairs, gaps={sorted(set(gap_ids.tolist()))}, tag='{args.tag}'")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    runs_log = {"created": datetime.now().isoformat(timespec="seconds"), "data": [str(p) for p in args.data],
                "tag": args.tag,
                "seed": args.seed, "val_frac": args.val_frac, "device": device,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "runs": []}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_type in args.models:
        best = None
        for run in RUNS[model_type]:
            model, val, train_s, epoch_times, fm, fs, em, es = train_one(
                model_type, run, feats, patches, labels, pair_ids, gap_ids,
                args.val_frac, args.seed, device)
            ckpt_path = args.out_dir / f"{model_type}_{args.tag}_{run['name']}.pt"
            save_checkpoint(ckpt_path, model_type, model, {
                "model_kwargs": (dict(in_dim=feats.shape[1], hidden=run["hidden"],
                                      dropout=run.get("dropout", 0.0))
                                 if model_type in ("mlp", "linear")
                                 else dict(patch=run.get("patch", PATCH_SIZE), channels=run["channels"])),
                "feat_mean": None if fm is None else fm.tolist(),
                "feat_std": None if fs is None else fs.tolist(),
                "extra_mean": em, "extra_std": es,
                "patch_size": PATCH_SIZE if model_type in ("mlp", "linear") else run.get("patch", PATCH_SIZE),
                "hyperparams": run, "seed": args.seed, "val_metrics": val,
                "train_time_s": round(train_s, 1), "epoch_times_s": epoch_times,
                "train_data": [str(p) for p in args.data],
                "train_tag": args.tag,
                "train_sequences": train_sequences,
            })
            entry = {"model": model_type, "run": run["name"], "val": val,
                     "train_time_s": round(train_s, 1), "checkpoint": str(ckpt_path)}
            runs_log["runs"].append(entry)
            log.info(f"[{model_type}/{run['name']}] val AUC={val['auc']} acc={val['acc']} "
                     f"P={val['precision']} R={val['recall']} per_gap={val['per_gap_auc']} | "
                     f"train {train_s:.1f} s ({run['epochs']} ep, mean {np.mean(epoch_times):.2f} s/ep)")
            if best is None or val["auc"] > best[0]["val"]["auc"]:
                best = (entry, ckpt_path)
        # best checkpoint is named by the training gap(s): <model>_<tag>_best.pt
        best_path = args.out_dir / f"{model_type}_{args.tag}_best.pt"
        import shutil
        shutil.copy(best[1], best_path)
        log.info(f"[{model_type}/{args.tag}] best run '{best[0]['run']}' (val AUC {best[0]['val']['auc']}) -> {best_path}")

    log_path = args.out_dir / f"runs_{stamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(runs_log, f, indent=2)
    log.info(f"run log (full hyperparams + seeds + timing for replication): {log_path}")


if __name__ == "__main__":
    main()
