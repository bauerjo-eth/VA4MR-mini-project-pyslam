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

The best run per model (by val AUC) is also copied to mlp_best.pt / cnn_best.pt, which is
what kitti_test_config.yaml points at. Test-split evaluation happens separately via
evaluation_pipeline.py --config geomatch/kitti_test_config.yaml.

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

from geomatch.learned_filter import (DEFAULT_CNN, DEFAULT_MLP, MODELS_DIR, PATCH_SIZE,
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


def load_data(path):
    z = np.load(path)
    return (z["mlp_features"].astype(np.float32), z["cnn_patches"].astype(np.float32) / 255.0,
            z["labels"].astype(np.float32), z["pair_ids"].astype(np.int64))


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
    # AUC via rank statistic (Mann-Whitney), robust and dependency-free
    order = np.argsort(prob)
    ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob) + 1)
    pos = y == 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    return dict(auc=round(float(auc), 4),
                acc=round((tp + tn) / len(y), 4),
                precision=round(tp / max(tp + fp, 1), 4),
                recall=round(tp / max(tp + fn, 1), 4),
                thresh=thresh)


def train_one(model_type, run, feats, patches, labels, pair_ids, val_frac, seed, device):
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
    return model, val, train_s, epoch_times, feat_mean, feat_std, extra_mean, extra_std


def main():
    parser = argparse.ArgumentParser(description="Train learned match filters (TRAIN split only).")
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data" / "training" / "kitti_train.npz")
    parser.add_argument("--models", nargs="+", default=["linear", "mlp", "cnn"],
                        choices=["linear", "mlp", "cnn"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available — training on CPU (slow).")
    log.info(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), "
             f"torch {torch.__version__}, cuda {torch.version.cuda}")

    # split guard: the training npz must come from the TRAIN split (00-08) only — its sidecar
    # .json records the sequences it was collected from; refuse TEST sequences (09-10)
    from geomatch.learned_filter import assert_train_split_only
    sidecar = Path(args.data).with_suffix(".json")
    if sidecar.exists():
        assert_train_split_only(json.load(open(sidecar))["sequences"])

    feats, patches, labels, pair_ids = load_data(args.data)
    log.info(f"data: {args.data}  {len(labels)} samples, {labels.mean():.1%} correct, "
             f"{len(np.unique(pair_ids))} pairs")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    runs_log = {"created": datetime.now().isoformat(timespec="seconds"), "data": str(args.data),
                "seed": args.seed, "val_frac": args.val_frac, "device": device,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "runs": []}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_type in args.models:
        best = None
        for run in RUNS[model_type]:
            model, val, train_s, epoch_times, fm, fs, em, es = train_one(
                model_type, run, feats, patches, labels, pair_ids, args.val_frac, args.seed, device)
            ckpt_path = args.out_dir / f"{model_type}_{run['name']}.pt"
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
                "train_data": str(args.data),
                "train_sequences": json.load(open(Path(args.data).with_suffix(".json")))["sequences"],
            })
            entry = {"model": model_type, "run": run["name"], "val": val,
                     "train_time_s": round(train_s, 1), "checkpoint": str(ckpt_path)}
            runs_log["runs"].append(entry)
            log.info(f"[{model_type}/{run['name']}] val AUC={val['auc']} acc={val['acc']} "
                     f"P={val['precision']} R={val['recall']} | train {train_s:.1f} s "
                     f"({run['epochs']} ep, mean {np.mean(epoch_times):.2f} s/ep)")
            if best is None or val["auc"] > best[0]["val"]["auc"]:
                best = (entry, ckpt_path)
        default_name = {"mlp": DEFAULT_MLP, "cnn": DEFAULT_CNN}.get(model_type, str(MODELS_DIR / f"{model_type}_best.pt"))
        best_path = args.out_dir / Path(default_name).name
        import shutil
        shutil.copy(best[1], best_path)
        log.info(f"[{model_type}] best run '{best[0]['run']}' (val AUC {best[0]['val']['auc']}) -> {best_path}")

    log_path = args.out_dir / f"runs_{stamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(runs_log, f, indent=2)
    log.info(f"run log (full hyperparams + seeds + timing for replication): {log_path}")


if __name__ == "__main__":
    main()
