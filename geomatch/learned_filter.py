"""
Learned match filters: an MLP on hand-crafted depth features and a CNN on raw depth patches.

Both models take a MatchContext (or its extracted features) and output, per putative match,
the probability that the match is geometrically correct. Training labels come from the
ground-truth epipolar error (oracle labels — used ONLY to create the training set, never at
inference). See geomatch/TRAINING.md for which data trained which model.

Pipeline:
  1. collect_training_data.py  ->  .npz with per-match features, patches, labels, pair ids
  2. train_filter.py           ->  .pt checkpoints (GPU, seeded, timed) in data/models/
  3. filtering.py              ->  @register("learned_mlp") / @register("learned_cnn")

Feature / patch extraction is shared between training and inference so both see identical inputs.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).parent / "data" / "models"
DEFAULT_MLP = str(MODELS_DIR / "mlp_best.pt")
DEFAULT_CNN = str(MODELS_DIR / "cnn_best.pt")

# depth range used to normalize log-depth patches (same clipping as descriptors.depth_to_image)
Z_MIN, Z_MAX = 1.0, 100.0
PATCH_SIZE = 32

MISSING_MODEL_HINT = (
    "learned filter weights not found at '{path}'. Train them first, on the TRAIN split only:\n"
    "  python -m geomatch.collect_training_data --config geomatch/kitti_train_config.yaml\n"
    "  python -m geomatch.train_filter.py --data geomatch/data/training/kitti_train.npz\n"
    "then point model_path at the produced checkpoint (default: {default}). See geomatch/TRAINING.md."
)


# ======================= feature extraction (shared train / inference) =======================

def _window_mean_std(depth, pts, k):
    """Mean and std of depth in a k x k window around each point (box filter on the full map)."""
    mean = cv2.boxFilter(depth, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(depth * depth, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    from geomatch.descriptors import depth_at
    m = depth_at(mean, pts)
    s = np.sqrt(np.maximum(depth_at(sq, pts) - m ** 2, 0.0))
    return m, s


def _window_range(depth, pts, k):
    """(max - min) of depth in a k x k window around each point: occlusion-boundary indicator."""
    dil = cv2.dilate(depth, np.ones((k, k), np.uint8))
    ero = cv2.erode(depth, np.ones((k, k), np.uint8))
    from geomatch.descriptors import depth_at
    return depth_at(dil, pts) - depth_at(ero, pts)


def _grad_mag(depth, pts):
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    from geomatch.descriptors import depth_at
    return np.hypot(depth_at(gx, pts), depth_at(gy, pts))


def build_mlp_features(ctx):
    """Per-match feature vector (N x 22) from the MatchContext. Order is fixed — the training
    script stores the normalization stats in the checkpoint, so do not reorder."""
    i1, i2 = ctx.i1, ctx.i2
    p1, p2 = ctx.pts1[i1], ctx.pts2[i2]
    z1, z2 = ctx.z1[i1], ctx.z2[i2]
    eps = 1e-6
    desc_dist = np.linalg.norm(ctx.des1[i1] - ctx.des2[i2], axis=1)

    h, w = ctx.depth1.shape
    m5a, s5a = _window_mean_std(ctx.depth1, p1, 5)
    m5b, s5b = _window_mean_std(ctx.depth2, p2, 5)
    m15a, s15a = _window_mean_std(ctx.depth1, p1, 15)
    m15b, s15b = _window_mean_std(ctx.depth2, p2, 15)

    feats = np.stack([
        z1, z2,                                                                    # 1-2   depth at the match
        np.abs(z1 - z2) / (z1 + z2 + eps),                                          # 3     relative depth
        desc_dist,                                                                   # 4     descriptor distance
        np.log(z1 + eps), np.log(z2 + eps),                                          # 5-6   log depth
        m5a, s5a, m5b, s5b,                                                          # 7-10  5x5 window stats
        m15a, s15a, m15b, s15b,                                                      # 11-14 15x15 window stats
        _grad_mag(ctx.depth1, p1), _grad_mag(ctx.depth2, p2),                        # 15-16 depth gradient
        _window_range(ctx.depth1, p1, 15), _window_range(ctx.depth2, p2, 15),         # 17-18 depth discontinuity
        p1[:, 0] / w, p1[:, 1] / h, p2[:, 0] / w, p2[:, 1] / h,                      # 19-22 normalized position
    ], axis=1)
    return feats.astype(np.float32)


def log_depth_image(depth):
    """Depth map -> float32 log-depth normalized to [0, 1] (fixed range, scale-invariant)."""
    z = np.clip(depth, Z_MIN, Z_MAX)
    v = (np.log(z) - np.log(Z_MIN)) / (np.log(Z_MAX) - np.log(Z_MIN))
    return v.astype(np.float32)


def extract_patch(img, x, y, P):
    """P x P patch of img centered at (x, y), reflect-padded at the borders."""
    r = P // 2
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    x0, y0 = xi - r, yi - r
    patch = np.full((P, P), img[max(0, y0), max(0, x0)], np.float32)  # corner fill, overwritten below
    # reflect-pad the whole image once per call is wasteful; pad just enough here:
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + P - w), max(0, y0 + P - h)
    if pad_l or pad_t or pad_r or pad_b:
        img_p = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        return img_p[y0 + pad_t:y0 + pad_t + P, x0 + pad_l:x0 + pad_l + P].astype(np.float32)
    return img[y0:y0 + P, x0:x0 + P].astype(np.float32)


def build_cnn_patches(ctx, patch_size=PATCH_SIZE):
    """Per-match depth patches: (N, 2, P, P) float32 in [0, 1]. Channel 0 = image 1 around
    pts1[i1], channel 1 = image 2 around pts2[i2]."""
    ld1, ld2 = log_depth_image(ctx.depth1), log_depth_image(ctx.depth2)
    p1, p2 = ctx.pts1[ctx.i1], ctx.pts2[ctx.i2]
    P = patch_size
    out = np.empty((len(p1), 2, P, P), np.float32)
    for k in range(len(p1)):
        out[k, 0] = extract_patch(ld1, p1[k, 0], p1[k, 1], P)
        out[k, 1] = extract_patch(ld2, p2[k, 0], p2[k, 1], P)
    return out


def build_cnn_extra(ctx):
    """Scalar extra input for the CNN head: the descriptor distance (N, 1)."""
    d = np.linalg.norm(ctx.des1[ctx.i1] - ctx.des2[ctx.i2], axis=1)
    return d.astype(np.float32)[:, None]


# ======================= models =======================

class MLP(nn.Module):
    """in_dim -> hidden... -> 1 logit. hidden=() gives plain logistic regression."""

    def __init__(self, in_dim, hidden=(64, 32), dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DepthPatchCNN(nn.Module):
    """2 x P x P depth patches -> conv stack -> concat extra -> 1 logit."""

    def __init__(self, patch=PATCH_SIZE, extra_dim=1, channels=(16, 32, 64)):
        super().__init__()
        layers, c_in = [], 2
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        flat = channels[-1] * (patch // (2 ** len(channels))) ** 2
        self.head = nn.Sequential(nn.Linear(flat + extra_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, extra):
        f = self.conv(x).flatten(1)
        return self.head(torch.cat([f, extra], dim=1)).squeeze(-1)


def make_model(model_type, **kwargs):
    if model_type in ("mlp", "linear"):  # linear = MLP with no hidden layers
        return MLP(**kwargs)
    if model_type == "cnn":
        return DepthPatchCNN(**kwargs)
    raise ValueError(f"unknown model_type '{model_type}'")


# ======================= checkpoints =======================

def save_checkpoint(path, model_type, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": model_type, "model_kwargs": meta["model_kwargs"],
                "state_dict": model.state_dict(), **meta}, path)


def load_model(path):
    """Load a checkpoint -> (model in eval mode on CPU, meta dict)."""
    if not os.path.exists(path):
        raise FileNotFoundError(MISSING_MODEL_HINT.format(path=path, default=DEFAULT_MLP))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ckpt["model_type"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model_type", "model_kwargs", "state_dict")}
    return model, meta


# ---------- train/test split guards ----------

# TEST split = KITTI odometry sequences 09-10 (held out; only 00-10 have GT poses).
# TRAIN split = 00-08. Learned filters must be trained on TRAIN and evaluated on TEST only.
TEST_SEQUENCES = {"09", "10"}

DEFAULT_FILTER_PATHS = {
    "learned_mlp": DEFAULT_MLP,
    "learned_cnn": DEFAULT_CNN,
    "learned_linear": str(MODELS_DIR / "linear_best.pt"),
}


def default_model_path(filter_type):
    """Default weights path of a registered learned filter (mirrors filtering.py defaults)."""
    return DEFAULT_FILTER_PATHS.get(filter_type)


def checkpoint_train_sequences(path):
    """Sequence names a checkpoint was trained on, read from its meta (empty if the file is
    missing — load_model() then raises the proper error later). Used for split-leak guards."""
    if not os.path.exists(path):
        return set()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {str(s) for s in (ckpt.get("train_sequences") or [])}


def assert_train_split_only(seq_names):
    """Raise ValueError if any of the given sequence names (e.g. 'kitti09') belongs to the
    held-out TEST split. Used so training data / training runs can never touch 09-10."""
    leak = [str(s) for s in seq_names if str(s).replace("kitti", "") in TEST_SEQUENCES]
    if leak:
        raise ValueError(
            f"TEST split leak: sequence(s) {leak} are the held-out TEST split (09-10). "
            f"Training data and training runs must use the TRAIN split (sequences 00-08) only; "
            f"evaluate on TEST via kitti_test_config.yaml. See geomatch/TRAINING.md.")


# ======================= inference (used by filtering.py) =======================

def predict_mlp(model, meta, features):
    """features (N, F) float32 -> keep-probabilities (N,) numpy. Runs on the model's device
    (GPU if the model was moved there — see filtering._get_learned_model)."""
    dev = next(model.parameters()).device
    mean = np.asarray(meta["feat_mean"], np.float32)
    std = np.asarray(meta["feat_std"], np.float32)
    x = torch.from_numpy((features - mean) / std).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()


def predict_cnn(model, meta, patches, extra):
    """patches (N, 2, P, P), extra (N, 1) -> keep-probabilities (N,) numpy. GPU if available."""
    dev = next(model.parameters()).device
    P = meta.get("patch_size", PATCH_SIZE)
    if patches.shape[2] != P:
        raise ValueError(f"patch size {patches.shape[2]} != model's {P}")
    em = float(meta.get("extra_mean", 0.0))
    es = float(meta.get("extra_std", 1.0))
    x = torch.from_numpy(patches).to(dev)
    e = torch.from_numpy((extra - em) / es).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x, e)).cpu().numpy()
