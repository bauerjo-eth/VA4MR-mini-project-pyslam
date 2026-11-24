"""
Shared plumbing of the benchmark: experiment file, KITTI sequences, caches, model paths.
Nothing here needs editing to add a new descriptor, network or filter.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from geomatch.cache_depth_pictures import cache_depth_maps, depth_folder, depth_path
from geomatch.components.backbones import BACKBONES, Keypoints
from geomatch.evaluation_pipeline import load_kitti_calib, load_poses

ROOT = Path(__file__).resolve().parents[1]           # geomatch/
BENCH_DATA = ROOT / "data" / "bench"                   # caches + models (gitignored)
MODELS = BENCH_DATA / "models"
RESULTS = ROOT / "results"
log = logging.getLogger("geomatch")


# ======================= experiment file =======================

def load_experiment(path):
    exp = yaml.safe_load(Path(path).read_text())
    d = exp["data"]
    leak = set(d["test"]) & (set(d["train"]) | set(d["val"]))
    if leak:
        raise ValueError(f"sequences {sorted(leak)} are in TEST and in TRAIN/VAL")
    exp.setdefault("matcher", {"type": "mutual_nn", "ratio": 0.8})
    return exp


def short_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def is_learned(spec):
    return isinstance(spec, dict) and "learned" in spec


def split_params(spec):
    p = dict(spec)
    return p.pop("type"), p


def desc_model_path(exp, backbone, depth_name):
    """Checkpoint of a learned depth descriptor. The file name hashes everything that defines
    the model (backbone, network spec, training settings, train/val sequences), so changing
    any of them trains a new model instead of silently reusing an old one. Changing the CODE
    of a network does not change the hash: use run.py --retrain then."""
    spec = exp["depth_descriptors"][depth_name]
    key = [exp["backbones"][backbone], backbone, spec, exp["training"]["descriptor"],
           exp["data"]["train"], exp["data"]["val"], exp["data"]["depth_model"]]
    return MODELS / f"desc_{backbone}_{depth_name}_{short_hash(key)}.pt"


def filter_model_path(exp, backbone, depth_name, filter_name):
    spec = exp["depth_descriptors"][depth_name]
    up = desc_model_path(exp, backbone, depth_name).stem if is_learned(spec) else spec
    key = [exp["backbones"][backbone], backbone, up, exp["matcher"], exp["filters"][filter_name],
           exp["training"]["filter"], exp["data"]["train"], exp["data"]["val"],
           exp["grading"]["correct_px"], exp["data"]["depth_model"]]
    return MODELS / f"filter_{backbone}_{depth_name}_{filter_name}_{short_hash(key)}.pt"


# ======================= KITTI sequences =======================

@dataclass
class Sequence:
    seq: str            # "09"
    base: Path
    depth_model: str

    def __post_init__(self):
        d = self.base / "sequences" / self.seq
        self.name = f"kitti{self.seq}"                      # also the depth-cache key
        self.image_dir = d / "image_0"
        self.poses = load_poses(self.base / "poses" / f"{self.seq}.txt")
        self.K = load_kitti_calib(d)
        self.n_frames = len(self.poses)
        self.depth_dir = depth_folder(self.depth_model, self.name)

    def image(self, i):
        """RGB (KITTI odometry image_0 is grey; converted like the evaluation pipeline)."""
        gray = cv2.imread(str(self.image_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def depth(self, i):
        return np.load(depth_path(self.depth_dir, i))

    def ensure_depth(self, frames, chunk=100):
        """Predict and cache Depth Anything depth for frames that are not cached yet."""
        missing = [i for i in sorted(set(frames)) if not depth_path(self.depth_dir, i).exists()]
        for c in range(0, len(missing), chunk):
            idx = missing[c:c + chunk]
            cache_depth_maps({i: self.image(i) for i in idx}, self.depth_model, self.name)

    def pairs(self, gap, step, start=0):
        """(i, i + gap) for anchors start, start + step, ... (same rule as evaluation_pipeline)."""
        return [(s, s + gap) for s in range(start, self.n_frames - gap, step)]


def sequences(exp, split):
    d = exp["data"]
    base = Path(d["base_path"])
    if not (base / "sequences").is_dir():
        raise FileNotFoundError(f"KITTI odometry not found at {base} (data.base_path)")
    return [Sequence(s, base, d["depth_model"]) for s in d[split]]


# ======================= backbone cache =======================

def backbone_cache_dir(exp, backbone):
    return BENCH_DATA / "keypoints" / f"{backbone}_{short_hash(exp['backbones'][backbone])}"


def keypoints(exp, backbone, seq, i):
    """Backbone output of one frame, computed once and cached on disk (float32, exact)."""
    path = backbone_cache_dir(exp, backbone) / seq.name / f"frame_{i:06d}.npz"
    if path.exists():
        z = np.load(path)
        return Keypoints(pts=z["pts"], des=z["des"], size=z["size"] if "size" in z else None,
                         angle=z["angle"] if "angle" in z else None)
    kp = BACKBONES[backbone](seq.image(i), **(exp["backbones"][backbone] or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {k: getattr(kp, k) for k in ("size", "angle") if getattr(kp, k) is not None}
    np.savez(path, pts=kp.pts, des=kp.des, **extra)
    return kp
