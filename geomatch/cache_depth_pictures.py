"""
Depth cache: predict each depth map once, save it, and reuse it in later runs.

The evaluation pipeline calls cache_depth_maps() once at the start. Depth maps are saved as

    geomatch/data/depth/<model>/<dataset>/frame_000050.npy

so different models and datasets never mix. Frames that are already saved are skipped, so
only the first run of a dataset pays for the Depth Anything 3 inference.

If the depth prediction code itself changes (not just the model), delete the folder so the
maps are computed again.
"""

import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geomatch.depth import predict_depth

DEPTH_DIR = Path(__file__).parent / "data" / "depth"
log = logging.getLogger("geomatch")


def depth_folder(model, dataset):
    """Folder for one model + dataset, e.g. data/depth/depth-anything_DA3METRIC-LARGE/kitti06."""
    return DEPTH_DIR / model.replace("/", "_") / dataset


def depth_path(folder, frame_id):
    return folder / f"frame_{frame_id:06d}.npy"


def cache_depth_maps(frames, model, dataset):
    """Make sure a depth map is saved for every frame. frames: {frame index: RGB image}.
    Returns the folder, to be passed to load_depth()."""
    folder = depth_folder(model, dataset)
    folder.mkdir(parents=True, exist_ok=True)
    missing = [i for i in sorted(frames) if not depth_path(folder, i).exists()]

    log.info(f"Depth maps: {len(frames) - len(missing)} cached, {len(missing)} to predict")
    log.debug(f"depth cache folder: {folder}")
    t0 = time.perf_counter()
    for i in tqdm(missing, desc="Predicting depth", unit="frame", disable=not missing):
        np.save(depth_path(folder, i), predict_depth(frames[i], model))
    if missing:
        log.debug(f"predicted {len(missing)} depth maps in {time.perf_counter() - t0:.1f} s")
    return folder


def load_depth(folder, frame_id):
    """Depth map HxW float32 of one frame, from the cache."""
    return np.load(depth_path(folder, frame_id))
