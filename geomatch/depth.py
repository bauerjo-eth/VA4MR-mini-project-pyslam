"""Depth Anything 3: image in, depth map out."""

import logging
import os
import warnings

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")  # DA3 otherwise prints timings for every image
warnings.filterwarnings("ignore", message=r"register_new_resolver\(\) is deprecated")  # harmless, from DA3

_model = None


def predict_depth(img_rgb, model_name="depth-anything/DA3METRIC-LARGE"):
    """RGB uint8 HxWx3 -> depth HxW float32 (same size as the image)."""
    global _model
    if _model is None:
        from depth_anything_3.api import DepthAnything3

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        _model = DepthAnything3.from_pretrained(model_name).to(device).eval()
        logging.getLogger("geomatch").debug(f"DA3 loaded on {device}")

    with torch.no_grad():
        depth = _model.inference([img_rgb]).depth[0]
    depth = np.asarray(depth.cpu() if torch.is_tensor(depth) else depth, dtype=np.float32)
    return cv2.resize(depth, (img_rgb.shape[1], img_rgb.shape[0]))
