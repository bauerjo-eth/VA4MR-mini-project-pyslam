"""
Descriptors: describe each keypoint so that the same 3D point looks alike in both images.

  pts   Nx2 pixel coordinates (x, y)
  des   NxD float32 descriptors (L2)
  depth HxW float32

Part 1 holds the algorithms. Part 2 is the registry: it maps a name from the evaluation
config to a descriptor method. To add your own method:

    @register_descriptor("my_descriptor")
    def my_descriptor(img, depth, **params):
        ...
        return pts, des          # Nx2 keypoints, NxD descriptors

and use `type: my_descriptor` in the config.
"""

import cv2
import numpy as np


# ======================= Part 1: algorithms =======================

# ---------- appearance baseline ----------

def detect(img_rgb, n=2000):
    """RootSIFT. Returns pts (Nx2), sizes (N), des (Nx128)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kps, des = cv2.SIFT_create(nfeatures=n).detectAndCompute(gray, None)
    des = np.sqrt(des / (des.sum(axis=1, keepdims=True) + 1e-7))  # RootSIFT
    pts = np.array([k.pt for k in kps], np.float32)
    sizes = np.array([k.size for k in kps], np.float32)
    return pts, sizes, des.astype(np.float32)


# ---------- depth helpers ----------

def depth_at(depth, pts):
    """Depth at pixel positions of keypoints (nearest pixel)."""
    h, w = depth.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return depth[y, x]


# ---------- WP2 : geometric descriptor, concatenated ----------

def depth_descriptor(depth, pts, sizes, grid=5, scale=3.0):
    """
    Relative depth around each keypoint: log(depth / center depth) on a grid x grid
    lattice spanning `scale * keypoint size`. Using log ratios makes it independent of the
    global depth scale. Returns N x grid^2, each row unit length.
    """
    r = np.linspace(-1, 1, grid)
    dx, dy = np.meshgrid(r, r)
    offsets = np.stack([dx.ravel(), dy.ravel()], 1)                      # G x 2
    samples = pts[:, None, :] + (scale * sizes)[:, None, None] * offsets  # N x G x 2

    z = depth_at(depth, samples.reshape(-1, 2)).reshape(len(pts), -1)
    zc = depth_at(depth, pts)[:, None]
    g = np.log(np.maximum(z, 1e-3) / np.maximum(zc, 1e-3))
    return (g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def concat(des, g, w):
    """[appearance | w * geometry]. w = 0 is exactly the appearance baseline."""
    return np.hstack([des, w * g]).astype(np.float32)


# ======================= Part 2: registry =======================

DESCRIPTORS = {}  # name -> function(img, depth, **params) -> (pts, des)


def register_descriptor(name):
    """Decorator: add a descriptor method to the registry under `name`."""
    def deco(fn):
        DESCRIPTORS[name] = fn
        return fn
    return deco


def available_descriptors():
    return sorted(DESCRIPTORS)


def compute_descriptors(name, img, depth, **params):
    """Run the registered descriptor method `name`. Returns pts (Nx2) and des (NxD)."""
    if name not in DESCRIPTORS:
        raise KeyError(f"unknown descriptor '{name}', available: {available_descriptors()}")
    return DESCRIPTORS[name](img, depth, **params)


@register_descriptor("rootsift")
def _rootsift(img, depth, n=2000):
    """Appearance only (baseline). Ignores the depth map."""
    pts, _, des = detect(img, n=n)
    return pts, des


@register_descriptor("rootsift_depth")
def _rootsift_depth(img, depth, n=2000, w=0.3, grid=5, scale=3.0):
    """[RootSIFT | w * depth descriptor]. w = 0 matches exactly like plain RootSIFT."""
    pts, sizes, des = detect(img, n=n)
    g = depth_descriptor(depth, pts, sizes, grid=grid, scale=scale)
    return pts, concat(des, g, w)
