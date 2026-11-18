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

def root_sift(des):
    """SIFT -> RootSIFT: L1-normalize, then square root. Each row then has unit L2 length."""
    return np.sqrt(des / (des.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)


def detect_keypoints(img_rgb, n=2000):
    """SIFT keypoints (cv2.KeyPoint: position, size, orientation) and RootSIFT descriptors."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kps, des = cv2.SIFT_create(nfeatures=n).detectAndCompute(gray, None)
    return kps, root_sift(des)


def detect(img_rgb, n=2000):
    """RootSIFT. Returns pts (Nx2), sizes (N), des (Nx128)."""
    kps, des = detect_keypoints(img_rgb, n)
    pts = np.array([k.pt for k in kps], np.float32)
    sizes = np.array([k.size for k in kps], np.float32)
    return pts, sizes, des


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


# ---------- WP2 : SIFT computed on the depth image ----------

def depth_to_image(depth, transform="log", z_min=1.0, z_max=100.0):
    """Depth map -> 8-bit image for SIFT, with a FIXED mapping, so the same surface looks the
    same in every frame (a per-image min/max rescaling would not).
      log:     equal depth ratios give equal steps -> changes near the camera count more,
               and a global scale error of the predicted depth cancels out
      inverse: 1/z, weights near points even more strongly
    Depths outside [z_min, z_max] metres are clipped."""
    z = np.clip(depth, z_min, z_max)
    if transform == "log":
        v = (np.log(z) - np.log(z_min)) / (np.log(z_max) - np.log(z_min))
    elif transform == "inverse":
        v = (1 / z - 1 / z_max) / (1 / z_min - 1 / z_max)
    else:
        raise ValueError(f"unknown depth transform '{transform}', use 'log' or 'inverse'")
    return np.round(255 * v).astype(np.uint8)  # OpenCV's SIFT needs an 8-bit image


def depth_sift(depth_img, kps):
    """RootSIFT descriptors computed on the depth image at the given image keypoints (same
    position, size and orientation), so they describe the same patch as the appearance
    descriptor. Describes the pattern of depth gradients = the local surface shape."""
    _, des = cv2.SIFT_create().compute(depth_img, kps)
    if des is None or len(des) != len(kps):
        raise RuntimeError("SIFT did not return one depth descriptor per keypoint")
    return root_sift(des)


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


@register_descriptor("depthsift")
def _depthsift(img, depth, n=2000, transform="log", z_min=1.0, z_max=100.0):
    """Depth only: keypoints from the image, descriptor from the depth image. Shows how
    distinctive the depth alone is."""
    kps, _ = detect_keypoints(img, n=n)
    pts = np.array([k.pt for k in kps], np.float32)
    return pts, depth_sift(depth_to_image(depth, transform, z_min, z_max), kps)


@register_descriptor("rootsift_depthsift")
def _rootsift_depthsift(img, depth, n=2000, w=0.5, transform="log", z_min=1.0, z_max=100.0):
    """[RootSIFT on the image | w * RootSIFT on the depth image], at the same keypoints. Both
    parts have unit length, so w is directly the weight of depth vs appearance."""
    kps, des = detect_keypoints(img, n=n)
    pts = np.array([k.pt for k in kps], np.float32)
    des_depth = depth_sift(depth_to_image(depth, transform, z_min, z_max), kps)
    return pts, concat(des, des_depth, w)
