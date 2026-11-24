"""
Backbones: find keypoints in an image and give each a visual descriptor.

Add your own:

    @backbone("my_backbone")
    def my_backbone(img_rgb, n=2000) -> Keypoints:
        ...

`size` / `angle` are optional (None if the detector has none); geometric depth descriptors
that need them fall back to a fixed size and angle 0.
"""

from dataclasses import dataclass

import cv2
import numpy as np

BACKBONES = {}


@dataclass
class Keypoints:
    pts: np.ndarray            # N x 2 float32, pixel x/y
    des: np.ndarray            # N x D float32, visual descriptor (unit length)
    size: np.ndarray = None    # N float32, keypoint diameter in px (SIFT), or None
    angle: np.ndarray = None   # N float32, orientation in degrees (SIFT), or None

    def cv_keypoints(self, default_size=8.0):
        """As cv2.KeyPoint list (for OpenCV descriptors); missing size/angle -> defaults."""
        n = len(self.pts)
        size = self.size if self.size is not None else np.full(n, default_size, np.float32)
        angle = self.angle if self.angle is not None else np.zeros(n, np.float32)
        return [cv2.KeyPoint(float(x), float(y), float(s), float(a))
                for (x, y), s, a in zip(self.pts, size, angle)]


def backbone(name):
    """Decorator: register fn(img_rgb, **params) -> Keypoints under `name`."""
    def deco(fn):
        BACKBONES[name] = fn
        return fn
    return deco


@backbone("sift")
def sift(img_rgb, n=2000):
    """SIFT keypoints (difference of Gaussians) + RootSIFT descriptors (128-d)."""
    from geomatch.descriptors import detect_keypoints
    kps, des = detect_keypoints(img_rgb, n=n)
    return Keypoints(pts=np.array([k.pt for k in kps], np.float32).reshape(-1, 2),
                     des=des.astype(np.float32),
                     size=np.array([k.size for k in kps], np.float32),
                     angle=np.array([k.angle for k in kps], np.float32))


@backbone("superpoint")
def superpoint(img_rgb, n=2000):
    """Frozen pretrained SuperPoint: keypoints from its heatmap (NMS 4 px) + 256-d descriptors."""
    from geomatch.superpoint import superpoint as sp
    pts, des = sp(img_rgb, n=n)
    return Keypoints(pts=pts, des=des)
