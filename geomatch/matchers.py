"""
Matchers: pair up the descriptors of image 1 with those of image 2.

Part 1 holds the algorithms. Part 2 is the registry: it maps a name from the evaluation
config to a matcher method. To add your own method:

    @register_matcher("my_matcher")
    def my_matcher(des1, des2, **params):
        ...
        return i1, i2            # index arrays: des1[i1[k]] is matched to des2[i2[k]]

and use `type: my_matcher` in the config.
"""

import cv2
import numpy as np


# ======================= Part 1: algorithms =======================

def match(des1, des2, ratio=0.8):
    """Nearest neighbour + Lowe ratio test. Returns (i1, i2) index arrays."""
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return np.array([m.queryIdx for m in good]), np.array([m.trainIdx for m in good])


# ======================= Part 2: registry =======================

MATCHERS = {}  # name -> function(des1, des2, **params) -> (i1, i2)


def register_matcher(name):
    """Decorator: add a matcher method to the registry under `name`."""
    def deco(fn):
        MATCHERS[name] = fn
        return fn
    return deco


def available_matchers():
    return sorted(MATCHERS)


def match_descriptors(name, des1, des2, **params):
    """Run the registered matcher method `name`. Returns index arrays (i1, i2)."""
    if name not in MATCHERS:
        raise KeyError(f"unknown matcher '{name}', available: {available_matchers()}")
    return MATCHERS[name](des1, des2, **params)


@register_matcher("mutual_nn")
def _mutual_nn(des1, des2, ratio=None):
    """Mutual nearest neighbours (L2): i and j are each other's closest descriptor. The usual
    matcher for SuperPoint descriptors. With `ratio`, also Lowe's ratio test in direction
    1 -> 2 (best distance < ratio * second best), the same rule as nn_ratio uses for SIFT."""
    d = (des1 ** 2).sum(1)[:, None] + (des2 ** 2).sum(1)[None] - 2 * des1 @ des2.T
    d = np.sqrt(np.maximum(d, 0))
    j = d.argmin(1)
    i = d.argmin(0)
    keep = i[j] == np.arange(len(des1))
    if ratio is not None and d.shape[1] > 1:
        two = np.partition(d, 1, axis=1)[:, :2]
        keep &= two[:, 0] < ratio * two[:, 1]
    return np.where(keep)[0], j[keep]


@register_matcher("nn_ratio")
def _nn_ratio(des1, des2, ratio=0.8):
    """Nearest neighbour + Lowe ratio test (baseline)."""
    return match(des1, des2, ratio=ratio)
