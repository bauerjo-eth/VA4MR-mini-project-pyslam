"""
Depth descriptors: the part appended to the visual descriptor,

    descriptor = [ visual (backbone) | w * depth descriptor ]

Two kinds:

1. GEOMETRIC (hand-designed): a function of the depth map at the keypoints.

    @geometric("my_desc")
    def my_desc(depth, img, kp, **params) -> np.ndarray:   # N x D, each row unit length
        ...

   In the experiment file: {type: my_desc, w: 0.3, <params>}. w is fixed.

2. LEARNED: a fully convolutional network over input channels; the benchmark reads its
   output map at the keypoints (bilinear), normalises, and learns w with it (InfoNCE on
   ground-truth correspondences, see geomatch/bench/train_descriptor.py).

    @network("my_cnn")
    class MyCnn(nn.Module):
        def __init__(self, in_ch, dim): ...
        def forward(self, x):            # B x in_ch x h x w  ->  B x dim x h' x w'
            ...

    @input_channels("my_input")
    def my_input(depth, img_gray) -> torch.Tensor:   # C x h x w (any resolution)
        ...

   In the experiment file: {learned: my_cnn, input: depth, dim: 64}. Pair every learned
   depth descriptor with the same network on input: image (the control), otherwise a gain
   cannot be attributed to depth.
"""

import numpy as np
import torch
import torch.nn as nn

GEOMETRIC = {}
NETWORKS = {}
INPUTS = {}


def geometric(name):
    def deco(fn):
        GEOMETRIC[name] = fn
        return fn
    return deco


def network(name):
    def deco(cls):
        NETWORKS[name] = cls
        return cls
    return deco


def input_channels(name):
    def deco(fn):
        INPUTS[name] = fn
        return fn
    return deco


# ======================= geometric (hand-designed) =======================

@geometric("log_grid")
def log_grid(depth, img, kp, grid=5, scale=3.0, default_size=8.0):
    """Gregor (WP2): log(depth / centre depth) on a grid x grid lattice spanning
    scale * keypoint size around the keypoint. Keypoints without a size use default_size."""
    from geomatch.descriptors import depth_descriptor
    size = kp.size if kp.size is not None else np.full(len(kp.pts), default_size, np.float32)
    return depth_descriptor(depth, kp.pts, size, grid=grid, scale=scale)


@geometric("depth_sift")
def depth_sift(depth, img, kp, transform="log", z_min=1.0, z_max=100.0, default_size=8.0):
    """Jonas (WP2): RootSIFT computed on the (log-)depth image at the same keypoints (same
    position, size, orientation). Keypoints without size/orientation: default_size, angle 0."""
    from geomatch.descriptors import depth_sift as dsift, depth_to_image
    return dsift(depth_to_image(depth, transform, z_min, z_max), kp.cv_keypoints(default_size))


# ======================= learned: input channels =======================

@input_channels("depth")
def depth_input(depth, img_gray):
    """Scale-free channels of DA3 depth at half resolution: d/dx log z, d/dy log z, log z minus
    its 15x15 local mean. A global scale error of the predicted depth cancels out."""
    from geomatch.depth_encoder import make_input
    return make_input(depth, "depth")


@input_channels("image")
def image_input(depth, img_gray):
    """CONTROL: the same channels computed on the grey image instead of depth."""
    from geomatch.depth_encoder import make_input
    return make_input(img_gray.astype(np.float32), "image")


@input_channels("depth+image")
def depth_image_input(depth, img_gray):
    """Both: 6 channels. If this beats depth and image alone, depth is (partly) complementary."""
    return torch.cat([depth_input(depth, img_gray), image_input(depth, img_gray)], 0)


# ======================= learned: networks =======================

def conv_block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


@network("cnn_small")
class CnnSmall(nn.Module):
    """The first WP2 network (0.59M parameters): 8 conv blocks, 2 downsamplings, dilated convs
    at the end (receptive field ~150 px at full resolution), 1x1 conv to dim."""

    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, 32), conv_block(32, 32),
            conv_block(32, 64, stride=2), conv_block(64, 64),
            conv_block(64, 128, stride=2), conv_block(128, 128),
            conv_block(128, 128, dilation=2), conv_block(128, 128, dilation=4),
            nn.Conv2d(128, dim, 1))

    def forward(self, x):
        return self.net(x)


@network("cnn_tiny")
class CnnTiny(nn.Module):
    """Small local network (~40k parameters, receptive field ~40 px): local surface shape only."""

    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.net = nn.Sequential(conv_block(in_ch, 32), conv_block(32, 32, stride=2),
                                 conv_block(32, 64), conv_block(64, 64, stride=2), nn.Conv2d(64, dim, 1))

    def forward(self, x):
        return self.net(x)
