"""
SuperPoint (Magic Leap, pretrained, frozen): image in, keypoints + 256-d descriptors out.

Uses the network definition and weights in thirdparty/superpoint (superpoint_v1.pth).
The weights are never trained here: WP2 keeps SuperPoint frozen and only learns the depth
branch on top of it (see depth_encoder.py).

    pts, des = superpoint(img_rgb, n=2000)   # Nx2 pixel coords (x, y), Nx256 unit-length float32
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SP_DIR = Path(__file__).resolve().parent.parent / "thirdparty" / "superpoint"
_net = None


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    """SuperPointNet with the pretrained weights, in eval mode, on the GPU if there is one."""
    global _net
    if _net is None:
        spec = importlib.util.spec_from_file_location("demo_superpoint", SP_DIR / "demo_superpoint.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        net = mod.SuperPointNet()
        net.load_state_dict(torch.load(SP_DIR / "superpoint_v1.pth", map_location="cpu"))
        _net = net.to(device()).eval()
        for p in _net.parameters():
            p.requires_grad_(False)  # frozen
    return _net


@torch.no_grad()
def superpoint(img_rgb, n=2000, conf_thresh=0.005, nms_radius=4, border=4):
    """Detect up to n keypoints (highest scores after NMS) and describe them.
    Returns pts (Nx2 float32, pixel x/y in the input image) and des (Nx256 float32, unit length)."""
    net = _load()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = gray.shape
    Hp, Wp = -(-H // 8) * 8, -(-W // 8) * 8  # pad to a multiple of 8 (the network downsamples 3x)
    x = torch.zeros(1, 1, Hp, Wp)
    x[0, 0, :H, :W] = torch.from_numpy(gray)
    semi, desc = net(x.to(device()))

    # detector head: 65 channels per 8x8 cell (64 positions + "no keypoint") -> full-res heatmap
    prob = F.softmax(semi, dim=1)[:, :-1]
    heat = F.pixel_shuffle(prob, 8)[0, 0, :H, :W]
    # NMS: keep only local maxima within nms_radius
    local_max = F.max_pool2d(heat[None, None], 2 * nms_radius + 1, stride=1, padding=nms_radius)[0, 0]
    keep = (heat == local_max) & (heat >= conf_thresh)
    keep[:border], keep[-border:], keep[:, :border], keep[:, -border:] = False, False, False, False
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = heat[ys, xs]
    order = torch.argsort(scores, descending=True)[:n]
    pts = torch.stack([xs[order], ys[order]], 1).float()

    des = sample_descriptors(desc, pts, Hp, Wp)
    return pts.cpu().numpy().astype(np.float32), des.cpu().numpy().astype(np.float32)


def sample_descriptors(desc_map, pts, H, W):
    """Bilinear readout of a 1xCxhxw descriptor map at pixel positions pts (Nx2, in an HxW
    image), as in the original SuperPoint demo. Returns NxC, unit length."""
    grid = pts.clone()
    grid[:, 0] = grid[:, 0] / (W / 2.0) - 1
    grid[:, 1] = grid[:, 1] / (H / 2.0) - 1
    d = F.grid_sample(desc_map, grid[None, None], mode="bilinear", align_corners=True)[0, :, 0].T
    return F.normalize(d, dim=1)
