"""
WP2: learned depth descriptor on top of frozen SuperPoint.

    descriptor = [ SuperPoint (256, frozen) | w * f_theta(depth) ]

f_theta is a small fully convolutional network over the whole (half-resolution) input map,
read out at the SuperPoint keypoints, like SuperPoint's own descriptor head but on depth.
Trainable parameters: theta, the weight w and the softmax temperature (training only).

Input modes (`mode`):
  depth  Depth Anything 3 depth, as scale-free channels: d/dx log z, d/dy log z and
         log z minus its local mean. A global scale error of the predicted depth cancels out.
  image  CONTROL: the same network on the grayscale image (intensity minus local mean and its
         gradients). If 'depth' beats 'image', the gain comes from the depth, not from simply
         training extra layers on KITTI.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 0.5   # inputs are processed at half resolution


def make_input(arr, mode):
    """Depth map (HxW metres) or grayscale image (HxW, 0..255) -> 3xH'xW' float32 tensor at
    half resolution. Computed with torch so it is identical in training and evaluation."""
    a = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)[None, None]
    a = F.interpolate(a, scale_factor=SCALE, mode="area")
    if mode == "depth":
        a = torch.log(a.clamp(min=0.5))
    elif mode == "image":
        a = a / 64.0  # roughly the same range as log depth
    else:
        raise ValueError(f"unknown mode '{mode}'")
    gx = F.pad(a[..., :, 1:] - a[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(a[..., 1:, :] - a[..., :-1, :], (0, 0, 0, 1))
    local = a - F.avg_pool2d(F.pad(a, (7, 7, 7, 7), mode="replicate"), 15, stride=1)
    return torch.cat([gx * 10, gy * 10, local], 1)[0]  # 3 x H' x W'


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
        nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class DepthEncoder(nn.Module):
    """Input 3 x H/2 x W/2 -> descriptor map dim x H/8 x W/8 (receptive field ~150 px at full
    resolution, so it sees the surroundings of a keypoint, not only its patch)."""

    def __init__(self, dim=64, mode="depth", w_init=0.5, tau_init=0.05):
        super().__init__()
        self.mode, self.dim = mode, dim
        self.net = nn.Sequential(
            block(3, 32), block(32, 32),
            block(32, 64, stride=2), block(64, 64),
            block(64, 128, stride=2), block(128, 128),
            block(128, 128, dilation=2), block(128, 128, dilation=4),
            nn.Conv2d(128, dim, 1))
        self.log_w = nn.Parameter(torch.tensor(float(np.log(w_init))))
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))

    @property
    def w(self):
        return self.log_w.exp()

    def forward(self, inp):
        """inp: B x 3 x h x w -> B x dim x h/4 x w/4."""
        return self.net(inp)

    def describe(self, fmap, pts, H, W):
        """Read the map of one image (1 x dim x h x w) at pixel positions pts (Nx2 tensor, in
        the full-resolution HxW image). Returns N x dim, unit length (NOT yet multiplied by w)."""
        # bilinear interpolation written out (grid_sample has no backward pass on Apple MPS)
        _, C, h, w = fmap.shape
        x = ((pts[:, 0] + 0.5) * w / W - 0.5).clamp(0, w - 1)
        y = ((pts[:, 1] + 0.5) * h / H - 0.5).clamp(0, h - 1)
        x0, y0 = x.floor().long().clamp(max=w - 2), y.floor().long().clamp(max=h - 2)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        flat = fmap[0].reshape(C, -1).T                      # (h*w) x C
        at = lambda yy, xx: flat[yy * w + xx]
        d = ((1 - fx) * (1 - fy) * at(y0, x0) + fx * (1 - fy) * at(y0, x0 + 1)
             + (1 - fx) * fy * at(y0 + 1, x0) + fx * fy * at(y0 + 1, x0 + 1))
        return F.normalize(d, dim=1)


def save_model(model, path, extra):
    torch.save({"state_dict": model.state_dict(), "dim": model.dim, "mode": model.mode, **extra}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DepthEncoder(dim=ck["dim"], mode=ck["mode"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
