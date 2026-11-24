"""
Full descriptor of a frame for one (backbone, depth descriptor) combination:

    [ visual (backbone) | w * depth descriptor ]

and the wrapper that turns any registered network into a learned depth descriptor
(LearnedBranch: network + learned weight w + temperature tau, read out at keypoints).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomatch.bench.common import is_learned, keypoints, split_params
from geomatch.components.depth_descriptors import GEOMETRIC, INPUTS, NETWORKS


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def desc_device_free():
    """Release cached GPU buffers (the MPS allocator otherwise grows with every new keypoint
    count it sees)."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class LearnedBranch(nn.Module):
    """network(input channels) -> map, read at keypoints, unit length; w and tau learned."""

    def __init__(self, net_name, input_name, dim=64, w_init=0.5, tau_init=0.05, **net_kwargs):
        super().__init__()
        self.net_name, self.input_name, self.dim, self.net_kwargs = net_name, input_name, dim, net_kwargs
        in_ch = INPUTS[input_name](np.ones((64, 64), np.float32), np.ones((64, 64), np.uint8)).shape[0]
        self.net = NETWORKS[net_name](in_ch=in_ch, dim=dim, **net_kwargs)
        self.log_w = nn.Parameter(torch.tensor(float(np.log(w_init))))
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))

    @property
    def w(self):
        return self.log_w.exp()

    @staticmethod
    def make_input(input_name, depth, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return INPUTS[input_name](depth, gray)

    def forward(self, inp):
        return self.net(inp)

    @staticmethod
    def describe(fmap, pts, H, W):
        """Bilinear read-out of a 1 x C x h x w map at full-resolution pixel positions pts
        (N x 2), unit length. Written out because grid_sample has no backward on Apple MPS."""
        _, C, h, w = fmap.shape
        x = ((pts[:, 0] + 0.5) * w / W - 0.5).clamp(0, w - 1)
        y = ((pts[:, 1] + 0.5) * h / H - 0.5).clamp(0, h - 1)
        x0, y0 = x.floor().long().clamp(max=w - 2), y.floor().long().clamp(max=h - 2)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        flat = fmap[0].reshape(C, -1).T
        at = lambda yy, xx: flat[yy * w + xx]
        d = ((1 - fx) * (1 - fy) * at(y0, x0) + fx * (1 - fy) * at(y0, x0 + 1)
             + (1 - fx) * fy * at(y0 + 1, x0) + fx * fy * at(y0 + 1, x0 + 1))
        return F.normalize(d, dim=1)

    def save(self, path, extra):
        torch.save({"state_dict": self.state_dict(), "net": self.net_name, "input": self.input_name,
                    "dim": self.dim, "net_kwargs": self.net_kwargs, **extra}, path)

    @classmethod
    def load(cls, path, dev="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(ck["net"], ck["input"], dim=ck["dim"], **ck["net_kwargs"])
        m.load_state_dict(ck["state_dict"])
        return m.to(dev).eval(), ck


def make_branch(spec):
    p = dict(spec)
    return LearnedBranch(p.pop("learned"), p.pop("input"), dim=p.pop("dim", 64), **p)


class Describer:
    """Computes full descriptors for one (backbone, depth descriptor) combination."""

    def __init__(self, exp, backbone, depth_name, model_path=None):
        self.exp, self.backbone, self.depth_name = exp, backbone, depth_name
        self.spec = exp["depth_descriptors"][depth_name]
        self.model = None
        if is_learned(self.spec):
            self.model, ck = LearnedBranch.load(model_path, device())
            self.trained_on = set(ck["train_sequences"]) | set(ck["val_sequences"])

    @torch.no_grad()
    def __call__(self, seq, i, depth=None, img=None, allow_trained=False):
        """-> (pts N x 2, descriptor N x D float32) for frame i of seq. A learned descriptor
        refuses sequences it was trained on, unless allow_trained (filter training on TRAIN)."""
        kp = keypoints(self.exp, self.backbone, seq, i)
        if self.spec is None:
            return kp.pts, kp.des
        depth = seq.depth(i) if depth is None else depth
        if is_learned(self.spec):
            if seq.name in self.trained_on and not allow_trained:
                raise ValueError(f"{self.depth_name}: model was trained/validated on {seq.name}")
            img = seq.image(i) if img is None else img
            dev = device()
            inp = LearnedBranch.make_input(self.model.input_name, depth, img)
            fmap = self.model(inp[None].to(dev))
            d = self.model.describe(fmap, torch.from_numpy(kp.pts).to(dev), *depth.shape[:2])
            g, w = d.cpu().numpy(), float(self.model.w)
        else:
            kind, params = split_params(self.spec)
            w = params.pop("w")
            img = seq.image(i) if img is None else img
            g = GEOMETRIC[kind](depth, img, kp, **params)
        return kp.pts, np.hstack([kp.des, w * g]).astype(np.float32)
