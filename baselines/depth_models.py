"""Depth models (WP1, reusable in WP2-WP4): one interface, depth in the pixel grid of the input.

predict(images_rgb) -> [(depth HxW float32, conf HxW or None), ...], one entry per image
- input: RGB uint8 (pySLAM datasets return BGR -> convert first)
- metric = True: depth in meters, else up to scale (align before use)
- multiview = True: all images in one forward pass -> common scale across the images
- invalid pixels: NaN

Models (name -> make_depth_model):
- DA3* (Depth Anything 3), loaded via pySLAM DepthEstimatorDepthAnythingV3
  - DA3METRIC-*: meters = net output * f / 300 (DA3 README), f = focal at DA3 processing resolution
    (as in DA3's own nested model); TUM fr1/desk: GT / raw 1.45, f_proc / 300 1.36, f_orig / 300 1.72
  - DA3NESTED-*: already meters
  - DA3MONO-*: monocular, up to scale
  - DA3-* (any-view, e.g. DA3-LARGE-1.1; original DA3-LARGE deprecated): multi-view, up to scale
  - pySLAM's own infer() gets BGR images and drops the metric factor -> model called directly here
- VGGT-OMEGA: VGGT-Omega 1B (512), multi-view, up to scale
  - checkpoint gated on Hugging Face: access request + `hf auth login`, then downloaded on first use
  - images with height / width outside [0.5, 2] are center-cropped by VGGT-Omega -> NaN outside the crop
- precision: DA3 and VGGT-Omega use bf16 / fp16 autocast on CUDA-type GPUs by themselves
  - FP32_ONLY (env DEPTH_FP32=1 / 0, default 1 on ROCm builds of PyTorch): both run in fp32 instead
  - reason: ROCm 10.2 under WSL2 on the Radeon RX 9070 XT (gfx1201) has no half-precision GEMM kernels
    (~500x slower); fp32 is fast there (DA3 metric 0.1 s / image)
- new model: subclass DepthModel, add it to make_depth_model
"""

import os
import types

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # DA3 logs every forward pass at INFO

import cv2
import numpy as np
import torch
from PIL import Image

DA3_CANONICAL_FOCAL = 300.0  # DA3 README: metric_depth = focal * net_output / 300
FP32_ONLY = os.environ.get("DEPTH_FP32", "1" if torch.version.hip else "0") == "1"


def _da3_forward_fp32(self, image, extrinsics=None, intrinsics=None, export_feat_layers=None, infer_gs=False):
    """DA3 api forward without its bf16 / fp16 autocast (otherwise the same call)."""
    with torch.no_grad():
        return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def resize_to(x, width, height):
    if x is None or x.shape[:2] == (height, width):
        return x
    return cv2.resize(x, (width, height), interpolation=cv2.INTER_LINEAR)


class DepthModel:
    name = ""
    metric = False
    multiview = False

    def predict(self, images_rgb):
        raise NotImplementedError


class DA3Model(DepthModel):
    def __init__(self, variant, camera):
        from pyslam.depth_estimation.depth_estimator_depth_anything_v3 import DepthEstimatorDepthAnythingV3

        key = variant.upper()
        self.name = variant
        self.camera = camera
        self.focal_scaling = key.startswith("DA3METRIC")
        self.metric = key.startswith(("DA3METRIC", "DA3NESTED"))
        self.multiview = not key.startswith(("DA3METRIC", "DA3MONO"))
        # pySLAM wrapper: model on pick_device() + model download (Hugging Face cache)
        estimator = DepthEstimatorDepthAnythingV3(device=pick_device(), camera=camera,
                                                  model_type=f"depth-anything/{variant}")
        self.model = estimator.model
        if FP32_ONLY:
            self.model.forward = types.MethodType(_da3_forward_fp32, self.model)

    def _forward(self, images_rgb):
        h, w = images_rgb[0].shape[:2]
        with torch.no_grad():
            pred = self.model.inference(list(images_rgb))
        out = []
        for k in range(len(images_rgb)):
            depth = to_numpy(pred.depth[k])
            if self.focal_scaling:
                hp, wp = depth.shape
                focal = 0.5 * (self.camera.fx * wp / w + self.camera.fy * hp / h)
                depth = depth * (focal / DA3_CANONICAL_FOCAL)
            conf = to_numpy(pred.conf[k]) if pred.conf is not None else None
            out.append((resize_to(depth, w, h), resize_to(conf, w, h)))
        return out

    def predict(self, images_rgb):
        if self.multiview:
            return self._forward(images_rgb)
        return [self._forward([img])[0] for img in images_rgb]


class VGGTOmegaModel(DepthModel):
    repo_id, filename = "facebook/VGGT-Omega", "vggt_omega_1b_512.pt"
    resolution, patch_size = 512, 16

    def __init__(self, name):
        from huggingface_hub import hf_hub_download
        from vggt_omega.models import VGGTOmega

        self.name = name
        self.multiview = True
        self.device = pick_device()
        path = hf_hub_download(self.repo_id, self.filename)  # gated: access request + `hf auth login`
        model = VGGTOmega(autocast=not FP32_ONLY)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model = model.to(self.device).eval()

    def _preprocess(self, img_rgb):
        """Same steps as vggt_omega.utils.load_fn.load_and_preprocess_images (balanced mode), plus crop box."""
        from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio

        pil = Image.fromarray(img_rgb)
        crop = _crop_to_supported_aspect_ratio(pil)
        (w, h), (cw, ch) = pil.size, crop.size
        box = ((w - cw) // 2, (h - ch) // 2, cw, ch)  # center crop (left, top, width, height)
        th, tw = _balanced_target_shape(ch / max(cw, 1), self.resolution, self.patch_size)
        tensor = torch.from_numpy(np.asarray(crop.resize((tw, th), Image.Resampling.BICUBIC), dtype=np.float32))
        return tensor.permute(2, 0, 1) / 255.0, box

    def predict(self, images_rgb):
        h, w = images_rgb[0].shape[:2]
        tensors, boxes = zip(*(self._preprocess(img) for img in images_rgb))
        with torch.inference_mode():
            pred = self.model(torch.stack(tensors).to(self.device))
        out = []
        for k, (x0, y0, cw, ch) in enumerate(boxes):
            depth = np.full((h, w), np.nan, np.float32)
            conf = np.full((h, w), np.nan, np.float32)
            depth[y0:y0 + ch, x0:x0 + cw] = resize_to(to_numpy(pred["depth"][0, k]).squeeze(-1), cw, ch)
            conf[y0:y0 + ch, x0:x0 + cw] = resize_to(to_numpy(pred["depth_conf"][0, k]), cw, ch)
            out.append((depth, conf))
        return out


def make_depth_model(name, camera):
    key = name.upper()
    if key.startswith("DA3"):
        return DA3Model(name, camera)
    if key == "VGGT-OMEGA":
        return VGGTOmegaModel(name)
    raise ValueError(f"unknown depth model: {name}")
