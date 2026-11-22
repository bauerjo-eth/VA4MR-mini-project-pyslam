"""GT depth + GT pose -> true point of a keypoint in another view (shared by the WP1 baselines, reusable in WP4).

- GT depth per frame: dataset depth (RGB-D, e.g. TUM Kinect) or KITTI raw LiDAR (kitti_lidar.py, sparse)
- sparse GT: gt_sample_radius_px > 0 -> median of the valid GT depths in a small window
- gt_reproject: point in ref -> GT depth -> 3D -> GT pose -> pixel in cur, only where visible in cur
"""

import os
import warnings

import numpy as np

from pyslam.utilities.geometry import inv_T

from kitti_lidar import KittiLidarDepth
from patch_geometry import project_points, sample_nearest


def finite(*arrays):
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a) & (a > 0)
    return ok


def sample_depth(depth, pts, radius=0):
    """GT depth at points: nearest pixel (radius 0) or median of the valid depths in the window (sparse GT)."""
    if radius == 0:
        return sample_nearest(depth, pts)
    h, w = depth.shape
    offs = np.arange(-radius, radius + 1)
    dy, dx = (o.ravel() for o in np.meshgrid(offs, offs, indexing="ij"))
    x = np.round(pts[:, 0]).astype(int)[:, None] + dx[None]
    y = np.round(pts[:, 1]).astype(int)[:, None] + dy[None]
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vals = np.where(inside, depth[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)], np.nan).astype(np.float64)
    vals[~(vals > 0)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no GT in the window -> NaN
        return np.nanmedian(vals, axis=1)


def gt_depth_source(params, dataset, camera):
    """GT depth per frame [m], NaN = invalid: dataset depth (RGB-D) or KITTI raw LiDAR."""
    src = params.get("gt_depth") or {"type": "dataset"}
    if src["type"] == "kitti_lidar":
        if not os.path.isdir(src["drive_path"]):
            raise SystemExit(f"KITTI raw LiDAR missing: {src['drive_path']} (download: configs/depth_kitti06_lidar.yaml)")
        return KittiLidarDepth(src["drive_path"], src["calib_path"], camera.width, camera.height, src.get("cam", 2))
    if src["type"] != "dataset":
        raise ValueError(f"unknown gt_depth type: {src['type']}")

    def from_dataset(frame_id):
        raw = dataset.getDepth(frame_id)
        if raw is None:
            raise SystemExit("dataset without depth: set sensor_type: rgbd, or gt_depth")
        return raw.astype(np.float32) * camera.depth_factor

    return from_dataset


def ray_at(rays, pts):
    """Ray of the nearest pixel (clipped to the image)."""
    h, w = rays.shape[:2]
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return rays[y, x]


def project_or_nan(X, camera):
    """Pixels of 3D points in front of the camera, NaN otherwise."""
    uv = np.full((len(X), 2), np.nan)
    front = np.isfinite(X).all(axis=1) & (X[:, 2] > 0)
    if front.any():
        uv[front] = project_points(X[front], camera)
    return uv


def gt_reproject(ref, cur, pts, rays, camera, params):
    """Points of ref -> GT depth + GT pose -> pixels in cur (NaN where unknown).

    ok: GT depth in ref, in front of cur and inside its image, visible in cur (GT depth there agrees within
    `visibility_rel_tol`, otherwise occluded or no GT)
    """
    T_cur_ref = inv_T(cur["T_w_c"]) @ ref["T_w_c"]
    R, t = T_cur_ref[:3, :3], T_cur_ref[:3, 3]
    radius = params.get("gt_sample_radius_px", 0)
    z_ref = sample_depth(ref["gt"], pts, radius)
    ok = finite(z_ref)
    X_cur = (z_ref[:, None] * ray_at(rays, pts)) @ R.T + t
    with np.errstate(invalid="ignore"):
        ok &= X_cur[:, 2] > 0
    # only rays inside the camera's field of view (distortion polynomial folds back outside)
    with np.errstate(invalid="ignore", divide="ignore"):
        xy = X_cur[:, :2] / X_cur[:, 2:3]
    lo, hi = rays[..., :2].reshape(-1, 2).min(0), rays[..., :2].reshape(-1, 2).max(0)
    ok &= np.all((xy >= lo) & (xy <= hi), axis=1)
    u_cur = np.full((len(pts), 2), np.nan)
    u_cur[ok] = project_or_nan(X_cur[ok], camera)
    h, w = cur["gt"].shape
    with np.errstate(invalid="ignore", divide="ignore"):
        ok &= (u_cur[:, 0] >= 0) & (u_cur[:, 0] < w) & (u_cur[:, 1] >= 0) & (u_cur[:, 1] < h)
        z_seen = sample_depth(cur["gt"], np.nan_to_num(u_cur), radius)
        ok &= np.abs(z_seen - X_cur[:, 2]) / X_cur[:, 2] < params["visibility_rel_tol"]
    u_cur[~ok] = np.nan
    return u_cur, ok, R, t


def gt_correspondences(ref, cur, pts, rays, camera, params):
    """Keypoints of ref -> GT depth + GT pose -> pixels in cur; keeps points visible in cur."""
    u_cur, ok, R, t = gt_reproject(ref, cur, pts, rays, camera, params)
    return pts[ok], u_cur[ok], R, t
