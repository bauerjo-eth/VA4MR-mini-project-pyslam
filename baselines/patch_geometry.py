"""Local geometry of keypoint patches from a depth map (WP1, input for WP2 / WP3).

Depth map -> point cloud (every valid pixel along its ray, lens distortion removed) -> surface normals with Open3D's
standard estimation (`estimate_normals`: PCA over the k nearest 3D neighbours, default k = 30 as in Open3D),
oriented towards the camera. Neighbours in 3D, not in the image: at object borders the normal comes from the keypoint's
own surface (the background is far away in 3D).
Per keypoint (nearest pixel):
- z           depth at the keypoint pixel
- X           3D point, camera frame (scales with depth)
- normal      unit surface normal, facing the camera
- view_angle  angle normal <-> direction to the camera [deg]; 0 = fronto-parallel
- valid       the keypoint pixel has a depth
normal and view_angle are scale-invariant -> usable with up-to-scale depth. Invalid entries: NaN (valid: False).
"""

import cv2
import numpy as np
import open3d as o3d


def pixel_rays(camera):
    """Ray (x, y, 1) per pixel, lens distortion removed; (H, W, 3). camera: pySLAM camera (K, D)."""
    w, h = int(camera.width), int(camera.height)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    dist = np.zeros(5) if camera.D is None else np.asarray(camera.D, dtype=np.float64)
    xy = cv2.undistortPoints(pix, np.asarray(camera.K, dtype=np.float64), dist).reshape(h, w, 2)
    return np.dstack([xy, np.ones((h, w))])


def project_points(X, camera):
    """3D points (camera frame, N x 3) -> pixels with lens distortion (N x 2)."""
    dist = np.zeros(5) if camera.D is None else np.asarray(camera.D, dtype=np.float64)
    uv, _ = cv2.projectPoints(np.ascontiguousarray(X, dtype=np.float64).reshape(-1, 1, 3),
                              np.zeros(3), np.zeros(3), np.asarray(camera.K, dtype=np.float64), dist)
    return uv.reshape(-1, 2)


def sample_nearest(img, pts):
    """Value at the nearest pixel; NaN outside the image."""
    h, w = img.shape[:2]
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(pts), np.nan, dtype=np.float64)
    out[inside] = img[y[inside], x[inside]]
    return out


def normal_map(depth, rays, knn=30):
    """Unit normal per pixel (H x W x 3, NaN without depth): Open3D estimate_normals on the back-projected depth
    map with `knn` nearest neighbours (30 = Open3D's default), oriented towards the camera centre."""
    h, w = depth.shape
    ok = np.isfinite(depth) & (depth > 0)
    out = np.full((h, w, 3), np.nan)
    if ok.sum() < 3:
        return out
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth[ok][:, None] * rays[ok]))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    out[ok] = np.asarray(pcd.normals)
    return out


def compute_patch_geometry(depth, rays, pts, normals=None):
    """depth HxW (0 / NaN = invalid), rays from pixel_rays, pts N x 2 pixels;
    normals: normal_map(depth, rays) if already computed (reuse for several keypoint sets)."""
    if normals is None:
        normals = normal_map(depth, rays)
    h, w = depth.shape
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    z = np.where(inside, depth[y, x], np.nan).astype(np.float64)
    valid = inside & np.isfinite(z) & (z > 0)
    z[~valid] = np.nan
    X = z[:, None] * rays[y, x]
    normal = np.where(valid[:, None], normals[y, x], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        to_cam = -X / np.linalg.norm(X, axis=1, keepdims=True)
        view_angle = np.degrees(np.arccos(np.clip(np.sum(normal * to_cam, axis=1), -1.0, 1.0)))
    return {"z": z, "X": X, "normal": normal, "view_angle": view_angle, "valid": valid}


def normal_angle_deg(n1, n2):
    """Unsigned angle between plane normals (sign-free), NaN-safe."""
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arccos(np.clip(np.abs(np.sum(n1 * n2, axis=1)), 0.0, 1.0)))
