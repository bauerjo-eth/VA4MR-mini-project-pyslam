"""KITTI raw LiDAR as GT depth in a rectified KITTI camera image (as in the Eigen-split evaluation).

- KITTI odometry 06 = raw 2011_09_30_drive_0020, frames 0..1100 (same index)
- calib: calib_cam_to_cam.txt (P_rect_0X, R_rect_00), calib_velo_to_cam.txt (R, T)
- point -> image: P_rect_0X @ R_rect_00 @ [R | T] @ X_velo, depth = 3rd coordinate
- only points in front of the LiDAR; several points on one pixel -> nearest kept
- sparse: ~5 % of the pixels, lower image part only (no LiDAR above), max range ~80 m
- object borders: LiDAR mounted higher / behind the camera -> background points can show through
"""

import numpy as np


def read_calib(path):
    """KITTI calib txt -> {key: float array}; non-numeric lines skipped."""
    out = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            try:
                out[key.strip()] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return out


class KittiLidarDepth:
    """depth(frame_id) -> HxW float32 depth [m] in camera `cam` (2 = left color), NaN = no LiDAR point."""

    def __init__(self, drive_path, calib_path, width, height, cam=2):
        c2c = read_calib(f"{calib_path}/calib_cam_to_cam.txt")
        v2c = read_calib(f"{calib_path}/calib_velo_to_cam.txt")
        velo_to_cam = np.eye(4)
        velo_to_cam[:3, :3], velo_to_cam[:3, 3] = v2c["R"].reshape(3, 3), v2c["T"]
        rect = np.eye(4)
        rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
        self.P = c2c[f"P_rect_0{cam}"].reshape(3, 4) @ rect @ velo_to_cam  # LiDAR -> image (3 x 4)
        self.drive_path, self.width, self.height = drive_path, int(width), int(height)

    def __call__(self, frame_id):
        scan = np.fromfile(f"{self.drive_path}/velodyne_points/data/{frame_id:010d}.bin", dtype=np.float32)
        X = scan.reshape(-1, 4)[:, :3].astype(np.float64)
        X = X[X[:, 0] > 0]  # in front of the LiDAR
        uvz = np.c_[X, np.ones(len(X))] @ self.P.T
        uvz = uvz[uvz[:, 2] > 0]
        u = np.round(uvz[:, 0] / uvz[:, 2]).astype(int)
        v = np.round(uvz[:, 1] / uvz[:, 2]).astype(int)
        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, z = u[inside], v[inside], uvz[inside, 2]
        depth = np.full((self.height, self.width), np.nan, dtype=np.float32)
        order = np.argsort(-z)  # far first -> nearest point per pixel written last
        depth[v[order], u[order]] = z[order]
        return depth
