"""
Grading functions: how wrong is each match, given the true camera motion?

Every function takes the matched points of both images and the ground-truth poses, and
returns one error in pixels per match. A match counts as "correct" if its error is below a
threshold (e.g. 4 px). Inputs:

  p1, p2  Nx2 matched pixel coordinates in image 1 and image 2 (row i of p1 matches row i of p2)
  T1, T2  4x4 ground-truth camera-to-world poses of image 1 and image 2
  K       3x3 camera intrinsics
  z1, z2  N predicted depths at p1 and p2 (only for reprojection_error)

Copied from demo.py (which stays unchanged); the only difference is that K is passed in
instead of being a global constant.
"""

import cv2
import numpy as np


def relative_pose(T1, T2):
    """4x4 transform that maps points from camera 1 coordinates to camera 2 coordinates."""
    return np.linalg.inv(T2) @ T1


def fundamental_matrix(T1, T2, K):
    """Fundamental matrix F from the true relative motion: x2^T F x1 = 0 for a correct match."""
    T21 = relative_pose(T1, T2)
    R, t = T21[:3, :3], T21[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    return Kinv.T @ tx @ R @ Kinv


# ---------- depth-independent: use only the true pose ----------

def epipolar_error(p1, p2, T1, T2, K):
    """Distance (px) of each p2 to the epipolar line of its partner p1."""
    F = fundamental_matrix(T1, T2, K)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    lines = x1 @ F.T                                   # epipolar lines in image 2
    return np.abs(np.sum(x2 * lines, 1)) / np.hypot(lines[:, 0], lines[:, 1])


def symmetric_epipolar_error(p1, p2, T1, T2, K):
    """Mean of the point-to-epipolar-line distances in both images (px)."""
    F = fundamental_matrix(T1, T2, K)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    l2 = x1 @ F.T                                      # epipolar lines in image 2
    l1 = x2 @ F                                        # epipolar lines in image 1
    d2 = np.abs(np.sum(x2 * l2, 1)) / np.hypot(l2[:, 0], l2[:, 1])
    d1 = np.abs(np.sum(x1 * l1, 1)) / np.hypot(l1[:, 0], l1[:, 1])
    return 0.5 * (d1 + d2)


def sampson_error(p1, p2, T1, T2, K):
    """Sampson distance (px): first-order approximation of the geometric error w.r.t. F."""
    F = fundamental_matrix(T1, T2, K)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx1 = x1 @ F.T                                     # F x1
    Ftx2 = x2 @ F                                      # F^T x2
    num = np.sum(x2 * Fx1, 1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(denom, 1e-12))


def triangulation_error(p1, p2, T1, T2, K):
    """Triangulate each match with the true pose, reproject the 3D point into both images,
    and return the mean pixel error. Points behind either camera get infinite error."""
    T21 = relative_pose(T1, T2)
    R, t = T21[:3, :3], T21[:3, 3]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # projection matrix of camera 1
    P2 = K @ T21[:3]                                   # projection matrix of camera 2
    Xh = cv2.triangulatePoints(P1, P2, p1.T, p2.T)     # 4xN homogeneous, camera 1 frame
    X = (Xh[:3] / Xh[3]).T                             # Nx3 in camera 1
    X2 = X @ R.T + t                                   # in camera 2

    def project(Xc):
        proj = Xc @ K.T
        return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)

    err = 0.5 * (np.linalg.norm(p1 - project(X), axis=1) + np.linalg.norm(p2 - project(X2), axis=1))
    err[(X[:, 2] <= 1e-6) | (X2[:, 2] <= 1e-6)] = np.inf  # behind a camera
    return err


# ---------- depth-dependent: also use the predicted depth ----------

def _reproject_one_way(p_src, p_dst, z_src, T_src, T_dst, K):
    """Lift p_src to 3D with its predicted depth, move it into the destination camera with
    the true pose, project it, and return the pixel distance to p_dst."""
    T_sd = relative_pose(T_src, T_dst)
    R, t = T_sd[:3, :3], T_sd[:3, 3]
    Kinv = np.linalg.inv(K)
    xs = np.c_[p_src, np.ones(len(p_src))]
    Xs = (xs @ Kinv.T) * z_src[:, None]                # Nx3 points in source camera
    Xd = Xs @ R.T + t                                  # in destination camera
    proj = Xd @ K.T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
    err = np.linalg.norm(p_dst - uv, axis=1)
    err[Xd[:, 2] <= 1e-6] = np.inf                     # behind destination camera
    return err


def reprojection_error(p1, p2, z1, z2, T1, T2, K):
    """Mean of the 1->2 and 2->1 reprojection errors (px), each using that point's own
    predicted depth. Mixes matching errors with depth errors."""
    return 0.5 * (_reproject_one_way(p1, p2, z1, T1, T2, K)
                  + _reproject_one_way(p2, p1, z2, T2, T1, K))
