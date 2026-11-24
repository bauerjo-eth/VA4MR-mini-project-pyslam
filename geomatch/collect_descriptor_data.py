"""
WP2 training data for the learned depth descriptor: SuperPoint keypoints/descriptors, Depth
Anything 3 depth maps, and ground-truth keypoint correspondences for image pairs.

Run from the repository root (needs the KITTI odometry dataset, see kitti_train_config.yaml):

    python -m geomatch.collect_descriptor_data --sequences 00 01 02 03 04 05 06 07 08

How a ground-truth correspondence is made (for every pair A, B):
  1. Depth scale: DA3 metric depth is off in scale, which moves a projected point along the
     epipolar line (tens of px at KITTI's forward motion). So the scale is corrected per pair:
     the one under which the most keypoints of A land on a keypoint of B (with consistent
     depth) is used for both depth maps (search 0.5-3, 1 % steps; see depth_scale).
     Triangulating matches instead was unreliable: wrong matches along the epipolar line (the
     error WP2 targets) triangulate to wrong depths.
  2. Every keypoint of A is lifted to 3D with its scaled DA3 depth, moved into camera B with
     the true pose and projected. It is kept only if it is in front of B, inside the image,
     and not occluded (its depth agrees with B's scaled DA3 depth within 10 %).
  3. Its match is the nearest keypoint of B within 3 px. The same is done from B to A, and
     only mutual matches whose keypoints are also within 2 px of each other's epipolar line
     are kept.
The true pose is used only to make these labels; it is never an input of the network.

Output (geomatch/data/wp2/, gitignored):
  sp/<seq>/frame_XXXXXX.npz   pts (Nx2 float32), des (Nx256 float16)
  pairs_<seq>.npz             per pair: frames, gap, depth scales, matches
Depth maps go to the shared depth cache (geomatch/data/depth/...), like the evaluation pipeline.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from geomatch.cache_depth_pictures import cache_depth_maps, load_depth
from geomatch.descriptors import depth_at
from geomatch.evaluation_metrics import epipolar_error, relative_pose
from geomatch.evaluation_pipeline import load_kitti_calib, load_poses, read_image_folder
from geomatch.superpoint import superpoint

OUT_DIR = Path(__file__).parent / "data" / "wp2"
DEPTH_MODEL = "depth-anything/DA3METRIC-LARGE"
log = logging.getLogger("geomatch")


def count_consistent(s, ptsA, zA_kp, ptsB, treeB, zB, TA, TB, K, shape, max_px=3.0, occl=0.10):
    """Number of A keypoints that, lifted with depth s * DA3 and moved with the true pose,
    land within max_px of a B keypoint and agree with B's depth s * DA3 (same scale for both
    frames: they are close in time, and the occlusion check then does not depend on s)."""
    H, W = shape
    uv, zp = project(ptsA, zA_kp * s, TA, TB, K)
    ok = (zp > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    if not ok.any():
        return 0
    uv, zp = uv[ok], zp[ok]
    vis = np.abs(zp / np.maximum(depth_at(zB, uv) * s, 1e-3) - 1) < occl
    if not vis.any():
        return 0
    dist, _ = treeB.query(uv[vis], distance_upper_bound=max_px)
    return int(np.isfinite(dist).sum())


def depth_scale(ptsA, ptsB, zA, zB, TA, TB, K, shape, center=None):
    """Scale s that makes DA3 depth metric for this pair, and how many keypoints agree with it.
    DA3's metric scale is off (Robin, WP1: 7-10 % too small, nearly constant); a wrong scale
    moves projected points along the epipolar line. The scale is the one under which the most
    keypoints land on a keypoint of the other image. center=None: search 0.5-3 (5 % steps) and
    refine (1 % steps); center=s0: search only s0 * [0.9, 1.1] in 1 % steps."""
    treeB = cKDTree(ptsB)
    zA_kp = depth_at(zA, ptsA)
    args = (ptsA, zA_kp, ptsB, treeB, zB, TA, TB, K, shape)
    if center is None:
        coarse = np.exp(np.linspace(np.log(0.5), np.log(3.0), 37))
        center = coarse[int(np.argmax([count_consistent(s, *args) for s in coarse]))]
        cands = center * np.exp(np.linspace(-0.05, 0.05, 11))
    else:
        cands = center * np.exp(np.linspace(-0.10, 0.10, 21))
    counts = [count_consistent(s, *args) for s in cands]
    k = int(np.argmax(counts))
    return float(cands[k]), counts[k]


def project(pts, z, T_src, T_dst, K):
    """Lift pixels with depth z in the source camera, project into the destination camera.
    Returns projected pixels (Nx2) and their depth in the destination camera."""
    rays = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T
    X = rays * z[:, None]
    T = relative_pose(T_src, T_dst)
    Xd = X @ T[:3, :3].T + T[:3, 3]
    uv = Xd @ K.T
    return uv[:, :2] / uv[:, 2:3], Xd[:, 2]


def one_way(ptsS, zS, sS, ptsD, zD, sD, TS, TD, K, shape, max_px=3.0, occl=0.10):
    """For every source keypoint: index of its destination keypoint, or -1."""
    H, W = shape
    out = -np.ones(len(ptsS), dtype=int)
    z = depth_at(zS, ptsS) * sS
    uv, zproj = project(ptsS, z, TS, TD, K)
    inside = (zproj > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
    idx = np.where(inside)[0]
    if len(idx) == 0:
        return out
    zdst = depth_at(zD, uv[idx]) * sD
    visible = np.abs(zproj[idx] / np.maximum(zdst, 1e-3) - 1) < occl
    idx = idx[visible]
    if len(idx) == 0:
        return out
    d2 = ((uv[idx, None, :] - ptsD[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    close = d2[np.arange(len(idx)), nn] < max_px ** 2
    out[idx[close]] = nn[close]
    return out


def correspondences(fa, fb, scales, TA, TB, K, shape):
    sA, sB = scales
    ab = one_way(fa["pts"], fa["depth"], sA, fb["pts"], fb["depth"], sB, TA, TB, K, shape)
    ba = one_way(fb["pts"], fb["depth"], sB, fa["pts"], fa["depth"], sA, TB, TA, K, shape)
    i = np.where(ab >= 0)[0]
    j = ab[i]
    mutual = ba[j] == i
    i, j = i[mutual], j[mutual]
    if len(i):
        ok = epipolar_error(fa["pts"][i], fb["pts"][j], TA, TB, K) < 2.0
        ok &= epipolar_error(fb["pts"][j], fa["pts"][i], TB, TA, K) < 2.0
        i, j = i[ok], j[ok]
    return np.stack([i, j], 1).astype(np.int32)


def collect_sequence(base, seq, gaps, step, n_kp, chunk=150):
    seq_dir = base / "sequences" / seq
    poses = load_poses(base / "poses" / f"{seq}.txt")
    K = load_kitti_calib(seq_dir)
    n_frames = len(list((seq_dir / "image_0").glob("*.png")))
    anchors = list(range(0, n_frames - max(gaps), step))
    pairs = [(a, a + g, g) for a in anchors for g in gaps]
    frames = sorted({f for a, b, _ in pairs for f in (a, b)})
    name = f"kitti{seq}"
    log.info(f"{name}: {n_frames} frames, {len(pairs)} pairs, {len(frames)} frames used")

    # depth (shared cache) + SuperPoint, in chunks so not all images sit in memory
    sp_dir = OUT_DIR / "sp" / name
    sp_dir.mkdir(parents=True, exist_ok=True)
    folder = None
    for c in range(0, len(frames), chunk):
        idxs = frames[c:c + chunk]
        imgs = read_image_folder(seq_dir / "image_0", set(idxs))
        folder = cache_depth_maps(imgs, DEPTH_MODEL, name)
        for i in idxs:
            p = sp_dir / f"frame_{i:06d}.npz"
            if not p.exists():
                pts, des = superpoint(imgs[i], n=n_kp)
                np.savez(p, pts=pts, des=des.astype(np.float16))
        log.info(f"  {name}: depth + SuperPoint done for {min(c + chunk, len(frames))}/{len(frames)} frames")

    # ground-truth correspondences
    shape = cv2.imread(str(seq_dir / "image_0" / "000000.png"), cv2.IMREAD_GRAYSCALE).shape
    feats = {}

    def feat(i):
        if i not in feats:
            d = np.load(sp_dir / f"frame_{i:06d}.npz")
            feats[i] = {"pts": d["pts"], "des": d["des"].astype(np.float32), "depth": load_depth(folder, i)}
        return feats[i]

    # pass 1: one DA3 scale for the sequence, from the smallest-gap pairs with many agreeing points
    small = [(a, b) for a, b, g in pairs if g == min(gaps)][::3]
    est = [depth_scale(feat(a)["pts"], feat(b)["pts"], feat(a)["depth"], feat(b)["depth"],
                       poses[a], poses[b], K, shape) for a, b in tqdm(small, desc=f"{name} scale", unit="pair")]
    good = [s for s, n in est if n >= 80] or [s for s, n in est if n >= 30] or [1.0]
    s_seq = float(np.median(good))
    log.info(f"  {name}: DA3 scale {s_seq:.3f} (median of {len(good)} of {len(est)} pairs, "
             f"IQR {np.percentile(good, 25):.3f}-{np.percentile(good, 75):.3f})")
    feats.clear()

    # pass 2: per pair, refine the scale within +-10 % and make the correspondences
    rec = {"fa": [], "fb": [], "gap": [], "sA": [], "sB": [], "n_match": [], "matches": []}
    skipped = 0
    for a, b, g in tqdm(pairs, desc=f"{name} correspondences", unit="pair"):
        fa, fb = feat(a), feat(b)
        s, n = depth_scale(fa["pts"], fb["pts"], fa["depth"], fb["depth"], poses[a], poses[b], K, shape,
                           center=s_seq)
        if n < 20:
            skipped += 1
            continue
        sc = (s, s)
        m = correspondences(fa, fb, sc, poses[a], poses[b], K, shape)
        if len(m) < 20:
            skipped += 1
            continue
        for k, v in zip(("fa", "fb", "gap", "sA", "sB", "n_match"), (a, b, g, sc[0], sc[1], len(m))):
            rec[k].append(v)
        rec["matches"].append(m)
        for old in [k for k in feats if k < a - max(gaps)]:
            del feats[old]

    offsets = np.cumsum([0] + [len(m) for m in rec["matches"]])
    np.savez(OUT_DIR / f"pairs_{name}.npz",
             fa=np.array(rec["fa"]), fb=np.array(rec["fb"]), gap=np.array(rec["gap"]),
             sA=np.array(rec["sA"]), sB=np.array(rec["sB"]),
             matches=np.concatenate(rec["matches"]) if rec["matches"] else np.zeros((0, 2), np.int32),
             offsets=offsets, K=K, shape=np.array(shape))
    stats = {"seq": seq, "pairs": len(rec["fa"]), "skipped": skipped,
             "median_matches": float(np.median(rec["n_match"])) if rec["n_match"] else 0,
             "scale_seq": s_seq,
             "per_gap_pairs": {int(g): int(np.sum(np.array(rec["gap"]) == g)) for g in gaps}}
    log.info(f"  {name}: {json.dumps(stats)}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="datasets/kitti_odometry/dataset")
    ap.add_argument("--sequences", nargs="+", default=[f"{i:02d}" for i in range(9)])
    ap.add_argument("--gaps", nargs="+", type=int, default=[5, 10, 20, 40])
    ap.add_argument("--step", type=int, default=10, help="frames between two anchor frames")
    ap.add_argument("--n_kp", type=int, default=2000)
    args = ap.parse_args()
    bad = set(args.sequences) & {"09", "10"}
    if bad:
        raise ValueError(f"sequences {sorted(bad)} are the TEST split and must not be used for training data")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_stats = [collect_sequence(Path(args.base), s, args.gaps, args.step, args.n_kp) for s in args.sequences]
    (OUT_DIR / "collect_stats.json").write_text(json.dumps(
        {"args": vars(args), "sequences": all_stats, "seconds": time.time() - t0}, indent=2))
    log.info(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
