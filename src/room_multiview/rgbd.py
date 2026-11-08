"""Measured-depth registration/fusion, explicitly separate from RGB-only SfM."""

import numpy as np

from .features import opencv
from .geometry import coverage
from .numeric import apply_pose, points_array


def backproject(pixels, depth_z, intrinsics):
    k = np.asarray(intrinsics, float)
    p = points_array(pixels, 2, "pixels")
    z = np.asarray(depth_z, float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or np.linalg.cond(k) > 1e10
        or np.shape(z) != (len(p),)
        or not np.isfinite(z).all()
        or np.any(z < 0)
        or np.max(z, initial=0) > 1e6
    ):
        raise ValueError("backprojection requires finite valid intrinsics and nonnegative depth")
    rays = np.einsum("nj,ij->ni", np.c_[p, np.ones(len(p))], np.linalg.inv(k), optimize=False)
    return rays * z[:, None]


def sample_depth(depth, pixels, intrinsics, min_depth=0.4, max_depth=5.0):
    pixels = points_array(pixels, 2, "pixels")
    depth = np.asarray(depth)
    if depth.ndim != 2 or not np.isfinite(depth).all() or min_depth < 0 or max_depth <= min_depth:
        raise ValueError("depth sampling requires a finite 2D depth map and valid depth range")
    xy = np.rint(pixels).astype(int)
    inside = (
        (xy[:, 0] >= 1)
        & (xy[:, 0] < depth.shape[1] - 1)
        & (xy[:, 1] >= 1)
        & (xy[:, 1] < depth.shape[0] - 1)
    )
    z = np.zeros(len(xy))
    valid = inside.copy()
    for index in np.flatnonzero(inside):
        x, y = xy[index]
        patch = depth[y - 1 : y + 2, x - 1 : x + 2]
        good = patch[(patch >= min_depth) & (patch <= max_depth)]
        # Reject depth discontinuities at foreground/background boundaries.
        if len(good) < 7 or np.ptp(good) > 0.06:
            valid[index] = False
        else:
            z[index] = depth[y, x]
            valid[index] &= min_depth <= z[index] <= max_depth
    return backproject(pixels, z, intrinsics), valid


def rigid_fit(first, second):
    a, b = points_array(first), points_array(second)
    if (
        len(a) < 3
        or a.shape != b.shape
        or min(np.linalg.matrix_rank(a - a.mean(axis=0)), np.linalg.matrix_rank(b - b.mean(axis=0)))
        < 2
    ):
        raise ValueError("rigid fit needs at least three noncollinear finite correspondences")
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd(np.einsum("ni,nj->ij", a - ac, b - bc, optimize=False))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = bc - rotation @ ac
    return pose


def transform(points, pose):
    return apply_pose(points, pose)


def register_rgbd(first, second, pixels_a, pixels_b, size, min_inliers=30, threshold=0.04):
    """Rigid camera-b-from-camera-a, with metric depth and no ground-truth poses."""
    first, second = points_array(first), points_array(second)
    pixels_a, pixels_b = points_array(pixels_a, 2), points_array(pixels_b, 2)
    if (
        first.shape != second.shape
        or len(pixels_a) != len(first)
        or len(pixels_b) != len(first)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("invalid RGB-D correspondence arrays or threshold")
    result = {
        "status": "rejected",
        "matches_with_depth": len(first),
        "mode": "sensor_assisted_rgbd",
    }
    if len(first) < min_inliers:
        return dict(result, reason="insufficient_valid_depth_correspondences")
    rng = np.random.default_rng(31)
    best = np.zeros(len(first), bool)
    for _ in range(500):
        ids = rng.choice(len(first), 3, replace=False)
        if (
            np.linalg.norm(np.cross(first[ids[1]] - first[ids[0]], first[ids[2]] - first[ids[0]]))
            < 1e-5
            or np.linalg.norm(
                np.cross(second[ids[1]] - second[ids[0]], second[ids[2]] - second[ids[0]])
            )
            < 1e-5
        ):
            continue
        pose = rigid_fit(first[ids], second[ids])
        inliers = np.linalg.norm(transform(first, pose) - second, axis=1) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < min_inliers or best.mean() < 0.35:
        return dict(result, reason="inconsistent_metric_correspondences", inliers=int(best.sum()))
    pose = rigid_fit(first[best], second[best])
    errors = np.linalg.norm(transform(first, pose) - second, axis=1)
    best = errors <= threshold
    spatial = min(
        coverage(np.asarray(pixels_a)[best], size), coverage(np.asarray(pixels_b)[best], size)
    )
    if best.sum() < min_inliers or spatial < 0.02:
        return dict(result, reason="poor_depth_inlier_spatial_support", coverage=spatial)
    return dict(
        result,
        status="accepted",
        camera_b_from_a=pose.tolist(),
        inlier_indices=np.flatnonzero(best).tolist(),
        inliers=int(best.sum()),
        coverage=spatial,
        median_3d_residual_metres=float(np.median(errors[best])),
        reason="robust_metric_rgbd_rigid_alignment",
        physically_verified=False,
    )


def dense_samples(depth, rgb, intrinsics, labels=None, confidence=None, stride=4):
    """Observed samples only: invalid depth stays missing; no inpainting."""
    cv = opencv()
    valid = (depth >= 0.4) & (depth <= 5) & np.isfinite(depth)
    maximum = cv.dilate(depth.astype(np.float32), np.ones((3, 3), np.uint8))
    minimum = cv.erode(depth.astype(np.float32), np.ones((3, 3), np.uint8))
    valid &= (maximum - minimum) < 0.08
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    x, y = x.ravel(), y.ravel()
    keep = valid[y, x]
    x, y = x[keep], y[keep]
    xyz = backproject(np.c_[x, y], depth[y, x], intrinsics)
    semantic = np.zeros(len(x), np.uint8)
    if labels is not None and confidence is not None:
        semantic = labels[y, x].copy()
        semantic[confidence[y, x] < 0.75] = 0
    return {
        "points": xyz,
        "colors": np.asarray(rgb)[y, x],
        "pixels": np.c_[x, y],
        "labels": semantic,
    }


def voxel_fuse(points, colors, labels, view_indices, voxel_size=0.025):
    """Average observed voxel samples; retain view support and semantic disagreement."""
    points = points_array(points)
    if not np.isfinite(voxel_size) or voxel_size < 1e-6:
        raise ValueError("voxel size must be finite and at least one micrometre")
    if len(points) == 0:
        return {
            "points": np.empty((0, 3)),
            "colors": np.empty((0, 3), np.uint8),
            "labels": np.zeros(0, np.uint8),
            "view_count": np.zeros(0, int),
            "spread": np.zeros(0),
        }
    cells = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    n = len(count)
    fused = np.column_stack(
        [np.bincount(inverse, weights=points[:, i], minlength=n) / count for i in range(3)]
    )
    color = np.column_stack(
        [np.bincount(inverse, weights=colors[:, i], minlength=n) / count for i in range(3)]
    ).astype(np.uint8)
    view_count = np.zeros(n, int)
    for v in np.unique(view_indices):
        view_count[np.unique(inverse[view_indices == v])] += 1
    votes = np.column_stack(
        [np.bincount(inverse, weights=(labels == code), minlength=n) for code in range(7)]
    )
    majority = votes.argmax(axis=1).astype(np.uint8)
    majority[votes.max(axis=1) / count < 0.8] = 0
    spread = np.sqrt(
        np.bincount(inverse, weights=np.sum((points - fused[inverse]) ** 2, axis=1)) / count
    )
    return {
        "points": fused,
        "colors": color,
        "labels": majority,
        "view_count": view_count,
        "spread": spread,
    }
