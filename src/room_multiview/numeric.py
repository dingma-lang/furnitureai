"""Bounded finite geometry inputs; avoid BLAS vector-warning paths on this Mac."""

import numpy as np


def points_array(value, dimensions=3, name="points", limit=1e8):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, dimensions), dtype=float)
    if (
        array.ndim != 2
        or array.shape[1] != dimensions
        or not np.isfinite(array).all()
        or np.max(np.abs(array)) > limit
    ):
        raise ValueError(f"{name} must be finite bounded Nx{dimensions} coordinates")
    return array


def rotation_matrix(value):
    r = np.asarray(value, float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or np.max(np.abs(r)) > 2:
        raise ValueError("invalid finite rotation matrix")
    if (
        not np.allclose(np.einsum("ki,kj->ij", r, r), np.eye(3), atol=1e-5)
        or abs(np.linalg.det(r) - 1) > 1e-5
    ):
        raise ValueError("rotation must be a proper orthonormal matrix")
    return r


def rigid_pose(value):
    pose = np.asarray(value, float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        or np.max(np.abs(pose[:3, 3])) > 1e6
    ):
        raise ValueError("invalid finite bounded rigid pose")
    rotation_matrix(pose[:3, :3])
    return pose


def apply_pose(points, pose):
    xyz = points_array(points)
    pose = rigid_pose(pose)
    return np.einsum("nj,ij->ni", xyz, pose[:3, :3], optimize=False) + pose[:3, 3]
