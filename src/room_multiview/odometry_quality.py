"""Reference-free, finite geometric checks for dense RGB-D alignment hypotheses."""

from dataclasses import asdict, dataclass

import numpy as np

from .numeric import apply_pose, rigid_pose


@dataclass(frozen=True)
class OdometryPolicy:
    depth_difference_metres: float = 0.04
    minimum_overlap: float = 0.30
    maximum_median_depth_error_metres: float = 0.02
    maximum_mean_intensity_error: float = 0.12
    minimum_correspondences: int = 500
    maximum_cycle_translation_metres: float = 0.03
    maximum_cycle_rotation_degrees: float = 1.0
    maximum_step_translation_metres: float = 0.20
    maximum_step_rotation_degrees: float = 15.0

    def to_dict(self):
        return asdict(self)


def motion_size(pose):
    pose = rigid_pose(pose)
    return float(np.linalg.norm(pose[:3, 3])), float(
        np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))
    )


def alignment_quality(depth_a, depth_b, gray_a, gray_b, k, pose, policy=OdometryPolicy(), stride=4):
    a, b = np.asarray(depth_a, float), np.asarray(depth_b, float)
    ga, gb = np.asarray(gray_a, float), np.asarray(gray_b, float)
    k = np.asarray(k, float)
    if (
        a.ndim != 2
        or a.shape != b.shape
        or ga.shape != a.shape
        or gb.shape != b.shape
        or not all(np.isfinite(x).all() for x in (a, b, ga, gb, k))
    ):
        raise ValueError("odometry quality requires matching finite depth/intensity arrays")
    if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid pinhole intrinsics")
    pose = rigid_pose(pose)
    y, x = np.mgrid[0 : a.shape[0] : stride, 0 : a.shape[1] : stride]
    x, y = x.ravel(), y.ravel()
    z = a[y, x]
    valid = (z >= 0.4) & (z <= 5)
    x, y, z = x[valid], y[valid], z[valid]
    p = np.c_[(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z]
    q = apply_pose(p, pose)
    positive = q[:, 2] > 0.1
    uv = np.zeros((len(q), 2))
    np.divide(q[:, :2], q[:, 2, None], out=uv, where=positive[:, None])
    uv = uv * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    target = np.rint(uv).astype(int)
    inside = (
        positive
        & (target[:, 0] >= 0)
        & (target[:, 0] < b.shape[1])
        & (target[:, 1] >= 0)
        & (target[:, 1] < b.shape[0])
    )
    ids = np.flatnonzero(inside)
    u, v = target[ids, 0], target[ids, 1]
    zb = b[v, u]
    depth_valid = (zb >= 0.4) & (zb <= 5)
    ids, u, v, zb = ids[depth_valid], u[depth_valid], v[depth_valid], zb[depth_valid]
    residual = np.abs(q[ids, 2] - zb)
    inlier = residual <= policy.depth_difference_metres
    count = int(inlier.sum())
    overlap = count / max(len(p), 1)
    photo = np.abs(ga[y[ids[inlier]], x[ids[inlier]]] - gb[v[inlier], u[inlier]])
    median = float(np.median(residual[inlier])) if count else None
    mean_intensity = float(np.mean(photo)) if count else None
    accepted = (
        count >= policy.minimum_correspondences
        and overlap >= policy.minimum_overlap
        and median <= policy.maximum_median_depth_error_metres
        and mean_intensity <= policy.maximum_mean_intensity_error
    )
    return {
        "accepted": accepted,
        "source_valid_samples": len(p),
        "correspondences": count,
        "overlap": float(overlap),
        "median_depth_error_metres": median,
        "mean_intensity_error": mean_intensity,
        "depth_residual_p95_metres": float(np.percentile(residual[inlier], 95)) if count else None,
    }


def validate_candidate(pose, reverse, quality_forward, quality_reverse, policy=OdometryPolicy()):
    try:
        pose = rigid_pose(pose)
        reverse = rigid_pose(reverse)
        step_t, step_r = motion_size(pose)
        cycle = np.einsum("ij,jk->ik", reverse, pose, optimize=False)
        cycle_t, cycle_r = motion_size(cycle)
    except ValueError as exc:
        return {"accepted": False, "reason": "invalid_rigid_transform", "message": str(exc)}
    reasons = []
    if not quality_forward["accepted"] or not quality_reverse["accepted"]:
        reasons.append("insufficient_bidirectional_dense_support")
    if (
        step_t > policy.maximum_step_translation_metres
        or step_r > policy.maximum_step_rotation_degrees
    ):
        reasons.append("step_motion_exceeds_local_tracking_policy")
    if (
        cycle_t > policy.maximum_cycle_translation_metres
        or cycle_r > policy.maximum_cycle_rotation_degrees
    ):
        reasons.append("forward_reverse_cycle_inconsistent")
    return {
        "accepted": not reasons,
        "reason": ";".join(reasons) if reasons else "passed_reference_free_local_checks",
        "step_translation_metres": step_t,
        "step_rotation_degrees": step_r,
        "cycle_translation_metres": cycle_t,
        "cycle_rotation_degrees": cycle_r,
    }
