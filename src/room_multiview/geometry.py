"""Calibrated two-view bootstrap and robust registration in one arbitrary-scale frame."""

from dataclasses import asdict, dataclass

import numpy as np

from .features import opencv
from .numeric import apply_pose, points_array, rigid_pose, rotation_matrix


@dataclass(frozen=True)
class Policy:
    min_matches: int = 40
    min_inliers: int = 25
    min_inlier_ratio: float = 0.35
    min_coverage: float = 0.015
    min_parallax_degrees: float = 0.75
    max_reprojection_pixels: float = 2.0
    max_rotation_disagreement_degrees: float = 8.0
    max_translation_direction_disagreement_degrees: float = 25.0

    def to_dict(self):
        return asdict(self)


def calibration(value):
    if value is None:
        return None
    k = np.asarray(value["intrinsics"], dtype=float)
    d = np.asarray(value.get("distortion", [0] * 5), dtype=float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or np.max(np.abs(k)) > 1e7
        or abs(np.linalg.det(k)) < 1e-8
        or np.linalg.cond(k) > 1e10
        or d.ndim != 1
        or len(d) not in (4, 5, 8, 12, 14)
        or not np.isfinite(d).all()
        or (len(d) > 0 and np.max(np.abs(d)) > 1e5)
        or not value.get("source")
        or value.get("status") not in ("measured", "dataset_calibrated", "estimated")
    ):
        raise ValueError("invalid sourced pinhole calibration")
    return {
        "intrinsics": k.tolist(),
        "distortion": d.tolist(),
        "source": value["source"],
        "status": value["status"],
    }


def normalized(pixels, cal):
    pixels = points_array(pixels, 2, "pixels")
    if not len(pixels):
        return np.empty((0, 2))
    cal = calibration(cal)
    return (
        opencv()
        .undistortPoints(
            np.asarray(pixels, float).reshape(-1, 1, 2),
            np.asarray(cal["intrinsics"]),
            np.asarray(cal["distortion"]),
        )
        .reshape(-1, 2)
    )


def centre(pose):
    pose = rigid_pose(pose)
    return -np.einsum("ji,j->i", pose[:3, :3], pose[:3, 3], optimize=False)


def coverage(pixels, size):
    pixels = points_array(pixels, 2, "pixels")
    if len(size) != 2 or not all(np.isfinite(v) and v > 0 for v in size):
        raise ValueError("invalid image dimensions")
    if len(pixels) < 3:
        return 0.0
    cv = opencv()
    return float(
        cv.contourArea(cv.convexHull(np.asarray(pixels, np.float32))) / (size[0] * size[1])
    )


def project(points, pose, cal):
    cv = opencv()
    points = points_array(points)
    pose = rigid_pose(pose)
    cal = calibration(cal)
    if not len(points):
        return np.empty((0, 2))
    projected = cv.projectPoints(
        np.asarray(points, float),
        cv.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        np.asarray(cal["intrinsics"]),
        np.asarray(cal["distortion"]),
    )[0].reshape(-1, 2)
    if not np.isfinite(projected).all() or np.max(np.abs(projected)) > 1e15:
        raise ValueError("camera projection is nonfinite or numerically unbounded")
    return projected


def triangulate(first, second, pose_a, pose_b, cal_a, cal_b, policy):
    first, second = points_array(first, 2), points_array(second, 2)
    if first.shape != second.shape:
        raise ValueError("triangulation correspondence lengths differ")
    if len(first) == 0:
        return np.empty((0, 3)), np.zeros(0, bool), np.zeros(0), np.zeros(0)
    cv = opencv()
    pa, pb = rigid_pose(pose_a), rigid_pose(pose_b)
    a, b = normalized(first, cal_a), normalized(second, cal_b)
    homogeneous = cv.triangulatePoints(pa[:3], pb[:3], a.T, b.T).T
    finite = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-10)
    points = np.zeros((len(first), 3))
    np.divide(homogeneous[:, :3], homogeneous[:, 3:4], out=points, where=finite[:, None])
    finite &= np.isfinite(points).all(axis=1) & (np.max(np.abs(points), axis=1) <= 1e8)
    points[~finite] = 0
    z1 = apply_pose(points, pa)[:, 2]
    z2 = apply_pose(points, pb)[:, 2]
    ray_a, ray_b = points - centre(pa), points - centre(pb)
    denominator = np.maximum(np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1), 1e-12)
    angles = np.degrees(np.arccos(np.clip(np.sum(ray_a * ray_b, axis=1) / denominator, -1, 1)))
    errors = np.maximum(
        np.linalg.norm(project(points, pa, cal_a) - first, axis=1),
        np.linalg.norm(project(points, pb, cal_b) - second, axis=1),
    )
    valid = finite & np.isfinite(points).all(axis=1) & (z1 > 0) & (z2 > 0)
    valid &= (angles >= policy.min_parallax_degrees) & (errors <= policy.max_reprojection_pixels)
    return points, valid, angles, errors


def estimate_pair(first, second, cal_a, cal_b, size_a, size_b, policy=Policy()):
    cv = opencv()
    first, second = points_array(first, 2), points_array(second, 2)
    if first.shape != second.shape:
        raise ValueError("pair correspondence lengths differ")
    result = {"status": "rejected", "matches": len(first), "reason": "insufficient_mutual_matches"}
    if len(first) < policy.min_matches:
        return result
    if min(
        np.linalg.matrix_rank(first - first.mean(axis=0)),
        np.linalg.matrix_rank(second - second.mean(axis=0)),
    ) < 2 or np.allclose(first, second, atol=1e-9):
        return dict(result, reason="degenerate_or_identical_correspondences")
    if cal_a is None or cal_b is None:
        _, mask = cv.findFundamentalMat(first, second, cv.FM_RANSAC, 1.5, 0.999)
        result.update(
            reason="camera_intrinsics_unknown_no_euclidean_3d",
            fundamental_inliers=int(mask.sum()) if mask is not None else 0,
        )
        return result
    a, b = normalized(first, cal_a), normalized(second, cal_b)
    focal = np.mean(
        [
            cal_a["intrinsics"][0][0],
            cal_a["intrinsics"][1][1],
            cal_b["intrinsics"][0][0],
            cal_b["intrinsics"][1][1],
        ]
    )
    essential, mask = cv.findEssentialMat(
        a, b, np.eye(3), method=cv.RANSAC, prob=0.999, threshold=1.5 / focal
    )
    if essential is None or essential.shape != (3, 3) or mask is None:
        result["reason"] = "essential_matrix_not_identifiable"
        return result
    essential_count = int(mask.sum())
    _, rotation, translation, pose_mask = cv.recoverPose(essential, a, b, np.eye(3), mask=mask)
    indices = np.flatnonzero(pose_mask.ravel() != 0)
    result.update(essential_inliers=essential_count, cheirality_inliers=len(indices))
    if len(indices) < policy.min_inliers or len(indices) / len(first) < policy.min_inlier_ratio:
        result["reason"] = "insufficient_cheirality_consensus_or_degenerate_motion"
        return result
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation.ravel()
    points, valid, angles, errors = triangulate(
        first[indices], second[indices], np.eye(4), pose, cal_a, cal_b, policy
    )
    indices = indices[valid]
    result.update(
        triangulated_inliers=len(indices),
        coverage=[coverage(first[indices], size_a), coverage(second[indices], size_b)],
    )
    if (
        len(indices) < policy.min_inliers
        or len(indices) / len(first) < policy.min_inlier_ratio
        or min(result["coverage"]) < policy.min_coverage
    ):
        result["reason"] = "insufficient_parallax_spatial_support_or_reprojection_consistency"
        return result
    _, hmask = cv.findHomography(first, second, cv.RANSAC, 3)
    hratio = float(hmask.mean()) if hmask is not None else 0
    result.update(
        status="accepted",
        reason="estimated_calibrated_two_view_geometry",
        camera_b_from_a=pose.tolist(),
        match_indices=indices.tolist(),
        points=points[valid].tolist(),
        median_parallax_degrees=float(np.median(angles[valid])),
        median_reprojection_pixels=float(np.median(errors[valid])),
        homography_inlier_ratio=hratio,
        planar_or_rotation_risk=hratio > 0.9,
        physically_verified=False,
        scale="unknown",
    )
    return result


def register_camera(points, pixels, cal, size, policy=Policy()):
    cv = opencv()
    points, pixels = points_array(points), points_array(pixels, 2)
    if len(points) != len(pixels):
        raise ValueError("PnP correspondence lengths differ")
    if cal is None or len(points) < policy.min_inliers:
        return {
            "status": "rejected",
            "reason": "insufficient_shared_3d_landmarks_or_missing_calibration",
        }
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return {"status": "rejected", "reason": "degenerate_3d_correspondences"}
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        np.asarray(points, float),
        np.asarray(pixels, float),
        np.asarray(cal["intrinsics"]),
        np.asarray(cal["distortion"]),
        iterationsCount=2000,
        reprojectionError=policy.max_reprojection_pixels,
        confidence=0.999,
        flags=cv.SOLVEPNP_EPNP,
    )
    if (
        not ok
        or inliers is None
        or len(inliers) < policy.min_inliers
        or len(inliers) / len(points) < policy.min_inlier_ratio
    ):
        return {"status": "rejected", "reason": "inconsistent_2d_3d_correspondences"}
    ids = inliers.ravel()
    rvec, tvec = cv.solvePnPRefineLM(
        np.asarray(points, float)[ids],
        np.asarray(pixels, float)[ids],
        np.asarray(cal["intrinsics"]),
        np.asarray(cal["distortion"]),
        rvec,
        tvec,
    )
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv.Rodrigues(rvec)[0], tvec.ravel()
    errors = np.linalg.norm(project(points, pose, cal) - pixels, axis=1)
    positive = apply_pose(points, pose)[:, 2] > 0
    good = positive & (errors <= policy.max_reprojection_pixels)
    spatial = coverage(np.asarray(pixels)[good], size)
    if good.sum() < policy.min_inliers or spatial < policy.min_coverage:
        return {"status": "rejected", "reason": "pnp_failed_post_refinement_consistency"}
    return {
        "status": "accepted",
        "camera_from_world": pose.tolist(),
        "inlier_indices": np.flatnonzero(good).tolist(),
        "coverage": spatial,
        "median_reprojection_pixels": float(np.median(errors[good])),
    }


def rotation_error(first, second):
    first, second = rotation_matrix(first), rotation_matrix(second)
    relative = np.einsum("ij,kj->ik", first, second, optimize=False)
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
