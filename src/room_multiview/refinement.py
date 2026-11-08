"""Optional robust sparse bundle refinement on a copy; no reference-pose inputs."""

import copy
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .features import opencv
from .geometry import centre


def refine_project(project, max_evaluations=50):
    cv = opencv()
    start = perf_counter()
    updated = copy.deepcopy(project)
    geometry = updated["shared_geometry"]
    camera_ids = list(geometry["cameras"])
    points = geometry["landmarks"]
    report = {
        "status": "skipped",
        "source_geometry_preserved": True,
        "reference_poses_used": False,
        "fixed_cameras": camera_ids[:2],
    }
    if len(camera_ids) < 2 or len(points) < 10:
        report["reason"] = "insufficient_registered_geometry"
        return updated, report
    if geometry.get("metric_scale") or any(
        v.get("status") == "measured" for v in project.get("measurements", {}).values()
    ):
        report["reason"] = "measured_constraints_require_explicit_constrained_refinement"
        return updated, report
    if len(points) > 20000:
        raise ValueError("CPU bundle refinement limited to 20000 landmarks per experiment")
    variable = camera_ids[2:]
    poses = {
        key: np.asarray(camera["camera_from_world"], float)
        for key, camera in geometry["cameras"].items()
    }
    packed = []
    for key in variable:
        packed.extend(cv.Rodrigues(poses[key][:3, :3])[0].ravel())
        packed.extend(poses[key][:3, 3])
    x0 = np.r_[packed, np.array([p["position"] for p in points]).ravel()]
    observations = []
    for index, point in enumerate(points):
        for obs in point["observations"]:
            if obs["view_id"] in poses:
                observations.append((index, obs["view_id"], obs["pixel"]))
    by_camera = {
        key: [
            (i, point, pixel) for i, (point, view, pixel) in enumerate(observations) if view == key
        ]
        for key in camera_ids
    }
    offset = 6 * len(variable)

    def unpack(x):
        current = {k: p.copy() for k, p in poses.items()}
        for i, key in enumerate(variable):
            current[key][:3, :3] = cv.Rodrigues(x[6 * i : 6 * i + 3])[0]
            current[key][:3, 3] = x[6 * i + 3 : 6 * i + 6]
        return current, x[offset:].reshape(-1, 3)

    def residual(x):
        current, xyz = unpack(x)
        out = np.zeros((len(observations), 2))
        for key, rows in by_camera.items():
            ids = np.array([r[1] for r in rows])
            cal = project["view_assets"][key]["calibration"]
            p = current[key]
            pixels = cv.projectPoints(
                xyz[ids],
                cv.Rodrigues(p[:3, :3])[0],
                p[:3, 3],
                np.array(cal["intrinsics"]),
                np.array(cal["distortion"]),
            )[0].reshape(-1, 2)
            r = pixels - np.array([r[2] for r in rows])
            z = (xyz[ids] @ p[:3, :3].T + p[:3, 3])[:, 2]
            r[z <= 0] += 1000
            out[[r[0] for r in rows]] = r
        return out.ravel()

    sparsity = lil_matrix((2 * len(observations), len(x0)), dtype=int)
    for i, (point, view, _) in enumerate(observations):
        sparsity[2 * i : 2 * i + 2, offset + 3 * point : offset + 3 * point + 3] = 1
        if view in variable:
            j = variable.index(view)
            sparsity[2 * i : 2 * i + 2, 6 * j : 6 * j + 6] = 1
    before = np.linalg.norm(residual(x0).reshape(-1, 2), axis=1)
    fitted = least_squares(
        residual,
        x0,
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_evaluations,
        ftol=1e-5,
    )
    after = np.linalg.norm(residual(fitted.x).reshape(-1, 2), axis=1)
    accepted = (
        np.isfinite(fitted.x).all()
        and np.median(after) <= np.median(before)
        and np.percentile(after, 95) <= np.percentile(before, 95) * 1.1 + 0.01
    )
    report.update(
        status="accepted" if accepted else "rejected",
        evaluations=fitted.nfev,
        optimizer_converged=bool(fitted.success),
        optimizer_message=fitted.message,
        before_median_pixels=float(np.median(before)),
        after_median_pixels=float(np.median(after)),
        before_p95_pixels=float(np.percentile(before, 95)),
        after_p95_pixels=float(np.percentile(after, 95)),
        seconds=perf_counter() - start,
        observation_count=len(observations),
        scope="First two poses fixed to preserve frame and baseline gauge; remaining cameras/points refined. Internal consistency, not accuracy proof.",
    )
    if accepted:
        current, xyz = unpack(fitted.x)
        for key in camera_ids:
            geometry["cameras"][key]["camera_from_world"] = current[key].tolist()
            geometry["cameras"][key]["centre"] = centre(current[key]).tolist()
            updated["coordinate_frames"]["transforms"]["camera_to_world"][key] = {
                "matrix": np.linalg.inv(current[key]).tolist(),
                "status": "estimated_bundle_refined",
            }
        for point, xyz_point in zip(points, xyz):
            point["position"] = xyz_point.tolist()
        # Previously fitted patches do not remain valid after moving their support.
        from .surfaces import plane_patches

        geometry["surface_patches"] = plane_patches(points)
        updated["pose_consistency_checks"] = []
        updated["refinement"] = report
        updated["fit_readiness"]["ready"] = False
    return updated, report
