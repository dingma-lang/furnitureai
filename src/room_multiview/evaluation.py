"""Reference poses stay outside reconstruction. Scale fit is evaluation-only."""

import numpy as np
from scipy.spatial.transform import Rotation

from furniture_bench.data import digest, read_json, write_json

from .geometry import centre, project, rotation_error


def evaluate(project_path, reference_path, output):
    state, reference = read_json(project_path), read_json(reference_path)
    geometry = state["shared_geometry"]
    cameras = geometry["cameras"]
    truth = {r["image_id"]: r for r in reference["poses"] if r["timestamp_delta_seconds"] <= 0.02}
    anchor = geometry.get("anchor_view_id")
    per_camera, residuals = [], []
    expected = {}
    if anchor in truth:
        origin = truth[anchor]
        r0 = Rotation.from_quat(origin["quaternion_xyzw"]).as_matrix()
        c0 = np.array(origin["position_metres"])
        for key in cameras:
            if key not in truth:
                continue
            row = truth[key]
            ri = Rotation.from_quat(row["quaternion_xyzw"]).as_matrix()
            expected[key] = r0.T @ (np.array(row["position_metres"]) - c0)
            pose = np.array(cameras[key]["camera_from_world"])
            angular = None
            if (
                key != anchor
                and np.linalg.norm(expected[key]) > 1e-6
                and np.linalg.norm(centre(pose)) > 1e-6
            ):
                angular = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(centre(pose), expected[key])
                                / (np.linalg.norm(centre(pose)) * np.linalg.norm(expected[key])),
                                -1,
                                1,
                            )
                        )
                    )
                )
            per_camera.append(
                {
                    "view_id": key,
                    "reference_timestamp_delta_seconds": row["timestamp_delta_seconds"],
                    "rotation_error_degrees": rotation_error(pose[:3, :3], ri.T @ r0),
                    "translation_direction_error_degrees": angular,
                    "estimated_centre_arbitrary_units": centre(pose).tolist(),
                    "reference_centre_in_anchor_metres": expected[key].tolist(),
                }
            )
    numerator = sum(np.dot(np.array(cameras[k]["centre"]), c) for k, c in expected.items())
    denominator = sum(np.dot(cameras[k]["centre"], cameras[k]["centre"]) for k in expected)
    fitted_scale = float(numerator / denominator) if denominator > 1e-10 and numerator > 0 else None
    if fitted_scale:
        for row in per_camera:
            error = np.linalg.norm(
                np.array(row["estimated_centre_arbitrary_units"]) * fitted_scale
                - row["reference_centre_in_anchor_metres"]
            )
            row["position_error_after_reference_scale_fit_metres"] = float(error)
    for p in geometry["landmarks"]:
        for o in p["observations"]:
            asset = state["view_assets"][o["view_id"]]
            pixels = project(
                [p["position"]], cameras[o["view_id"]]["camera_from_world"], asset["calibration"]
            )[0]
            residuals.append(float(np.linalg.norm(pixels - o["pixel"])))
    report = {
        "project_sha256": digest(project_path),
        "reference_sha256": digest(reference_path),
        "input_views": len(state["views"]),
        "registered_views": len(cameras),
        "unregistered_views": [v["view_id"] for v in state["views"] if v["view_id"] not in cameras],
        "landmarks": len(geometry["landmarks"]),
        "supported_semantic_plane_patches": len(geometry["surface_patches"]),
        "reprojection_pixels": {
            "median": float(np.median(residuals)) if residuals else None,
            "p95": float(np.percentile(residuals, 95)) if residuals else None,
            "observations": len(residuals),
            "is_independent_accuracy": False,
        },
        "reference_camera_errors": per_camera,
        "evaluation_only_reference_scale_fit": fitted_scale,
        "reference_scale_fed_to_reconstruction": False,
        "physical_room_dimension_accuracy": None,
        "room_completeness": None,
        "structural_segmentation_accuracy": None,
        "pair_rejections": [
            {"views": p["views"], "reason": p["geometry"]["reason"]}
            for p in state["pair_reports"]
            if p["geometry"]["status"] != "accepted"
        ],
        "geometric_conflicts": state["geometric_conflicts"],
        "runtime": state["runtime"],
        "note": "Motion-capture reference errors concern cameras only. Reprojection consistency is not independent 3D accuracy. No room surfaces/dimensions ground truth. Reference scale fit is not an available product measurement.",
    }
    write_json(output, report)
    return report
