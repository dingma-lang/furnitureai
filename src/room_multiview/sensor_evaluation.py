"""Held-out RGB-D/camera evaluation. References never enter sensor registration."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from furniture_bench.data import digest, load_image, load_manifest, read_json, write_json

from .geometry import rotation_error
from .rgbd import dense_samples, transform
from .sensor_pipeline import load_sensor


def evaluate_sensor(project_path, heldout_manifest, reference_path, output):
    state = read_json(project_path)
    sensor = state["sensor_reconstruction"]
    rows = load_manifest(heldout_manifest, "rooms")
    if any(r["image_id"] in sensor["sources"] for r in rows):
        raise ValueError("held-out geometry views overlap reconstruction inputs")
    input_hashes = {s["image_sha256"] for s in sensor["sources"].values()}
    input_depth_hashes = {s["depth_sha256"] for s in sensor["sources"].values()}
    if any(
        digest(r["resolved_file"]) in input_hashes
        or r.get("sensor_depth", {}).get("sha256") in input_depth_hashes
        for r in rows
    ):
        raise ValueError("held-out image/depth bytes overlap reconstruction inputs")
    poses = {
        p["image_id"]: p
        for p in read_json(reference_path)["poses"]
        if p["timestamp_delta_seconds"] <= 0.02
    }
    anchor = sensor["anchor_view_id"]
    if anchor not in poses:
        raise ValueError("anchor needs independent reference pose for evaluation")

    def pose(key):
        p = poses[key]
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(p["quaternion_xyzw"]).as_matrix()
        m[:3, 3] = p["position_metres"]
        return m

    anchor_from_reference = np.linalg.inv(pose(anchor))
    camera_errors = []
    for key, camera in sensor["cameras"].items():
        if key not in poses:
            continue
        expected = anchor_from_reference @ pose(key)
        estimated = np.linalg.inv(camera["camera_from_sensor_world"])
        camera_errors.append(
            {
                "view_id": key,
                "gauge_anchor": key == anchor,
                "rotation_error_degrees": rotation_error(estimated[:3, :3], expected[:3, :3]),
                "translation_error_metres": float(
                    np.linalg.norm(estimated[:3, 3] - expected[:3, 3])
                ),
                "scale_fitted_from_evaluation": False,
            }
        )
    reference_points = []
    for row in rows:
        key = row["image_id"]
        if key not in poses:
            continue
        depth, cal = load_sensor(row, heldout_manifest)
        samples = dense_samples(
            depth, load_image(row["resolved_file"]), cal["intrinsics"], stride=6
        )
        reference_points.append(transform(samples["points"], anchor_from_reference @ pose(key)))
    if not reference_points:
        raise ValueError("no timestamp-associated held-out reference geometry")
    reference = np.concatenate(reference_points)
    cloud_file = sensor["point_cloud"]["file"]
    if digest(cloud_file) != sensor["point_cloud"]["sha256"]:
        raise ValueError("fused cloud bytes changed")
    with np.load(cloud_file) as cloud:
        predicted = cloud["points"]
        view_count = cloud["view_count"]
    tree = cKDTree(reference)
    distance = tree.query(predicted)[0]
    reverse = cKDTree(predicted).query(reference)[0]

    # Report one-way distances plus overlap-limited diagnostics, not full room completeness.
    def stats(values):
        return {
            "samples": len(values),
            "median_metres": float(np.median(values)) if len(values) else None,
            "p95_metres": float(np.percentile(values, 95)) if len(values) else None,
            "fraction_within_5cm": float(np.mean(values <= 0.05)) if len(values) else None,
            "fraction_within_10cm": float(np.mean(values <= 0.1)) if len(values) else None,
        }

    plane_checks = []
    for plane in sensor["surface_planes"]:
        ids = plane["support_voxel_indices"]
        support = predicted[ids]
        distances, nearest = tree.query(support)
        reference_support = np.unique(nearest[distances < 0.1])
        q = reference[reference_support]
        entry = {
            "plane_id": plane["id"],
            "surface_hypothesis": plane["surface_hypothesis"],
            "heldout_nearest_surface_distance": stats(distances),
            "reference_support_points": len(q),
            "normal_error_degrees": None,
            "reference_local_plane_rms_metres": None,
            "reference_semantic_identity_verified": False,
        }
        if len(q) >= 100:
            centre = q.mean(axis=0)
            _, singular, v = np.linalg.svd(q - centre, full_matrices=False)
            residual = float(np.sqrt(np.mean(np.sum((q - centre) * v[-1], axis=1) ** 2)))
            entry["reference_local_plane_rms_metres"] = residual
            if residual <= 0.035 and singular[1] / np.sqrt(len(q)) > 0.1:
                entry["normal_error_degrees"] = float(
                    np.degrees(np.arccos(np.clip(abs(np.dot(plane["normal"], v[-1])), -1, 1)))
                )
        plane_checks.append(entry)
    report = {
        "mode": "sensor_assisted_rgbd",
        "project_sha256": digest(project_path),
        "reference_sha256": digest(reference_path),
        "heldout_manifest_sha256": digest(heldout_manifest),
        "reference_scale_fit": None,
        "reference_poses_used_for_inference": False,
        "registered_views": len(sensor["cameras"]),
        "input_views": len(sensor["sources"]),
        "failed_view_fraction": 1 - len(sensor["cameras"]) / len(sensor["sources"]),
        "camera_errors": camera_errors,
        "predicted_to_heldout_sensor_geometry": stats(distance),
        "heldout_sensor_geometry_to_predicted": stats(reverse),
        "semantic_plane_checks": plane_checks,
        "boundary_alignment_error": None,
        "verified_physical_dimension_error": None,
        "structural_completeness": None,
        "furniture_localization_error": None,
        "sensor_uncertainty_metres": None,
        "multi_view_supported_voxel_fraction": float(np.mean(view_count >= 2)),
        "limitations": [
            "Held-out frames and optical poses are separate from inference; sensor/calibration family is shared, so systematic sensor bias is not independently tested.",
            "Different fields of view make full-cloud nearest-neighbour distances a mixture of alignment error and missing overlap.",
            "Local held-out plane fits use geometric proximity, not human structural labels; semantic identity is unverified.",
            "No ground-truth room boundary, dimensions or complete mesh; no furniture-placement accuracy claim.",
        ],
    }
    write_json(output, report)
    np.savez_compressed(
        Path(output).with_suffix(".npz"),
        reference_points=reference,
        predicted_to_reference_distance=distance,
        reference_to_predicted_distance=reverse,
    )
    return report
