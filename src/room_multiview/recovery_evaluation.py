"""Separate reference-pose audit before any recovered camera map is fused."""

import numpy as np
from scipy.spatial.transform import Rotation

from furniture_bench.data import digest, read_json, write_json

from .geometry import rotation_error
from .numeric import rigid_pose


def summarize_cameras(cameras, anchor, references, primary_ids, baseline_ids):
    truth = {r["image_id"]: r for r in references if r["timestamp_delta_seconds"] <= 0.02}
    if anchor not in truth:
        raise ValueError("anchor reference pose unavailable")

    def world_pose(key):
        row = truth[key]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(row["quaternion_xyzw"]).as_matrix()
        pose[:3, 3] = row["position_metres"]
        return pose

    anchor_from_world = np.linalg.inv(world_pose(anchor))
    rows = []
    for key, camera in cameras.items():
        if key not in truth:
            continue
        estimated = np.linalg.inv(rigid_pose(camera["camera_from_sensor_world"]))
        expected = anchor_from_world @ world_pose(key)
        rows.append(
            {
                "view_id": key,
                "anchor": key == anchor,
                "is_original_phase5_input": key in primary_ids,
                "is_common_with_baseline": key in baseline_ids,
                "rotation_error_degrees": rotation_error(estimated[:3, :3], expected[:3, :3]),
                "translation_error_metres": float(
                    np.linalg.norm(estimated[:3, 3] - expected[:3, 3])
                ),
                "estimated_centre_metres": estimated[:3, 3].tolist(),
                "reference_centre_metres": expected[:3, 3].tolist(),
                "reference_scale_fitted": False,
            }
        )

    def stats(selected):
        selected = [r for r in selected if not r["anchor"]]
        translation = np.array([r["translation_error_metres"] for r in selected])
        angle = np.array([r["rotation_error_degrees"] for r in selected])
        return {
            "evaluated_non_anchor_views": len(selected),
            "translation_rmse_metres": float(np.sqrt(np.mean(translation**2)))
            if len(selected)
            else None,
            "translation_median_metres": float(np.median(translation)) if len(selected) else None,
            "translation_max_metres": float(translation.max()) if len(selected) else None,
            "rotation_mean_degrees": float(angle.mean()) if len(selected) else None,
            "rotation_max_degrees": float(angle.max()) if len(selected) else None,
        }

    relative = []
    keys = [key for key in cameras if key in truth]
    for a, b in zip(keys, keys[1:]):
        expected = np.linalg.inv(world_pose(a)) @ world_pose(b)
        estimated = np.array(cameras[a]["camera_from_sensor_world"]) @ np.linalg.inv(
            cameras[b]["camera_from_sensor_world"]
        )
        relative.append(
            {
                "views": [a, b],
                "rotation_error_degrees": rotation_error(estimated[:3, :3], expected[:3, :3]),
                "translation_error_metres": float(
                    np.linalg.norm(estimated[:3, 3] - expected[:3, 3])
                ),
            }
        )
    return {
        "registered_views": len(cameras),
        "registered_original_phase5_views": sum(k in primary_ids for k in cameras),
        "original_phase5_inputs": len(primary_ids),
        "registered_original_fraction": sum(k in primary_ids for k in cameras) / len(primary_ids),
        "all_registered": stats(rows),
        "original_phase5": stats([r for r in rows if r["is_original_phase5_input"]]),
        "common_with_phase5_baseline": stats([r for r in rows if r["is_common_with_baseline"]]),
        "per_camera": rows,
        "relative_pose_errors": relative,
        "reference_poses_used_for_inference": False,
        "evaluation_scale_fit": None,
    }


def evaluate_variants(worker_output, reference_path, baseline_project, primary_manifest, output):
    from pathlib import Path

    worker_output = Path(worker_output)
    references = read_json(reference_path)["poses"]
    baseline = read_json(baseline_project)["sensor_reconstruction"]
    primary = {r["image_id"] for r in read_json(primary_manifest)["images"]}
    result = {
        "reference_sha256": digest(reference_path),
        "primary_manifest_sha256": digest(primary_manifest),
        "baseline_project_sha256": digest(baseline_project),
        "variants": {},
    }
    result["baseline"] = summarize_cameras(
        baseline["cameras"],
        baseline["anchor_view_id"],
        references,
        primary,
        set(baseline["cameras"]),
    )
    for name in ("hybrid", "hybrid-icp", "pose-graph"):
        path = worker_output / (name + ".json")
        variant = read_json(path)
        result["variants"][name] = summarize_cameras(
            variant["cameras"],
            variant["anchor_view_id"],
            references,
            primary,
            set(baseline["cameras"]),
        )
        result["variants"][name]["variant_sha256"] = digest(path)
        result["variants"][name]["optimizer_adopted"] = variant.get("optimizer_adopted")
    write_json(output, result)
    return result
