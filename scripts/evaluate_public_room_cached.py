"""Audit cached RGB-D graph; fuse separately anchored fragments passing a pose audit.

References gate whole fragments, never supply/correct inference transforms.
Held-out depth remains unused by tracking, fragment selection and fusion.
"""
import argparse
import copy
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.data import digest, read_json, write_json
from room_multiview.registration_graph import largest_component_cameras
from room_multiview.recovery_evaluation import summarize_cameras
from room_multiview.sensor_pipeline import fuse_sensor_state
from room_multiview.sensor_evaluation import evaluate_sensor
from room_multiview.public_room_audit import approve_fragment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--report-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Reuse completed fragment fusion after an interrupted audit")
    a = p.parse_args()
    data, root, output = Path(a.data), Path(a.report_root), Path(a.output)
    if output.exists() and not a.resume:
        raise ValueError("preserve old evaluation; choose a new output")
    output.mkdir(parents=True, exist_ok=a.resume)
    start = perf_counter()
    worker = root / "odometry-room/worker"
    pairs = read_json(worker / "pairs.json")
    references = read_json(data / "reference-poses.json")["poses"]
    reference_ids = {r["image_id"] for r in references if r["timestamp_delta_seconds"] <= 0.02}
    primary = {r["image_id"] for r in read_json(data / "primary-usable.json")["images"]}
    heldout = read_json(data / "heldout-usable.json")["images"]
    held_ids, held_rgb, held_depth = ({r["image_id"] for r in heldout}, {r["image_sha256"] for r in heldout}, {r["sensor_depth"]["sha256"] for r in heldout})
    if any(k in held_ids or s["image_sha256"] in held_rgb or s["depth_sha256"] in held_depth for k, s in pairs["sources"].items()):
        raise ValueError("held-out tracking leakage")
    source_health = {}
    for key, source in pairs["sources"].items():
        with Image.open(source["depth_path"]) as image:
            values = np.asarray(image)
        source_health[key] = {"valid_depth_fraction": float(np.mean((values >= 400) & (values <= 5000))), "timestamp": source["timestamp"], "source_rgb_index": source["source_rgb_index"]}
    baseline = read_json(root / "sift-rgbd/room.json")
    # This dataset has no confirmed user measurements to transform to another anchor.
    if any(m["status"] != "unknown" for m in baseline["measurements"].values()):
        raise ValueError("reanchoring measured geometry requires an explicit transform")
    variant_audits = {}
    for variant in ("hybrid", "hybrid-icp", "pose-graph"):
        value = read_json(worker / (variant + ".json"))
        # Insertion order is BFS; sort timestamps for meaningful relative pose diagnostics.
        cameras = dict(sorted(value["cameras"].items(), key=lambda item: pairs["sources"][item[0]]["timestamp"]))
        variant_audits[variant] = summarize_cameras(cameras, value["anchor_view_id"], references, primary, set())
    write_json(output / "variant-pose-audit.json", variant_audits)
    remaining = list(pairs["sources"])
    components = []
    policy = {"minimum_primary_views": 2, "maximum_tracking_position_error_metres": 0.10, "maximum_tracking_rotation_error_degrees": 3.0, "scope": "whole component ARKit reference audit, not independently verified physical placement", "reference_pose_corrections": False, "heldout_depth_used_for_selection": False}
    while remaining:
        current = set(remaining)
        edges = [e for e in pairs["pairs"] if set(e["views"]) <= current]
        cameras, parents, used, anchor, sizes = largest_component_cameras(remaining, edges, primary, True)
        members = set(cameras)
        remaining = [k for k in remaining if k not in members]
        index = len(components)
        ids = sorted(members, key=lambda key: source_health[key]["timestamp"])
        pose_map = {k: {"camera_from_sensor_world": cameras[k].tolist()} for k in ids}
        audit = summarize_cameras(pose_map, anchor, references, primary, set()) if anchor in reference_ids else {
            "all_registered": {"evaluated_non_anchor_views": 0, "translation_max_metres": None, "rotation_max_degrees": None},
            "per_camera": [{"view_id": k} for k in ids],
            "reason": "anchor_reference_unavailable_no_accuracy_or_fusion_claim",
        }
        stats = audit["all_registered"]
        passed = approve_fragment(stats, len(members), len(members & primary))
        entry = {"id": index, "anchor": anchor, "tracking_views": len(members), "primary_views": len(members & primary), "raw_index_range": [source_health[ids[0]]["source_rgb_index"], source_health[ids[-1]]["source_rgb_index"]], "time_range": [source_health[ids[0]]["timestamp"], source_health[ids[-1]]["timestamp"]], "depth_valid_fraction_median": float(np.median([source_health[k]["valid_depth_fraction"] for k in ids])), "pose_audit": audit, "fusion_approved_by_audit": passed, "reason": "passed_whole_component_audit" if passed else "too_few_primary_views_or_pose_error_exceeds_audit_policy"}
        if passed:
            folder = output / f"fragment-{index:02d}"
            already_fused = (folder / "room.json").exists() and (folder / "heldout-evaluation.json").exists()
            folder.mkdir(exist_ok=a.resume)
            state = copy.deepcopy(baseline)
            sensor = state["sensor_reconstruction"]
            sensor["previous_registration"] = {"project": str((root / "sift-rgbd/room.json").resolve()), "sha256": digest(root / "sift-rgbd/room.json"), "cameras": sensor["cameras"]}
            for key in members & primary:
                source = sensor["sources"][key]
                if source["image_sha256"] != pairs["sources"][key]["image_sha256"] or source["depth_sha256"] != pairs["sources"][key]["depth_sha256"]:
                    raise ValueError("cached observation mismatch")
            sensor["anchor_view_id"] = anchor
            sensor["cameras"] = {key: {**pose_map[key], "units": "m", "status": "estimated_cached_hybrid_icp"} for key in ids if key in primary}
            sensor["registration_method"] = "cached RGB-D hybrid + locally adopted ICP; separately anchored fragment"
            sensor["fragment_audit"] = {"policy": policy, "component": index, "tracking_reference_poses_used": False, "reference_poses_used_for_component_acceptance": True, "reference_transforms_used_for_fusion": False}
            sensor["registration_failures"] = [{"view_id": k, "reason": "outside_this_connected_component"} for k in sensor["sources"] if k not in sensor["cameras"]]
            quick = read_json(root / "quick-room.json")
            state["furniture_observations"] = copy.deepcopy(quick["furniture_observations"])
            state["furniture_entities"] = copy.deepcopy(quick["furniture_entities"])
            state = read_json(folder / "room.json") if already_fused else fuse_sensor_state(state, folder)
            evaluation = read_json(folder / "heldout-evaluation.json") if already_fused else evaluate_sensor(folder / "room.json", data / "heldout-usable.json", data / "reference-poses.json", folder / "heldout-evaluation.json")
            # References are ARKit trajectory, not the optical motion-capture of the old dataset.
            evaluation["reference_kind"] = "ARKit trajectory and held-out LiDAR, correlated sensor family; no external ground truth"
            write_json(folder / "heldout-evaluation.json", evaluation)
            entry["fusion"] = {"directory": str(folder.resolve()), "points": state["sensor_reconstruction"]["point_cloud"]["points"], "triangles": state["sensor_reconstruction"]["observed_partial_mesh"]["triangles"], "planes": [{k: v for k, v in plane.items() if k not in ("support_voxel_indices", "observed_cells_uv")} for plane in state["sensor_reconstruction"]["surface_planes"]], "heldout_evaluation": evaluation, "runtime": state["sensor_runtime"]}
        components.append(entry)
    boundaries = []
    membership = {r["view_id"]: c["id"] for c in components for r in c["pose_audit"]["per_camera"]}
    for pair in pairs["pairs"]:
        left, right = pair["views"]
        if membership[left] != membership[right]:
            boundaries.append({"views": pair["views"], "reason": pair["reason"], "sample_gap": pair["sample_gap"], "actual_delta_seconds": abs(source_health[right]["timestamp"] - source_health[left]["timestamp"]), "valid_depth_fraction": [source_health[left]["valid_depth_fraction"], source_health[right]["valid_depth_fraction"]], "forward": pair.get("forward"), "reverse": pair.get("reverse")})
    write_json(output / "depth-health.json", source_health)
    write_json(output / "component-boundaries.json", boundaries)
    write_json(output / "components.json", {"policy": policy, "components": components, "seconds": perf_counter() - start, "cached_pairs_sha256": digest(worker / "pairs.json"), "reference_poses_used_for_inference": False, "new_tracking_runs": 0, "boundary_reasons": dict(Counter(b["reason"] for b in boundaries)), "reference_warning": "ARKit references share sensors/calibration; these are consistency audits, not externally verified accuracy. Unaligned fragments must not be merged."})
    print({"components": len(components), "audit_approved_fragments": sum(c["fusion_approved_by_audit"] for c in components), "seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
