"""Attach partial photo evidence without altering an editable metric room."""

from __future__ import annotations

import copy

from .model import validate_scene


def attach_photo_estimate(scene, estimate, reviewer, evidence, demo_only=False):
    from room_layout.scene import validate_photo_scene

    validate_scene(scene)
    validate_photo_scene(estimate)
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or not isinstance(evidence, str)
        or not evidence.strip()
    ):
        raise ValueError("photo/room correspondence needs reviewer and evidence")
    if scene.get("provenance", {}).get("derived_from_photo") is False and not demo_only:
        raise ValueError("an independent demonstration room requires demo_only=true")
    updated = copy.deepcopy(scene)
    references = updated.setdefault("photo_structure_estimates", [])
    if any(r["photo_room_id"] == estimate["room_id"] for r in references):
        raise ValueError("photo estimate already attached")
    references.append(
        {
            "photo_room_id": estimate["room_id"],
            "views": copy.deepcopy(estimate["views"]),
            "coordinate_frames": copy.deepcopy(estimate["coordinate_frames"]),
            "reviewer": reviewer,
            "evidence": evidence,
            "correspondence_status": "demo_only" if demo_only else "user_supplied_unverified",
            "metric_geometry_applied": False,
            "physical_accuracy_verified": False,
        }
    )
    if "shared_geometry" in estimate:
        references[-1]["shared_geometry"] = copy.deepcopy(estimate["shared_geometry"])
        references[-1]["shared_geometry_role"] = (
            "estimated_reference_only_no_metric_room_transform_applied"
        )
    if "sensor_reconstruction" in estimate:
        references[-1]["sensor_reconstruction"] = copy.deepcopy(estimate["sensor_reconstruction"])
        references[-1]["sensor_geometry_role"] = "measured_depth_with_estimated_pose_reference_only_no_collision_envelope"
    return validate_scene(updated)
