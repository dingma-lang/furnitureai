"""Observation-to-room correspondences without inferring metric geometry from pixels."""

from __future__ import annotations

import copy
import hashlib
import json
import math

from .data import require_string, validate_prediction


def make_observation(record, detection_ids, observation_id):
    require_string(observation_id, "observation_id")
    require_string(record.get("image_sha256"), "image_sha256")
    if len(record["image_sha256"]) != 64:
        raise ValueError("observation must reference an image SHA-256")
    width, height = record["width"], record["height"]
    if not width > 0 or not height > 0:
        raise ValueError("image dimensions must be positive")
    by_id = {d["detection_id"]: d for d in record["detections"]}
    if not detection_ids or len(set(detection_ids)) != len(detection_ids):
        raise ValueError("select unique detection IDs")
    if any(d not in by_id for d in detection_ids):
        raise ValueError("unknown detection ID")
    selected = [by_id[d] for d in detection_ids]
    for detection in selected:
        validate_prediction(detection)
        if detection["box"][2] > width or detection["box"][3] > height:
            raise ValueError("observation box is outside the source image")
    if len({d["label"] for d in selected}) != 1:
        raise ValueError("fragment grouping requires a reviewed common category")
    return {
        "schema_version": 1,
        "observation_id": observation_id,
        "image_id": record["image_id"],
        "image_sha256": record["image_sha256"],
        "image_frame": {
            "width": width,
            "height": height,
            "units": "pixels",
            "orientation": "EXIF-corrected",
        },
        "label_hypothesis": selected[0]["label"],
        "image_regions": [
            {
                "detection_id": d["detection_id"],
                "box_xyxy": list(d["box"]),
                "model_score": d["score"],
            }
            for d in selected
        ],
        "grouping_review": "pending" if len(selected) > 1 else "not_required",
        "estimated_pose": None,
        "physical_dimensions": None,
        "room_entity_id": None,
        "link_status": "unlinked",
    }


def _metric_field(value, field):
    if value is None:
        return
    if value.get("units") != "m":
        raise ValueError(f"{field} must use metres")
    require_string(value.get("source"), f"{field}.source")
    status = value.get("status")
    if status not in ("estimated", "measured", "manufacturer_specified", "demo_specified"):
        raise ValueError(f"unsupported {field} provenance status")
    key = "position" if field == "estimated_pose" else "size"
    values = value.get(key)
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        )
    ):
        raise ValueError(f"{field}.{key} needs three finite values")
    if field == "physical_dimensions" and any(v <= 0 for v in values):
        raise ValueError("physical dimensions must be positive")
    if status == "measured":
        require_string(value.get("measurement_evidence"), "measurement_evidence")
    if status == "estimated":
        require_string(value.get("method"), "estimation method")
        uncertainty = value.get("uncertainty_metres")
        if (
            isinstance(uncertainty, bool)
            or not isinstance(uncertainty, (int, float))
            or not math.isfinite(uncertainty)
            or uncertainty < 0
        ):
            raise ValueError("estimated fields require explicit nonnegative uncertainty_metres")


def link_observation(
    observation,
    room,
    entity_id,
    reviewer,
    evidence,
    estimated_pose=None,
    physical_dimensions=None,
    demo_only=False,
):
    """Record a reviewed link; never overwrite a room entity's transforms or size."""
    from room_reconstruction.model import validate_scene

    validate_scene(room)
    require_string(reviewer, "reviewer")
    require_string(evidence, "link evidence")
    entity = next((e for e in room["furniture"] if e["id"] == entity_id), None)
    if entity is None:
        raise ValueError("unknown room furniture entity")
    if entity["category"] != observation["label_hypothesis"]:
        raise ValueError("category mismatch requires explicit observation review first")
    if len(observation["image_regions"]) > 1 and observation.get("grouping_review") != "reviewed":
        raise ValueError("fragment grouping must be reviewed before linking")
    if room.get("provenance", {}).get("derived_from_photo") is False and not demo_only:
        raise ValueError("link to the independent demo room must be explicitly demo_only")
    _metric_field(estimated_pose, "estimated_pose")
    _metric_field(physical_dimensions, "physical_dimensions")
    if not demo_only and any(
        v is not None and v["status"] == "demo_specified"
        for v in (estimated_pose, physical_dimensions)
    ):
        raise ValueError("demo-specified values require demo_only=true")
    linked = copy.deepcopy(observation)
    linked.update(
        room_snapshot_sha256=room_fingerprint(room),
        room_id=room["room_id"],
        room_entity_id=entity_id,
        link_status="demonstration_only" if demo_only else "reviewed_correspondence",
        link_review={"reviewer": reviewer, "evidence": evidence},
        estimated_pose=copy.deepcopy(estimated_pose),
        physical_dimensions=copy.deepcopy(physical_dimensions),
    )
    return linked


def room_fingerprint(room):
    canonical = json.dumps(room, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def link_matches_room(link, room):
    """A room edit/replacement makes the old snapshot link require explicit review."""
    return (
        link.get("room_id") == room["room_id"]
        and link.get("room_snapshot_sha256") == room_fingerprint(room)
        and any(e["id"] == link.get("room_entity_id") for e in room["furniture"])
    )
