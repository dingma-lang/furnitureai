"""Partial multi-view photo-room schema and explicit metric integration gate."""

from __future__ import annotations

import copy

MEASUREMENT_KEYS = ("room_width", "room_length", "ceiling_height", "camera_height")


def unknown_measurement(reason):
    return {
        "status": "unknown",
        "value": None,
        "units": "m",
        "source": None,
        "confidence": None,
        "reason": reason,
    }


def photo_scene(image_id, image_sha256, width, height, structure, camera_metadata, furniture):
    from furniture_bench.room_links import make_observation

    record = {
        "image_id": image_id,
        "image_sha256": image_sha256,
        "width": width,
        "height": height,
        "detections": furniture,
    }
    observations = [
        make_observation(record, [d["detection_id"]], f"{image_id}:{d['detection_id']}")
        for d in furniture
    ]
    scene = {
        "schema_version": 2,
        "kind": "partial_photo_room",
        "room_id": f"photo-room:{image_id}",
        "views": [
            {
                "view_id": image_id,
                "image_sha256": image_sha256,
                "image_size_pixels": [width, height],
                "camera": {
                    "metadata": camera_metadata,
                    "calibration_status": "uncalibrated",
                    "intrinsics": None,
                    "camera_from_room": None,
                    "perspective": structure["camera_perspective"],
                },
                "visible_structure": copy.deepcopy(structure),
            }
        ],
        "coordinate_frames": {
            "image_pixels": {
                "units": "pixels",
                "origin": "top-left",
                "orientation": "EXIF-corrected",
            },
            "camera_relative_3d": {"units": "unknown", "available": False},
            "world_room": {"units": "unresolved", "available": False},
            "physical_metres": {"available": False},
            "transforms": {
                "image_to_camera_3d": None,
                "camera_to_world": None,
                "world_to_metres": None,
            },
        },
        "floor": {
            "image_region_ids": [r["id"] for r in structure["regions"] if r["surface"] == "floor"],
            "world_polygon": None,
            "hidden_extent": "unknown",
            "status": "estimated_visible_regions_only",
        },
        "ceiling": {
            "image_region_ids": [
                r["id"] for r in structure["regions"] if r["surface"] == "ceiling"
            ],
            "world_geometry": None,
            "status": "estimated_visible_regions_only",
        },
        "walls": {
            "image_region_ids": [r["id"] for r in structure["regions"] if r["surface"] == "wall"],
            "world_planes": None,
            "plane_count": None,
            "status": "semantic_regions_not_separate_planes",
        },
        "openings": [
            {
                "image_region_id": r["id"],
                "kind": r["surface"],
                "world_geometry": None,
                "status": "semantic_candidate_requires_review",
            }
            for r in structure["regions"]
            if r["surface"] in ("door", "window")
        ],
        "measurements": {
            key: unknown_measurement("No supplied verified reference for this photograph")
            for key in MEASUREMENT_KEYS
        },
        "reference_objects": [],
        "wall_measurements": [],
        "furniture_observations": observations,
        "furniture_entities": [
            {
                "id": "candidate:" + o["observation_id"],
                "category_hypothesis": o["label_hypothesis"],
                "observation_ids": [o["observation_id"]],
                "status": "unreviewed_candidate_may_be_duplicate_or_fragment",
                "camera_position": None,
                "world_position": None,
                "rotation_degrees": None,
                "physical_dimensions": None,
                "metric_room_entity_id": None,
            }
            for o in observations
        ],
        "metric_scene": None,
        "reconstruction_status": "partial_visible_structure",
        "physical_accuracy_verified": False,
        "fit_readiness": {
            "ready": False,
            "reasons": [
                "World-space closed floor and wall planes unresolved",
                "Camera intrinsics/pose and metric transform unknown",
                "Furniture centres, rotations and dimensions unknown",
            ],
        },
        "future_views": {
            "supported_input_kinds": [
                "perspective_photo",
                "additional_photos",
                "scan_video",
                "compatible_lidar",
            ],
            "registration_status": "not_implemented",
            "hidden_geometry_completion": "not_attempted",
        },
    }
    for key in ("floor", "ceiling", "walls"):
        if not scene[key]["image_region_ids"]:
            scene[key]["status"] = "unknown_no_visible_region"
    return validate_photo_scene(scene)


def validate_photo_scene(scene):
    if scene.get("schema_version") != 2 or scene.get("kind") != "partial_photo_room":
        raise ValueError("not a partial photo-room v2 document")
    if not scene.get("views") or len({v["view_id"] for v in scene["views"]}) != len(scene["views"]):
        raise ValueError("photo-room needs unique view IDs")
    for measurement in list(scene["measurements"].values()) + scene.get("wall_measurements", []):
        if measurement.get("status") not in ("unknown", "estimated", "measured"):
            raise ValueError("measurement status must be unknown, estimated or measured")
        if measurement["status"] == "unknown" and measurement.get("value") is not None:
            raise ValueError("unknown measurements cannot carry invented values")
        if measurement["status"] != "unknown":
            from room_reconstruction.model import number

            number(measurement.get("value"), "measurement", positive=True)
            if not measurement.get("source"):
                raise ValueError("measurement requires source")
            confidence = number(measurement.get("confidence"), "measurement confidence")
            if not 0 <= confidence <= 1 or measurement.get("units") != "m":
                raise ValueError("measurement needs confidence in [0,1] and metre units")
            if measurement["status"] == "measured" and not measurement.get("evidence"):
                raise ValueError("measured values require physical measurement evidence")
    for wall in scene.get("wall_measurements", []):
        if not isinstance(wall.get("reference"), str) or not wall["reference"].strip():
            raise ValueError("wall length requires a user-labelled wall reference")
    return scene


def require_metric_scene(scene):
    validate_photo_scene(scene)
    if not scene["fit_readiness"]["ready"] or scene.get("metric_scene") is None:
        raise ValueError(
            "Photo structure is not a metric room; physical fit/replacement is unavailable"
        )
    from room_reconstruction.model import validate_scene

    return validate_scene(copy.deepcopy(scene["metric_scene"]))
