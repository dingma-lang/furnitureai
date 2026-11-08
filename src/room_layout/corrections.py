"""Corrections are saved as separate, source-bound artifacts, never as auto outputs."""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone

from .scene import validate_photo_scene


def apply_corrections(scene, correction):
    validate_photo_scene(scene)
    if not correction.get("reviewer") or not correction.get("reason"):
        raise ValueError("correction needs reviewer and reason")
    by_id = {v["view_id"]: v for v in scene["views"]}
    view_id = correction.get("view_id")
    if view_id not in by_id or correction.get("image_sha256") != by_id[view_id]["image_sha256"]:
        raise ValueError("correction references a different view/image")
    updated = copy.deepcopy(scene)
    view = next(v for v in updated["views"] if v["view_id"] == view_id)
    width, height = view["image_size_pixels"]
    for key in ("corner_corrections", "boundary_corrections"):
        for item in correction.get(key, []):
            points = [item["pixel"]] if key == "corner_corrections" else item["points"]
            if key == "boundary_corrections" and len(points) < 2:
                raise ValueError("a corrected boundary needs at least two points")
            for point in points:
                if (
                    len(point) != 2
                    or not all(
                        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                        for v in point
                    )
                    or not (0 <= point[0] < width and 0 <= point[1] < height)
                ):
                    raise ValueError("correction point outside its image")
        view.setdefault("user_corrections", {})[key] = copy.deepcopy(correction.get(key, []))
    for key, value in correction.get("measurements", {}).items():
        if key not in updated["measurements"]:
            raise ValueError("unknown room measurement name")
        updated["measurements"][key] = copy.deepcopy(value)
    updated.setdefault("wall_measurements", []).extend(
        copy.deepcopy(correction.get("wall_measurements", []))
    )
    for reference in correction.get("reference_objects", []):
        if not reference.get("source") or reference.get("status") not in ("measured", "estimated"):
            raise ValueError("reference object needs source and measured/estimated status")
        if reference["status"] == "measured" and not reference.get("evidence"):
            raise ValueError("reference measurement evidence required")
        if len(reference.get("size_metres", [])) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in reference["size_metres"]
        ):
            raise ValueError("reference dimensions must be finite and positive")
        confidence = reference.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("reference object needs confidence in [0,1]")
        updated["reference_objects"].append(copy.deepcopy(reference))
    camera = correction.get("camera", {})
    if set(camera) - {"intrinsics", "camera_from_room", "source", "evidence"}:
        raise ValueError("unknown camera correction fields")
    if camera:
        if not camera.get("source"):
            raise ValueError("camera input requires source")
        # Supplied matrices are partial estimates, never automatic calibration proof.
        import numpy as np

        for key, shape in (("intrinsics", (3, 3)), ("camera_from_room", (4, 4))):
            if key in camera:
                matrix = np.asarray(camera[key], dtype=float)
                if (
                    matrix.shape != shape
                    or not np.isfinite(matrix).all()
                    or np.linalg.matrix_rank(matrix) != shape[0]
                ):
                    raise ValueError("invalid camera matrix")
                view["camera"][key] = matrix.tolist()
        view["camera"]["provided_calibration_source"] = copy.deepcopy(camera)
        view["camera"]["calibration_status"] = "externally_supplied_unverified"
    verdict = correction.get("geometry_verdict", "unreviewed")
    if verdict not in ("unreviewed", "accepted_as_estimate", "rejected"):
        raise ValueError("invalid geometry verdict")
    updated["review"] = {
        "reviewer": correction["reviewer"],
        "reason": correction["reason"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "automatic_predictions_preserved": True,
        "used_for_training": False,
    }
    updated["physical_accuracy_verified"] = False
    updated["fit_readiness"]["ready"] = False
    return validate_photo_scene(updated)
