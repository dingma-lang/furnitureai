"""Immutable progressive state, observation preservation, metric-scale provenance."""

import copy
import math

import numpy as np

from room_layout.scene import validate_photo_scene


def merge_document(project, incoming):
    """Keep complete source documents, including corrections and furniture associations."""
    validate_photo_scene(incoming)
    updated = copy.deepcopy(project) if project is not None else copy.deepcopy(incoming)
    updated.setdefault("view_documents", {})
    updated.setdefault("evidence_conflicts", [])
    present = {v["view_id"]: v for v in updated["views"]}
    for view in incoming["views"]:
        key = view["view_id"]
        if key in present:
            if present[key]["image_sha256"] != view["image_sha256"]:
                raise ValueError("existing view ID references changed image bytes")
            # Existing reviewed view/camera/corrections win; never replace on resume.
        else:
            updated["views"].append(copy.deepcopy(view))
        updated["view_documents"].setdefault(key, copy.deepcopy(incoming))
    for name, measurement in incoming["measurements"].items():
        old = updated["measurements"].get(name)
        if not old or old["status"] == "unknown":
            updated["measurements"][name] = copy.deepcopy(measurement)
        elif measurement["status"] != "unknown" and old != measurement:
            conflict = {
                "kind": "measurement_conflict",
                "field": name,
                "preserved": old,
                "incoming": measurement,
                "resolution": "needs_user_review",
            }
            if conflict not in updated["evidence_conflicts"]:
                updated["evidence_conflicts"].append(copy.deepcopy(conflict))
    for field, key in (("furniture_observations", "observation_id"), ("furniture_entities", "id")):
        existing = {v[key]: v for v in updated.get(field, [])}
        for value in incoming.get(field, []):
            if value[key] not in existing:
                updated.setdefault(field, []).append(copy.deepcopy(value))
                existing[value[key]] = value
    for field in ("reference_objects", "wall_measurements"):
        for value in incoming.get(field, []):
            identity = "reference" if field == "wall_measurements" else "id"
            previous = next(
                (
                    r
                    for r in updated.get(field, [])
                    if value.get(identity) is not None and r.get(identity) == value[identity]
                ),
                None,
            )
            if previous is not None and previous != value:
                updated["evidence_conflicts"].append(
                    {
                        "kind": field + "_conflict",
                        "preserved": copy.deepcopy(previous),
                        "incoming": copy.deepcopy(value),
                        "resolution": "needs_user_review",
                    }
                )
                continue
            if value not in updated.setdefault(field, []):
                updated[field].append(copy.deepcopy(value))
    updated["physical_accuracy_verified"] = False
    updated["fit_readiness"]["ready"] = False
    return validate_photo_scene(updated)


def establish_scale(project, measurement):
    """A measured distance must identify actual reconstructed landmarks, not a bbox."""
    required = ("source", "evidence")
    value, confidence = measurement.get("value"), measurement.get("confidence")
    if (
        measurement.get("kind") != "landmark_distance"
        or measurement.get("units") != "m"
        or measurement.get("status") != "measured"
        or not all(measurement.get(k) for k in required)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError(
            "scale requires a positive sourced measured landmark distance and confidence"
        )
    points = {p["id"]: p for p in project["shared_geometry"]["landmarks"]}
    ids = measurement.get("landmark_ids", [])
    if len(ids) != 2 or len(set(ids)) != 2 or any(i not in points for i in ids):
        raise ValueError("scale references two distinct existing landmarks")
    distance = np.linalg.norm(np.asarray(points[ids[0]]["position"]) - points[ids[1]]["position"])
    if distance < 1e-8:
        raise ValueError("scale landmarks are coincident")
    factor = float(value / distance)
    old = project["shared_geometry"].get("metric_scale")
    if old and abs(factor / old["metres_per_unit"] - 1) > 0.05:
        raise ValueError("new scale conflicts with preserved measurement; resolve explicitly")
    updated = copy.deepcopy(project)
    updated["shared_geometry"]["metric_scale"] = {
        "metres_per_unit": factor,
        "measurement": copy.deepcopy(measurement),
        "geometry_status": "estimated_geometry_with_measured_scale",
        "physical_accuracy_verified": False,
    }
    updated["coordinate_frames"]["transforms"]["world_to_metres"] = {
        "scale": factor,
        "rotation": np.eye(3).tolist(),
        "translation": [0, 0, 0],
        "source": copy.deepcopy(measurement),
    }
    updated["coordinate_frames"]["physical_metres"]["available"] = True
    updated["fit_readiness"]["ready"] = False
    return updated


def photo_guidance(project):
    recommendations, observation_summary = [], []
    cameras = project.get("shared_geometry", {}).get("cameras", {})
    for view in project["views"]:
        structure = view["visible_structure"]
        fractions = {r["surface"]: r.get("pixel_fraction", 0) for r in structure.get("regions", [])}
        observation_summary.append(
            {
                "view_id": view["view_id"],
                "estimated_visible_pixel_fractions": fractions,
                "corner_candidates": len(structure.get("corner_candidates", [])),
                "world_surface_coverage": None,
            }
        )
        if fractions.get("floor", 0) < 0.03:
            recommendations.append(
                {
                    "view_id": view["view_id"],
                    "surface": "floor",
                    "advice": "Take a lower-tilted overlapping view showing visible floor-wall junctions around furniture.",
                }
            )
        if fractions.get("ceiling", 0) < 0.01:
            recommendations.append(
                {
                    "view_id": view["view_id"],
                    "surface": "ceiling",
                    "advice": "Add an upward-tilted overlapping view if ceiling height or upper corners matter.",
                }
            )
        if view["view_id"] not in cameras:
            recommendations.append(
                {
                    "view_id": view["view_id"],
                    "surface": "alignment",
                    "advice": "Add an intermediate view sharing distinctive static features with a registered view; avoid large viewpoint jumps.",
                }
            )
    if not any(v["visible_structure"].get("corner_candidates") for v in project["views"]):
        recommendations.append(
            {
                "surface": "corners",
                "advice": "Photograph a visible room junction from a shifted position; do not assume hidden corners exist at fixed pixels.",
            }
        )
    for recommendation in recommendations:
        recommendation.update(status="heuristic_not_verified", predicts_specific_hidden_wall=False)
    return {
        "observations": observation_summary,
        "recommendations": recommendations,
        "general_guidance": "Keep roughly 50–70% overlap, move sideways as well as rotate, keep lighting/static objects stable. This overlap target is heuristic.",
        "complete_room_coverage": "unknown",
        "unknown_wall_count": True,
        "note": "Missing semantic detections do not prove that a surface is absent.",
    }
