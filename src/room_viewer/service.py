"""Stateless editing and rendering adapters for the existing room schemas."""
import copy
from pathlib import Path

import numpy as np

from furniture_bench.data import digest, read_json
from room_layout.scene import validate_photo_scene
from room_reconstruction.clearance import assess_arrangement
from room_reconstruction.collision import solid_wall_parts, wall_body
from room_reconstruction.demo import demo_scene
from room_reconstruction.model import (
    propose_replacement,
    remove_furniture,
    validate_scene,
    vector,
    wall_length,
)

from .photo_objects import conceptual

ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "phase51": "reports/phase5-registration-recovery/fused-hybrid/room.json",
    "phase52": "reports/phase5-public-room/cached-rgbd-audit/fragment-08/room.json",
    "single-photo": "reports/phase5-public-room/quick-room.json",
}
POLICY = {"minimum_furniture_gap": 0.30, "minimum_wall_gap": 0.05, "front_access_depths": {"sofa": 0.50, "chair": 0.45}, "source": "Prototype demonstration preferences; not accessibility or building regulations"}


def editable(scene):
    return scene.get("schema_version") == 1 and (scene.get("provenance", {}).get("kind") == "manually_specified_demonstration" or conceptual(scene) or scene.get("provenance", {}).get("kind") == "destination_plan")


def validate_document(scene):
    if scene.get("schema_version") == 1:
        validate_scene(scene)
    else:
        validate_photo_scene(scene)
    return scene


def presentation(scene, observed=None):
    validate_document(scene)
    result = {"scene": scene, "editable": editable(scene), "physical_placement_verified": False, "observed": observed or {}, "status": "disabled", "assessment": None}
    if scene["schema_version"] == 1:
        result["wall_parts"] = [part for wall in scene["walls"] for part in solid_wall_parts(wall, scene["openings"])]
        walls = {w["id"]: w for w in scene["walls"]}
        result["opening_bodies"] = [{"kind": o["kind"], "id": o["id"], **wall_body(walls[o["wall_id"]], o["offset"], o["offset"] + o["width"], o["bottom"], o["bottom"] + o["height"], thickness=0.025)} for o in scene["openings"]]
        if editable(scene) and scene.get("provenance", {}).get("kind") != "destination_plan" and not conceptual(scene) and not any(f.get("source_observation_ids") for f in scene["furniture"]):
            assessment = assess_arrangement(scene, POLICY)
            result["assessment"] = assessment
            result["status"] = "invalid" if not assessment["geometrically_valid"] else "restricted" if not assessment["clearance_targets_met"] else "valid"
    result["geometry_notice"] = "Specified demonstration dimensions; gaps are calculated examples, not room measurements." if editable(scene) else "Unverified reconstruction / imported geometry. Unknown surfaces remain absent. Physical placement and distance checks disabled."
    if conceptual(scene) or any(f.get("source_observation_ids") for f in scene.get("furniture", [])):
        result["geometry_notice"] = "Editable design visualization only. User-chosen layout, furniture sizes and poses are not a calibrated photographic reconstruction. Physical fit checks disabled."
    if scene.get("provenance", {}).get("kind") == "destination_plan":
        result["geometry_notice"] = "Destination design drawing; physical placement unverified. Missing heights use visualization proportions only."
    return result


def edit_scene(scene, action):
    validate_document(scene)
    if not editable(scene):
        raise ValueError("Editing and metric placement are disabled for unverified scenes")
    updated = copy.deepcopy(scene)
    kind = action.get("type")
    if scene.get("provenance", {}).get("kind") == "destination_plan" and kind == "resize":
        raise ValueError("Edit destination room boundaries in the 2D plan")
    proposal = None
    if kind in ("update", "replace", "remove"):
        original = next((f for f in scene["furniture"] if f["id"] == action.get("id")), None)
        if original is None:
            raise ValueError("Select an existing furniture object")
        if kind == "remove":
            updated = remove_furniture(scene, original["id"])
        else:
            replacement = copy.deepcopy(original)
            for field in ("position", "size", "rotation_degrees"):
                if field in action:
                    replacement[field] = action[field]
            if original.get("source_observation_ids"):
                evidence = replacement.setdefault("design_evidence", {})
                if replacement["size"] != original["size"]:
                    evidence["dimensions"] = {"status": "unknown", "value": None, "units": "m", "source": "Edited design size; original evidence retained in project history"}
                    evidence["visualization_size"] = copy.deepcopy(replacement["size"])
                if replacement["position"] != original["position"]:
                    evidence["position"] = {"status": "unknown", "physical_position": None, "design_coordinates": copy.deepcopy(replacement["position"]), "source": "User-edited design position"}
            if kind == "replace":
                if original["category"] not in ("sofa", "chair", "table") and not original.get("source_observation_ids"):
                    raise ValueError("Prototype replacements support sofas, chairs and tables")
                replacement["prototype_variant"] = "replacement"
                if conceptual(scene) or scene.get("provenance", {}).get("kind") == "destination_plan" or original.get("source_observation_ids"):
                    updated["furniture"] = [replacement if f["id"] == original["id"] else f for f in updated["furniture"]]
                    proposal = {"physical_placement_verified": False, "accepted": None, "reason": "Design placeholder replacement only; physical fit not evaluated"}
                else:
                    updated, proposal = propose_replacement(scene, original["id"], replacement)
                if proposal["accepted"] is False:
                    # Retain an explicitly invalid draft for visual feedback, not an accepted placement.
                    updated = remove_furniture(scene, original["id"])
                    updated["furniture"].append(replacement)
            else:
                updated["furniture"] = [replacement if f["id"] == original["id"] else f for f in updated["furniture"]]
    elif kind == "resize":
        dimensions = list(vector(action.get("dimensions"), 3, "dimensions", positive=True))
        old = scene["dimensions"]
        def coordinate(value, axis):
            if value < 0:
                return value
            if value > old[axis]:
                return dimensions[axis] + value - old[axis]
            return value / old[axis] * dimensions[axis]
        old_lengths = {w["id"]: wall_length(w) for w in scene["walls"]}
        for wall in updated["walls"]:
            wall["start"] = [coordinate(v, axis) for axis, v in enumerate(wall["start"])]
            wall["end"] = [coordinate(v, axis) for axis, v in enumerate(wall["end"])]
            wall["height"] = dimensions[2] if wall["height"] == old[2] else min(wall["height"], dimensions[2])
        for opening in updated["openings"]:
            wall = next(w for w in updated["walls"] if w["id"] == opening["wall_id"])
            opening["offset"] *= wall_length(wall) / old_lengths[wall["id"]]
        updated["dimensions"] = dimensions
        updated["floor"]["boundary"] = [[0, 0], [dimensions[0], 0], dimensions[:2], [0, dimensions[1]]]
        if conceptual(updated):
            updated["provenance"]["previous_dimension_evidence"] = copy.deepcopy(scene["provenance"].get("dimension_evidence"))
            updated["provenance"]["dimension_evidence"] = {"status": "unknown", "value": None, "units": "m", "source": "Edited design extents; no renewed physical measurement confirmation"}
            updated["provenance"]["visualization_extents"] = {"values": dimensions, "role": "display_only", "physical_measurements": False}
    else:
        raise ValueError("Unknown scene operation")
    validate_scene(updated)
    result = presentation(updated)
    result["replacement_proposal"] = proposal
    return result


def observed_geometry(scene, project):
    """Read only report-owned, hash-verified files from a trusted preset, never upload paths."""
    sensor = scene.get("sensor_reconstruction", {})
    result = {"coordinate_system": "sensor-camera-x-right-y-down-z-forward", "points": [], "colors": [], "mesh_points": [], "triangles": []}
    for field, key in (("point_cloud", "file"), ("observed_partial_mesh", "file")):
        spec = sensor.get(field, {})
        if not spec.get(key):
            continue
        path = Path(spec[key]).resolve()
        if not path.is_relative_to((ROOT / "reports").resolve()) or not path.is_file():
            raise ValueError("Observed geometry must be an existing report asset")
        if spec.get("sha256") and digest(path) != spec["sha256"]:
            raise ValueError("Observed geometry hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            points = archive["points"]
            if not np.isfinite(points).all():
                raise ValueError("Nonfinite observed points")
            if field == "point_cloud":
                step = max(1, int(np.ceil(len(points) / 30000)))
                result["points"] = points[::step].tolist()
                result["colors"] = archive["colors"][::step].tolist()
                result["source_point_count"] = len(points)
            else:
                result["mesh_points"] = points.tolist()
                result["triangles"] = archive["triangles"].tolist()
    if not sensor and scene.get("shared_geometry", {}).get("landmarks"):
        result["points"] = [r["position"] for r in scene["shared_geometry"]["landmarks"]]
        result["colors"] = [[110, 190, 225]] * len(result["points"])
    result["units"] = "sensor metres, physical accuracy unverified" if sensor else "unknown relative scale"
    return result


def load_preset(name):
    if name == "demo":
        return presentation(demo_scene())
    if name not in PRESETS:
        raise ValueError("Unknown scene preset")
    path = ROOT / PRESETS[name]
    scene = read_json(path)
    return presentation(scene, observed_geometry(scene, path))
