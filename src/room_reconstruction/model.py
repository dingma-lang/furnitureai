from __future__ import annotations

import copy
import json
import math
from pathlib import Path

EPS = 1e-8


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def vector(value, length, name, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have {length} coordinates")
    return [number(v, name, positive) for v in value]


def require_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IDs must be nonempty strings")


def validate_furniture(item):
    require_id(item.get("id"))
    require_id(item.get("category"))
    vector(item.get("position"), 3, "furniture.position")
    vector(item.get("size"), 3, "furniture.size", positive=True)
    number(item.get("rotation_degrees"), "rotation_degrees")


def wall_length(wall):
    return math.dist(wall["start"], wall["end"])


def validate_scene(scene):
    if not isinstance(scene, dict) or scene.get("schema_version") != 1:
        raise ValueError("unsupported room schema")
    if scene.get("units") != "m" or scene.get("coordinate_system") != "right-handed-z-up":
        raise ValueError("v1 requires metres and right-handed-z-up coordinates")
    require_id(scene.get("room_id"))
    width, depth, height = vector(scene.get("dimensions"), 3, "dimensions", positive=True)
    floor = scene["floor"]
    expected_boundary = [[0, 0], [width, 0], [width, depth], [0, depth]]
    if scene.get("provenance", {}).get("kind") == "destination_plan":
        from .polygon import polygon
        polygon(floor.get("boundary"))
        if floor.get("z") != 0:
            raise ValueError("Destination floor must be at z=0")
    elif floor.get("boundary") != expected_boundary or floor.get("z") != 0:
        raise ValueError("v1 supports a rectangular floor at z=0, matching dimensions")
    number(floor.get("thickness"), "floor thickness", positive=True)
    ids = set()

    def unique_id(item):
        require_id(item.get("id"))
        if item["id"] in ids:
            raise ValueError(f"duplicate scene ID: {item['id']}")
        ids.add(item["id"])

    unique_id(floor)
    walls = scene.get("walls")
    furniture = scene.get("furniture")
    openings = scene.get("openings")
    if not isinstance(walls, list) or not walls:
        raise ValueError("walls must be a nonempty list")
    if not isinstance(furniture, list) or not isinstance(openings, list):
        raise ValueError("furniture and openings must be lists")
    wall_map = {}
    for wall in walls:
        unique_id(wall)
        vector(wall.get("start"), 2, "wall start")
        vector(wall.get("end"), 2, "wall end")
        if wall_length(wall) <= EPS:
            raise ValueError("wall must have positive length")
        number(wall.get("thickness"), "wall thickness", positive=True)
        wall_height = number(wall.get("height"), "wall height", positive=True)
        if wall_height > height + EPS:
            raise ValueError("wall exceeds room height")
        # Clearance direction is explicit so interior walls need no winding assumption.
        if wall.get("interior_side") not in ("left", "right"):
            raise ValueError("wall interior_side must be left or right")
        wall_map[wall["id"]] = wall
    for opening in openings:
        unique_id(opening)
        if opening.get("kind") not in ("door", "window"):
            raise ValueError("opening kind must be door or window")
        wall = wall_map.get(opening.get("wall_id"))
        if wall is None:
            raise ValueError("opening references an unknown wall")
        offset = number(opening.get("offset"), "opening offset")
        bottom = number(opening.get("bottom"), "opening bottom")
        span = number(opening.get("width"), "opening width", positive=True)
        rise = number(opening.get("height"), "opening height", positive=True)
        clearance = number(opening.get("interior_clearance_depth", 0), "opening clearance")
        if offset < 0 or offset + span > wall_length(wall) + EPS:
            raise ValueError("opening extends past wall endpoints")
        if bottom < 0 or bottom + rise > wall["height"] + EPS or clearance < 0:
            raise ValueError("invalid opening height or clearance")
        if opening["kind"] == "door" and bottom != 0:
            raise ValueError("door bottom must be at floor level")
    for i, first in enumerate(openings):
        for second in openings[i + 1 :]:
            if first["wall_id"] != second["wall_id"]:
                continue
            overlap_u = min(
                first["offset"] + first["width"], second["offset"] + second["width"]
            ) - max(first["offset"], second["offset"])
            overlap_z = min(
                first["bottom"] + first["height"], second["bottom"] + second["height"]
            ) - max(first["bottom"], second["bottom"])
            if overlap_u > EPS and overlap_z > EPS:
                raise ValueError("wall openings overlap")
    for item in furniture:
        unique_id(item)
        validate_furniture(item)
    return scene


def load_scene(path):
    return validate_scene(json.loads(Path(path).read_text(encoding="utf-8")))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def remove_furniture(scene, object_id):
    validate_scene(scene)
    if not any(item["id"] == object_id for item in scene["furniture"]):
        raise ValueError(f"unknown furniture ID: {object_id}")
    updated = copy.deepcopy(scene)
    updated["furniture"] = [item for item in updated["furniture"] if item["id"] != object_id]
    return updated


def propose_replacement(scene, object_id, replacement):
    from .collision import check_scene

    original = next((item for item in scene["furniture"] if item["id"] == object_id), None)
    if original is None:
        raise ValueError(f"unknown furniture ID: {object_id}")
    validate_furniture(replacement)
    if replacement["category"] != original["category"]:
        raise ValueError("replacement must preserve the selected furniture category")
    candidate = remove_furniture(scene, object_id)
    candidate["furniture"].append(copy.deepcopy(replacement))
    validate_scene(candidate)
    check = check_scene(candidate)
    report = {
        "removed_id": object_id,
        "replacement_id": replacement["id"],
        "original_size": original["size"],
        "replacement_size": replacement["size"],
        "accepted": check["fits"],
        "placement_check": check,
    }
    # Never mutate the original room, including when a proposal fails.
    return (candidate if check["fits"] else copy.deepcopy(scene)), report
