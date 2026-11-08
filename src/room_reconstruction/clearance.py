"""Arrangement targets are separate from collision-free geometric fit."""

from __future__ import annotations

import math

from .collision import bodies_overlap, check_scene, footprint, polygons_overlap, solid_wall_parts
from .model import EPS, number, validate_scene


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)
    fraction = max(0.0, min(1.0, fraction))
    return math.hypot(point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy)


def polygon_gap(first, second):
    if polygons_overlap(first, second):
        return 0.0
    distances = []
    for a, b in ((first, second), (second, first)):
        for point in a:
            distances.extend(
                _point_segment_distance(point, b[i], b[(i + 1) % len(b)]) for i in range(len(b))
            )
    return min(distances)


def vertical_overlap(a, b):
    return (
        min(a["position"][2] + a["size"][2], b["position"][2] + b["size"][2])
        - max(a["position"][2], b["position"][2])
        > EPS
    )


def validate_policy(policy):
    allowed = {
        "minimum_furniture_gap",
        "minimum_wall_gap",
        "front_access_depths",
        "passages",
        "source",
    }
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError("unknown arrangement policy fields")
    for key in ("minimum_furniture_gap", "minimum_wall_gap"):
        value = number(policy.get(key, 0), key)
        if value < 0:
            raise ValueError("minimum clearances must be nonnegative")
    depths = policy.get("front_access_depths", {})
    if not isinstance(depths, dict):
        raise ValueError("front_access_depths must map categories to distances")
    for depth in depths.values():
        if number(depth, "front access depth") < 0:
            raise ValueError("front access depth must be nonnegative")
    if not isinstance(policy.get("passages", []), list):
        raise ValueError("passages must be a list")
    for passage in policy.get("passages", []):
        if (
            not isinstance(passage.get("between"), list)
            or len(passage["between"]) != 2
            or passage["between"][0] == passage["between"][1]
        ):
            raise ValueError("passage must reference two different furniture IDs")
        number(passage.get("minimum_width"), "passage minimum_width", positive=True)


def assess_arrangement(scene, policy):
    validate_scene(scene)
    validate_policy(policy)
    geometry = check_scene(scene)
    items = {item["id"]: item for item in scene["furniture"]}
    for passage in policy.get("passages", []):
        if any(key not in items for key in passage["between"]):
            raise ValueError("passage references unknown furniture")
    warnings, gaps, wall_gaps, access_zones = [], [], [], []
    floor_width, floor_depth, room_height = scene["dimensions"]
    wall_parts = {wall["id"]: solid_wall_parts(wall, scene["openings"]) for wall in scene["walls"]}
    for index, first in enumerate(scene["furniture"]):
        for second in scene["furniture"][index + 1 :]:
            if not vertical_overlap(first, second):
                continue
            gap = polygon_gap(footprint(first), footprint(second))
            row = {"object_ids": [first["id"], second["id"]], "gap_metres": gap}
            gaps.append(row)
            minimum = policy.get("minimum_furniture_gap", 0)
            if gap < minimum - EPS:
                warnings.append(
                    {"type": "restrictive_furniture_gap", **row, "target_metres": minimum}
                )
        for wall_id, parts in wall_parts.items():
            candidates = [
                polygon_gap(footprint(first), footprint(part))
                for part in parts
                if vertical_overlap(first, part)
            ]
            if not candidates:
                continue
            gap = min(candidates)
            row = {"object_id": first["id"], "wall_id": wall_id, "gap_metres": gap}
            wall_gaps.append(row)
            minimum = policy.get("minimum_wall_gap", 0)
            if gap < minimum - EPS:
                warnings.append({"type": "restrictive_wall_gap", **row, "target_metres": minimum})
        access_depth = policy.get("front_access_depths", {}).get(first["category"], 0)
        if not access_depth:
            continue
        angle = math.radians(first["rotation_degrees"])
        offset = (first["size"][1] + access_depth) / 2
        zone = {
            "position": [
                first["position"][0] + math.sin(angle) * offset,
                first["position"][1] - math.cos(angle) * offset,
                0,
            ],
            "size": [first["size"][0], access_depth, min(room_height, 1.9)],
            "rotation_degrees": first["rotation_degrees"],
        }
        obstacles = []
        if any(
            x < -EPS or y < -EPS or x > floor_width + EPS or y > floor_depth + EPS
            for x, y in footprint(zone)
        ):
            obstacles.append("outside_room")
        obstacles.extend(
            second["id"]
            for second in scene["furniture"]
            if second["id"] != first["id"] and bodies_overlap(zone, second)
        )
        obstacles.extend(
            f"wall:{wall_id}"
            for wall_id, parts in wall_parts.items()
            if any(bodies_overlap(zone, part) for part in parts)
        )
        row = {
            "object_id": first["id"],
            "requested_depth_metres": access_depth,
            "footprint": footprint(zone),
            "obstacles": obstacles,
        }
        access_zones.append(row)
        if obstacles:
            warnings.append({"type": "potential_front_access_obstruction", **row})
    passage_checks = []
    for passage in policy.get("passages", []):
        first, second = [items[key] for key in passage["between"]]
        gap = polygon_gap(footprint(first), footprint(second))
        row = {
            **passage,
            "local_gap_metres": gap,
            "meets_target": gap >= passage["minimum_width"] - EPS,
            "method": "minimum horizontal gap between nominated objects; not a full traversable route",
        }
        passage_checks.append(row)
        if not row["meets_target"]:
            warnings.append({"type": "restrictive_nominated_passage", **row})
    return {
        "schema_version": 1,
        "geometrically_valid": geometry["fits"],
        "geometry_check": geometry,
        "clearance_targets_met": not warnings,
        "comfort_verified": False,
        "arrangement_assessment": "restricted_by_configured_targets"
        if warnings
        else "configured_targets_met_only",
        "policy": policy,
        "warnings": warnings,
        "pair_gaps": gaps,
        "wall_gaps": wall_gaps,
        "front_access_zones": access_zones,
        "passage_checks": passage_checks,
        "limits": "Targets are user preferences, not building-code or accessibility certification. Front is local -y. No complete walking-path or dynamic door simulation.",
    }
