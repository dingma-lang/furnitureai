from __future__ import annotations

import math

from .model import EPS, validate_scene, wall_length


def footprint(body):
    """Counterclockwise oriented rectangle; position is floor-centre, not 3D centre."""
    x, y, _ = body["position"]
    width, depth, _ = body["size"]
    radians = math.radians(body["rotation_degrees"])
    cosine, sine = math.cos(radians), math.sin(radians)
    return [
        (x + dx * cosine - dy * sine, y + dx * sine + dy * cosine)
        for dx, dy in [
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2),
        ]
    ]


def polygons_overlap(first, second):
    """Separating-axis test. Edge/corner contact is allowed within EPS metres."""
    for polygon in (first, second):
        for i, a in enumerate(polygon):
            b = polygon[(i + 1) % len(polygon)]
            nx, ny = -(b[1] - a[1]), b[0] - a[0]
            norm = math.hypot(nx, ny)
            nx, ny = nx / norm, ny / norm
            projection_a = [p[0] * nx + p[1] * ny for p in first]
            projection_b = [p[0] * nx + p[1] * ny for p in second]
            if (
                min(max(projection_a), max(projection_b))
                - max(min(projection_a), min(projection_b))
                <= EPS
            ):
                return False
    return True


def bodies_overlap(first, second):
    bottom_a, bottom_b = first["position"][2], second["position"][2]
    top_a, top_b = bottom_a + first["size"][2], bottom_b + second["size"][2]
    if min(top_a, top_b) - max(bottom_a, bottom_b) <= EPS:
        return False
    return polygons_overlap(footprint(first), footprint(second))


def wall_body(wall, start, end, bottom, top, normal_offset=0, thickness=None):
    length = wall_length(wall)
    ux = (wall["end"][0] - wall["start"][0]) / length
    uy = (wall["end"][1] - wall["start"][1]) / length
    middle = (start + end) / 2
    return {
        "position": [
            wall["start"][0] + ux * middle - uy * normal_offset,
            wall["start"][1] + uy * middle + ux * normal_offset,
            bottom,
        ],
        "size": [end - start, thickness or wall["thickness"], top - bottom],
        "rotation_degrees": math.degrees(math.atan2(uy, ux)),
    }


def solid_wall_parts(wall, openings):
    """Subtract actual opening rectangles in wall-local (distance, height) space."""
    openings = [o for o in openings if o["wall_id"] == wall["id"]]
    breaks = sorted(
        {
            0.0,
            wall_length(wall),
            *(o["offset"] for o in openings),
            *(o["offset"] + o["width"] for o in openings),
        }
    )
    parts = []
    for start, end in zip(breaks, breaks[1:]):
        if end - start <= EPS:
            continue
        middle = (start + end) / 2
        holes = sorted(
            (o["bottom"], o["bottom"] + o["height"])
            for o in openings
            if o["offset"] <= middle <= o["offset"] + o["width"]
        )
        bottom = 0.0
        for hole_bottom, hole_top in holes:
            if hole_bottom > bottom + EPS:
                parts.append(wall_body(wall, start, end, bottom, hole_bottom))
            bottom = max(bottom, hole_top)
        if wall["height"] > bottom + EPS:
            parts.append(wall_body(wall, start, end, bottom, wall["height"]))
    return parts


def opening_clearance(wall, opening):
    depth = opening.get("interior_clearance_depth", 0)
    side = 1 if wall["interior_side"] == "left" else -1
    offset = side * (wall["thickness"] / 2 + depth / 2)
    return wall_body(
        wall,
        opening["offset"],
        opening["offset"] + opening["width"],
        opening["bottom"],
        opening["bottom"] + opening["height"],
        normal_offset=offset,
        thickness=depth,
    )


def check_scene(scene):
    validate_scene(scene)
    width, depth, height = scene["dimensions"]
    violations = []
    walls = {wall["id"]: wall for wall in scene["walls"]}
    wall_parts = {key: solid_wall_parts(wall, scene["openings"]) for key, wall in walls.items()}
    for item in scene["furniture"]:
        if scene.get("provenance", {}).get("kind") == "destination_plan":
            from .polygon import contains
            if not contains(scene["floor"]["boundary"], footprint(item)):
                violations.append({"type": "outside_polygon_floor", "object_id": item["id"]})
        if any(
            x < -EPS or y < -EPS or x > width + EPS or y > depth + EPS for x, y in footprint(item)
        ):
            violations.append({"type": "outside_floor", "object_id": item["id"]})
        if item["position"][2] < -EPS:
            violations.append({"type": "below_floor", "object_id": item["id"]})
        if item["position"][2] + item["size"][2] > height + EPS:
            violations.append({"type": "above_ceiling", "object_id": item["id"]})
        for wall_id, parts in wall_parts.items():
            if any(bodies_overlap(item, part) for part in parts):
                violations.append(
                    {"type": "wall_overlap", "object_id": item["id"], "wall_id": wall_id}
                )
        for opening in scene["openings"]:
            if opening.get("interior_clearance_depth", 0) > 0 and bodies_overlap(
                item, opening_clearance(walls[opening["wall_id"]], opening)
            ):
                violations.append(
                    {
                        "type": "opening_clearance",
                        "object_id": item["id"],
                        "opening_id": opening["id"],
                    }
                )
    for i, first in enumerate(scene["furniture"]):
        for second in scene["furniture"][i + 1 :]:
            if bodies_overlap(first, second):
                violations.append(
                    {"type": "furniture_overlap", "object_ids": [first["id"], second["id"]]}
                )
    return {
        "fits": not violations,
        "violations": violations,
        "method": "oriented rectangular prisms; SAT footprints + vertical overlap",
        "touching_allowed": True,
        "tolerance_metres": EPS,
        "scope": "static placement; no entry-path, accessibility, support or structural simulation",
    }
