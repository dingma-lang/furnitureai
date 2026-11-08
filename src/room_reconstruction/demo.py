from __future__ import annotations

import copy
from pathlib import Path

from .collision import check_scene, footprint
from .export import export_obj, floorplan_svg
from .model import propose_replacement, remove_furniture, save_json, validate_scene


def demo_scene():
    scene = {
        "schema_version": 1,
        "room_id": "known-dimensions-demo",
        "units": "m",
        "coordinate_system": "right-handed-z-up",
        "provenance": {
            "kind": "manually_specified_demonstration",
            "derived_from_photo": False,
            "note": "All dimensions and furniture poses are specified examples, not measurements of the inference photograph.",
        },
        "dimensions": [6.0, 4.5, 2.8],
        "floor": {
            "id": "floor",
            "boundary": [[0, 0], [6, 0], [6, 4.5], [0, 4.5]],
            "z": 0,
            "thickness": 0.15,
        },
        "walls": [
            {
                "id": "south",
                "start": [0, -0.075],
                "end": [6, -0.075],
                "height": 2.8,
                "thickness": 0.15,
                "interior_side": "left",
            },
            {
                "id": "east",
                "start": [6.075, 0],
                "end": [6.075, 4.5],
                "height": 2.8,
                "thickness": 0.15,
                "interior_side": "left",
            },
            {
                "id": "north",
                "start": [6, 4.575],
                "end": [0, 4.575],
                "height": 2.8,
                "thickness": 0.15,
                "interior_side": "left",
            },
            {
                "id": "west",
                "start": [-0.075, 4.5],
                "end": [-0.075, 0],
                "height": 2.8,
                "thickness": 0.15,
                "interior_side": "left",
            },
        ],
        "openings": [
            {
                "id": "door-south",
                "kind": "door",
                "wall_id": "south",
                "offset": 0.5,
                "width": 0.9,
                "bottom": 0,
                "height": 2.1,
                "interior_clearance_depth": 0.9,
            },
            {
                "id": "window-north",
                "kind": "window",
                "wall_id": "north",
                "offset": 1.8,
                "width": 2.4,
                "bottom": 0.9,
                "height": 1.35,
                "interior_clearance_depth": 0,
            },
        ],
        "furniture": [
            {
                "id": "sofa-original",
                "category": "sofa",
                "position": [3, 3.72, 0],
                "rotation_degrees": 0,
                "size": [2.1, 0.85, 0.85],
            },
            {
                "id": "coffee-table",
                "category": "table",
                "position": [3, 2.45, 0],
                "rotation_degrees": 0,
                "size": [1.2, 0.6, 0.45],
            },
            {
                "id": "chair",
                "category": "chair",
                "position": [5, 2.2, 0],
                "rotation_degrees": 90,
                "size": [0.75, 0.85, 0.9],
            },
            {
                "id": "cabinet",
                "category": "cabinet",
                "position": [0.35, 2.7, 0],
                "rotation_degrees": 0,
                "size": [0.45, 1.4, 1.5],
            },
            {
                "id": "floor-lamp",
                "category": "lamp",
                "position": [4.65, 3.85, 0],
                "rotation_degrees": 0,
                "size": [0.35, 0.35, 1.6],
            },
        ],
    }
    return validate_scene(scene)


def replacement_sofa():
    return {
        "id": "sofa-replacement",
        "category": "sofa",
        "position": [3, 3.72, 0],
        "rotation_degrees": 0,
        "size": [2.8, 1.05, 0.9],
        "provenance": "Manually specified example dimensions; no catalog product claim",
    }


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = demo_scene()
    removed = remove_furniture(before, "sofa-original")
    replacement = replacement_sofa()
    after, accepted = propose_replacement(before, "sofa-original", replacement)
    too_wide = copy.deepcopy(replacement)
    too_wide.update(id="sofa-too-wide", size=[4.9, 1.05, 0.9])
    _, rejected_furniture = propose_replacement(before, "sofa-original", too_wide)
    wall_overlap = copy.deepcopy(replacement)
    wall_overlap.update(id="sofa-wall-overlap", position=[5.4, 3.72, 0])
    _, rejected_wall = propose_replacement(before, "sofa-original", wall_overlap)
    if not check_scene(before)["fits"] or not accepted["accepted"]:
        raise RuntimeError("demonstration's valid placement unexpectedly failed")
    if rejected_furniture["accepted"] or rejected_wall["accepted"]:
        raise RuntimeError("demonstration's invalid proposal unexpectedly passed")
    for name, scene, title in (
        ("before", before, "1. Original sofa"),
        ("sofa-removed", removed, "2. Sofa removed"),
        ("after", after, "3. Larger replacement sofa"),
    ):
        save_json(output / f"{name}.json", scene)
        check = check_scene(scene)
        save_json(output / f"{name}-check.json", check)
        floorplan_svg(scene, output / f"{name}.svg", title, check)
        export_obj(scene, output / f"{name}.obj")
    save_json(output / "replacement-sofa.json", replacement)
    save_json(output / "rejected-too-wide-sofa.json", too_wide)
    save_json(output / "rejected-wall-overlap-sofa.json", wall_overlap)

    def bounds(item):
        corners = footprint(item)
        return (
            min(x for x, _ in corners),
            min(y for _, y in corners),
            max(x for x, _ in corners),
            max(y for _, y in corners),
        )

    items = {item["id"]: item for item in after["furniture"]}
    sofa_bounds = bounds(items["sofa-replacement"])
    table_bounds = bounds(items["coffee-table"])
    lamp_bounds = bounds(items["floor-lamp"])
    report = {
        "kind": "known_dimension_geometric_demonstration",
        "derived_from_photo": False,
        "dimensions_metres": before["dimensions"],
        "accepted_replacement": accepted,
        "rejected_furniture_collision": rejected_furniture,
        "rejected_wall_collision": rejected_wall,
        "accepted_clearances_metres": {
            "sofa_to_north_interior_face": after["dimensions"][1] - sofa_bounds[3],
            "sofa_to_coffee_table": sofa_bounds[1] - table_bounds[3],
            "sofa_to_floor_lamp": lamp_bounds[0] - sofa_bounds[2],
        },
        "limitations": [
            "All metric geometry and furniture sizes are manually specified demo inputs.",
            "Furniture represented as oriented cuboid envelopes, not detailed meshes.",
            "Static placement only; does not prove transport through doorways or walking accessibility.",
            "No room reconstruction, depth, camera calibration, hidden geometry or dimensions inferred from the photograph.",
        ],
    }
    save_json(output / "replacement-report.json", report)
    return report
