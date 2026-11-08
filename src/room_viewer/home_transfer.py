"""Additive destination-home planning. No inventory cloning or physical-fit certification."""

import copy
import hashlib
import io
import json
import math
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from room_reconstruction.clearance import polygon_gap
from room_reconstruction.collision import (
    footprint,
    opening_clearance,
    polygons_overlap,
    solid_wall_parts,
)
from room_reconstruction.model import number, validate_scene, vector
from room_reconstruction.polygon import area, contains, overlaps, polygon

from .inventory import active_dimensions, dimension, text
from .photo_objects import DEFAULT_SIZES
from .photo_projects import ID, image_copies, now
from .service import ROOT

STATUSES = (
    "keep_and_place",
    "keep_not_place",
    "consider_selling",
    "consider_replacing",
    "undecided",
)
SOURCES = (
    "user_calibrated_plan",
    "landlord_dimensioned_plan",
    "user_measured_room",
    "conceptual_drawing",
)
ASSET = re.compile(r"^[0-9a-f]{32}\.(png|jpg|pdf)$")


def uid():
    return uuid4().hex


def point(value):
    p = vector(value, 2, "point")
    if max(map(abs, p)) > 10000:
        raise ValueError("Coordinate exceeds planning limits")
    return p


def scale_check(plan):
    refs = plan.get("calibrations", [])
    if not refs:
        return {
            "status": "unscaled",
            "metres_per_pixel": None,
            "inconsistent": False,
            "physical_accuracy_verified": False,
        }
    latest = refs[-1]
    checks = latest.get("checks", [])
    errors = [abs(c["metres_per_pixel"] / latest["metres_per_pixel"] - 1) for c in checks]
    return {
        "status": latest["source"],
        "metres_per_pixel": latest["metres_per_pixel"],
        "inconsistent": any(e > 0.05 for e in errors),
        "relative_discrepancies": errors,
        "physical_accuracy_verified": False,
        "notice": "Drawing scale only; uniformity, distortion and building accuracy are not verified.",
    }


def reference(plan, body):
    p1, p2 = point(body["p1"]), point(body["p2"])
    if (
        any(not (0 <= p[0] <= plan["width"] and 0 <= p[1] <= plan["height"]) for p in (p1, p2))
        or math.dist(p1, p2) < 10
    ):
        raise ValueError("Choose reference endpoints inside the image at least 10 pixels apart")
    value = dimension(body["value"], body["unit"])
    source = body.get("source", "user_calibrated_plan")
    if source not in SOURCES or not text(body.get("description", "")):
        raise ValueError("Describe calibration provenance and uncertainty")
    return {
        "id": uid(),
        "p1": p1,
        "p2": p2,
        "original_value": body["value"],
        "unit": body["unit"],
        "metres": value,
        "metres_per_pixel": value / math.dist(p1, p2),
        "source": source,
        "description": text(body["description"]),
        "recorded_at": now(),
        "independently_verified": False,
        "checks": [],
    }


def validate_home(home):
    if home.get("schema_version") != 1:
        raise ValueError("Unsupported destination schema; no automatic migration")
    for plan in [home.get("floorplan"), *home.get("floorplan_history", [])]:
        if plan is None:
            continue
        for key in ("original", "preview"):
            if not isinstance(plan.get(key), str) or not ASSET.fullmatch(plan[key]):
                raise ValueError("Invalid floor-plan asset token")
            if not re.fullmatch(r"[0-9a-f]{64}", plan.get(key + "_sha256", "")):
                raise ValueError("Invalid floor-plan digest")
        for key in ("width", "height"):
            if not 16 <= number(plan.get(key), key, True) <= 1600:
                raise ValueError("Invalid normalized floor-plan dimensions")
        if (
            not isinstance(plan.get("page"), int)
            or not isinstance(plan.get("pages"), int)
            or not 0 <= plan["page"] < plan["pages"] <= 30
        ):
            raise ValueError("Invalid PDF page metadata")
        number(plan.get("rotation_degrees", 0), "plan rotation")
        for ref in plan.get("calibrations", []):
            for measurement in [ref, *ref.get("checks", [])]:
                rebuilt = reference(
                    plan,
                    {
                        "p1": measurement["p1"],
                        "p2": measurement["p2"],
                        "value": measurement["original_value"],
                        "unit": measurement["unit"],
                        "source": measurement["source"],
                        "description": measurement["description"],
                    },
                )
                if not math.isclose(
                    number(measurement["metres_per_pixel"], "scale", True),
                    rebuilt["metres_per_pixel"],
                    rel_tol=1e-10,
                ):
                    raise ValueError("Forged or inconsistent stored calibration conversion")
    if not isinstance(home.get("layouts"), dict) or not 1 <= len(home["layouts"]) <= 40:
        raise ValueError("Use 1–40 layout alternatives")
    if home["active_layout"] not in home["layouts"]:
        raise ValueError("Missing active alternative")
    if home.get("preferred_layout") is not None and home["preferred_layout"] not in home["layouts"]:
        raise ValueError("Missing preferred alternative")
    if not isinstance(home.get("rooms"), dict) or len(home["rooms"]) > 40:
        raise ValueError("Use at most 40 rooms")
    for rid, room in home["rooms"].items():
        if not ID.fullmatch(rid) or room["id"] != rid:
            raise ValueError("Invalid room ID")
        if not text(room.get("name", ""), 120):
            raise ValueError("Name each destination room")
        poly = polygon(room["boundary"])
        if min(v for p in poly for v in p) < 0 or max(v for p in poly for v in p) > 100:
            raise ValueError("Room coordinates must be within 0–100 planning units")
        if room["evidence_status"] not in (
            "unknown",
            "estimated",
            "user_supplied",
            "user_measured",
        ):
            raise ValueError("Invalid room evidence classification")
        if (
            room.get("height") is not None
            and not 0.1 <= number(room["height"], "height", True) <= 20
        ):
            raise ValueError("Invalid room height")
        if room.get("trace_pixels") is not None:
            polygon(room["trace_pixels"])
        for feature in room.get("features", []):
            index = feature["edge"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(poly)
                or feature["kind"] not in ("door", "window")
            ):
                raise ValueError("Invalid architectural feature edge/type")
            span = math.dist(poly[index], poly[(index + 1) % len(poly)])
            if (
                number(feature["offset"], "offset") < 0
                or number(feature["width"], "width", True) + feature["offset"] > span + 1e-8
            ):
                raise ValueError("Opening exceeds its wall segment")
            if feature.get("height") is not None and number(
                feature["height"], "opening height", True
            ) > (room.get("height") or 20):
                raise ValueError("Opening exceeds room height")
            if feature.get("bottom") is not None and number(feature["bottom"], "bottom") < 0:
                raise ValueError("Opening bottom must be nonnegative")
            if feature["kind"] == "door" and feature.get("swing", "left") not in (
                "left",
                "right",
                "unknown",
            ):
                raise ValueError("Unknown door swing")
        for obstacle in room.get("obstacles", []):
            if not contains(poly, polygon(obstacle["boundary"])):
                raise ValueError("Fixed obstacle must be inside its room")
    for lid, layout in home["layouts"].items():
        if not ID.fullmatch(lid) or layout["id"] != lid or not text(layout["name"], 120):
            raise ValueError("Invalid alternative")
        for item_id, status in layout["decisions"].items():
            if not re.fullmatch(r"inv-[0-9a-f]{32}", item_id) or status not in STATUSES:
                raise ValueError("Invalid inventory planning decision")
        ids = set()
        owned = set()
        for item in layout["placements"]:
            hypothetical = item.get("kind") == "proposed_replacement"
            placement_reference = "proposal:" + str(item.get("proposal_id")) if hypothetical else item.get("inventory_item_id")
            if hypothetical and item.get("inventory_item_id") is not None:
                raise ValueError("A hypothetical replacement is not owned inventory")
            if (
                not ID.fullmatch(item["id"])
                or item["id"] in ids
                or placement_reference in owned
                or item["room_id"] not in home["rooms"]
            ):
                raise ValueError("Duplicate or invalid layout placement")
            ids.add(item["id"])
            owned.add(placement_reference)
            if not hypothetical and not re.fullmatch(r"inv-[0-9a-f]{32}", item["inventory_item_id"]):
                raise ValueError("Invalid owned item reference")
            pos = vector(item["position"], 3, "placement")
            size = vector(item["size"], 3, "size", True)
            if max(map(abs, pos)) > 1000 or max(size) > 30:
                raise ValueError("Placement exceeds bounds")
            number(item["rotation_degrees"], "rotation")
        from .replacements import validate_layout
        validate_layout(layout)
    number(home.get("walking_clearance", 0.6), "walking clearance", True)
    if home["walking_clearance"] > 5:
        raise ValueError("Walking clearance must be at most 5m")

    for rid in home["rooms"]:
        room_scene(home, rid)

    # Reject forged verification flags anywhere in imported data.
    def verify(value):
        if isinstance(value, dict):
            for key, v in value.items():
                if (
                    key
                    in (
                        "physical_accuracy_verified",
                        "physical_placement_verified",
                        "independently_verified",
                    )
                    and v is not False
                ):
                    raise ValueError("Software cannot certify physical verification")
                verify(v)
        elif isinstance(value, list):
            for v in value:
                verify(v)

    verify(home)
    json.dumps(home, allow_nan=False)
    return home


def room_scene(home, room_id):
    room = home["rooms"][room_id]
    poly = room["boundary"]
    height = room["height"] or 2.6
    walls = []
    openings = []
    for i, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1])):
        walls.append(
            {
                "id": f"wall-{room_id}-{i}",
                "start": a,
                "end": b,
                "height": height,
                "thickness": 0.10,
                "interior_side": "left" if area(poly) > 0 else "right",
            }
        )
    for f in room["features"]:
        bottom = 0 if f["kind"] == "door" else f.get("bottom") if f.get("bottom") is not None else 1
        rise = f.get("height") or min(2 if f["kind"] == "door" else 1, height - bottom)
        openings.append(
            {
                "id": f["id"],
                "kind": f["kind"],
                "wall_id": walls[f["edge"]]["id"],
                "offset": f["offset"],
                "width": f["width"],
                "bottom": bottom,
                "height": rise,
                "interior_clearance_depth": home["walking_clearance"] if f["kind"] == "door" else 0,
                "swing": f.get("swing", "unknown"),
            }
        )
    furniture = [
        copy.deepcopy(p)
        for p in home["layouts"][home["active_layout"]]["placements"]
        if p["room_id"] == room_id
    ]
    scene = {
        "schema_version": 1,
        "units": "m",
        "coordinate_system": "right-handed-z-up",
        "room_id": room_id,
        "dimensions": [max(p[0] for p in poly), max(p[1] for p in poly), height],
        "floor": {"id": "floor-" + room_id, "boundary": poly, "z": 0, "thickness": 0.10},
        "walls": walls,
        "openings": openings,
        "furniture": furniture,
        "provenance": {
            "kind": "destination_plan",
            "height_status": "user_supplied" if room["height"] else "unknown_visualization_only",
            "source": "Explicit destination design; never photographic camera calibration",
        },
        "physical_accuracy_verified": False,
        "destination_project_id": home["destination_project_id"],
        "fixed_obstacles": room["obstacles"],
    }
    return validate_scene(scene)


def assess(home):
    layout = home["layouts"][home["active_layout"]]
    warnings = []
    gaps = []
    scale = scale_check(home.get("floorplan") or {})
    rooms = list(home["rooms"].values())
    for i, room in enumerate(rooms):
        for other in rooms[i + 1 :]:
            if overlaps(room["boundary"], other["boundary"]):
                warnings.append(
                    {"type": "overlapping_room_boundaries", "room_ids": [room["id"], other["id"]]}
                )
    for rid, room in home["rooms"].items():
        items = [p for p in layout["placements"] if p["room_id"] == rid]
        scene = room_scene(home, rid)
        walls = {w["id"]: w for w in scene["walls"]}
        for i, item in enumerate(items):
            fp = footprint(item)
            if not contains(room["boundary"], fp):
                warnings.append({"type": "outside_room_or_across_wall", "id": item["id"]})
            for obstacle in room["obstacles"]:
                if overlaps(fp, obstacle["boundary"]):
                    warnings.append(
                        {
                            "type": "fixed_obstacle_overlap",
                            "id": item["id"],
                            "obstacle_id": obstacle["id"],
                        }
                    )
            for wall in scene["walls"]:
                if any(
                    polygons_overlap(fp, footprint(part))
                    for part in solid_wall_parts(wall, scene["openings"])
                ):
                    warnings.append(
                        {"type": "wall_overlap", "id": item["id"], "wall_id": wall["id"]}
                    )
            for op in scene["openings"]:
                if op["kind"] == "door" and polygons_overlap(
                    fp, footprint(opening_clearance(walls[op["wall_id"]], op))
                ):
                    warnings.append(
                        {
                            "type": "doorway_or_swing_region_obstructed",
                            "id": item["id"],
                            "opening_id": op["id"],
                        }
                    )
            for other in items[i + 1 :]:
                gap = polygon_gap(fp, footprint(other))
                gaps.append(
                    {
                        "object_ids": [item["id"], other["id"]],
                        "design_gap": gap,
                        "units": "m" if scale["metres_per_pixel"] else "conceptual_units",
                    }
                )
                if polygons_overlap(fp, footprint(other)):
                    warnings.append(
                        {"type": "furniture_overlap", "object_ids": [item["id"], other["id"]]}
                    )
                elif gap < home["walking_clearance"]:
                    warnings.append(
                        {
                            "type": "restricted_clearance",
                            "object_ids": [item["id"], other["id"]],
                            "design_gap": gap,
                        }
                    )
    return {
        "physical_placement_verified": False,
        "status": "conceptual_warning" if warnings else "no_model_conflict_detected",
        "warnings": warnings,
        "gaps": gaps,
        "scale": scale,
        "notice": "Calculated design geometry only. No certified physical fit, accessibility, fire safety, building-code or transport claim. Door reserve is conservative, not a swept-leaf simulation.",
    }


class TransferStore:
    def __init__(self, projects, inventory):
        self.projects, self.inventory, self.lock = projects, inventory, projects.lock

    def read(self, pid):
        project = self.projects.read(pid)
        if not project.get("home_transfer"):
            raise ValueError("Not a destination-home project")
        validate_home(project["home_transfer"])
        return project

    def public(self, project):
        result = self.projects.public(project)
        home = result["home_transfer"]
        plan = home.get("floorplan")
        if plan:
            plan["preview_url"] = f"/api/transfer/{project['id']}/assets/{plan['preview']}"
        result["transfer_assessment"] = assess(home)
        return result

    def create(self, body):
        with self.lock:
            project = self.projects.read(self.projects.create(body.get("name", "New home"))["id"])
            lid = uid()
            project["home_transfer"] = {
                "schema_version": 1,
                "destination_project_id": project["id"],
                "floorplan": None,
                "floorplan_history": [],
                "rooms": {},
                "layouts": {
                    lid: {"id": lid, "name": "Layout A", "placements": [], "decisions": {}}
                },
                "active_layout": lid,
                "preferred_layout": None,
                "walking_clearance": 0.6,
                "history": [],
                "redo": [],
                "physical_accuracy_verified": False,
            }
            self.projects.write(project)
            return self.public(project)

    def action(self, pid, body):
        with self.lock:
            project = self.read(pid)
            home = project["home_transfer"]
            op = body["operation"]
            if op in ("undo", "redo"):
                source = home["history"] if op == "undo" else home["redo"]
                target = home["redo"] if op == "undo" else home["history"]
                if not source:
                    raise ValueError("No operation to " + op)
                current = {
                    k: copy.deepcopy(v) for k, v in home.items() if k not in ("history", "redo")
                }
                previous = source.pop()
                target.append(current)
                history, redo = home["history"], home["redo"]
                home = previous
                home.update(history=history, redo=redo)
                project["home_transfer"] = home
            else:
                snapshot = {
                    k: copy.deepcopy(v) for k, v in home.items() if k not in ("history", "redo")
                }
                layout = home["layouts"][home["active_layout"]]
                if op in ("calibrate", "check_calibration"):
                    plan = home.get("floorplan")
                    if not plan:
                        raise ValueError("Import a floor plan first")
                    ref = reference(plan, body)
                    if op == "check_calibration":
                        if not plan["calibrations"]:
                            raise ValueError("Establish initial scale first")
                        plan["calibrations"][-1]["checks"].append(ref)
                    else:
                        # Rescaling existing traces/placements silently would invalidate user edits.
                        plan["calibrations"].append(ref)
                        for room in home["rooms"].values():
                            room["calibration_needs_review"] = True
                elif op == "rotate_plan":
                    angle = number(body["degrees"], "rotation")
                    home["floorplan"]["rotation_degrees"] = angle % 360
                elif op == "room":
                    rid = body.get("room_id") or uid()
                    old = home["rooms"].get(rid, {})
                    boundary = polygon(body["boundary"])
                    trace = None
                    if body.get("coordinate_space") == "image_pixels":
                        plan = home.get("floorplan") or {}
                        scale = scale_check(plan)
                        if scale["metres_per_pixel"] is None:
                            raise ValueError("Calibrate the plan before tracing metric geometry")
                        if any(
                            not (0 <= p[0] <= plan["width"] and 0 <= p[1] <= plan["height"])
                            for p in boundary
                        ):
                            raise ValueError("Trace must stay inside plan image")
                        trace = copy.deepcopy(boundary)
                        boundary = [
                            [p[0] * scale["metres_per_pixel"], p[1] * scale["metres_per_pixel"]]
                            for p in boundary
                        ]
                    room = {
                        **copy.deepcopy(old),
                        "id": rid,
                        "name": text(body.get("name", "Room"), 120),
                        "boundary": boundary,
                        "trace_pixels": trace
                        if trace is not None
                        else old.get("trace_pixels")
                        if boundary == old.get("boundary")
                        else None,
                        "trace_reference": {
                            "preview": home["floorplan"]["preview"],
                            "page": home["floorplan"]["page"],
                            "sha256": home["floorplan"]["preview_sha256"],
                        }
                        if trace is not None
                        else old.get("trace_reference"),
                        "geometry_basis": "traced_scaled_drawing"
                        if trace is not None
                        else "explicit_user_design_coordinates",
                        "evidence_status": body.get("evidence_status", "estimated"),
                        "evidence_description": text(
                            body.get("evidence_description", "User traced design")
                        ),
                        "height": body.get("height"),
                        "features": copy.deepcopy(body.get("features", old.get("features", []))),
                        "obstacles": copy.deepcopy(body.get("obstacles", old.get("obstacles", []))),
                        "calibration_needs_review": False,
                        "calibration_id": (
                            (home.get("floorplan") or {}).get("calibrations") or [{}]
                        )[-1].get("id"),
                        "physical_accuracy_verified": False,
                    }
                    for feature in room["features"]:
                        feature.setdefault("id", uid())
                    for obstacle in room["obstacles"]:
                        obstacle.setdefault("id", uid())
                    if (
                        old
                        and old.get("trace_pixels")
                        and room.get("trace_pixels") != old.get("trace_pixels")
                    ):
                        room.setdefault("prior_trace_evidence", []).append(
                            {
                                "pixels": copy.deepcopy(old["trace_pixels"]),
                                "reference": copy.deepcopy(old.get("trace_reference")),
                                "calibration_id": old.get("calibration_id"),
                                "recorded_at": now(),
                            }
                        )
                    home["rooms"][rid] = room
                elif op == "wall_length":
                    room = home["rooms"][body["room_id"]]
                    edge = body["edge"]
                    if (
                        isinstance(edge, bool)
                        or not isinstance(edge, int)
                        or not 0 <= edge < len(room["boundary"])
                    ):
                        raise ValueError("Choose an existing wall edge")
                    length = dimension(body["value"], body["unit"])
                    description = text(body.get("description", ""))
                    if not description:
                        raise ValueError("Describe the wall measurement source")
                    a = room["boundary"][edge]
                    j = (edge + 1) % len(room["boundary"])
                    b = room["boundary"][j]
                    distance = math.dist(a, b)
                    room.setdefault("wall_measurements", []).append(
                        {
                            "edge": edge,
                            "value": body["value"],
                            "unit": body["unit"],
                            "metres": length,
                            "description": description,
                            "previous_end": copy.deepcopy(b),
                            "recorded_at": now(),
                            "independently_verified": False,
                        }
                    )
                    room["boundary"][j] = [a[i] + (b[i] - a[i]) * length / distance for i in (0, 1)]
                    room["calibration_needs_review"] = True
                    room["evidence_status"] = "user_supplied"
                elif op == "remove_room":
                    rid = body["room_id"]
                    del home["rooms"][rid]
                    for alt in home["layouts"].values():
                        alt["placements"] = [p for p in alt["placements"] if p["room_id"] != rid]
                elif op == "alternative":
                    lid = uid()
                    alt = (
                        copy.deepcopy(layout)
                        if body.get("duplicate")
                        else {"placements": [], "decisions": {}}
                    )
                    alt.update(id=lid, name=text(body.get("name", "Alternative"), 120))
                    # New scene IDs, same physical inventory references.
                    from .replacements import duplicate
                    duplicate(alt, lid, project["id"])
                    home["layouts"][lid] = alt
                    if body.get("duplicate"):
                        from .furniture_budget import duplicate_scenario
                        duplicate_scenario(project, home["active_layout"], lid)
                    home["active_layout"] = lid
                elif op == "switch":
                    home["active_layout"] = body["layout_id"]
                elif op == "rename_alternative":
                    layout["name"] = text(body["name"], 120)
                elif op == "prefer":
                    home["preferred_layout"] = home["active_layout"]
                elif op == "decision":
                    self.inventory.get(self.inventory.read(), body["inventory_item_id"])
                    layout["decisions"][body["inventory_item_id"]] = body["status"]
                elif op == "place":
                    if body.get("approximate_acknowledged") is not True:
                        raise ValueError("Acknowledge this is a hypothetical placement")
                    item = self.inventory.get(self.inventory.read(), body["inventory_item_id"])
                    if any(
                        p["inventory_item_id"] == item["inventory_item_id"]
                        for p in layout["placements"]
                    ):
                        raise ValueError("This owned item is already placed in this alternative")
                    if body["room_id"] not in home["rooms"]:
                        raise ValueError("Choose destination room")
                    try:
                        size, dims = active_dimensions(item)
                        complete = True
                    except ValueError:
                        size = copy.deepcopy(DEFAULT_SIZES[item["category"]])
                        dims = []
                        complete = False
                    placement = {
                        "id": uid(),
                        "inventory_item_id": item["inventory_item_id"],
                        "room_id": body["room_id"],
                        "name": item["display_name"],
                        "category": item["category"],
                        "position": vector(body["position"], 3, "explicit design position"),
                        "rotation_degrees": number(body.get("rotation_degrees", 0), "rotation"),
                        "size": size,
                        "dimension_status": "recorded_unverified"
                        if complete
                        else "unknown_approximate_placeholder",
                        "dimension_evidence": copy.deepcopy(dims),
                        "physical_placement_verified": False,
                        "role": "hypothetical_layout_not_physical_move",
                    }
                    layout["placements"].append(placement)
                    layout["decisions"][item["inventory_item_id"]] = "keep_and_place"
                elif op in ("update", "remove"):
                    p = next((p for p in layout["placements"] if p["id"] == body["id"]), None)
                    if p is None:
                        raise ValueError("Placement not found")
                    if op == "remove":
                        layout["placements"].remove(p)
                    else:
                        for field in ("position", "rotation_degrees", "room_id"):
                            if field in body:
                                p[field] = copy.deepcopy(body[field])
                        if "size" in body:
                            p["size"] = body["size"]
                            p["dimension_status"] = "layout_specific_estimate"
                elif op == "sync_scene":
                    scene = body["scene"]
                    rid = scene["room_id"]
                    validate_scene(scene)
                    if (
                        rid not in home["rooms"]
                        or scene["provenance"]["kind"] != "destination_plan"
                    ):
                        raise ValueError("Not the current destination geometry")
                    original = {p["id"]: p for p in layout["placements"] if p["room_id"] == rid}
                    new = []
                    for f in scene["furniture"]:
                        if f["id"] not in original:
                            raise ValueError("Add inventory through the library")
                        p = copy.deepcopy(original[f["id"]])
                        p["position"] = f["position"]
                        p["rotation_degrees"] = f["rotation_degrees"]
                        if p["size"] != f["size"]:
                            p.update(size=f["size"], dimension_status="layout_specific_estimate")
                        new.append(p)
                    layout["placements"] = [
                        p for p in layout["placements"] if p["room_id"] != rid
                    ] + new
                elif op == "clearance":
                    home["walking_clearance"] = body["value"]
                else:
                    raise ValueError("Unknown transfer operation")
                home["history"] = (home["history"] + [snapshot])[-40:]
                home["redo"] = []
            from .replacements import synchronize, validate_project
            for alt in home["layouts"].values():
                synchronize(alt)
            validate_home(home)
            validate_project(project)
            self.projects.write(project)
            return self.public(project)

    def upload(self, pid, raw, name, rights, page=0):
        if rights is not True:
            raise ValueError("Confirm permission for local floor-plan processing")
        if not raw or len(raw) > 15_000_000:
            raise ValueError("Plan must be smaller than 15 MB")
        if isinstance(page, bool) or not isinstance(page, int) or not 0 <= page < 30:
            raise ValueError("Page must be 0–29")
        count = 1
        if raw.startswith(b"%PDF-"):
            from PyPDF2 import PdfReader

            try:
                reader = PdfReader(io.BytesIO(raw), strict=False)
                if (
                    reader.is_encrypted
                    or not 1 <= len(reader.pages) <= 30
                    or page >= len(reader.pages)
                ):
                    raise ValueError("Locked PDF, excessive pages or invalid selected page")
                (ROOT / ".cache" / "floorplan-swift-modules").mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as folder:
                    src = Path(folder) / "input.pdf"
                    dest = Path(folder) / "preview.png"
                    src.write_bytes(raw)
                    result = subprocess.run(
                        [
                            "swift",
                            "-module-cache-path",
                            str(ROOT / ".cache" / "floorplan-swift-modules"),
                            str(ROOT / "scripts/helpers/render_floorplan.swift"),
                            str(src),
                            str(page),
                            str(dest),
                        ],
                        capture_output=True,
                        timeout=45,
                    )
                    if result.returncode or not dest.exists():
                        raise ValueError("Local PDFKit renderer could not render this PDF")
                    fmt, normalized, size, _ = image_copies(dest.read_bytes())
                    count = len(reader.pages)
            except Exception as exc:
                raise ValueError(
                    "PDF is locked, unsupported, corrupt or could not be rendered locally: "
                    + str(exc)[:200]
                ) from exc
            suffix = "pdf"
        else:
            fmt, normalized, size, _ = image_copies(raw)
            suffix = "jpg" if fmt == "JPEG" else "png"
            if page != 0:
                raise ValueError("An image has only one page")
        with self.lock:
            project = self.read(pid)
            home = project["home_transfer"]
            folder = self.projects.folder(pid) / "floorplans"
            folder.mkdir(exist_ok=True)
            token = uid()
            original = f"{token}.{suffix}"
            preview = f"{uid()}.png"
            (folder / original).write_bytes(raw)
            (folder / preview).write_bytes(normalized)
            if home["floorplan"]:
                home["floorplan_history"].append(copy.deepcopy(home["floorplan"]))
            home["floorplan"] = {
                "original": original,
                "preview": preview,
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "preview_sha256": hashlib.sha256(normalized).hexdigest(),
                "name": Path(str(name)).name[:120],
                "width": size[0],
                "height": size[1],
                "page": page,
                "pages": count,
                "rotation_degrees": 0,
                "calibrations": [],
                "rights": "User-attested permission for private local processing only",
                "created_at": now(),
            }
            for room in home["rooms"].values():
                room["calibration_needs_review"] = True
            home["history"] = []
            home["redo"] = []
            self.projects.write(project)
            return self.public(project)

    def asset(self, pid, token):
        project = self.read(pid)
        if not ASSET.fullmatch(token):
            raise ValueError("Invalid private plan token")
        records = [
            project["home_transfer"]["floorplan"],
            *project["home_transfer"]["floorplan_history"],
        ]
        for record in records:
            if record:
                for field in ("original", "preview"):
                    if record[field] == token:
                        path = self.projects.folder(pid) / "floorplans" / token
                        if (
                            hashlib.sha256(path.read_bytes()).hexdigest()
                            != record[field + "_sha256"]
                        ):
                            raise ValueError("Plan asset digest mismatch")
                        return path
        raise ValueError("Unknown private plan token")

    def export(self, pid):
        with self.lock:
            project = self.read(pid)
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(
                    "destination.json",
                    json.dumps({"bundle_version": 1, "project": project}, allow_nan=False),
                )
                records = [
                    project["home_transfer"]["floorplan"],
                    *project["home_transfer"]["floorplan_history"],
                ]
                seen = set()
                for plan in records:
                    if plan:
                        for key in ("original", "preview"):
                            token = plan[key]
                            if token not in seen:
                                z.write(self.asset(pid, token), "floorplans/" + token)
                                seen.add(token)
            return stream.getvalue()

    def import_bundle(self, raw):
        if len(raw) > 64_000_000:
            raise ValueError("Destination ZIP too large")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                infos = z.infolist()
                names = [i.filename for i in infos]
                if (
                    len(names) != len(set(names))
                    or len(names) > 100
                    or sum(i.file_size for i in infos) > 100_000_000
                ):
                    raise ValueError("Unsafe destination archive size/duplicates")
                for i in infos:
                    if i.file_size > 20_000_000 or (
                        i.filename != "destination.json"
                        and not re.fullmatch(r"floorplans/[0-9a-f]{32}\.(png|jpg|pdf)", i.filename)
                    ):
                        raise ValueError("Unsafe destination archive path")
                doc = json.loads(z.read("destination.json"))
                project = doc["project"]
                if (
                    doc["bundle_version"] != 1
                    or project["project_file_version"] != 1
                    or project.get("photos")
                    or project.get("room")
                ):
                    raise ValueError("Unsupported destination-only bundle")
                home = validate_home(project["home_transfer"])
                from .replacements import validate_project
                validate_project(project)
                if project.get("furniture_budget"):
                    from .furniture_budget import validate_budget
                    validate_budget(project["furniture_budget"], project["id"])
                home["history"] = []
                home["redo"] = []
                allowed = {"destination.json"}
                for record in [home["floorplan"], *home["floorplan_history"]]:
                    if not record:
                        continue
                    for key in ("original", "preview"):
                        token = record[key]
                        if not ASSET.fullmatch(token):
                            raise ValueError("Unsafe asset token")
                        data = z.read("floorplans/" + token)
                        allowed.add("floorplans/" + token)
                        if hashlib.sha256(data).hexdigest() != record[key + "_sha256"]:
                            raise ValueError("Plan hash mismatch")
                    if record["preview"].endswith(".png"):
                        image_copies(z.read("floorplans/" + record["preview"]))
                if set(names) != allowed:
                    raise ValueError("Unexpected private assets")
                project["imported_from_id"] = project["id"]
                project["id"] = uid()
                home["destination_project_id"] = project["id"]
                from .replacements import remap
                remap(project)
                if project.get("furniture_budget"):
                    from .furniture_budget import remap_budget
                    remap_budget(project)
                with self.lock:
                    for name in names:
                        if name != "destination.json":
                            path = self.projects.folder(project["id"]) / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(z.read(name))
                    self.projects.write(project)
                    return self.public(project)
        except (zipfile.BadZipFile, KeyError, OSError, TypeError) as exc:
            raise ValueError("Invalid destination bundle") from exc
