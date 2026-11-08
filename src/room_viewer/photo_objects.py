"""Explicit conceptual scene setup and stable observation-to-placeholder identity.

No function projects image coordinates or estimates physical sizes from a photo.
"""
import copy
from uuid import uuid4

from room_reconstruction.demo import demo_scene
from room_reconstruction.model import number, validate_scene, vector

CONCEPT_KINDS = {"user_supplied_layout", "approximate_design_workspace"}
DEFAULT_SIZES = {"sofa": [2.1, .85, .85], "chair": [.75, .85, .9], "table": [1.2, .6, .45], "desk": [1.2, .6, .75], "bed": [1.5, 2, .6], "cabinet": [.8, .4, 1.5], "lamp": [.35, .35, 1.6]}


def conceptual(scene):
    return scene.get("schema_version") == 1 and scene.get("provenance", {}).get("kind") in CONCEPT_KINDS


def setup_room(body):
    if body.get("layout") != "rectangular":
        raise ValueError("Choose the supported rectangular design layout explicitly")
    status = body.get("dimension_status")
    if status not in ("measured", "estimated", "unknown"):
        raise ValueError("Classify room dimensions as measured, estimated or unknown")
    source = str(body.get("source", "")).strip()
    if not source:
        raise ValueError("Describe the source of room dimensions or the approximate visualization")
    if status == "measured" and (body.get("confirmed") is not True or not str(body.get("evidence", "")).strip()):
        raise ValueError("Confirm supplied physical measurements and describe their evidence")
    if body.get("approximate_acknowledged") is not True:
        raise ValueError("Acknowledge that this layout is a design workspace, not a calibrated photographic reconstruction")
    dimensions = list(vector(body.get("dimensions"), 3, "layout dimensions", positive=True))
    if max(dimensions) > 100:
        raise ValueError("Prototype workspace dimensions must be at most 100")
    # Reuse the existing valid metric model, removing all unrelated demo content.
    scene = demo_scene()
    scene.update(room_id="design-" + uuid4().hex, dimensions=dimensions, furniture=[], openings=[])
    scene["floor"]["boundary"] = [[0, 0], [dimensions[0], 0], dimensions[:2], [0, dimensions[1]]]
    thickness = .15
    w, d, h = dimensions
    for wall, start, end in zip(scene["walls"], [[0, -.075], [w + .075, 0], [w, d + .075], [-.075, d]], [[w, -.075], [w + .075, d], [0, d + .075], [-.075, 0]]):
        wall.update(start=start, end=end, height=h, thickness=thickness)
    measurement = {"status": status, "value": dimensions if status != "unknown" else None, "units": "m", "source": source, "evidence": body.get("evidence"), "user_confirmed": body.get("confirmed") is True}
    scene["provenance"] = {"kind": "approximate_design_workspace" if status == "unknown" else "user_supplied_layout", "derived_from_photo": False, "note": "User-chosen rectangular workspace; no recovered room shape or camera calibration", "dimension_evidence": measurement, "visualization_extents": {"values": dimensions, "role": "display_only" if status == "unknown" else "user_supplied", "physical_measurements": status == "measured"}}
    scene["physical_accuracy_verified"] = False
    scene["fit_readiness"] = {"ready": False, "reasons": ["No validated photograph-to-room camera alignment", "Furniture poses and dimensions require independent verification", "Rectangular layout is user supplied, not reconstructed"]}
    return validate_scene(scene)


def create_placeholder(project, body):
    room = project.get("room")
    if not room or room.get("schema_version") != 1:
        raise ValueError("Set up an explicit design room first; photo geometry remains unresolved")
    if not conceptual(room) and room.get("provenance", {}).get("kind") != "manually_specified_demonstration":
        raise ValueError("This room is not an editable design workspace")
    observation = next((o for o in project["observations"] if o["observation_id"] == body.get("observation_id")), None)
    if observation is None:
        raise ValueError("Confirm this detection before creating a 3D placeholder")
    key = observation["observation_id"]
    association = project["associations"].get(key, {})
    if association.get("entity_id") and any(f["id"] == association["entity_id"] for f in room["furniture"]):
        raise ValueError("This observation already has an active linked object")
    if body.get("approximate_acknowledged") is not True:
        raise ValueError("Acknowledge that placeholder dimensions and position are not inferred measurements")
    status = body.get("dimension_status", "unknown")
    if status not in ("unknown", "estimated", "measured"):
        raise ValueError("Invalid furniture dimension classification")
    source = str(body.get("source", "")).strip()
    if not source:
        raise ValueError("Describe the source of placeholder dimensions")
    if status == "measured" and (body.get("confirmed") is not True or not str(body.get("evidence", "")).strip()):
        raise ValueError("Measured furniture dimensions need confirmation and evidence")
    size = list(vector(body.get("size"), 3, "placeholder size", positive=True))
    position = list(vector(body.get("position"), 3, "design position"))
    rotation = number(body.get("rotation_degrees", 0), "rotation")
    if max(size) > 100 or max(map(abs, position)) > 1000:
        raise ValueError("Placeholder exceeds prototype visualization limits")
    entity_id = "photo-object-" + uuid4().hex
    item = {"id": entity_id, "category": observation["label_hypothesis"], "size": size, "position": position, "rotation_degrees": rotation, "placeholder": True, "source_observation_ids": [key], "display_label": observation.get("display_label", observation["label_hypothesis"]), "design_evidence": {"dimensions": {"status": status, "value": size if status != "unknown" else None, "units": "m", "source": source, "evidence": body.get("evidence"), "confirmed": body.get("confirmed") is True}, "position": {"status": "unknown", "physical_position": None, "design_coordinates": position, "source": "User-chosen design position; no image projection"}, "orientation": {"status": "unknown", "physical_rotation": None}, "visualization_size": size, "physical_placement_verified": False}, "product_details": {"status": "unknown", "product_id": None, "identity_verified": False}}
    room["furniture"].append(item)
    validate_scene(room)
    project.setdefault("photo_objects", {})[entity_id] = {"entity": copy.deepcopy(item), "observation_ids": [key], "room_id": room["room_id"], "state": "active", "geometry_verified": False}
    project["associations"][key] = {"status": "editable_placeholder", "entity_id": entity_id, "room_id": room["room_id"], "geometry_applied": False, "reason": "Identity link only. Design coordinates and size do not establish photographic physical placement."}
    return item


def sync_objects(project):
    room = project.get("room") or {}
    items = {f["id"]: f for f in room.get("furniture", [])}
    for entity_id, record in project.get("photo_objects", {}).items():
        if record["room_id"] != room.get("room_id"):
            record["state"] = "other_room"
        elif entity_id in items:
            previous = record["entity"]
            item = items[entity_id]
            evidence = item.get("design_evidence", {})
            if item["size"] != previous["size"]:
                evidence["dimensions"] = {"status": "unknown", "value": None, "units": "m", "source": "Edited design size; previous measured evidence is retained in history"}
                evidence["visualization_size"] = copy.deepcopy(item["size"])
            if item["position"] != previous["position"]:
                evidence["position"] = {"status": "unknown", "physical_position": None, "design_coordinates": copy.deepcopy(item["position"]), "source": "User-edited design position"}
            item["design_evidence"] = evidence
            record.update(entity=copy.deepcopy(item), state="active")
        else:
            record["state"] = "removed"
        for observation_id in record["observation_ids"]:
            association = project["associations"].get(observation_id)
            if association and association.get("entity_id") == entity_id and association["status"] not in ("observation_not_confirmed", "needs_review"):
                association["status"] = "editable_placeholder" if record["state"] == "active" else "placeholder_removed" if record["state"] == "removed" else "placeholder_other_room"
