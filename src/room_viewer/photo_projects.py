"""Private local projects, validated image assets and append-only review history."""
import copy
import io
import json
import math
import re
import threading
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from furniture_bench.corrections import save_correction
from furniture_bench.data import CATEGORIES, digest, read_json
from furniture_bench.geometry import validate_box
from furniture_bench.room_links import link_matches_room, make_observation
from room_layout.scene import photo_scene
from room_multiview.project import merge_document
from room_reconstruction.integration import associate_observation

from .photo_objects import create_placeholder, setup_room, sync_objects
from .service import ROOT, validate_document

MAX_BYTES = 15_000_000
MAX_PIXELS = 24_000_000
ASSETS = {"original.jpg", "original.png", "processing.png", "semantic.png", "depth.png", "depth-values.npz"}
ALIASES = {"coffee_table": "table", "bookshelf": "cabinet"}
LABELS = (*CATEGORIES, *ALIASES)
ID = re.compile(r"^[0-9a-f]{32}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def image_copies(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose a JPEG or PNG smaller than 15 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ("JPEG", "PNG"):
                    raise ValueError("Only JPEG and PNG images are supported")
                if source.width * source.height > MAX_PIXELS or min(source.size) < 16:
                    raise ValueError("Image must be at least 16 pixels per side and at most 24 megapixels")
                fmt = source.format
                source.verify()
            with Image.open(io.BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                oriented_size = image.size
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format="PNG")
                return fmt, output.getvalue(), image.size, oriented_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("This image is corrupt, unsupported or exceeds safe image limits") from exc


def empty_room(photo):
    return photo_scene(photo["id"], photo["sha256"], photo["width"], photo["height"], {"regions": [], "camera_perspective": {"status": "unknown"}, "source": "pending_analysis"}, {}, [])


def validate_project_room(room):
    validate_document(room)
    if room.get("schema_version") == 2:
        from room_multiview.geometry import calibration
        for view in room["views"]:
            camera = view.get("camera", {})
            if camera.get("intrinsics") is not None:
                calibration({"intrinsics": camera["intrinsics"], "distortion": camera.get("distortion", [0] * 5), "source": camera.get("calibration_source", ""), "status": camera.get("calibration_status", "unknown")})
    return room


def clean_public(value):
    if isinstance(value, dict):
        return {k: clean_public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_public(v) for v in value]
    if isinstance(value, str) and (value.startswith("/Users/") or value.startswith("/private/") or value.startswith("/tmp/")):
        return "<local reference retained in saved project>"
    return value


def restore_private_references(incoming, previous):
    """A sanitized browser round trip must not erase opaque saved provenance."""
    if incoming == "<local reference retained in saved project>" and previous is not None:
        return copy.deepcopy(previous)
    if isinstance(incoming, dict) and isinstance(previous, dict):
        return {key: restore_private_references(value, previous.get(key)) for key, value in incoming.items()}
    if isinstance(incoming, list) and isinstance(previous, list):
        result = []
        for index, value in enumerate(incoming):
            old = previous[index] if index < len(previous) else None
            if isinstance(value, dict):
                identity = next((key for key in ("id", "view_id", "observation_id", "detection_id") if key in value), None)
                if identity:
                    old = next((v for v in previous if isinstance(v, dict) and v.get(identity) == value[identity]), None)
            result.append(restore_private_references(value, old))
        return result
    return copy.deepcopy(incoming)


class ProjectStore:
    def __init__(self, root=None):
        self.root = Path(root or ROOT / ".local-projects")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def folder(self, project_id):
        if not isinstance(project_id, str) or not ID.fullmatch(project_id):
            raise ValueError("Invalid project ID")
        return self.root / project_id

    def read(self, project_id):
        path = self.folder(project_id) / "project.json"
        if not path.is_file():
            raise ValueError("Project not found")
        return read_json(path)

    def write(self, project):
        project["revision"] = project.get("revision", 0) + 1
        project["updated_at"] = now()
        atomic_json(self.folder(project["id"]) / "project.json", project)

    def create(self, name="Untitled room"):
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Project name must contain 1–120 characters")
        with self.lock:
            project = {"project_file_version": 1, "id": uuid4().hex, "name": name.strip(), "photos": [], "observations": [], "associations": {}, "measurements": [], "room": None, "created_at": now(), "consent": {"training_authorized": False}, "geometry": {"physical_accuracy_verified": False, "camera_alignment": "unresolved", "scale": "unknown"}}
            self.write(project)
            return self.public(project)

    def public(self, project):
        result = clean_public(copy.deepcopy(project))
        for photo in result["photos"]:
            photo["assets"] = {name: f"/api/projects/{project['id']}/assets/{photo['id']}/{name}" for name in ASSETS if (self.folder(project["id"]) / "photos" / photo["id"] / name).is_file()}
        return result

    def listing(self):
        with self.lock:
            return [{**{k: p.get(k) for k in ("id", "name", "updated_at", "revision")}, "home_transfer": bool(p.get("home_transfer"))} for path in sorted(self.root.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True) for p in [read_json(path)]]

    def upload(self, project_id, raw, filename, rights_confirmed, replace_id=None):
        if rights_confirmed is not True:
            raise ValueError("Confirm that you may process this photo locally")
        fmt, normalized, size, oriented_size = image_copies(raw)
        with self.lock:
            project = self.read(project_id)
            photo_id = uuid4().hex
            folder = self.folder(project_id) / "photos" / photo_id
            folder.mkdir(parents=True)
            original_name = "original.jpg" if fmt == "JPEG" else "original.png"
            (folder / original_name).write_bytes(raw)
            (folder / "processing.png").write_bytes(normalized)
            photo = {"id": photo_id, "name": Path(str(filename).replace("\\", "/")).name[:120], "sha256": digest(folder / "processing.png"), "original_sha256": digest(folder / original_name), "original_asset": original_name, "width": size[0], "height": size[1], "oriented_original_size": list(oriented_size), "processing_copy": "EXIF-corrected RGB PNG; longest side <=1600 pixels", "created_at": now(), "rights": {"source": "Local user upload", "license": "User-attested local processing only; no redistribution or training consent", "allowed_for_benchmark": True}, "stages": {}, "predictions": None, "reviewed": [], "corrections": [], "review_undo": [], "processing_state": "not_started", "deleted": False}
            if replace_id:
                self.photo(project, replace_id)["deleted"] = True
            project["photos"].append(photo)
            incoming = empty_room(photo)
            if project["room"] is None:
                project["room"] = incoming
            elif project["room"].get("schema_version") == 2:
                project["room"] = merge_document(project["room"], incoming)
            self.write(project)
            return self.public(project)

    @staticmethod
    def photo(project, photo_id):
        photo = next((p for p in project["photos"] if p["id"] == photo_id), None)
        if photo is None:
            raise ValueError("Photo not found")
        return photo

    def asset(self, project_id, photo_id, name):
        self.folder(project_id)
        if not ID.fullmatch(photo_id) or name not in ASSETS:
            raise ValueError("Unknown image asset")
        path = self.folder(project_id) / "photos" / photo_id / name
        if not path.is_file():
            raise ValueError("Image asset unavailable")
        return path

    def refresh_observations(self, project, photo):
        active = {d["detection_id"]: d for d in photo["reviewed"] if d.get("confirmed") and d["review_status"] != "rejected"}
        old = {o["observation_id"]: o for o in project["observations"]}
        other = [o for o in project["observations"] if o["image_id"] != photo["id"]]
        for detection in active.values():
            safe = {**detection, "score": detection["score"] if detection["score"] is not None else 0.0}
            record = {"image_id": photo["id"], "image_sha256": photo["sha256"], "width": photo["width"], "height": photo["height"], "detections": [safe]}
            observation_id = photo["id"] + ":" + detection["detection_id"]
            observation = make_observation(record, [detection["detection_id"]], observation_id)
            if detection["score"] is None:
                observation["image_regions"][0]["model_score"] = None
            observation["display_label"] = detection.get("display_label", detection["label"])
            other.append(observation)
            if observation_id not in project["associations"]:
                project["associations"][observation_id] = {"status": "unresolved", "entity_id": None, "reason": "No reviewed 3D correspondence or verified position; a 2D box does not define a 3D pose"}
            elif observation_id in old and old[observation_id] != observation:
                project["associations"][observation_id]["status"] = "needs_review"
            elif project["associations"][observation_id]["status"] == "observation_not_confirmed":
                project["associations"][observation_id]["status"] = "needs_review" if project["associations"][observation_id].get("entity_id") else "unresolved"
        project["observations"] = other
        for key in project["associations"]:
            if key.startswith(photo["id"] + ":") and key not in {o["observation_id"] for o in other}:
                project["associations"][key]["status"] = "observation_not_confirmed"
        if project["room"] and project["room"].get("schema_version") == 2:
            project["room"]["furniture_observations"] = copy.deepcopy(other)
        if photo.get("product_matches"):
            from .product_matching import observation_signature
            for detection in photo["reviewed"]:
                record = photo["product_matches"].get(detection["detection_id"])
                if not record:
                    continue
                current = detection.get("confirmed") and detection["review_status"] != "rejected" and record.get("observation_signature") == observation_signature(photo, detection)
                if record.get("state") == "complete" and not current:
                    record.update(state="stale", message="Observation changed; request current product candidates")
                    inventory = project.get("inventory", {}).get(photo["id"] + ":" + detection["detection_id"])
                    if inventory:
                        inventory["needs_review"] = True
                elif record.get("state") == "stale" and current:
                    record["state"] = "complete"

    def correct(self, project, body):
        photo = self.photo(project, body["photo_id"])
        if body["operation"] == "undo":
            if not photo["review_undo"]:
                raise ValueError("No photo correction to undo")
            photo["reviewed"] = photo["review_undo"].pop()
            photo["corrections"].append({"action": "undo", "recorded_at": now(), "used_for_training": False})
            self.refresh_observations(project, photo)
            return
        before = copy.deepcopy(photo["reviewed"])
        operation = body["operation"]
        if operation not in ("add", "update", "reject", "confirm"):
            raise ValueError("Unknown correction")
        if operation == "add":
            detection = {"detection_id": "user-" + uuid4().hex[:12], "score": None, "source": "user", "label": "chair", "box": [0, 0, 1, 1]}
        else:
            detection = copy.deepcopy(next((d for d in photo["reviewed"] if d["detection_id"] == body.get("detection_id")), None))
            if detection is None:
                raise ValueError("Select an existing detection")
        label = body.get("label", detection.get("display_label", detection["label"]))
        if label not in LABELS:
            raise ValueError("Unsupported furniture category")
        if "display_name" in body:
            name=body["display_name"]
            if not isinstance(name,str) or len(name)>160:
                raise ValueError("Furniture name must contain at most 160 characters")
            detection["display_name"]=name.strip()
        detection["label"] = ALIASES.get(label, label)
        detection["display_label"] = label
        box = list(validate_box(body.get("box", detection["box"])))
        if box[0] < 0 or box[1] < 0 or box[2] > photo["width"] or box[3] > photo["height"]:
            raise ValueError("Bounding box must stay inside the processing image")
        detection.update(box=box, review_status="confirmed" if operation == "confirm" else "rejected" if operation == "reject" else "modified", confirmed=operation == "confirm")
        correction = save_correction(self.folder(project["id"]) / "corrections.jsonl", {"image_id": photo["id"], "image_sha256": photo["sha256"], "reviewer": body.get("reviewer", "Local user"), "action": "delete" if operation == "reject" else "add" if operation == "add" else "update", "detection_id": detection["detection_id"], "label": detection["label"], "box": box, "review_status": detection["review_status"], "display_label": label, "display_name": detection.get("display_name")})
        photo["review_undo"].append(before)
        if operation == "add":
            photo["reviewed"].append(detection)
        else:
            photo["reviewed"] = [detection if d["detection_id"] == detection["detection_id"] else d for d in photo["reviewed"]]
        photo["corrections"].append(correction)
        self.refresh_observations(project, photo)

    def action(self, project_id, body):
        with self.lock:
            project = self.read(project_id)
            kind = body["type"]
            if kind in ("room", "setup_room", "create_placeholder", "restore_placeholder"):
                project.setdefault("editing_history", []).append({"recorded_at": now(), "operation": kind, "room": copy.deepcopy(project.get("room")), "photo_objects": copy.deepcopy(project.get("photo_objects", {})), "associations": copy.deepcopy(project["associations"])})
            if kind == "correct":
                self.correct(project, body)
            elif kind == "setup_room":
                room = setup_room(body)
                if project.get("room"):
                    project.setdefault("room_history", []).append({"room": copy.deepcopy(project["room"]), "recorded_at": now(), "reason": "Explicit design layout setup; previous observations preserved"})
                project["room"] = room
                project["geometry"].update(physical_accuracy_verified=False, camera_alignment="unresolved", scale="unknown" if body["dimension_status"] == "unknown" else "user_supplied_not_calibrated")
                project["measurements"].append({"id": uuid4().hex, "reference": "design room dimensions", **copy.deepcopy(room["provenance"]["dimension_evidence"]), "applied_to_geometry": body["dimension_status"] != "unknown", "physical_accuracy_verified": False, "recorded_at": now()})
                sync_objects(project)
            elif kind == "create_placeholder":
                item = create_placeholder(project, body)
                if body["observation_id"] in project.get("inventory", {}):
                    item["product_details"] = copy.deepcopy(project["inventory"][body["observation_id"]])
                    project["photo_objects"][item["id"]]["entity"]["product_details"] = copy.deepcopy(item["product_details"])
                if body.get("dimension_status") == "measured":
                    project["measurements"].append({"id": uuid4().hex, "reference": item["id"] + " dimensions", **copy.deepcopy(item["design_evidence"]["dimensions"]), "recorded_at": now(), "physical_accuracy_verified": False})
            elif kind == "restore_placeholder":
                record = project.get("photo_objects", {}).get(body.get("entity_id"))
                if record is None or record["state"] != "removed" or project["room"]["room_id"] != record["room_id"]:
                    raise ValueError("Choose a removed placeholder from the current room")
                project["room"]["furniture"].append(copy.deepcopy(record["entity"]))
                sync_objects(project)
            elif kind == "undo_design":
                if not project.get("editing_history"):
                    raise ValueError("No saved design operation to undo")
                snapshot = project["editing_history"].pop()
                project["room"] = snapshot["room"]
                project["photo_objects"] = snapshot["photo_objects"]
                for key, value in project["associations"].items():
                    if key not in snapshot["associations"] and value.get("status") in ("editable_placeholder", "placeholder_removed", "placeholder_other_room"):
                        project["associations"][key] = {"status": "unresolved", "entity_id": None, "reason": "Creation undone; physical position remains unknown"}
                for key, value in snapshot["associations"].items():
                    if key in {o["observation_id"] for o in project["observations"]}:
                        project["associations"][key] = value
                sync_objects(project)
            elif kind == "product_choice":
                from .product_matching import choose_product
                choose_product(project, body)
            elif kind == "rename":
                name = body.get("name", "").strip()
                if not name or len(name) > 120:
                    raise ValueError("Project name must contain 1–120 characters")
                project["name"] = name
            elif kind == "remove_photo":
                self.photo(project, body["photo_id"])["deleted"] = True
            elif kind == "measurement":
                value = body.get("value")
                status = body.get("status", "measured")
                if status not in ("measured", "estimated", "unknown"):
                    raise ValueError("Invalid measurement classification")
                if status != "unknown" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                    raise ValueError("Measurement must be a positive finite number")
                if not body.get("source", "").strip() or not body.get("reference", "").strip():
                    raise ValueError("Describe the measurement reference and source")
                if status == "measured" and (not body.get("confirmed") or not body.get("evidence", "").strip()):
                    raise ValueError("Confirm direct measurement and describe the evidence")
                project["measurements"].append({"id": uuid4().hex, "reference": body["reference"], "value": None if status == "unknown" else value, "units": "m", "status": status, "source": body["source"], "evidence": body.get("evidence"), "confirmed": body.get("confirmed", False), "recorded_at": now(), "applied_to_geometry": False})
            elif kind == "room":
                validate_project_room(body["room"])
                if project.get("room") and (project["room"].get("room_id") != body["room"].get("room_id") or project["room"].get("schema_version") != body["room"].get("schema_version")):
                    project.setdefault("room_history", []).append({"recorded_at": now(), "room": copy.deepcopy(project["room"]), "reason": "Explicitly attached a different room; previous photo geometry retained"})
                same_room = project.get("room") and project["room"].get("room_id") == body["room"].get("room_id") and project["room"].get("schema_version") == body["room"].get("schema_version")
                project["room"] = restore_private_references(body["room"], project["room"]) if same_room else copy.deepcopy(body["room"])
                for association in project["associations"].values():
                    if association.get("link") and not link_matches_room(association["link"], project["room"]):
                        association["status"] = "needs_review"
                sync_objects(project)
            elif kind == "associate":
                observation = next((o for o in project["observations"] if o["observation_id"] == body["observation_id"]), None)
                if observation is None:
                    raise ValueError("Confirm the detection before linking it")
                if project["room"].get("schema_version") != 1:
                    raise ValueError("No dimensioned room entity exists; observation remains unresolved")
                room = copy.deepcopy(project["room"])
                for entity in room["furniture"]:
                    entity["photo_observations"] = [r for r in entity.get("photo_observations", []) if r["observation_id"] != observation["observation_id"]]
                updated, link = associate_observation(room, observation, body["entity_id"], body.get("reviewer", "Local user"), body.get("evidence", ""), demo_only=body.get("demo_only") is True)
                project["room"] = updated
                for association in project["associations"].values():
                    if association.get("link") and not link_matches_room(association["link"], updated):
                        association["status"] = "needs_review"
                project["associations"][observation["observation_id"]] = {"status": "demonstration_only" if body.get("demo_only") else "reviewed_correspondence", "entity_id": body["entity_id"], "link": link, "geometry_applied": False}
            else:
                raise ValueError("Unknown project operation")
            self.write(project)
            return self.public(project)

    def export(self, project_id):
        with self.lock:
            project = self.read(project_id)
            if project.get("home_transfer"):
                raise ValueError("Use Plan a Move destination export to include private floor-plan assets")
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("project.json", json.dumps(project, allow_nan=False))
                for photo in project["photos"]:
                    for name in ASSETS:
                        path = self.folder(project_id) / "photos" / photo["id"] / name
                        if path.is_file():
                            archive.write(path, f"photos/{photo['id']}/{name}")
            return stream.getvalue()

    def import_bundle(self, raw):
        if len(raw) > 64_000_000:
            raise ValueError("Project archive must be smaller than 64 MB")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                infos = archive.infolist()
                if len(infos) > 1000 or sum(i.file_size for i in infos) > 120_000_000:
                    raise ValueError("Project archive exceeds extraction limits")
                names = [i.filename for i in infos]
                if len(set(names)) != len(names):
                    raise ValueError("Duplicate project archive paths")
                for info in infos:
                    parts = PurePosixPath(info.filename).parts
                    if info.file_size > 20_000_000 or (info.filename != "project.json" and not (len(parts) == 3 and parts[0] == "photos" and ID.fullmatch(parts[1]) and parts[2] in ASSETS)):
                        raise ValueError("Unexpected or unsafe project archive path")
                project = json.loads(archive.read("project.json"))
                if project.get("home_transfer"):
                    raise ValueError("Import destination bundles through Plan a Move")
                if project.get("project_file_version") != 1:
                    raise ValueError("Unsupported project version; archive left unchanged")
                validate_project_room(project["room"]) if project.get("room") else None
                if len({p["id"] for p in project["photos"]}) != len(project["photos"]):
                    raise ValueError("Duplicate photograph IDs")
                for photo in project["photos"]:
                    if not ID.fullmatch(photo["id"]) or photo["original_asset"] not in ("original.jpg", "original.png"):
                        raise ValueError("Invalid stored photo identity")
                    original = archive.read(f"photos/{photo['id']}/{photo['original_asset']}")
                    original_format, _, _, _ = image_copies(original)
                    if photo["original_asset"] != ("original.jpg" if original_format == "JPEG" else "original.png"):
                        raise ValueError("Original photo type differs from its stored asset")
                    processing = archive.read(f"photos/{photo['id']}/processing.png")
                    processing_format, _, size, _ = image_copies(processing)
                    if processing_format != "PNG" or list(size) != [photo["width"], photo["height"]]:
                        raise ValueError("Processing image dimensions or format mismatch")
                    detections = photo.get("reviewed", [])
                    if not isinstance(detections, list) or len({d["detection_id"] for d in detections}) != len(detections):
                        raise ValueError("Invalid or duplicate review detection IDs")
                    for detection in detections:
                        box = validate_box(detection["box"])
                        if detection["label"] not in CATEGORIES or detection.get("review_status") not in ("unreviewed", "confirmed", "modified", "rejected") or box[0] < 0 or box[1] < 0 or box[2] > photo["width"] or box[3] > photo["height"]:
                            raise ValueError("Invalid saved furniture observation")
                        score = detection.get("score")
                        if score is not None and (isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1):
                            raise ValueError("Invalid saved confidence score")
                    import hashlib
                    if hashlib.sha256(processing).hexdigest() != photo["sha256"] or hashlib.sha256(original).hexdigest() != photo["original_sha256"]:
                        raise ValueError("Photo hash mismatch in project archive")
                    for stage in photo.get("stages", {}).values():
                        if stage.get("state") in ("queued", "running"):
                            stage.update(state="interrupted", message="Previous processing was interrupted; request a retry")
                    photo["processing_state"] = "saved"
                project["imported_from_id"] = project["id"]
                project["id"] = uuid4().hex
                json.dumps(project, allow_nan=False)
                with self.lock:
                    folder = self.folder(project["id"])
                    for info in infos:
                        if info.filename != "project.json":
                            path = folder / info.filename
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(archive.read(info))
                    self.write(project)
                    return self.public(project)
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            raise ValueError("Invalid or incomplete project archive") from exc
