"""Local owned furniture identities and field-level evidence; no inferred physical scale."""

import copy
import hashlib
import io
import json
import math
import re
from urllib.parse import urlsplit
from uuid import uuid4

from furniture_bench.data import CATEGORIES, digest, read_json
from room_reconstruction.model import validate_scene

from .photo_objects import DEFAULT_SIZES, sync_objects
from .photo_projects import atomic_json, image_copies, now

ITEM_ID = re.compile(r"^inv-[0-9a-f]{32}$")
HEX = re.compile(r"^[0-9a-f]{32}$")
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
CONDITIONS = ("new", "like_new", "good", "fair", "poor", "unknown")
FIELDS = ("width", "depth", "height", "material", "color", "upholstery", "weight")
ORIGINS = (
    "user_measured",
    "user_entered",
    "estimated",
    "catalog_reported",
    "manufacturer_document",
)


def text(value, maximum=2000):
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise ValueError("Expected bounded text")
    return value.strip()


def identifier(value):
    if not isinstance(value, str) or not ITEM_ID.fullmatch(value):
        raise ValueError("Invalid inventory item ID")
    return value


def url(value):
    value = text(value, 2000)
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Use an HTTP(S) document/product URL without credentials")
    return value


def positive(value, maximum=30):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"Value must be finite, positive and at most {maximum}")
    return float(value)


def dimension(value, unit):
    if unit not in UNITS:
        raise ValueError("Supported length units: m, cm, mm, in, ft")
    metric = positive(positive(value, 100000) * UNITS[unit])
    if metric < 0.001:
        raise ValueError("Furniture/component dimensions must be at least 1 mm")
    return metric


def empty_item(name, category):
    if category not in CATEGORIES:
        raise ValueError("Unsupported inventory category")
    name = text(name, 160)
    if not name:
        raise ValueError("Item name is required")
    return {
        "schema_version": 1,
        "inventory_item_id": "inv-" + uuid4().hex,
        "scope": "local_owner",
        "owner_label": "Local owner",
        "created_at": now(),
        "updated_at": now(),
        "display_name": name,
        "category": category,
        "quantity": 1,
        "notes": "",
        "archived": False,
        "identification": {
            "status": "unknown",
            "catalog": None,
            "user_identification": None,
            "supporting_document_ids": [],
        },
        "specifications": {field: {"evidence": [], "active_evidence_id": None} for field in FIELDS},
        "condition": {
            "grade": "unknown",
            "stains": "",
            "scratches": "",
            "wear": "",
            "repairs": "",
            "purchase_year": None,
            "original_price": None,
            "source": "user_recorded_not_AI_damage_detection",
        },
        "documents": [],
        "owned_photos": [],
        "links": [],
        "history": [],
        "future": {"transfer": None, "valuation": None, "marketplace": None, "transport": None},
    }


def add_evidence(item, body):
    field = body.get("field")
    if field not in FIELDS or body.get("source") not in ORIGINS:
        raise ValueError("Choose a supported field and evidence source")
    source = body["source"]
    description = text(body.get("description", ""))
    if not description:
        raise ValueError("Describe the specification source/evidence")
    document_id = body.get("document_id")
    if source == "manufacturer_document" and not any(
        d["id"] == document_id for d in item["documents"]
    ):
        raise ValueError("Manufacturer-reported values require an attached documentation record")
    value = body.get("value")
    unit = body.get("unit")
    metric = None
    if field in ("width", "depth", "height"):
        metric = dimension(value, unit)
    elif field == "weight":
        if unit not in ("kg", "lb"):
            raise ValueError("Use kg or lb for weight")
        metric = positive(value, 20000) * (0.45359237 if unit == "lb" else 1)
    else:
        value = text(value, 500)
        if not value:
            raise ValueError("Empty specification; leave unknown instead")
    evidence = {
        "id": uuid4().hex,
        "value": value,
        "original_unit": unit,
        "metric_value": metric,
        "source": source,
        "description": description,
        "document_id": document_id,
        "status": "user_attested_measurement"
        if source == "user_measured"
        else "estimated"
        if source == "estimated"
        else "reported_not_independently_verified",
        "independently_verified": False,
        "recorded_at": now(),
        "scope": body.get("scope", "assembled"),
    }
    if evidence["scope"] not in ("assembled", "component"):
        raise ValueError("Specification scope must be assembled or component")
    item["specifications"][field]["evidence"].append(evidence)
    if body.get("make_active", True):
        item["specifications"][field]["active_evidence_id"] = evidence["id"]
    return evidence


def active_dimensions(item):
    values, evidence = [], []
    for field in ("width", "depth", "height"):
        record = item["specifications"][field]
        selected = next(
            (e for e in record["evidence"] if e["id"] == record["active_evidence_id"]), None
        )
        if not selected or selected["scope"] != "assembled":
            raise ValueError("Select active assembled width, depth and height evidence first")
        values.append(dimension(selected["value"], selected["original_unit"]))
        evidence.append(copy.deepcopy(selected))
    return values, evidence


def attach_catalog(item, product, candidate=None):
    previous = copy.deepcopy(item["identification"])
    item["identification"] = {
        "status": "user_confirmed_catalog_listing",
        "catalog": {
            "dataset": "ABO",
            "source_listing_id": product["product_id"],
            "marketplace": product.get("domain_name", "amazon.com"),
            "reported_manufacturer": product.get("manufacturer"),
            "reported_brand": product.get("brand"),
            "reported_model_identifier": product.get("model_number"),
            "variant": copy.deepcopy(product.get("specifications", {}).get("color")),
            "product_url": "https://www.amazon.com/dp/" + product["product_id"],
            "catalog_image_ids": [i["image_id"] for i in product.get("images", [])],
            "source_rights": copy.deepcopy(product.get("rights", {})),
            "exact_identity_verified": False,
            "confirmation": "User-selected possible listing association, not independent household identity proof",
            "retrieval_similarity_not_probability": candidate.get("similarity")
            if candidate
            else None,
        },
        "supporting_document_ids": [],
        "user_identification": None,
    }
    item["history"].append({"operation": "catalog_association", "previous": previous, "at": now()})
    record = product.get("source_record", {})
    specifications = product.get("specifications", {})
    # ABO length is retained as reported length, never silently assumed to be depth.
    for field in ("width", "height"):
        raw = (record.get("item_dimensions") or specifications.get("item_dimensions") or {}).get(
            field
        )
        if raw:
            normalized = raw.get("normalized_value", raw)
            units = {
                "inches": "in",
                "centimeters": "cm",
                "millimeters": "mm",
                "meters": "m",
                "feet": "ft",
            }
            unit = units.get(normalized.get("unit"))
            if unit and normalized.get("value") is not None:
                try:
                    add_evidence(
                        item,
                        {
                            "field": field,
                            "value": normalized["value"],
                            "unit": unit,
                            "source": "catalog_reported",
                            "description": "ABO item_dimensions."
                            + field
                            + "; source listing "
                            + product["product_id"],
                            "make_active": item["specifications"][field]["active_evidence_id"]
                            is None,
                        },
                    )
                except ValueError:
                    pass

    def reported(values):
        if isinstance(values, list):
            return next(
                (r.get("value") for r in values if r.get("language_tag") == "en_US"),
                values[0].get("value") if values else None,
            )
        return None

    for field in ("material", "color"):
        value = reported(record.get(field) or specifications.get(field))
        if value:
            add_evidence(
                item,
                {
                    "field": field,
                    "value": str(value),
                    "source": "catalog_reported",
                    "description": "ABO reported metadata " + product["product_id"],
                    "make_active": item["specifications"][field]["active_evidence_id"] is None,
                },
            )
    weight = record.get("item_weight")
    if isinstance(weight, list) and weight:
        value = weight[0].get("normalized_value", weight[0])
        unit = {"pounds": "lb", "kilograms": "kg"}.get(value.get("unit"))
        if unit and value.get("value"):
            add_evidence(
                item,
                {
                    "field": "weight",
                    "value": value["value"],
                    "unit": unit,
                    "source": "catalog_reported",
                    "description": "ABO item_weight; not transport capacity verification",
                    "make_active": item["specifications"]["weight"]["active_evidence_id"] is None,
                },
            )
    item["catalog_reported_raw_specifications"] = copy.deepcopy(specifications)
    item["catalog_dimension_mapping_note"] = (
        "Reported length is retained in raw metadata. Depth remains unknown until user explicitly establishes its meaning. Catalog measurements do not verify this household item's identity."
    )


def validate_item(item):
    if item.get("schema_version") != 1:
        raise ValueError("Unsupported inventory schema; original left unchanged")
    identifier(item.get("inventory_item_id"))
    empty_item(item["display_name"], item["category"])
    if (
        isinstance(item.get("quantity"), bool)
        or not isinstance(item.get("quantity"), int)
        or not 1 <= item["quantity"] <= 1000
    ):
        raise ValueError("Quantity must be an integer from 1 to 1000")
    if item.get("scope") != "local_owner" or not isinstance(item.get("archived"), bool):
        raise ValueError("Invalid owner scope/archive state")
    text(item.get("owner_label", ""), 160)
    text(item.get("notes", ""), 10000)
    for field in FIELDS:
        record = item["specifications"][field]
        ids = []
        for e in record["evidence"]:
            if not isinstance(e.get("id"), str) or not HEX.fullmatch(e["id"]):
                raise ValueError("Invalid evidence ID")
            ids.append(e["id"])
            if e.get("independently_verified") is not False:
                raise ValueError("This prototype cannot import independent verification claims")
            trial = empty_item("validation", item["category"])
            trial["documents"] = item["documents"]
            check = add_evidence(
                trial,
                {
                    "field": field,
                    "value": e["value"],
                    "unit": e.get("original_unit"),
                    "source": e["source"],
                    "description": e["description"],
                    "document_id": e.get("document_id"),
                    "scope": e.get("scope"),
                },
            )
            if check["metric_value"] != e.get("metric_value") or check["status"] != e.get("status"):
                raise ValueError("Evidence conversion/status mismatch")
        if len(ids) != len(set(ids)) or (
            record["active_evidence_id"] is not None and record["active_evidence_id"] not in ids
        ):
            raise ValueError("Invalid active/duplicate evidence ID")
    if item["condition"]["grade"] not in CONDITIONS:
        raise ValueError("Invalid condition")
    for field in ("stains", "scratches", "wear", "repairs"):
        text(item["condition"].get(field, ""), 3000)
    year = item["condition"].get("purchase_year")
    if year is not None and (
        isinstance(year, bool) or not isinstance(year, int) or not 1800 <= year <= 2100
    ):
        raise ValueError("Invalid purchase year")
    price = item["condition"].get("original_price")
    if price is not None:
        positive(price.get("amount"), 10000000)
        if not re.fullmatch(r"[A-Z]{3}", price.get("currency", "")):
            raise ValueError("Use a three-letter purchase-price currency")
    for document in item["documents"]:
        if not HEX.fullmatch(document["id"]):
            raise ValueError("Invalid documentation ID")
        url(document["url"])
        text(document["title"], 200)
        text(document["permission"], 1000)
    identity = item["identification"]
    if identity.get("status") not in (
        "unknown",
        "ai_suggested_candidate",
        "user_confirmed_catalog_listing",
        "user_supplied_unverified",
        "documentation_supported_unverified",
    ):
        raise ValueError("Unsupported identity status; no automatic exact identity verification")
    if identity.get("catalog"):
        catalog = identity["catalog"]
        if catalog.get("exact_identity_verified") is not False or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,120}", catalog.get("source_listing_id", "")
        ):
            raise ValueError("Invalid external listing ID or unsupported verification claim")
        url(catalog["product_url"])
    json.dumps(item, allow_nan=False)
    return item


class InventoryStore:
    def __init__(self, projects):
        self.projects, self.lock = projects, projects.lock
        self.path = projects.root / "owned-inventory.json"
        self.journal = projects.root / "owned-inventory-transaction.json"
        with self.lock:
            self.recover()

    def recover(self):
        if self.journal.exists():
            transaction = read_json(self.journal)
            if transaction.get("schema_version") != 1:
                raise ValueError("Unsupported inventory transaction journal")
            project = transaction["project"]
            for item in transaction["library"]["items"].values():
                validate_item(item)
            path = self.projects.folder(project["id"]) / "project.json"
            atomic_json(path, project)
            atomic_json(self.path, transaction["library"])
            self.journal.unlink()

    def read(self):
        self.recover()
        if not self.path.exists():
            return {"schema_version": 1, "items": {}}
        value = read_json(self.path)
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported saved library version; no automatic migration")
        return value

    def listing(self):
        with self.lock:
            return copy.deepcopy(self.read())

    def get(self, data, item_id):
        identifier(item_id)
        if item_id not in data["items"]:
            raise ValueError("Inventory item not found")
        return data["items"][item_id]

    def project(self, body):
        return self.projects.read(body["project_id"])

    def observation(self, project, observation_id):
        observation = next(
            (o for o in project["observations"] if o["observation_id"] == observation_id), None
        )
        if observation is None:
            raise ValueError("Choose a currently confirmed photo observation")
        return observation

    def link(self, item, project, observation_id=None, scene_id=None):
        if not observation_id and not scene_id:
            raise ValueError("Choose a confirmed observation or a scene object to link")
        if observation_id:
            self.observation(project, observation_id)
        if scene_id and not any(
            f["id"] == scene_id for f in (project.get("room") or {}).get("furniture", [])
        ):
            raise ValueError("Scene object not present in project")
        link = {
            "project_id": project["id"],
            "observation_id": observation_id,
            "scene_object_id": scene_id,
            "role": "design_reference_not_physical_location",
            "physical_location_verified": False,
        }
        if link not in item["links"]:
            item["links"].append(link)

    def action(self, body):
        with self.lock:
            data = self.read()
            operation = body.get("operation")
            if operation == "create":
                project = self.project(body) if body.get("project_id") else None
                observation_id = body.get("observation_id")
                if observation_id:
                    self.observation(project, observation_id)
                    existing = next(
                        (
                            i
                            for i in data["items"].values()
                            if any(
                                link_record["project_id"] == project["id"]
                                and link_record["observation_id"] == observation_id
                                for link_record in i["links"]
                            )
                        ),
                        None,
                    )
                    if existing:
                        return {
                            "item": existing,
                            "library": data,
                            "project": self.projects.public(project),
                        }
                item = empty_item(
                    body.get("display_name", "My furniture"), body.get("category", "sofa")
                )
                if project and observation_id:
                    scene_id = project["associations"].get(observation_id, {}).get("entity_id")
                    self.link(
                        item,
                        project,
                        observation_id,
                        scene_id
                        if any(
                            f["id"] == scene_id
                            for f in (project.get("room") or {}).get("furniture", [])
                        )
                        else None,
                    )
                data["items"][item["inventory_item_id"]] = item
            else:
                item = self.get(data, body.get("inventory_item_id"))
            previous = copy.deepcopy(item)
            if operation == "edit":
                for field in (
                    "display_name",
                    "notes",
                    "owner_label",
                    "quantity",
                    "category",
                    "archived",
                ):
                    if field in body:
                        item[field] = copy.deepcopy(body[field])
                if "condition" in body:
                    for field in (
                        "grade",
                        "stains",
                        "scratches",
                        "wear",
                        "repairs",
                        "purchase_year",
                        "original_price",
                    ):
                        if field in body["condition"]:
                            item["condition"][field] = copy.deepcopy(body["condition"][field])
            elif operation == "evidence":
                add_evidence(item, body)
            elif operation == "select_evidence":
                if body.get("field") not in FIELDS:
                    raise ValueError("Unknown evidence field")
                item["specifications"][body["field"]]["active_evidence_id"] = body.get(
                    "evidence_id"
                )
            elif operation == "catalog":
                project = self.project(body)
                observation_id = body["observation_id"]
                self.observation(project, observation_id)
                saved = project.get("inventory", {}).get(observation_id)
                if not saved or saved.get("needs_review"):
                    raise ValueError("Save a current possible product match first")
                pid = saved["candidate"]["product_id"]
                from .product_matching import CATALOG

                product = next(
                    (p for p in read_json(CATALOG)["products"] if p["product_id"] == pid), None
                )
                if not product:
                    raise ValueError("Product missing from current licensed catalog")
                attach_catalog(item, product, saved["candidate"])
                self.link(item, project, observation_id)
            elif operation == "clear_catalog":
                item["identification"] = {
                    "status": "unknown",
                    "catalog": None,
                    "supporting_document_ids": [],
                    "user_identification": None,
                }
            elif operation == "user_identity":
                item["identification"]["user_identification"] = text(body.get("identification", ""))
                item["identification"]["status"] = "user_supplied_unverified"
            elif operation == "document":
                document = {
                    "id": uuid4().hex,
                    "url": url(body.get("url", "")),
                    "title": text(body.get("title", ""), 200),
                    "permission": text(body.get("permission", "unknown"), 1000),
                    "kind": body.get("kind", "specification"),
                    "retrieval_status": "link_only_not_retrieved",
                    "date_accessed": None,
                    "recorded_at": now(),
                    "supports_identity": body.get("supports_identity") is True,
                    "identity_support_note": text(body.get("identity_support_note", "")),
                    "disassembly": "unknown; assembly instructions do not establish safe reverse disassembly",
                }
                if document["supports_identity"] and not document["identity_support_note"]:
                    raise ValueError("Describe documentary identity/variant support")
                item["documents"].append(document)
                if document["supports_identity"]:
                    item["identification"]["supporting_document_ids"].append(document["id"])
                    item["identification"]["status"] = "documentation_supported_unverified"
            elif operation == "link":
                project = self.project(body)
                observation_id = body.get("observation_id")
                for other in data["items"].values():
                    if other["inventory_item_id"] != item["inventory_item_id"] and any(
                        link_record["project_id"] == project["id"]
                        and (
                            (observation_id and link_record["observation_id"] == observation_id)
                            or (
                                body.get("scene_object_id")
                                and link_record["scene_object_id"] == body["scene_object_id"]
                            )
                        )
                        for link_record in other["links"]
                    ):
                        raise ValueError(
                            "Observation or scene object already belongs to another inventory item; unlink it explicitly first"
                        )
                self.link(item, project, observation_id, body.get("scene_object_id"))
            elif operation == "unlink":
                item["links"] = [
                    link_record
                    for link_record in item["links"]
                    if not (
                        link_record["project_id"] == body.get("project_id")
                        and (
                            not body.get("observation_id")
                            or link_record["observation_id"] == body["observation_id"]
                        )
                    )
                ]
            elif operation in ("place", "apply_dimensions"):
                project = self.project(body)
                room = project.get("room")
                if not room or room.get("schema_version") != 1:
                    raise ValueError("Set up an editable design room first")
                from .service import presentation

                if not presentation(room)["editable"]:
                    raise ValueError(
                        "Inventory cannot turn partial/unverified reconstruction into an editable complete room"
                    )
                if body.get("acknowledge_approximate_position") is not True:
                    raise ValueError(
                        "Acknowledge that dimensions do not verify room geometry or position"
                    )
                if item["archived"]:
                    raise ValueError("Restore the archived item first")
                try:
                    size, evidence = active_dimensions(item)
                except ValueError:
                    if operation != "place":
                        raise
                    size, evidence = copy.deepcopy(DEFAULT_SIZES[item["category"]]), []
                scene_id = body.get("scene_object_id")
                entity = next((f for f in room["furniture"] if f["id"] == scene_id), None)
                if operation == "apply_dimensions" and entity is None:
                    raise ValueError("Choose an existing scene object")
                if entity and any(
                    link_record["project_id"] == project["id"]
                    and link_record["scene_object_id"] == scene_id
                    for other in data["items"].values()
                    if other["inventory_item_id"] != item["inventory_item_id"]
                    for link_record in other["links"]
                ):
                    raise ValueError("Scene object already references another item")
                project.setdefault("editing_history", []).append(
                    {
                        "recorded_at": now(),
                        "operation": operation,
                        "room": copy.deepcopy(room),
                        "photo_objects": copy.deepcopy(project.get("photo_objects", {})),
                        "associations": copy.deepcopy(project["associations"]),
                    }
                )
                if entity is None:
                    existing = next(
                        (
                            link_record
                            for link_record in item["links"]
                            if link_record["project_id"] == project["id"]
                            and link_record["scene_object_id"]
                            and any(
                                f["id"] == link_record["scene_object_id"] for f in room["furniture"]
                            )
                        ),
                        None,
                    )
                    if existing:
                        raise ValueError(
                            "Item already has a scene reference in this project; apply to it instead"
                        )
                    scene_id = "inventory-object-" + uuid4().hex
                    entity = {
                        "id": scene_id,
                        "category": item["category"],
                        "size": size,
                        "position": [room["dimensions"][0] / 2, room["dimensions"][1] / 2, 0],
                        "rotation_degrees": 0,
                        "placeholder": True,
                        "display_label": item["display_name"],
                        "source_observation_ids": [],
                    }
                    room["furniture"].append(entity)
                original_size = copy.deepcopy(entity["size"])
                entity.update(size=size, inventory_item_id=item["inventory_item_id"])
                validate_scene(room)
                sync_objects(project)
                entity["design_evidence"] = {
                    "dimensions": {
                        "status": "unknown"
                        if not evidence
                        else "measured"
                        if all(e["source"] == "user_measured" for e in evidence)
                        else "estimated",
                        "value": size if evidence else None,
                        "units": "m",
                        "source": "Explicitly applied inventory evidence"
                        if evidence
                        else "Default category proportions for approximate visualization only",
                        "evidence": evidence,
                        "independently_verified": False,
                    },
                    "position": {
                        "status": "unknown",
                        "physical_position": None,
                        "design_coordinates": copy.deepcopy(entity["position"]),
                    },
                    "physical_placement_verified": False,
                    "visualization_size": size,
                }
                if scene_id in project.get("photo_objects", {}):
                    project["photo_objects"][scene_id]["entity"] = copy.deepcopy(entity)
                project.setdefault("inventory_dimension_history", []).append(
                    {
                        "inventory_item_id": item["inventory_item_id"],
                        "scene_object_id": scene_id,
                        "original_size": original_size,
                        "applied_size": size,
                        "evidence": evidence,
                        "at": now(),
                        "physical_position_verified": False,
                    }
                )
                project["revision"] = project.get("revision", 0) + 1
                project["updated_at"] = now()
                pending_project = project
                self.link(item, project, scene_id=scene_id)
            elif operation != "create":
                raise ValueError("Unknown inventory action")
            validate_item(item)
            if previous != item:
                # Keep histories bounded without recursively copying earlier histories.
                snapshot = {k: v for k, v in previous.items() if k != "history"}
                item["history"].append({"operation": operation, "at": now(), "before": snapshot})
            item["updated_at"] = now()
            if "pending_project" in locals():
                atomic_json(
                    self.journal, {"schema_version": 1, "project": pending_project, "library": data}
                )
                self.recover()
            else:
                atomic_json(self.path, data)
            return {
                "item": item,
                "library": data,
                "project": self.projects.public(project)
                if "project" in locals() and project
                else None,
            }

    def export(self):
        data = self.listing()
        # Metadata only by default. Local photo bytes, project IDs and asset tokens stay private.
        for item in data["items"].values():
            item["links"] = []
            item["owned_photos"] = []
            item["history"] = []
        data["export_note"] = (
            "Portable inventory metadata; private photographs and project placement links excluded. Same IDs do not imply simultaneous physical locations."
        )
        return data

    def import_data(self, body):
        if (
            body.get("schema_version") != 1
            or not isinstance(body.get("items"), dict)
            or len(body["items"]) > 1000
        ):
            raise ValueError("Unsupported inventory export")
        with self.lock:
            data = self.read()
            added, skipped = 0, 0
            for key, raw in body["items"].items():
                item = copy.deepcopy(raw)
                if key != item.get("inventory_item_id"):
                    raise ValueError("Item map/identity mismatch")
                if item.get("owned_photos") or item.get("links") or item.get("history"):
                    raise ValueError(
                        "Portable import must exclude local asset tokens, links and private history"
                    )
                validate_item(item)
                if key in data["items"]:
                    skipped += (
                        1  # Never overwrite an existing item or confirmed decisions on import.
                    )
                else:
                    data["items"][key] = item
                    added += 1
            atomic_json(self.path, data)
            return {"library": data, "added": added, "existing_ids_preserved": skipped}

    def upload_photo(self, item_id, raw, rights):
        if rights is not True:
            raise ValueError(
                "Confirm local processing rights; no training or redistribution consent implied"
            )
        fmt, _, dimensions, _ = image_copies(raw)
        with self.lock:
            data = self.read()
            item = self.get(data, item_id)
            asset_id = uuid4().hex
            folder = self.projects.root / "inventory-assets" / item_id
            folder.mkdir(parents=True, exist_ok=True)
            suffix = ".jpg" if fmt == "JPEG" else ".png"
            path = folder / (asset_id + suffix)
            path.write_bytes(raw)
            item["owned_photos"].append(
                {
                    "id": asset_id,
                    "suffix": suffix,
                    "sha256": digest(path),
                    "dimensions": dimensions,
                    "rights": "User attests local processing; private, no training/sharing consent",
                    "at": now(),
                }
            )
            atomic_json(self.path, data)
            return {"item": item, "library": data}

    def asset(self, item_id, asset_id):
        identifier(item_id)
        if not HEX.fullmatch(asset_id):
            raise ValueError("Invalid inventory asset ID")
        with self.lock:
            item = self.get(self.read(), item_id)
            record = next((p for p in item["owned_photos"] if p["id"] == asset_id), None)
            if not record:
                raise ValueError("Unknown private inventory asset")
            path = self.projects.root / "inventory-assets" / item_id / (asset_id + record["suffix"])
            if digest(path) != record["sha256"]:
                raise ValueError("Inventory asset hash changed")
            return path

    def extract_document(self, item_id, document_id, raw, rights):
        if rights is not True or len(raw) > 3_000_000 or not raw.startswith(b"%PDF-"):
            raise ValueError(
                "Supply a small PDF with affirmative permission for local text extraction"
            )
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise ValueError(
                "Optional PyPDF2 is unavailable; use manual documented specification entry"
            ) from exc
        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted or len(reader.pages) > 30:
                raise ValueError(
                    "Use an unencrypted PDF of at most 30 pages, or enter specifications manually"
                )
            content = "\n".join((page.extract_text() or "")[:8000] for page in reader.pages)[:60000]
        except Exception as exc:
            raise ValueError(
                "PDF text extraction failed; original inventory is unchanged; use manual entry"
            ) from exc
        with self.lock:
            data = self.read()
            item = self.get(data, item_id)
            document = next((d for d in item["documents"] if d["id"] == document_id), None)
            if document is None:
                raise ValueError("Attach a document record first")
            document.update(
                retrieval_status="user_supplied_pdf_text_extracted"
                if content.strip()
                else "no_extractable_text_manual_entry_required",
                date_accessed=now(),
                document_sha256=hashlib.sha256(raw).hexdigest(),
                text_excerpt=content[:12000],
                extraction_provenance="Actual user-supplied PDF text; no automatic identity/specification verification",
                specification_suggestions=[],
            )
            for field in ("width", "depth", "height"):
                match = re.search(
                    r"\b" + field + r"\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|inches|in|ft|m)\b",
                    content,
                    re.I,
                )
                if match:
                    unit = "in" if match.group(2).lower() == "inches" else match.group(2).lower()
                    try:
                        metric = dimension(float(match.group(1)), unit)
                        document["specification_suggestions"].append(
                            {
                                "field": field,
                                "value": float(match.group(1)),
                                "unit": unit,
                                "metric_value": metric,
                                "status": "extracted_requires_user_review",
                                "matched_text": match.group(0),
                            }
                        )
                    except ValueError:
                        pass
            atomic_json(self.path, data)
            return {"item": item, "library": data}
