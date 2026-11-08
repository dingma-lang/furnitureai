"""Explicit hypothetical furniture replacements; no ownership, identity or purchase inference."""

import copy
import hashlib
import json
from uuid import uuid4

from furniture_bench.data import read_json
from room_reconstruction.collision import footprint
from room_reconstruction.model import number, validate_furniture, vector

from .furniture_budget import empty_scenario, money, rate, validate_budget
from .inventory import (
    add_evidence,
    attach_catalog,
    empty_item,
    identifier,
    text,
    url,
    validate_item,
)
from .photo_objects import DEFAULT_SIZES
from .photo_projects import ID, now


def uid():
    return uuid4().hex


def proposals(layout):
    wrapper = layout.get("replacement_planning", {"schema_version": 1, "proposals": {}})
    if (
        wrapper.get("schema_version") != 1
        or not isinstance(wrapper.get("proposals"), dict)
        or len(wrapper["proposals"]) > 100
    ):
        raise ValueError("Unsupported replacement schema or too many proposals")
    return wrapper["proposals"]


def temporary_item(p):
    item = empty_item(p["name"], p["category"])
    item["specifications"] = copy.deepcopy(p["specifications"])
    return item


def size(p):
    values = []
    evidence = []
    for i, field in enumerate(("width", "depth", "height")):
        r = p["specifications"][field]
        e = next(
            (
                e
                for e in r["evidence"]
                if e["id"] == r["active_evidence_id"] and e["scope"] == "assembled"
            ),
            None,
        )
        values.append(e["metric_value"] if e else None)
        evidence.append(copy.deepcopy(e))
    return (
        [v if v is not None else DEFAULT_SIZES[p["category"]][i] for i, v in enumerate(values)],
        values,
        evidence,
    )


def candidate(p, pose=None):
    display, known, evidence = size(p)
    pose = pose or p["pose"]
    return {
        "id": p["scene_placement_id"],
        "inventory_item_id": None,
        "kind": "proposed_replacement",
        "prototype_variant": "replacement",
        "proposal_id": p["id"],
        "replaces_inventory_item_id": p["original_inventory_item_id"],
        "room_id": p["original_placement"]["room_id"],
        "name": p["name"],
        "category": p["category"],
        "position": vector(pose["position"], 3, "design position"),
        "rotation_degrees": number(pose["rotation_degrees"], "rotation"),
        "size": display,
        "dimension_status": "recorded_unverified"
        if all(v is not None for v in known)
        else "mixed_source_and_approximate",
        "known_dimensions_m": known,
        "dimension_evidence": evidence,
        "physical_placement_verified": False,
        "role": "hypothetical_purchase_not_owned",
    }


def validate_layout(layout, pid=None):
    records = proposals(layout)
    active = {p["id"]: p for p in layout["placements"]}
    originals = set()
    scene_ids = set()
    for key, p in records.items():
        if (
            key != p["id"]
            or not ID.fullmatch(key)
            or p["layout_id"] != layout["id"]
            or (pid and p["destination_project_id"] != pid)
        ):
            raise ValueError("Invalid replacement proposal identity")
        if not ID.fullmatch(p["scene_placement_id"]) or p["scene_placement_id"] in scene_ids:
            raise ValueError("Duplicate proposed scene ID")
        scene_ids.add(p["scene_placement_id"])
        if p["original_placement"]["id"] in originals:
            raise ValueError("Edit the existing proposal for this original placement")
        originals.add(p["original_placement"]["id"])
        if (
            p["original_placement"].get("inventory_item_id") != p["original_inventory_item_id"]
            or p["original_placement"].get("kind") == "proposed_replacement"
        ):
            raise ValueError("Replacement must reference an original owned placement")
        if p.get("catalog_candidate"):
            c = p["catalog_candidate"]
            listing = c.get("identification", {}).get("catalog") or {}
            if listing.get("product_url"):
                url(listing["product_url"])
            if c.get("identity_verified") is not False or not c.get("image_url", "").startswith(
                "/api/catalog/images/"
            ):
                raise ValueError("Invalid or externally hosted catalog candidate reference")
        identifier(p["original_inventory_item_id"])
        if not ID.fullmatch(p["original_placement"]["id"]):
            raise ValueError("Invalid recoverable original placement ID")
        for body in (p["original_placement"], candidate(p)):
            if max(map(abs, body["position"])) > 1000 or max(body["size"]) > 30:
                raise ValueError("Replacement geometry exceeds prototype bounds")
        validate_furniture(p["original_placement"])
        validate_item(temporary_item(p))
        validate_furniture(candidate(p))
        if p["status"] not in ("proposed", "applied", "reverted", "removed"):
            raise ValueError("Unknown replacement state")
        if (
            p.get("physical_placement_verified") is not False
            or p.get("ownership_verified") is not False
        ):
            raise ValueError("Hypothetical replacement cannot claim ownership or physical fit")
        if p.get("purchase_event_id") is not None and not ID.fullmatch(p["purchase_event_id"]):
            raise ValueError("Invalid purchase reference")
        if p["status"] == "applied":
            placed = active.get(p["scene_placement_id"])
            if (
                not placed
                or placed.get("proposal_id") != key
                or p["original_placement"]["id"] in active
            ):
                raise ValueError("Applied replacement relationship is inconsistent")
        elif p["scene_placement_id"] in active:
            raise ValueError("Unexpected inactive replacement placement")
    for placed in active.values():
        if placed.get("kind") == "proposed_replacement":
            p = records.get(placed.get("proposal_id"))
            if (
                not p
                or p["status"] != "applied"
                or p["scene_placement_id"] != placed["id"]
                or placed.get("inventory_item_id") is not None
                or placed.get("replaces_inventory_item_id") != p["original_inventory_item_id"]
            ):
                raise ValueError("Orphaned or forged hypothetical placement")


def validate_project(project):
    for lid, layout in project["home_transfer"]["layouts"].items():
        validate_layout(layout, project["id"])
        scenario = project.get("furniture_budget", {}).get("scenarios", {}).get(lid, {})
        for p in proposals(layout).values():
            if p.get("purchase_event_id"):
                event = scenario.get("purchases", {}).get(p["purchase_event_id"])
                if not event or event.get("replacement_of") != p["original_inventory_item_id"]:
                    raise ValueError(
                        "Linked purchase missing or belongs to another item; unlink replacement first"
                    )


def synchronize(layout):
    active = {p["id"] for p in layout["placements"]}
    for p in proposals(layout).values():
        if p["status"] == "applied" and p["scene_placement_id"] not in active:
            p["status"] = "removed"


def duplicate(layout, new_id, pid):
    records = proposals(layout)
    placement_ids = {p["id"]: uid() for p in layout["placements"]}
    for p in records.values():
        placement_ids.setdefault(p["original_placement"]["id"], uid())
        placement_ids.setdefault(p["scene_placement_id"], uid())
    proposal_ids = {key: uid() for key in records}
    for placed in layout["placements"]:
        placed["id"] = placement_ids[placed["id"]]
        if placed.get("proposal_id"):
            placed["proposal_id"] = proposal_ids[placed["proposal_id"]]
    if records:
        new = {}
        for old, p in records.items():
            p.update(
                id=proposal_ids[old],
                layout_id=new_id,
                destination_project_id=pid,
                scene_placement_id=placement_ids[p["scene_placement_id"]],
                last_applied_token=None,
            )
            p["original_placement"]["id"] = placement_ids[p["original_placement"]["id"]]
            new[p["id"]] = p
        layout["replacement_planning"]["proposals"] = new


def remap(project):
    for layout in project["home_transfer"]["layouts"].values():
        for p in proposals(layout).values():
            p["destination_project_id"] = project["id"]
    validate_project(project)


class ReplacementStore:
    def __init__(self, transfer):
        self.transfer = transfer
        self.projects = transfer.projects
        self.inventory = transfer.inventory
        self.lock = transfer.lock

    def read(self, pid):
        p = self.transfer.read(pid)
        validate_project(p)
        if p.get("furniture_budget"):
            validate_budget(p["furniture_budget"], pid)
        return p

    def candidates(self, iid):
        item = self.inventory.get(self.inventory.read(), iid)
        eligible = {}
        for link in item["links"]:
            if not link.get("observation_id"):
                continue
            try:
                source = self.projects.read(link["project_id"])
                photo_id, detection_id = link["observation_id"].split(":", 1)
                photo = self.projects.photo(source, photo_id)
                matches = photo.get("product_matches", {}).get(detection_id, {})
                from .product_matching import observation_signature

                detection = next(
                    (
                        d
                        for d in photo.get("reviewed", [])
                        if d["detection_id"] == detection_id
                        and d.get("confirmed")
                        and d.get("review_status") != "rejected"
                    ),
                    None,
                )
                if (
                    matches.get("state") != "complete"
                    or photo.get("deleted")
                    or not detection
                    or matches.get("observation_signature")
                    != observation_signature(photo, detection)
                ):
                    continue
                for c in matches.get("candidates", []):
                    decision = matches.get("decisions", {}).get(c["product_id"], {}).get("decision")
                    if decision != "reject":
                        eligible[c["product_id"]] = {
                            "source_project_id": source["id"],
                            "observation_id": link["observation_id"],
                            "similarity_not_probability": c.get("similarity"),
                            "decision": decision or "unconfirmed",
                        }
            except (ValueError, KeyError):
                continue
        current = item["identification"].get("catalog")
        if current:
            eligible.setdefault(
                current["source_listing_id"],
                {"source": "Owned item user-selected possible listing; not exact identity"},
            )
        from .product_matching import CATALOG

        data = read_json(CATALOG)
        out = []
        for product in data["products"]:
            if product["product_id"] not in eligible:
                continue
            temp = empty_item("Possible replacement", item["category"])
            attach_catalog(temp, product)
            out.append(
                {
                    "product_id": product["product_id"],
                    "category": product["label"],
                    "name": product.get("name")
                    or product.get("item_name")
                    or product["product_id"],
                    "image_url": "/api/catalog/images/" + product["images"][0]["image_id"],
                    "identification": temp["identification"],
                    "specifications": temp["specifications"],
                    "raw_source_specifications": product.get("specifications", {}),
                    "source_reference": eligible[product["product_id"]],
                    "identity_verified": False,
                }
            )
        return out

    def public(self, project, lid):
        from .furniture_budget import BudgetStore

        return {
            "project": self.transfer.public(project),
            "layout_id": lid,
            "proposals": copy.deepcopy(proposals(project["home_transfer"]["layouts"][lid])),
            "financial": BudgetStore(self.transfer).public(project),
        }

    def edit(self, p, body):
        p["name"] = text(body.get("name", p["name"]), 160)
        category = body.get("category", p["category"])
        if category not in DEFAULT_SIZES or not p["name"]:
            raise ValueError("Choose furniture name and supported category")
        p["category"] = category
        for field in ("retailer", "notes", "color", "material", "user_identity"):
            if field in body:
                p[field] = text(body[field], 3000)
        if "product_url" in body:
            p["product_url"] = url(body["product_url"]) if body["product_url"] else None
        if body.get("candidate_id"):
            selected = next(
                (
                    c
                    for c in self.candidates(p["original_inventory_item_id"])
                    if c["product_id"] == body["candidate_id"]
                ),
                None,
            )
            if selected is None:
                raise ValueError("Choose an existing eligible production retrieval candidate")
            if (p.get("catalog_candidate") or {}).get("product_id") != selected["product_id"]:
                # Append rather than overwriting manual/conflicting source evidence.
                for field, record in selected["specifications"].items():
                    p["specifications"][field]["evidence"].extend(copy.deepcopy(record["evidence"]))
                    if not p["specifications"][field]["active_evidence_id"]:
                        p["specifications"][field]["active_evidence_id"] = record[
                            "active_evidence_id"
                        ]
            p["catalog_candidate"] = selected
        if body.get("clear_candidate"):
            p["catalog_candidate"] = None
        temp = temporary_item(p)
        for field, entry in body.get("dimensions", {}).items():
            if field not in ("width", "depth", "height"):
                raise ValueError("Unknown dimension field")
            record = temp["specifications"][field]
            if entry.get("value") in ("", None):
                record["active_evidence_id"] = None
                continue
            old = next(
                (e for e in record["evidence"] if e["id"] == record["active_evidence_id"]), None
            )
            values = {**entry, "field": field, "scope": "assembled"}
            if (
                old
                and old["value"] == entry["value"]
                and old["original_unit"] == entry["unit"]
                and old["source"] == entry["source"]
                and old["description"] == entry["description"]
            ):
                continue
            add_evidence(temp, values)
        for field, eid in body.get("select_evidence", {}).items():
            record = temp["specifications"][field]
            if eid is not None and not any(e["id"] == eid for e in record["evidence"]):
                raise ValueError("Evidence reference not found")
            record["active_evidence_id"] = eid
        p["specifications"] = temp["specifications"]
        if "position" in body:
            p["pose"]["position"] = vector(body["position"], 3, "preview design position")
        if "rotation_degrees" in body:
            p["pose"]["rotation_degrees"] = number(body["rotation_degrees"], "rotation")
        p["modified_at"] = now()
        return p

    def preview(self, pid, body):
        from .home_transfer import assess, validate_home

        with self.lock:
            project = self.read(pid)
            home = project["home_transfer"]
            lid = body["layout_id"]
            layout = home["layouts"][lid]
            p = proposals(layout)[body["proposal_id"]]
            pose = {
                "position": body.get("position", p["pose"]["position"]),
                "rotation_degrees": body.get("rotation_degrees", p["pose"]["rotation_degrees"]),
            }
            placed = candidate(p, pose)
            pose = {"position": placed["position"], "rotation_degrees": placed["rotation_degrees"]}
            token = hashlib.sha256(
                json.dumps(
                    {
                        "proposal": p,
                        "placements": layout["placements"],
                        "room": home["rooms"][placed["room_id"]],
                        "pose": pose,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            draft = copy.deepcopy(home)
            draft["active_layout"] = lid
            alt = draft["layouts"][lid]
            alt["placements"] = [
                o
                for o in alt["placements"]
                if o["id"] not in (p["original_placement"]["id"], p["scene_placement_id"])
            ] + [placed]
            proposals(alt)[p["id"]]["status"] = "applied"
            validate_home(draft)
            return {
                "preview_token": token,
                "original": copy.deepcopy(p["original_placement"]),
                "candidate": placed,
                "original_footprint": footprint(p["original_placement"]),
                "candidate_footprint": footprint(placed),
                "room": copy.deepcopy(home["rooms"][placed["room_id"]]),
                "assessment": assess(draft),
                "proposal": copy.deepcopy(p),
                "saved_layout_mutated": False,
                "physical_placement_verified": False,
            }

    def action(self, pid, body):
        from .home_transfer import validate_home

        with self.lock:
            project = self.read(pid)
            home = project["home_transfer"]
            lid = body["layout_id"]
            layout = home["layouts"][lid]
            op = body["operation"]
            before = {k: copy.deepcopy(v) for k, v in home.items() if k not in ("history", "redo")}
            records = layout.setdefault(
                "replacement_planning", {"schema_version": 1, "proposals": {}}
            )["proposals"]
            if op == "create":
                key = body["proposal_id"]
                if not isinstance(key, str) or not ID.fullmatch(key):
                    raise ValueError("Use a stable proposal ID")
                if key in records:
                    return self.public(project, lid)
                original = next(
                    (
                        p
                        for p in layout["placements"]
                        if p["id"] == body["original_placement_id"] and p.get("inventory_item_id")
                    ),
                    None,
                )
                if not original:
                    raise ValueError("Select an owned furniture placement")
                if any(p["original_placement"]["id"] == original["id"] for p in records.values()):
                    raise ValueError("Edit the existing proposal for this placement")
                self.inventory.get(self.inventory.read(), original["inventory_item_id"])
                p = {
                    "id": key,
                    "layout_id": lid,
                    "destination_project_id": pid,
                    "original_inventory_item_id": original["inventory_item_id"],
                    "original_placement": copy.deepcopy(original),
                    "scene_placement_id": uid(),
                    "purchase_event_id": None,
                    "name": "Proposed " + original["category"],
                    "category": original["category"],
                    "specifications": empty_item("Proposal", original["category"])[
                        "specifications"
                    ],
                    "pose": {
                        "position": copy.deepcopy(original["position"]),
                        "rotation_degrees": original["rotation_degrees"],
                    },
                    "catalog_candidate": None,
                    "user_identity": "",
                    "status": "proposed",
                    "retailer": "",
                    "product_url": None,
                    "color": "",
                    "material": "",
                    "notes": "",
                    "created_at": now(),
                    "modified_at": now(),
                    "physical_placement_verified": False,
                    "ownership_verified": False,
                    "last_applied_token": None,
                }
                records[key] = self.edit(p, body)
            else:
                p = records.get(body.get("proposal_id"))
                if p is None:
                    raise ValueError("Replacement proposal not found")
                if op == "update":
                    self.edit(p, body)
                elif op == "apply":
                    if body.get("acknowledge_approximate") is not True:
                        raise ValueError(
                            "Acknowledge hypothetical placement and unresolved physical fit"
                        )
                    if (
                        p["status"] == "applied"
                        and bool(body.get("preview_token"))
                        and body.get("preview_token") == p["last_applied_token"]
                    ):
                        return self.public(project, lid)
                    preview = self.preview(pid, body)
                    if (
                        not body.get("preview_token")
                        or preview["preview_token"] != body["preview_token"]
                    ):
                        raise ValueError(
                            "Preview is stale; preview the current proposal and layout again"
                        )
                    original = next(
                        (
                            o
                            for o in layout["placements"]
                            if o["id"] == p["original_placement"]["id"]
                        ),
                        None,
                    )
                    if (
                        p["status"] in ("proposed", "reverted")
                        and original != p["original_placement"]
                    ):
                        raise ValueError(
                            "Original placement changed; create/review a new proposal before applying"
                        )
                    if any(
                        o.get("inventory_item_id") == p["original_inventory_item_id"]
                        and o["id"] != p["original_placement"]["id"]
                        for o in layout["placements"]
                    ):
                        raise ValueError("Original owned item already has another placement")
                    layout["placements"] = [
                        o
                        for o in layout["placements"]
                        if o["id"] not in (p["original_placement"]["id"], p["scene_placement_id"])
                    ] + [preview["candidate"]]
                    p.update(
                        status="applied",
                        last_applied_token=body["preview_token"],
                        pose={
                            "position": preview["candidate"]["position"],
                            "rotation_degrees": preview["candidate"]["rotation_degrees"],
                        },
                    )
                elif op == "revert":
                    existing = next(
                        (
                            o
                            for o in layout["placements"]
                            if o.get("inventory_item_id") == p["original_inventory_item_id"]
                        ),
                        None,
                    )
                    if existing is not None and existing != p["original_placement"]:
                        raise ValueError(
                            "Another original-item placement exists; do not overwrite it"
                        )
                    layout["placements"] = [
                        o
                        for o in layout["placements"]
                        if o["id"] not in (p["scene_placement_id"], p["original_placement"]["id"])
                    ] + [copy.deepcopy(p["original_placement"])]
                    p["status"] = "reverted"
                elif op == "remove":
                    if p["status"] == "applied":
                        raise ValueError(
                            "Revert the applied replacement before removing its proposal"
                        )
                    del records[p["id"]]
                elif op in ("link_purchase", "unlink_purchase"):
                    if op == "unlink_purchase":
                        p["purchase_event_id"] = None
                    else:
                        budget = project.setdefault(
                            "furniture_budget",
                            {"schema_version": 1, "currency": "USD", "scenarios": {}},
                        )
                        s = budget["scenarios"].setdefault(lid, empty_scenario(pid, lid))
                        event_id = body.get("purchase_event_id") or p.get("purchase_event_id")
                        if not event_id:
                            matching = [
                                e["id"]
                                for e in s["purchases"].values()
                                if e.get("replacement_of") == p["original_inventory_item_id"]
                            ]
                            if len(matching) > 1:
                                raise ValueError(
                                    "Select an existing purchase explicitly; multiple assumptions target this item"
                                )
                            event_id = matching[0] if matching else uid()
                        if body.get("purchase_event_id") and event_id not in s["purchases"]:
                            raise ValueError("Purchase event does not exist")
                        if (
                            event_id in s["purchases"]
                            and s["purchases"][event_id].get("replacement_of")
                            != p["original_inventory_item_id"]
                        ):
                            raise ValueError("Purchase belongs to another item")
                        if any(
                            other["id"] != p["id"] and other.get("purchase_event_id") == event_id
                            for other in records.values()
                        ):
                            raise ValueError("Purchase already linked to another proposal")
                        if event_id not in s["purchases"]:
                            snapshot = {
                                k: copy.deepcopy(v)
                                for k, v in s.items()
                                if k not in ("history", "redo")
                            }
                            s["history"] = (s["history"] + [snapshot])[-40:]
                            s["redo"] = []
                            source = "Explicit replacement proposal link; all prices, fees and tax treatment remain unknown"

                            def m():
                                return money(None, "unknown", source)

                            s["purchases"][event_id] = {
                                "id": event_id,
                                "proposed_replacement_id": event_id,
                                "replacement_of": p["original_inventory_item_id"],
                                "description": p["name"],
                                "retailer": p["retailer"],
                                "product_url": p["product_url"],
                                "catalog_candidate": {
                                    "product_id": p["catalog_candidate"]["product_id"],
                                    "exact_identity_verified": False,
                                }
                                if p["catalog_candidate"]
                                else None,
                                "quantity": 1,
                                "unit_price": m(),
                                "discount": m(),
                                "goods_taxable": None,
                                "tax": rate(None, "", source),
                                "fees": {
                                    f: {"amount": m(), "taxable": None}
                                    for f in ("delivery", "assembly", "other")
                                },
                                "notes": "Hypothetical proposal; not a completed purchase",
                                "source": source,
                                "modified_at": now(),
                            }
                            s["modified_at"] = now()
                        p["purchase_event_id"] = event_id
                else:
                    raise ValueError("Unknown replacement operation")
            home["history"] = (home["history"] + [before])[-40:]
            home["redo"] = []
            validate_home(home)
            validate_project(project)
            if project.get("furniture_budget"):
                validate_budget(project["furniture_budget"], pid)
            result = self.public(project, lid)
            self.projects.write(project)
            result["project"]["revision"] = project["revision"]
            result["project"]["updated_at"] = project["updated_at"]
            result["financial"]["project_revision"] = project["revision"]
            return result
