"""Local product retrieval jobs sharing the existing serial model queue."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from furniture_bench.data import digest, read_json

from .photo_projects import atomic_json, now
from .service import ROOT

CATALOG = ROOT / "data/catalogs/abo-phase6d/catalog.json"
INDEX = ROOT / "reports/phase6d-product-identification/benchmark/index"
MODEL = ROOT / ".cache/models/siglip2-base-75de2d55"
RETAILERS = ("IKEA", "Amazon", "Wayfair", "Walmart", "Target")


def observation_signature(photo, detection):
    return hashlib.sha256(json.dumps({"photo": photo["sha256"], "detection": detection["detection_id"], "box": detection["box"], "label": detection["label"]}, sort_keys=True).encode()).hexdigest()


class ProductMatching:
    def __init__(self, store, jobs):
        self.store, self.jobs = store, jobs
        self.pending = set()
        with store.lock:
            for item in store.listing():
                project = store.read(item["id"])
                changed = False
                for photo in project["photos"]:
                    for record in photo.get("product_matches", {}).values():
                        if record.get("state") in ("queued", "running"):
                            record.update(state="error", message="Previous product search interrupted; request another search")
                            changed = True
                if changed:
                    store.write(project)

    def submit(self, project_id, body):
        if not (INDEX / "metadata.json").is_file():
            raise ValueError("The licensed product catalog has not been indexed yet")
        with self.store.lock:
            project = self.store.read(project_id)
            photo = self.store.photo(project, body["photo_id"])
            detection = next((d for d in photo["reviewed"] if d["detection_id"] == body["detection_id"]), None)
            if not detection or not detection.get("confirmed") or detection["review_status"] == "rejected":
                raise ValueError("Confirm a furniture observation before requesting product candidates")
            preferences = body.get("preferred_retailers", [])
            if not isinstance(preferences, list) or any(p not in RETAILERS for p in preferences):
                raise ValueError("Unsupported retailer preference")
            if body.get("retailer_only") is True and not preferences:
                raise ValueError("Choose a retailer before enabling retailer-only filtering")
            settings = {"preferred_retailers": sorted(set(preferences)), "retailer_only": body.get("retailer_only") is True, "category": detection["label"], "catalog_fingerprint": digest(INDEX / "metadata.json"), "model_checkpoint": MODEL.name}
            signature = observation_signature(photo, detection)
            key = hashlib.sha256(json.dumps({"signature": signature, "settings": settings}, sort_keys=True).encode()).hexdigest()
            token = (project_id, photo["id"], detection["detection_id"])
            if token in self.pending:
                return self.store.public(project)
            current = photo.setdefault("product_matches", {}).get(detection["detection_id"])
            if current and current.get("cache_key") == key and current.get("state") == "complete":
                return self.store.public(project)
            photo["product_matches"][detection["detection_id"]] = {"state": "queued", "cache_key": key, "observation_signature": signature, "settings": settings, "decisions": copy.deepcopy(current.get("decisions", {})) if current else {}, "exact_identification": "No verified exact match."}
            self.store.write(project)
            self.pending.add(token)
            self.jobs.pool.submit(self.execute, project_id, photo["id"], copy.deepcopy(detection), key, signature, settings)
            return self.store.public(project)

    def execute(self, project_id, photo_id, detection, key, signature, settings):
        cache = self.store.root / "product-cache" / key
        cache.mkdir(parents=True, exist_ok=True)
        try:
            with self.store.lock:
                project = self.store.read(project_id)
                photo = self.store.photo(project, photo_id)
                photo["product_matches"][detection["detection_id"]]["state"] = "running"
                self.store.write(project)
            if (cache / "result.json").exists():
                result = read_json(cache / "result.json")
                result["cached"] = True
            else:
                with Image.open(self.store.asset(project_id, photo_id, "processing.png")) as source:
                    source.convert("RGB").crop(tuple(detection["box"])).save(cache / "crop.png")
                embedding_cache = self.store.root / "product-embeddings" / signature
                atomic_json(cache / "request.json", {"index": str(INDEX), "model": str(MODEL), "image": str(cache / "crop.png"), "embedding_cache": str(embedding_cache), **settings})
                env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
                attempt = sum(1 for _ in cache.glob("worker-*.log"))
                with (cache / f"worker-{attempt:03d}.log").open("w") as log:
                    completed = subprocess.run([str(Path(sys.executable).absolute()), "-X", "faulthandler", "-m", "room_viewer.product_worker", "--request", str(cache / "request.json"), "--output", str(cache / "result.json")], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
                if completed.returncode != 0:
                    raise RuntimeError(f"Product worker exit {completed.returncode}; inspect saved local log")
                result = read_json(cache / "result.json")
                result["cached"] = False
            with self.store.lock:
                project = self.store.read(project_id)
                photo = self.store.photo(project, photo_id)
                record = photo["product_matches"][detection["detection_id"]]
                record.update(result, state="complete", observation_signature=signature, cache_key=key, completed_at=now())
                current = next((d for d in photo["reviewed"] if d["detection_id"] == detection["detection_id"]), None)
                if not current or not current.get("confirmed") or observation_signature(photo, current) != signature:
                    record.update(state="stale", message="Observation changed during processing; request current candidates")
                self.store.write(project)
        except Exception as exc:
            with self.store.lock:
                project = self.store.read(project_id)
                photo = self.store.photo(project, photo_id)
                photo["product_matches"][detection["detection_id"]].update(state="error", message="Product matching could not finish. Recognition and your design remain saved.", error_type=type(exc).__name__, technical_error=str(exc))
                self.store.write(project)
        finally:
            with self.store.lock:
                self.pending.discard((project_id, photo_id, detection["detection_id"]))


def choose_product(project, body):
    photo = next((p for p in project["photos"] if p["id"] == body["photo_id"]), None)
    detection = next((d for d in photo["reviewed"] if d["detection_id"] == body["detection_id"]), None) if photo else None
    if not detection or not detection.get("confirmed") or detection["review_status"] == "rejected":
        raise ValueError("Confirm a current observation before saving a product candidate")
    record = photo.get("product_matches", {}).get(detection["detection_id"], {})
    if record.get("state") != "complete" or record.get("observation_signature") != observation_signature(photo, detection):
        raise ValueError("Candidates are unavailable or stale; request current product matches")
    candidate = next((c for c in record["candidates"] if c["product_id"] == body["product_id"]), None)
    if candidate is None:
        raise ValueError("Choose a returned catalog candidate")
    decision = body.get("decision")
    if decision not in ("confirm", "reject"):
        raise ValueError("Product decision must be confirm or reject")
    value = {"decision": decision, "recorded_at": now(), "reviewer": body.get("reviewer", "Local user"), "identity_verified": False, "claim": "user_selected_possible_product_match", "evidence": "User preference/selection only; no independent exact-product evidence supplied"}
    record.setdefault("decisions", {})[candidate["product_id"]] = value
    observation_id = photo["id"] + ":" + detection["detection_id"]
    project.setdefault("inventory", {})
    if decision == "confirm":
        details = {"candidate": copy.deepcopy(candidate), "review": value, "observation_id": observation_id, "identity_verified": False, "geometry_applied": False}
        previous = project["inventory"].get(observation_id)
        if previous:
            project.setdefault("product_history", []).append(copy.deepcopy(previous))
        project["inventory"][observation_id] = details
        association = project["associations"].get(observation_id, {})
        entity_id = association.get("entity_id")
        if entity_id in project.get("photo_objects", {}):
            project["photo_objects"][entity_id]["entity"]["product_details"] = copy.deepcopy(details)
        for entity in (project.get("room") or {}).get("furniture", []):
            if entity["id"] == entity_id:
                entity["product_details"] = copy.deepcopy(details)
                if entity_id in project.get("photo_objects", {}):
                    project["photo_objects"][entity_id]["entity"]["product_details"] = copy.deepcopy(details)
    elif project["inventory"].get(observation_id, {}).get("candidate", {}).get("product_id") == candidate["product_id"]:
        project.setdefault("product_history", []).append(project["inventory"].pop(observation_id))
        entity_id = project["associations"].get(observation_id, {}).get("entity_id")
        if entity_id in project.get("photo_objects", {}):
            project["photo_objects"][entity_id]["entity"]["product_details"] = {"status": "user_rejected", "identity_verified": False}
        for entity in (project.get("room") or {}).get("furniture", []):
            if entity["id"] == entity_id:
                entity["product_details"] = {"status": "user_rejected", "identity_verified": False}
                if entity_id in project.get("photo_objects", {}):
                    project["photo_objects"][entity_id]["entity"]["product_details"] = copy.deepcopy(entity["product_details"])


def catalog_asset(image_id):
    catalog = read_json(CATALOG)
    image = next((i for p in catalog["products"] for i in p["images"] if i["image_id"] == image_id), None)
    if image is None:
        raise ValueError("Unknown catalog photograph")
    path = (CATALOG.parent / image["file"]).resolve()
    if not path.is_relative_to(CATALOG.parent.resolve()) or digest(path) != image["sha256"]:
        raise ValueError("Catalog asset integrity check failed")
    return path
