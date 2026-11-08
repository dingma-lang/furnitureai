from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from .geometry import validate_box

CATEGORIES = ("sofa", "chair", "table", "bed", "desk", "cabinet", "lamp")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_image(path):
    """All boxes and masks use EXIF-transposed RGB pixel coordinates."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def validate_label(label):
    if label not in CATEGORIES:
        raise ValueError(f"unsupported label {label!r}; choose from {CATEGORIES}")


def require_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_rights(row):
    rights = row.get("rights", {})
    if not isinstance(rights, dict) or rights.get("allowed_for_benchmark") is not True:
        raise ValueError("each image needs rights.allowed_for_benchmark=true")
    for name in ("source", "license"):
        require_string(rights.get(name), f"rights.{name}")


def annotation_is_human_reviewed(row):
    annotation = row.get("annotation", {})
    return (
        annotation.get("status") == "human_reviewed"
        and annotation.get("human_reviewed") is True
        and all(
            isinstance(annotation.get(key), str) and annotation[key].strip()
            for key in ("reviewer", "reviewed_at", "evidence")
        )
    )


def load_manifest(path, kind: str):
    """Validate metadata without opening or downloading images."""
    if kind not in ("rooms", "catalog"):
        raise ValueError("manifest kind must be rooms or catalog")
    path = Path(path).resolve()
    document = read_json(path)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    key = "images" if kind == "rooms" else "products"
    rows = document.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest needs a nonempty {key} list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest entries must be objects")
        id_key = "image_id" if kind == "rooms" else "product_id"
        require_string(row.get(id_key), id_key)
        if row[id_key] in seen:
            raise ValueError(f"duplicate {id_key}: {row[id_key]}")
        seen.add(row[id_key])
        require_string(row.get("file"), "file")
        validate_rights(row)
        if kind == "rooms":
            if row.get("annotation", {}).get(
                "status"
            ) == "human_reviewed" and not annotation_is_human_reviewed(row):
                raise ValueError(
                    "human-reviewed annotations require reviewer, date, evidence and explicit attestation"
                )
            for region in row.get("ignore_regions", []):
                region["box"] = list(validate_box(region.get("box", [])))
                require_string(region.get("reason"), "ignore region reason")
            if (
                "image_sha256" in row
                and digest(
                    row["resolved_file"] if "resolved_file" in row else path.parent / row["file"]
                )
                != row["image_sha256"]
            ):
                raise ValueError("image bytes differ from annotated manifest fingerprint")
            objects = row.get("objects")
            if objects is not None:
                if not isinstance(objects, list):
                    raise ValueError("objects must be null or a list")
                object_ids = set()
                for obj in objects:
                    if not isinstance(obj, dict):
                        raise ValueError("annotations must be objects")
                    require_string(obj.get("object_id"), "object_id")
                    if obj["object_id"] in object_ids:
                        raise ValueError("duplicate object_id in image")
                    object_ids.add(obj["object_id"])
                    validate_label(obj.get("label"))
                    obj["box"] = list(validate_box(obj.get("box", [])))
                    if "identity_verified" in obj and not isinstance(
                        obj["identity_verified"], bool
                    ):
                        raise ValueError("identity_verified must be a boolean")
                    if obj.get("identity_verified") is True:
                        require_string(obj.get("product_id"), "verified product_id")
        else:
            validate_label(row.get("label"))
        row["resolved_file"] = str((path.parent / row["file"]).resolve())
    return rows


def validate_prediction(detection):
    validate_label(detection.get("label"))
    validate_box(detection.get("box", []))
    score = float(detection.get("score", -1))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("detection score must be finite and in [0, 1]")
