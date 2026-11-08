"""Provenance-aware offline dataset review, independent of production retrieval."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .data import digest, read_json, write_json
from .retrieval_experiments import category_scores

QUALITY_LABELS = (
    "whole_object",
    "whole_object_occluded",
    "staged_interior",
    "multiple_objects",
    "material_closeup",
    "component_part",
    "packaging_diagram_instruction",
    "ambiguous",
)
ROLES = (
    "catalog_reference",
    "catalog_query",
    "household_query",
    "unknown_query",
    "synthetic",
    "reserved",
)


def fingerprint(path):
    """Conservative crop-normalized screening; not proof of independent capture."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        a = np.asarray(image)
        mask = np.any(a < 240, axis=2)
        y, x = np.nonzero(mask)
        if len(x) and mask.mean() >= 0.03:
            image = image.crop((int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)))
        rgb = np.asarray(image.resize((32, 32)), dtype=np.int16)
        gray = np.asarray(image.convert("L").resize((9, 8)))
    return {"sha256": digest(path), "rgb": rgb, "bits": gray[:, 1:] > gray[:, :-1]}


def near_duplicate(a, b):
    return a["sha256"] == b["sha256"] or (
        np.count_nonzero(a["bits"] != b["bits"]) <= 6
        and float(np.mean(np.abs(a["rgb"] - b["rgb"]))) < 14
    )


def _path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Image path must remain inside dataset root")
    return path


def validate_manifest(manifest, root, require_evaluable=False):
    """Check supplied evidence, never turn an attestation into independent verification."""
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("images"), list):
        raise ValueError("Expected schema_version 1 and images list")
    images = manifest["images"]
    ids, hashes = set(), set()
    gallery = {
        (r.get("identity_namespace"), r.get("product_id"))
        for r in images
        if r.get("role") == "catalog_reference"
    }
    warnings = []
    household_sessions = {}
    query_splits = {}
    for row in images:
        image_id = row.get("image_id")
        if not image_id or image_id in ids:
            raise ValueError("Image IDs must be unique and nonempty")
        ids.add(image_id)
        if row.get("role") not in ROLES:
            raise ValueError("Unknown dataset role")
        rights = row.get("rights", {})
        if not all(rights.get(k) for k in ["source", "source_url", "license", "permission_basis"]):
            raise ValueError("Source, license and explicit permission basis are required")
        if rights.get("allowed_for_evaluation") is not True:
            raise ValueError("Affirmative evaluation permission is required")
        if not rights["source_url"].startswith(("https://", "http://", "permission:")):
            raise ValueError("Source URL must be an HTTP URL or permission record reference")
        if rights["permission_basis"] == "owner_permission" and not rights.get("permission_record"):
            raise ValueError("Owner permission requires a retained permission record")
        path = _path(root, row.get("file", ""))
        if not path.is_file() or digest(path) != row.get("sha256"):
            raise ValueError("Image file missing or hash mismatch")
        if row["sha256"] in hashes:
            raise ValueError("Duplicate image bytes across roles/identities")
        hashes.add(row["sha256"])
        with Image.open(path) as image:
            if list(image.size) != row.get("dimensions"):
                raise ValueError("Image dimensions do not match source")
        annotation = row.get("quality_annotation", {})
        if annotation.get("label", "ambiguous") not in QUALITY_LABELS:
            raise ValueError("Invalid quality label")
        if annotation and not annotation.get("provenance"):
            raise ValueError("Quality annotation provenance required")
        if annotation.get("status") == "human_reviewed" and not annotation.get("reviewer"):
            raise ValueError("Human review requires an identified reviewer")
        identity = row.get("identity_evidence", {})
        key = (row.get("identity_namespace"), row.get("product_id"))
        if row.get("manufacturer_model_verified") is True and not (
            identity.get("status") == "verified"
            and identity.get("evidence_references")
            and identity.get("verified_by")
        ):
            raise ValueError("Verified manufacturer/model claim requires supporting evidence")
        if not all(key):
            warnings.append({"image_id": image_id, "reason": "missing_product_identity"})
        if row["role"] == "catalog_query" and key not in gallery:
            raise ValueError("Catalog-present query identity has no gallery reference")
        if row["role"] == "unknown_query" and (not all(key) or key in gallery):
            raise ValueError("Unknown query requires a documented identity absent from gallery")
        if row["role"] == "household_query":
            eligible = (
                identity.get("status") == "verified"
                and identity.get("verification_method")
                and identity.get("evidence_references")
                and identity.get("verified_by")
                and row.get("session_id")
            )
            if not eligible:
                warnings.append({"image_id": image_id, "reason": "household_identity_unverified"})
            if require_evaluable and not eligible:
                raise ValueError("Verified household identity/session evidence is required")
            if row.get("catalog_present") is True and key not in gallery:
                raise ValueError("Exact household identity claimed present but absent from gallery")
            session = row.get("session_id")
            split = row.get("split")
            if session and split:
                if session in household_sessions and household_sessions[session] != split:
                    raise ValueError("Household session crosses development/heldout split")
                household_sessions[session] = split
        if row["role"] in ("catalog_query", "household_query", "unknown_query") and row.get(
            "split"
        ):
            if key in query_splits and query_splits[key] != row["split"]:
                raise ValueError("Query product identity crosses development/heldout split")
            query_splits[key] = row["split"]
        if row["role"] == "synthetic" and not (
            row.get("derived_from_image_id") and row.get("synthetic_transformation")
        ):
            raise ValueError("Synthetic images require explicit source and transformation")
        if require_evaluable and not all(key):
            raise ValueError("Evaluation requires product identity")
    return {"images": len(images), "warnings": warnings, "independent_rights_verification": False}


def apply_reviews(manifest, patch):
    """Apply attested human review to a copy; preserve identity, rights and original predictions."""
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if patch.get("manifest_signature") != signature:
        raise ValueError("Reviews belong to a different manifest")
    result = copy.deepcopy(manifest)
    by_id = {r["image_id"]: r for r in result["images"]}
    seen = set()
    for review in patch.get("reviews", []):
        image_id = review.get("image_id")
        if image_id in seen or image_id not in by_id:
            raise ValueError("Unknown or repeated reviewed image ID")
        seen.add(image_id)
        row = by_id[image_id]
        if review.get("sha256") != row["sha256"]:
            raise ValueError("Reviewed image hash mismatch")
        if review.get("label") not in QUALITY_LABELS or not review.get("reviewer"):
            raise ValueError("Review requires a valid label and named reviewer")
        if review.get("attested_human_review") is not True:
            raise ValueError("Explicit human review attestation required")
        for field in ["whole_product_visible", "suitable_query", "suitable_gallery"]:
            if review.get(field) not in ("yes", "no", "unknown"):
                raise ValueError("Review visibility/suitability must be yes/no/unknown")
        row.setdefault("quality_history", []).append(copy.deepcopy(row.get("quality_annotation")))
        row["quality_annotation"] = {
            **{
                k: review[k]
                for k in [
                    "label",
                    "reviewer",
                    "whole_product_visible",
                    "suitable_query",
                    "suitable_gallery",
                ]
            },
            "status": "human_reviewed",
            "confidence": review.get("confidence", "unspecified"),
            "provenance": "Named reviewer attestation imported; not independently verified by software",
            "notes": review.get("notes", ""),
        }
    return result


def category_condition(scores, categories, predicted, confirmed=None, mode="global"):
    """Experimental scenarios only; unconfirmed detections never trigger strict production filtering."""
    if mode == "global":
        return np.array(scores, copy=True)
    if mode == "predicted_soft":
        return category_scores(scores, categories, predicted, "soft")[0]
    if mode == "predicted_strict_diagnostic":
        return category_scores(scores, categories, predicted, "strict")[0]
    if mode in ("confirmed", "oracle"):
        if confirmed is None or len(confirmed) != len(scores):
            raise ValueError("One category evidence record per query required")
        labels = []
        for record in confirmed:
            usable = mode == "oracle" or (
                record.get("confirmed_by") and record.get("evidence_source") == "user_confirmation"
            )
            labels.append({"label": record.get("label") if usable else None, "score": 1})
        return category_scores(scores, categories, labels, "strict")[0]
    raise ValueError("Unknown category scenario")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--require-evaluable", action="store_true")
    parser.add_argument("--reviews")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise ValueError("Output exists; preserve the previous manifest")
    manifest = read_json(args.manifest)
    if args.reviews:
        if not args.output or Path(args.output).exists():
            raise ValueError("Supply a new output path to preserve original manifest")
        manifest = apply_reviews(manifest, read_json(args.reviews))
    report = validate_manifest(manifest, args.root, args.require_evaluable)
    if args.output:
        write_json(args.output, manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
