"""Provisional image/gallery review from existing visual references and genuine embeddings."""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from furniture_bench.data import read_json, write_json
from furniture_bench.dataset_quality import QUALITY_LABELS, validate_manifest

ROOT = Path("reports/phase6d2-data-quality")


def csv_file(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    records = read_json(ROOT / "vector-image-rows.json")["images"]
    vectors = np.load(ROOT / "image-vectors.npy")
    assert len(vectors) == len(records)
    historical = read_json(
        "reports/phase6d-product-identification/benchmark-validated/rankings.json"
    )["queries"]
    reviewed = read_json(
        "reports/phase6d1-retrieval-optimization/failure-analysis/visual-review.json"
    )["query_types"]
    query_ids = {q["product_id"]: q["image_id"] for q in historical}
    mapping = {
        "whole_object": "whole_object",
        "part_detail": "component_part",
        "material_closeup": "material_closeup",
        "staged_room": "staged_interior",
        "dimension_diagram": "packaging_diagram_instruction",
    }
    prototypes = {query_ids[pid]: mapping[label] for pid, label in reviewed.items()}
    proto_indices = [i for i, row in enumerate(records) if row["image_id"] in prototypes]
    labels = sorted(set(prototypes.values()))
    similarities = np.einsum("qd,pd->qp", vectors, vectors[proto_indices], optimize=False)
    class_scores = np.stack(
        [
            similarities[
                :,
                [
                    j
                    for j, i in enumerate(proto_indices)
                    if prototypes[records[i]["image_id"]] == label
                ],
            ].max(axis=1)
            for label in labels
        ],
        axis=1,
    )
    products = {}
    for base in [
        Path("data/catalogs/abo-phase6d"),
        Path("data/catalogs/abo-phase6d1"),
        ROOT / "fresh-catalog",
    ]:
        products.update(
            {p["product_id"]: p for p in read_json(base / "selected-products.json")["products"]}
        )
    gallery_ids = set()
    query_roles = {}
    for catalog, queries in [
        (read_json("data/catalogs/abo-phase6d/catalog.json")["products"], historical),
        (
            read_json("reports/phase6d1-retrieval-optimization/gallery.json")["products"],
            read_json("reports/phase6d1-retrieval-optimization/protocol.json")["queries"],
        ),
        (
            read_json(ROOT / "fresh-gallery.json")["products"],
            read_json(ROOT / "fresh-protocol.json")["queries"],
        ),
    ]:
        gallery_ids.update(i["image_id"] for p in catalog for i in p["images"])
        query_roles.update(
            {
                q["image_id"]: (
                    "catalog_query" if q["true_product_in_catalog"] else "unknown_query",
                    q.get("split", "historical_examined"),
                )
                for q in queries
            }
        )
    rows, manifest_rows = [], []
    for i, record in enumerate(records):
        order = np.argsort(-class_scores[i], kind="stable")
        margin = float(class_scores[i, order[0]] - class_scores[i, order[1]])
        top = labels[order[0]]
        suggested = top if margin >= 0.03 and class_scores[i, order[0]] >= 0.65 else "ambiguous"
        original_visual = record["image_id"] in prototypes
        if original_visual:
            suggested = prototypes[record["image_id"]]
        provenance = (
            "Prior assistant visual description; NOT owner/human verification"
            if original_visual
            else "Provisional nearest SigLIP image-reference heuristic; 20 assistant-described sofa queries; biased small reference bank, no training or quality ground truth"
        )
        annotation = {
            "label": suggested,
            "status": "provisional",
            "provenance": provenance,
            "confidence": "unverified",
            "relative_margin_not_probability": margin,
            "whole_product_visible": "unknown",
            "suitable_query": "unknown",
            "suitable_gallery": "unknown",
        }
        role, split = query_roles.get(
            record["image_id"],
            (
                "catalog_reference" if record["image_id"] in gallery_ids else "reserved",
                "reference_or_reserved",
            ),
        )
        product = products[record["product_id"]]
        with Image.open(record["file"]) as image:
            dimensions = list(image.size)
        row = {
            "image_id": record["image_id"],
            "product_id": record["product_id"],
            "category": record["label"],
            "dataset": record["dataset"],
            "width": dimensions[0],
            "height": dimensions[1],
            "source_url": record["source_url"],
            "license": "CC BY4.0",
            "sha256": record["sha256"],
            "role": role,
            "split": split,
            "quality_classification": suggested,
            "whole_product_visible": "unknown",
            "suitable_query": "unknown",
            "suitable_gallery": "unknown",
            "annotation_confidence": "unverified",
            "relative_margin_not_probability": margin,
            "annotation_provenance": provenance,
            "review_state": "needs_human_review",
            "file": record["file"],
        }
        rows.append(row)
        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "product_id": row["product_id"],
                "identity_namespace": "ABO_listing",
                "category": row["category"],
                "file": str(Path(row["file"]).relative_to(Path.cwd())),
                "sha256": row["sha256"],
                "dimensions": dimensions,
                "dataset": row["dataset"],
                "role": role,
                "split": split,
                "manufacturer_reported": product.get("manufacturer"),
                "model_reported": product.get("model_number"),
                "manufacturer_model_verified": False,
                "camera_view": None,
                "furniture_condition": None,
                "session_id": None,
                "identity_evidence": {
                    "status": "dataset_listing_association",
                    "verification_method": "official ABO record links source image ID to item_id; no independent manufacturer/model verification",
                    "evidence_references": [
                        str(Path(row["file"]).parent.parent / "selected-products.json")
                    ],
                    "verified_by": None,
                },
                "rights": {
                    "source": "Amazon.com / Amazon Berkeley Objects",
                    "source_url": row["source_url"],
                    "license": "CC BY4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "permission_basis": "official_open_license",
                    "allowed_for_evaluation": True,
                    "attribution": "Amazon.com; ABO dataset creators, see retained ATTRIBUTION.md and official dataset page",
                },
                "quality_annotation": annotation,
            }
        )
    manifest = {
        "schema_version": 1,
        "scope": "Three distinct catalog cohorts; analyze per cohort, never pool as a paired before/after benchmark",
        "household_verified_queries": 0,
        "synthetic_images_in_this_manifest": 0,
        "images": manifest_rows,
        "quality_labels": list(QUALITY_LABELS),
        "human_verified_quality_images": 0,
    }
    validation = validate_manifest(manifest, Path.cwd())
    write_json(ROOT / "licensed-dataset-manifest.json", manifest)
    write_json(ROOT / "manifest-validation.json", validation)
    csv_file(ROOT / "dataset-quality-audit.csv", rows)
    queue = [
        {
            "kind": "image_quality",
            "image_id": r["image_id"],
            "product_id": r["product_id"],
            "suggestion": r["quality_classification"],
            "reason": "All automated/assistant labels need owner verification",
            "state": "unreviewed",
            "source": r["file"],
        }
        for r in rows
    ]
    # Visual similarity can identify review candidates, never merge listing IDs.
    product_ids = list(products)
    model_groups = {}
    for pid, product in products.items():
        if product.get("brand") and product.get("model_number"):
            key = (str(product["brand"]).casefold(), str(product["model_number"]).casefold())
            model_groups.setdefault(key, []).append(pid)
    centers = np.stack(
        [
            vectors[[j for j, row in enumerate(records) if row["product_id"] == pid]].mean(axis=0)
            for pid in product_ids
        ]
    )
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pair_scores = np.einsum("qd,pd->qp", centers, centers, optimize=False)
    np.fill_diagonal(pair_scores, -np.inf)
    audits = []
    for i, pid in enumerate(product_ids):
        product_rows = [r for r in rows if r["product_id"] == pid]
        refs = [r for r in product_rows if r["role"] == "catalog_reference"]
        nearest = int(np.argmax(pair_scores[i]))
        possible_duplicate = float(pair_scores[i, nearest]) >= 0.95
        counts = Counter(r["quality_classification"] for r in refs)
        model_key = (
            str(products[pid].get("brand")).casefold(),
            str(products[pid].get("model_number")).casefold(),
        )
        shared = [other for other in model_groups.get(model_key, []) if other != pid]
        audits.append(
            {
                "product_id": pid,
                "category": products[pid]["label"],
                "dataset": product_rows[0]["dataset"],
                "retained_images": len(product_rows),
                "active_gallery_images": len(refs),
                "main_image_retained": products[pid]["source_record"].get("main_image_id")
                in [r["image_id"] for r in product_rows],
                "verified_whole_object_references": 0,
                "provisional_whole_object_references": counts["whole_object"],
                "provisional_closeup_or_part_references": counts["material_closeup"]
                + counts["component_part"],
                "provisional_staged_references": counts["staged_interior"],
                "missing_whole_object_status": "unknown_pending_human_review",
                "view_angles_verified": False,
                "identity_scope": "ABO_listing_only",
                "reported_model": products[pid].get("model_number"),
                "reported_brand": products[pid].get("brand"),
                "same_reported_brand_model_other_listings": ";".join(shared),
                "manufacturer_model_independently_verified": False,
                "nearest_other_listing": product_ids[nearest],
                "nearest_product_mean_cosine": float(pair_scores[i, nearest]),
                "possible_duplicate_unverified": possible_duplicate,
                "merged": False,
                "review_state": "unreviewed",
            }
        )
        if shared:
            queue.append(
                {
                    "kind": "shared_reported_brand_model",
                    "image_id": "",
                    "product_id": pid,
                    "suggestion": ";".join(shared),
                    "reason": "Reported shared model may conceal color/size variants; documentary review required",
                    "state": "unreviewed",
                    "source": "official ABO selected records",
                }
            )
        if possible_duplicate:
            queue.append(
                {
                    "kind": "possible_identity_variant",
                    "image_id": "",
                    "product_id": pid,
                    "suggestion": product_ids[nearest],
                    "reason": "High embedding similarity is not identity evidence",
                    "state": "unreviewed",
                    "source": "official ABO selected records",
                }
            )
    queue.append(
        {
            "kind": "household_identity",
            "image_id": "",
            "product_id": "unknown",
            "suggestion": "",
            "reason": "Licensed room photos have no independently verified product identity",
            "state": "blocked_missing_identity_evidence",
            "source": "data/detection-pilot/photos",
        }
    )
    csv_file(ROOT / "product-gallery-audit.csv", audits)
    csv_file(ROOT / "manual-review-queue.csv", queue)
    old_result = read_json("reports/phase6d1-retrieval-optimization/final-evaluation-aligned.json")
    by_image = {r["image_id"]: r for r in rows}
    exploratory = []
    for label in QUALITY_LABELS:
        indices = [
            j
            for j, q in enumerate(old_result["queries"])
            if q["split"] == "development"
            and q["true_product_in_catalog"]
            and by_image[q["image_id"]]["quality_classification"] == label
        ]
        if indices:
            ranks = [old_result["baseline_ranks"][j] for j in indices]
            exploratory.append(
                {
                    "scope": "examined_Phase6D1_development_only",
                    "quality_label_provisional": label,
                    "queries": len(indices),
                    "recall1": float(np.mean(np.array(ranks) == 1)),
                    "recall5": float(np.mean(np.array(ranks) <= 5)),
                    "causality_verified": False,
                }
            )
    csv_file(ROOT / "exploratory-quality-retrieval.csv", exploratory)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    display = {
        "signature": signature,
        "images": [
            {**row, "image_url": os.path.relpath(Path.cwd() / row["file"], ROOT.resolve())}
            for row in manifest_rows
        ],
    }
    template = Path("viewer/dataset-review.html").read_text()
    (ROOT / "image-review.html").write_text(
        template.replace("__DATASET__", json.dumps(display).replace("<", "\\u003c"))
    )
    folder = ROOT / "failure-gallery"
    folder.mkdir(exist_ok=True)
    # Diagnostic historical cases and provisional examples, not new held-out selection.
    for label in QUALITY_LABELS:
        examples = [
            r
            for r in rows
            if r["dataset"] == "abo-phase6d" and r["quality_classification"] == label
        ][:12]
        if not examples:
            continue
        canvas = Image.new("RGB", (1200, 920), "#f4f4ef")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 8),
            "PROVISIONAL " + label + " — owner review required; no verified quality ground truth",
            fill="black",
            font=ImageFont.load_default(size=16),
        )
        for n, row in enumerate(examples):
            x, y = (n % 4) * 300, 40 + (n // 4) * 290
            with Image.open(row["file"]) as source:
                thumbnail = ImageOps.contain(source.convert("RGB"), (280, 225))
            canvas.paste(thumbnail, (x + 10 + (280 - thumbnail.width) // 2, y))
            draw.text(
                (x + 10, y + 230),
                row["product_id"] + " / " + row["image_id"],
                fill="black",
                font=ImageFont.load_default(size=12),
            )
        canvas.save(folder / (label + ".png"))
    counts = Counter(r["quality_classification"] for r in rows)
    write_json(
        ROOT / "quality-summary.json",
        {
            "images": len(rows),
            "products": len(products),
            "provisional_counts": dict(counts),
            "gallery_provisional_counts": dict(
                Counter(
                    r["quality_classification"] for r in rows if r["role"] == "catalog_reference"
                )
            ),
            "owner_verified_whole_object_images": 0,
            "whole_visibility_unknown_images": len(rows),
            "prototype_images": len(proto_indices),
            "prototype_label_counts": dict(Counter(prototypes.values())),
            "classification_method": "Nearest image reference per class, fixed min cosine0.65 and class margin0.03; uncalibrated, sofa-biased; unsupported occlusion/multiple-object classes left unresolved, no training",
            "manual_queue_rows": len(queue),
            "possible_duplicate_listing_flags": sum(
                r["possible_duplicate_unverified"] for r in audits
            ),
            "independent_household_queries": 0,
        },
    )
    print({"images": len(rows), "provisional_counts": counts, "owner_verified": 0})


if __name__ == "__main__":
    main()
