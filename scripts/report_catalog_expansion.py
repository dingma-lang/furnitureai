"""Catalog provenance, scalable duplicate diagnostics and figures from saved evidence."""

import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from furniture_bench.catalog_expansion import atomic_json, validate_nested_galleries
from furniture_bench.data import digest, read_json
from furniture_bench.dataset_quality import fingerprint

ROOT = Path("reports/phase6d3-dataset-expansion")


def csv_file(name, rows):
    with (ROOT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    products = (
        read_json(ROOT / "existing-products.json")["products"]
        + read_json(ROOT / "new-products.json")["products"]
    )
    prior = [p for p in products if p["source_cohort"] != "phase6d3"]
    new = [p for p in products if p["source_cohort"] == "phase6d3"]
    protocol = read_json(ROOT / "scaling-protocol.json")
    results = read_json(ROOT / "scaling-evaluation.json")
    embedding = read_json(ROOT / "embedding-report.json")
    lock = read_json(ROOT / "scaling-protocol-lock.json")
    assert digest(ROOT / "scaling-protocol.json") == lock["protocol_sha256"]
    galleries = {
        size: read_json(ROOT / f"gallery-{size}.json")["products"]
        for size in protocol["gallery_ids"]
    }
    validate_nested_galleries(galleries, protocol["queries"])
    ledger = read_json(ROOT / "acquired/download-ledger.json")
    ledger_by_id = {r["image_id"]: r for r in ledger["entries"].values()}
    source = read_json(ROOT / "approved-source.json")
    query_by_id = {q["image_id"]: q for q in protocol["queries"]}
    owner = defaultdict(list)
    unique = {}
    for p in products:
        for i in p["images"]:
            owner[i["image_id"]].append(p["product_id"])
            if i["image_id"] in unique:
                assert unique[i["image_id"]]["sha256"] == i["sha256"]
            unique[i["image_id"]] = i
    images = []
    for image_id, image in unique.items():
        assert digest(image["file"]) == image["sha256"]
        with Image.open(image["file"]) as im:
            dimensions = list(im.size)
        row = ledger_by_id.get(image_id, {})
        query = query_by_id.get(image_id)
        role = (
            "known_query"
            if query and query["true_product_in_catalog"]
            else "unknown_query"
            if query
            else "reference_or_reserved"
        )
        images.append(
            {
                "image_id": image_id,
                "source_product_ids": sorted(owner[image_id]),
                "file": image["file"],
                "sha256": image["sha256"],
                "dimensions": dimensions,
                "source_original_dimensions_reported": image.get("original_dimensions"),
                "source_url": image["source_url"],
                "dataset_source": source["name"],
                "license": source["license"],
                "license_evidence": source["license_evidence"],
                "license_sha256": source["license_sha256"],
                "attribution": source["attribution"],
                "download_timestamp": row.get("downloaded_at"),
                "timestamp_provenance": "download ledger"
                if row
                else "not recorded in older selection, left unknown",
                "commercial_training": True,
                "commercial_evaluation": True,
                "retain_images": True,
                "permission_basis": "official publisher CC BY4.0 copyright grant; other rights separate",
                "image_role": role,
                "annotation_status": "provisional",
                "annotation_provenance": "Dataset role from frozen protocol, visible-content quality not human verified",
                "split": "frozen_scaling_evaluation" if query else "catalog_reference_or_reserved",
                "identity_scope": "dataset listing ID; not independently verified manufacturer/model",
            }
        )
    atomic_json(
        ROOT / "licensed-dataset-manifest.json",
        {
            "schema_version": 3,
            "source": source,
            "products_manifest": "catalog-products.json",
            "product_count": len(products),
            "unique_image_count": len(images),
            "product_image_associations": sum(len(p["images"]) for p in products),
            "images": images,
            "household_verified_queries": 0,
            "identity_note": "Retain exact ABO item_id + amazon.com marketplace; different listing variants remain distinct. Shared image IDs map to all source product IDs without invented identity equivalence.",
            "catalog_format_backward_compatible": "expanded-catalog.json schema2 with additive fields; image paths relative to manifest, source records separate from vector arrays",
            "license_evidence_hashes": {
                p.name: digest(p) for p in (ROOT / "license-evidence").iterdir() if p.is_file()
            },
        },
    )
    coverage = []
    for category in sorted({p["selection_category"] for p in products}):
        old = [p for p in prior if p["selection_category"] == category]
        added = [p for p in new if p["selection_category"] == category]
        group = old + added
        coverage.append(
            {
                "category": category,
                "old_listing_identities": len(old),
                "new_listing_identities": len(added),
                "total_listing_identities": len(group),
                "product_image_associations": sum(len(p["images"]) for p in group),
                "unique_source_images": len({i["image_id"] for p in group for i in p["images"]}),
                "main_image_candidates_retained": sum(
                    any(
                        i["image_id"] == p["source_record"].get("main_image_id")
                        for i in p["images"]
                    )
                    for p in group
                ),
                "human_verified_whole_object_images": 0,
            }
        )
    csv_file("category-coverage.csv", coverage)
    duplicates = []

    def flag(kind, a, b, ia="", ib="", similarity="", reason=""):
        duplicates.append(
            {
                "kind": kind,
                "product_id_a": a,
                "product_id_b": b,
                "image_id_a": ia,
                "image_id_b": ib,
                "similarity_not_identity_probability": similarity,
                "reason": reason,
                "review_state": "needs_human_documentary_review",
                "merged": False,
                "deleted": False,
            }
        )

    for image_id, ids in owner.items():
        ids = sorted(set(ids))
        if len(ids) > 1:
            for other in ids[1:]:
                flag(
                    "shared_source_image_id",
                    ids[0],
                    other,
                    image_id,
                    image_id,
                    reason="Same official image reused by different listings; keep listing variants separate",
                )
    hashes = defaultdict(list)
    for image in images:
        hashes[image["sha256"]].append(image["image_id"])
    for ids in hashes.values():
        if len(ids) > 1:
            for other in ids[1:]:
                flag(
                    "exact_bytes_different_image_id",
                    owner[ids[0]][0],
                    owner[other][0],
                    ids[0],
                    other,
                    reason="Byte-identical exports retained once per source ID and flagged",
                )
    fingerprints = [fingerprint(i["file"]) for i in images]
    bits = np.stack([f["bits"] for f in fingerprints])
    for n, fp in enumerate(fingerprints):
        candidates = (
            np.flatnonzero(np.count_nonzero(bits[n + 1 :] != fp["bits"], axis=(1, 2)) <= 6) + n + 1
        )
        for j in candidates:
            if fp["sha256"] == fingerprints[j]["sha256"]:
                continue
            difference = float(np.mean(abs(fp["rgb"] - fingerprints[j]["rgb"])))
            if difference < 14:
                a, b = images[n], images[j]
                flag(
                    "crop_perceptual_near_duplicate",
                    owner[a["image_id"]][0],
                    owner[b["image_id"]][0],
                    a["image_id"],
                    b["image_id"],
                    difference,
                    "dHash distance<=6 and cropped RGB mean difference<14; heuristic, preserve alternative views",
                )
    # Product centroids provide a review queue; not a merge/recognition rule.
    rows = read_json(ROOT / "vector-image-rows.json")["images"]
    vectors = np.load(ROOT / "image-vectors.npy")
    vector_by_id = {r["image_id"]: v for r, v in zip(rows, vectors)}
    centers = np.stack(
        [np.mean([vector_by_id[i["image_id"]] for i in p["images"]], axis=0) for p in products]
    )
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    sim = np.einsum("qd,pd->qp", centers, centers, optimize=False)
    for a, b in zip(*np.where(np.triu(sim >= 0.95, k=1))):
        flag(
            "high_embedding_similarity_listings",
            products[a]["product_id"],
            products[b]["product_id"],
            similarity=float(sim[a, b]),
            reason="Mean image cosine>=0.95; possible variant or similar appearance, not verified equivalent identity",
        )
    model_groups, name_groups = defaultdict(list), defaultdict(list)
    for p in products:
        if p.get("brand") and p.get("model_number"):
            model_groups[(str(p["brand"]).casefold(), str(p["model_number"]).casefold())].append(
                p["product_id"]
            )
        if p.get("name"):
            name_groups[" ".join(p["name"].casefold().split())].append(p["product_id"])
    for kind, groups in [
        ("same_reported_brand_model", model_groups),
        ("same_normalized_reported_name", name_groups),
    ]:
        for ids in groups.values():
            for other in ids[1:]:
                flag(
                    kind,
                    ids[0],
                    other,
                    reason="Reported text shared; size/color/upholstery may differ; no manufacturer authentication",
                )
    csv_file("possible-duplicates.csv", duplicates)
    # Provisional sampled content hints reuse prior assistant visual descriptions only.
    old = read_json("reports/phase6d-product-identification/benchmark-validated/rankings.json")[
        "queries"
    ]
    query_map = {q["product_id"]: q["image_id"] for q in old}
    annotations = read_json(
        "reports/phase6d1-retrieval-optimization/failure-analysis/visual-review.json"
    )["query_types"]
    prototypes = [(vector_by_id[query_map[pid]], label) for pid, label in annotations.items()]
    selected = []
    for category in sorted({p["selection_category"] for p in new}):
        selected.extend([p for p in new if p["selection_category"] == category][:3])
    # Add sofas whose third photo resembles a reviewed material close-up.
    material = np.stack([v for v, label in prototypes if label == "material_closeup"])
    sofa_order = sorted(
        [p for p in new if p["selection_category"] == "sofa"],
        key=lambda p: -float((material @ vector_by_id[p["images"][2]["image_id"]]).max()),
    )
    selected.extend(
        p for p in sofa_order[:8] if p["product_id"] not in {x["product_id"] for x in selected}
    )
    cards = []
    for p in selected:
        tiles = []
        for image in p["images"]:
            score, label = max(
                (float(v @ vector_by_id[image["image_id"]]), label) for v, label in prototypes
            )
            relative = (
                Path(image["file"]).relative_to(ROOT.resolve())
                if Path(image["file"]).is_relative_to(ROOT.resolve())
                else Path(image["file"])
            )
            role = (
                "Official main whole-object CANDIDATE"
                if image["image_id"] == p["source_record"].get("main_image_id")
                else "Alternate/detail UNRESOLVED"
            )
            tiles.append(
                f'<div><img loading="lazy" src="{html.escape(str(relative))}" alt="{html.escape(p["product_id"])}"><p>{html.escape(image["image_id"])}<br>{role}<br>PROVISIONAL visual-reference hint: {label} ({score:.3f}; not probability)</p></div>'
            )
        cards.append(
            f'<section><h2>{html.escape(p["product_id"])} · {html.escape(p["selection_category"])}</h2><p>{html.escape(p.get("name") or "Name unavailable")}</p><div class="photos">'
            + "".join(tiles)
            + "</div></section>"
        )
    (ROOT / "image-quality-sample.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Expanded catalog review sample</title><style>body{font:15px system-ui;background:#f4f5ef;margin:24px;color:#223336}section{background:white;border:1px solid #ccc;margin:18px 0;padding:16px}.photos{display:flex;gap:24px}.photos>div{width:31%}img{width:100%;height:200px;object-fit:contain}p{overflow-wrap:anywhere}</style><h1>Expanded furniture reference sample</h1><p>Amazon.com / ABO, CC BY 4.0. Resized contact sheet; originals unchanged. All image-role/content hints are provisional. No human verification, identity merging or automated filtering. Stratified category examples plus difficult close-up-like sofas.</p><p><a href="possible-duplicates.csv">Variant/duplicate review queue</a> · <a href="license-evidence/README.md">Attribution and license</a></p>'
        + "".join(cards)
        + "</html>"
    )
    sizes = sorted(results["sizes"], key=int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = [int(s) for s in sizes]
    axes[0].plot(
        x,
        [results["sizes"][s]["metrics"]["recall1"] for s in sizes],
        marker="o",
        label="Recall@1 (50 known)",
    )
    axes[0].plot(
        x,
        [results["sizes"][s]["metrics"]["recall5"] for s in sizes],
        marker="o",
        label="Recall@5 (50 known)",
    )
    axes[0].plot(
        x,
        [results["sizes"][s]["metrics"]["sofa_recall5"] for s in sizes],
        marker="o",
        label="Sofa Recall@5 (14)",
    )
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].plot(
        x,
        [results["sizes"][s]["metrics"]["product_search_batch_wall_seconds"] * 1000 for s in sizes],
        marker="o",
        label="70-query full product ranking",
    )
    axes[1].plot(
        x,
        [
            results["sizes"][s]["metrics"]["faiss_search_batch_seconds_including_ipc"] * 1000
            for s in sizes
        ],
        marker="o",
        label="FAISS batch incl. IPC",
    )
    axes[1].set_ylabel("Milliseconds")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Nested gallery listing identities")
        ax.grid(alpha=0.2)
    fig.suptitle("Frozen common catalog-query set; no household accuracy claim")
    fig.tight_layout()
    fig.savefig(ROOT / "catalog-scaling-chart.png", dpi=160)
    plt.close(fig)
    summary = read_json(ROOT / "download-summary.json")
    assert (
        summary["new_image_objects_started"] <= 4000
        and summary["image_bytes_transferred"] <= 1_500_000_000
    )
    summary.update(
        unique_total_images=len(images),
        total_image_associations=sum(len(p["images"]) for p in products),
        new_product_image_associations=sum(len(p["images"]) for p in new),
        existing_source_images_reused=1140,
        shared_image_source_associations=sum(len(p["images"]) for p in products) - len(images),
        downloaded_object_checksums={
            r["image_id"]: r["sha256"] for r in ledger["entries"].values()
        },
        retained_image_file_bytes=sum(Path(i["file"]).stat().st_size for i in images),
        working_tree_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
        metadata_bytes_new=sum(p.stat().st_size for p in (ROOT / "metadata").glob("*.gz")),
    )
    atomic_json(ROOT / "download-summary.json", summary)
    atomic_json(
        ROOT / "expansion-audit-summary.json",
        {
            "products": len(products),
            "unique_images": len(images),
            "provisional_sample_products": len(selected),
            "duplicate_queue_rows": len(duplicates),
            "duplicate_kind_counts": dict(Counter(r["kind"] for r in duplicates)),
            "new_category_counts": dict(Counter(p["selection_category"] for p in new)),
            "human_verified_whole_object_count": None,
            "quality_claim": "Main-image priority and provisional hints, not verified whole-object coverage",
            "query_exclusions": len(protocol["exclusions"]),
            "model": embedding["model_signature"],
        },
    )
    print(
        {
            "products": len(products),
            "images": len(images),
            "duplicate_flags": len(duplicates),
            "sample_products": len(selected),
        }
    )


if __name__ == "__main__":
    main()
