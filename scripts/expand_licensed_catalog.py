"""Plan/resume selective official ABO expansion; never alter production manifests."""

import argparse
import copy
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from furniture_bench.catalog_expansion import BudgetExceeded, DownloadStore, atomic_json
from furniture_bench.data import digest, read_json

ROOT = Path("reports/phase6d3-dataset-expansion")
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/"
TYPES = {
    "SOFA": "sofa",
    "CHAIR": "chair",
    "TABLE": "table",
    "BED": "bed",
    "BED_FRAME": "bed",
    "DESK": "desk",
    "CABINET": "cabinet",
    "WARDROBE": "cabinet",
    "SHELF": "shelving",
    "BOOKCASE": "shelving",
    "LAMP": "lamp",
}
WEIGHTS = {
    "sofa": 250,
    "chair": 200,
    "table": 160,
    "bed": 110,
    "desk": 70,
    "cabinet": 110,
    "shelving": 100,
    "lamp": 40,
}


def english(values):
    if not values:
        return None
    return next(
        (v.get("value") for v in values if v.get("language_tag") == "en_US"),
        next(
            (v.get("value") for v in values if str(v.get("language_tag", "")).startswith("en")),
            values[0].get("value"),
        ),
    )


def previous_products():
    products = []
    for base in [
        Path("data/catalogs/abo-phase6d"),
        Path("data/catalogs/abo-phase6d1"),
        Path("reports/phase6d2-data-quality/fresh-catalog"),
    ]:
        for raw in read_json(base / "selected-products.json")["products"]:
            product = copy.deepcopy(raw)
            product["selection_category"] = TYPES.get(
                product["source_record"].get("product_type", [{}])[0].get("value"), product["label"]
            )
            product["source_cohort"] = base.name
            for image in product["images"]:
                image["file"] = str((base / image["file"]).resolve())
                if digest(image["file"]) != image["sha256"]:
                    raise ValueError("Existing source hash mismatch")
            products.append(product)
    if len({p["product_id"] for p in products}) != len(products):
        raise ValueError("Prior listing identity collision")
    return products


def source():
    return {
        "name": "Amazon Berkeley Objects",
        "license": "CC BY 4.0",
        "license_evidence": str((ROOT / "license-evidence/LICENSE-CC-BY-4.0.txt").resolve()),
        "license_sha256": digest(ROOT / "license-evidence/LICENSE-CC-BY-4.0.txt"),
        "attribution": "Amazon.com; ABO dataset creators (see license-evidence/README.md)",
        "commercial_training": True,
        "commercial_evaluation": True,
        "retain_images": True,
        "allowed_hosts": [
            "amazon-berkeley-objects.s3.amazonaws.com",
            "amazon-berkeley-objects.s3.us-east-1.amazonaws.com",
        ],
        "official_registry": "https://registry.opendata.aws/amazon-berkeley-objects/",
        "rights_basis": "Publisher states CC BY4.0 applies to all assets; retain attribution/license, no endorsement; other rights not independently authenticated",
    }


def plan():
    if (ROOT / "acquisition-plan.json").exists():
        raise ValueError("Plan already saved; reuse frozen selection")
    previous = previous_products()
    excluded_ids = {p["product_id"] for p in previous}
    with gzip.open("data/catalogs/abo-phase6d/images.csv.gz", "rt") as stream:
        mapping = {r["image_id"]: r for r in csv.DictReader(stream)}
    paths = [
        Path("data/catalogs/abo-phase6d/listings_0.json.gz"),
        Path("data/catalogs/abo-phase6d1/listings_1.json.gz"),
    ] + sorted((ROOT / "metadata").glob("listings_*.json.gz"))
    groups, rejected = defaultdict(list), Counter()
    seen = set(excluded_ids)
    for path in paths:
        shard_sha256 = digest(path)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                kind = row.get("product_type", [{}])[0].get("value")
                category = TYPES.get(kind)
                if not category:
                    continue
                if row.get("domain_name") != "amazon.com":
                    rejected["other_marketplace_not_in_current_identity_namespace"] += 1
                    continue
                if row["item_id"] in seen:
                    rejected["existing_or_repeated_listing_id"] += 1
                    continue
                name = english(row.get("item_name"))
                if not name or not row.get("main_image_id"):
                    rejected["missing_name_or_main_image_metadata"] += 1
                    continue
                if any(
                    s in name.casefold()
                    for s in [
                        "swatch",
                        "replacement cover",
                        "cover only",
                        "slipcover",
                        "replacement leg",
                        "hardware kit",
                    ]
                ):
                    rejected["explicit_accessory_or_cover_title"] += 1
                    continue
                ids = list(dict.fromkeys([row["main_image_id"]] + row.get("other_image_id", [])))
                images = [
                    {
                        "image_id": i,
                        "source_url": BASE + "images/small/" + mapping[i]["path"],
                        "original_dimensions": [
                            int(mapping[i]["width"]),
                            int(mapping[i]["height"]),
                        ],
                    }
                    for i in ids
                    if i in mapping
                    and min(int(mapping[i]["width"]), int(mapping[i]["height"])) >= 128
                ]
                if len(images) < 3:
                    rejected["fewer_than_three_source_images_with_dimension_metadata"] += 1
                    continue
                groups[category].append(
                    {
                        "product_id": row["item_id"],
                        "source_record": row,
                        "selection_category": category,
                        "images": images[:6],
                        "metadata_shard": str(path),
                        "metadata_sha256": shard_sha256,
                    }
                )
                seen.add(row["item_id"])
    for values in groups.values():
        values.sort(key=lambda p: p["product_id"])
    counts = Counter(p["selection_category"] for p in previous)
    ordered = []
    while any(groups.values()):
        category = min((c for c in groups if groups[c]), key=lambda c: (counts[c] / WEIGHTS[c], c))
        ordered.append(groups[category].pop(0))
        counts[category] += 1
    atomic_json(ROOT / "existing-products.json", {"products": previous})
    atomic_json(ROOT / "approved-source.json", source())
    atomic_json(
        ROOT / "acquisition-plan.json",
        {
            "target_total_identities": 1040,
            "existing": len(previous),
            "candidates": ordered,
            "candidate_counts": dict(Counter(p["selection_category"] for p in ordered)),
            "metadata_exclusions": dict(rejected),
            "metadata_files": [{"file": str(p), "sha256": digest(p)} for p in paths],
            "image_limit": 4000,
            "byte_limit": 1_500_000_000,
            "main_image_policy": "Official main_image_id first when available; candidate for whole-object, not a verified quality label. Three distinct source IDs and byte hashes preferred; near-duplicates retained and flagged, never automatically merged.",
        },
    )
    print(
        {
            "existing": len(previous),
            "candidates": len(ordered),
            "counts": dict(Counter(p["selection_category"] for p in ordered)),
        }
    )


def acquire():
    plan = read_json(ROOT / "acquisition-plan.json")
    previous = read_json(ROOT / "existing-products.json")["products"]
    store = DownloadStore(
        ROOT / "acquired",
        read_json(ROOT / "approved-source.json"),
        plan["image_limit"],
        plan["byte_limit"],
    )
    file = ROOT / "new-products.json"
    products = read_json(file)["products"] if file.exists() else []
    completed = {p["product_id"] for p in products}
    image_cache = {i["image_id"]: i for p in previous + products for i in p["images"]}
    failures = (
        read_json(ROOT / "acquisition-failures.json")
        if (ROOT / "acquisition-failures.json").exists()
        else []
    )
    stop = "target_reached"
    try:
        for candidate in plan["candidates"]:
            if len(previous) + len(products) >= plan["target_total_identities"]:
                break
            if candidate["product_id"] in completed:
                continue
            images, hashes = [], set()
            for image in candidate["images"]:
                try:
                    if image["image_id"] in image_cache:
                        saved = image_cache[image["image_id"]]
                    else:
                        saved = {**image, **store.fetch(image["image_id"], image["source_url"])}
                        image_cache[image["image_id"]] = saved
                    if saved["sha256"] in hashes:
                        failures.append(
                            {
                                "product_id": candidate["product_id"],
                                "image_id": image["image_id"],
                                "reason": "same_byte_hash_within_listing_retained_in_download_store_not_selected_as_distinct_view",
                            }
                        )
                        continue
                    images.append({**image, **saved})
                    hashes.add(saved["sha256"])
                    if len(images) == 3:
                        break
                except BudgetExceeded:
                    raise
                except (OSError, ValueError) as exc:
                    failures.append(
                        {
                            "product_id": candidate["product_id"],
                            "image_id": image["image_id"],
                            "reason": str(exc),
                        }
                    )
            if len(images) < 3:
                failures.append(
                    {
                        "product_id": candidate["product_id"],
                        "reason": "not_three_distinct_downloaded_photos",
                    }
                )
                continue
            row = candidate["source_record"]
            category = candidate["selection_category"]
            products.append(
                {
                    "product_id": candidate["product_id"],
                    "source_product_id": row["item_id"],
                    "domain_name": row["domain_name"],
                    "label": "cabinet" if category == "shelving" else category,
                    "selection_category": category,
                    "name": english(row.get("item_name")),
                    "brand": english(row.get("brand")),
                    "manufacturer": english(row.get("manufacturer")),
                    "model_number": english(row.get("model_number")),
                    "retailer": "Amazon",
                    "source_record": row,
                    "source_cohort": "phase6d3",
                    "metadata_shard": candidate["metadata_shard"],
                    "metadata_sha256": candidate["metadata_sha256"],
                    "specifications": {
                        k: row.get(k) for k in ["item_dimensions", "material", "color", "style"]
                    },
                    "images": images,
                    "rights": {
                        "source": "Amazon.com / Amazon Berkeley Objects",
                        "license": "CC BY 4.0",
                        "allowed_for_benchmark": True,
                        "allowed_for_commercial_training": True,
                        "allowed_for_commercial_evaluation": True,
                        "license_evidence": str(
                            (ROOT / "license-evidence/LICENSE-CC-BY-4.0.txt").resolve()
                        ),
                        "attribution": source()["attribution"],
                    },
                    "manufacturer_model_independently_verified": False,
                }
            )
            completed.add(candidate["product_id"])
            atomic_json(file, {"products": products})
            atomic_json(ROOT / "acquisition-failures.json", failures)
            print(len(previous) + len(products), category, candidate["product_id"], flush=True)
        else:
            stop = "eligible_candidates_exhausted"
    except BudgetExceeded as exc:
        stop = str(exc)
    atomic_json(ROOT / "acquisition-failures.json", failures)
    atomic_json(
        ROOT / "download-summary.json",
        {
            "stop_reason": stop,
            "existing_products": len(previous),
            "new_products": len(products),
            "total_products": len(previous) + len(products),
            "new_image_objects_started": len(store.state["entries"]),
            "completed_downloads": sum(
                r["status"] == "complete" for r in store.state["entries"].values()
            ),
            "image_bytes_transferred": store.state["transferred_bytes"],
            "incomplete_downloads": [
                r for r in store.state["entries"].values() if r["status"] != "complete"
            ],
            "failure_events": len(failures),
            "resumption": "Per-object Range/If-Range; server200 restarts safely; completed objects revalidated by SHA256; ledger counts partial/retried bytes and files",
            "checksum_scope": "Locally computed SHA256 and image decoder validation, no publisher per-object checksum supplied",
            "new_model_weights": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "acquire"])
    args = parser.parse_args()
    (plan if args.stage == "plan" else acquire)()
