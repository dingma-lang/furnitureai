"""Freeze fresh identity-level development/held-out roles before inference."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from audit_abo_queries import fingerprint

from furniture_bench.data import digest, read_json, write_json


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    data = Path("data/catalogs/abo-phase6d1")
    if (root / "protocol.json").exists():
        raise ValueError("Protocol already frozen; preserve it")
    products = read_json(data / "selected-products.json")["products"]
    old = read_json("data/catalogs/abo-phase6d/selected-products.json")["products"]
    assert not {p["product_id"] for p in products} & {p["product_id"] for p in old}
    assert not {i["image_id"] for p in products for i in p["images"]} & {
        i["image_id"] for p in old for i in p["images"]
    }
    rng = np.random.default_rng(6117)
    groups = defaultdict(list)
    roles = {}
    for product in products:
        groups[product["label"]].append(product)
    for category, items in sorted(groups.items()):
        items = sorted(items, key=lambda p: p["product_id"])
        order = rng.permutation(len(items))
        for n, index in enumerate(order):
            roles[items[index]["product_id"]] = {
                "split": "development" if n % 2 == 0 else "heldout",
                "absent": False,
            }
        for split in ["development", "heldout"]:
            subset = [items[i] for i in order if roles[items[i]["product_id"]]["split"] == split]
            n = max(1, len(subset) // 5)
            for p in subset[:n]:
                roles[p["product_id"]]["absent"] = True
    gallery = [
        {**p, "images": p["images"][:2]} for p in products if not roles[p["product_id"]]["absent"]
    ]
    gm = [(p["product_id"], i, fingerprint(data / i["file"])) for p in gallery for i in p["images"]]
    exclusions = []
    queries = []
    for product in products:
        image = product["images"][2]
        h, rgb = fingerprint(data / image["file"])
        reasons = []
        for pid, other, (hh, rr) in gm:
            if image["sha256"] == other["sha256"] or (
                np.count_nonzero(h != hh) <= 6 and float(np.mean(abs(rgb - rr))) < 14
            ):
                reasons.append(
                    {
                        "gallery_product_id": pid,
                        "gallery_image_id": other["image_id"],
                        "reason": "byte_or_crop_normalized_near_duplicate",
                    }
                )
        if reasons:
            exclusions.append(
                {
                    "product_id": product["product_id"],
                    "image_id": image["image_id"],
                    "reasons": reasons,
                }
            )
            continue
        queries.append(
            {
                "product_id": product["product_id"],
                "label": product["label"],
                "file": str((data / image["file"]).resolve()),
                "image_id": image["image_id"],
                "sha256": image["sha256"],
                "split": roles[product["product_id"]]["split"],
                "true_product_in_catalog": not roles[product["product_id"]]["absent"],
            }
        )
    detector_subset = []
    for split in ["development", "heldout"]:
        for category in sorted(groups):
            eligible = sorted(
                [
                    q
                    for q in queries
                    if q["split"] == split
                    and q["label"] == category
                    and q["true_product_in_catalog"]
                ],
                key=lambda q: q["product_id"],
            )[:2]
            detector_subset.extend(q["image_id"] for q in eligible)
    protocol = {
        "seed": 6117,
        "source": "Fresh official ABO listings shard1; none of the examined 140 product identities or image IDs",
        "fresh_products": len(products),
        "gallery_products": len(gallery),
        "gallery_images": len(gm),
        "queries": queries,
        "identity_roles": roles,
        "exclusions": exclusions,
        "counts": {
            split: dict(
                Counter(
                    (q["label"] + "_" + ("present" if q["true_product_in_catalog"] else "absent"))
                    for q in queries
                    if q["split"] == split
                )
            )
            for split in ["development", "heldout"]
        },
        "predicted_category_query_image_ids": detector_subset,
        "predicted_category_subset_policy": "first two eligible present queries per category per split; fixed before detector inference",
        "individual_configs": [
            "original+max",
            "original+normalized_centroid",
            "original+top2_mean",
            "original+main_only",
            "white_margin_trim+max",
            "margin10+max",
            "letterbox+max",
        ],
        "crop_semantics": "White-margin foreground heuristic only on nearly white borders; not GT furniture bbox or instance segmentation",
        "instance_mask_experiment": "Unavailable: no reliable per-instance mask supplied; UPerNet room masks not used as furniture masks",
        "selection_objective": "development mean of overall R@5 and sofa R@5; ties R@1 then simpler baseline order",
        "combination_limit": "one combination of best individual preprocessing and best individual aggregation, selected on development only",
        "heldout_policy": "One final evaluation after development configuration freeze; no re-selection",
        "app_adoption_gate": "Held-out paired R@5 improvement bootstrap95% lower bound >0, sofa R@5 nonnegative, R@1 nonnegative; otherwise retain Phase6D production retrieval",
        "bootstrap_seed": 6118,
        "bootstrap_samples": 5000,
        "physical_identity_scope": "dataset listing identities, not exact household manufacturer recognition",
        "metadata_reranking": "Not used: query brand/specifications not available from the image; using known IDs/filenames forbidden",
        "annotation_rights": "ABO images and metadata Amazon.com CC BY4.0, same official license as Phase6D; source photos unchanged",
    }
    write_json(root / "protocol.json", protocol)
    write_json(root / "gallery.json", {"products": gallery, "data_root": str(data.resolve())})
    rows = [
        {
            "image_id": q["image_id"],
            "file": q["file"],
            "rights": {
                "allowed_for_benchmark": True,
                "source": "Amazon.com / ABO official licensed catalog",
                "license": "CC BY4.0",
            },
        }
        for q in queries
        if q["image_id"] in detector_subset
    ]
    write_json(root / "detector-input.json", {"schema_version": 1, "images": rows})
    selection = read_json(data / "selection.json")
    selection["source_listings"] = (
        "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/listings/metadata/listings_1.json.gz"
    )
    selection["source_shard_correction"] = (
        "Generic preparer source label was shard0; this fresh file and source hash are shard1, corrected before inference"
    )
    selection["provisional_gallery_products"] = selection.get("gallery_products")
    selection["gallery_products"] = len(gallery)
    selection["actual_absent_query_products"] = sum(role["absent"] for role in roles.values())
    selection["protocol_note"] = "Canonical manifests mirror frozen seed6117 protocol roles"
    for filename in ["catalog.json", "queries.json"]:
        source = data / filename
        provisional = data / ("provisional-" + filename)
        if source.exists() and not provisional.exists():
            source.rename(provisional)
    write_json(data / "catalog.json", {"schema_version": 2, "products": gallery})
    write_json(data / "queries.json", {"schema_version": 2, "queries": queries})
    write_json(data / "selection.json", selection)
    if (data / "listings_0.json.gz").exists():
        (data / "listings_0.json.gz").rename(data / "listings_1.json.gz")
    write_json(
        root / "protocol-lock.json",
        {
            "protocol_sha256": digest(root / "protocol.json"),
            "gallery_sha256": digest(root / "gallery.json"),
            "ranking_results_seen_before_lock": False,
        },
    )
    print(
        {k: protocol[k] for k in ["fresh_products", "gallery_products", "gallery_images", "counts"]}
    )


if __name__ == "__main__":
    main()
