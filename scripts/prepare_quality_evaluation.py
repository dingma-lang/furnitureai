"""Freeze a fresh catalog-only study before model scoring; preserve earlier benchmarks."""

from collections import Counter
from pathlib import Path

import numpy as np

from furniture_bench.data import digest, read_json, write_json
from furniture_bench.dataset_quality import fingerprint, near_duplicate

ROOT = Path("reports/phase6d2-data-quality")


def main():
    if (ROOT / "protocol-lock.json").exists():
        raise ValueError("Protocol already frozen")
    data = ROOT / "fresh-catalog"
    products = read_json(data / "selected-products.json")["products"]
    previous = read_json(ROOT / "prior-identities-excluded.json")["products"]
    old_ids = {p["product_id"] for p in previous}
    old_images = {i["image_id"] for p in previous for i in p["images"]}
    old_hashes = {i["sha256"] for p in previous for i in p["images"]}
    assert not old_ids & {p["product_id"] for p in products}
    assert not old_images & {i["image_id"] for p in products for i in p["images"]}
    assert not old_hashes & {i["sha256"] for p in products for i in p["images"]}
    for name in ["LICENSE-CC-BY-4.0.txt", "ATTRIBUTION.md", "ABO-README.md"]:
        (data / name).write_bytes((Path("data/catalogs/abo-phase6d") / name).read_bytes())
    prior_fps = []
    for folder in ["abo-phase6d", "abo-phase6d1"]:
        base = Path("data/catalogs") / folder
        for p in read_json(base / "selected-products.json")["products"]:
            prior_fps.extend(fingerprint(base / i["file"]) for i in p["images"])
    excluded, eligible = [], []
    for product in products:
        fps = [fingerprint(data / i["file"]) for i in product["images"]]
        cross = any(near_duplicate(a, b) for a in fps for b in prior_fps)
        internal = any(near_duplicate(fps[a], fps[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        if cross or internal:
            excluded.append(
                {
                    "product_id": product["product_id"],
                    "reason": "prior_crop_near_duplicate"
                    if cross
                    else "within_product_crop_near_duplicate",
                }
            )
        else:
            eligible.append(product)
    # A category-stratified identity split; references may contain both splits' identities.
    rng = np.random.default_rng(6227)
    roles = {}
    for label in sorted({p["label"] for p in eligible}):
        group = sorted([p for p in eligible if p["label"] == label], key=lambda p: p["product_id"])
        group = [group[i] for i in rng.permutation(len(group))]
        for n, product in enumerate(group):
            roles[product["product_id"]] = {
                "split": "development" if n % 2 == 0 else "heldout",
                "absent": False,
            }
        for split in ["development", "heldout"]:
            selected = [p for p in group if roles[p["product_id"]]["split"] == split]
            for product in selected[: max(1, len(selected) // 5)]:
                roles[product["product_id"]]["absent"] = True
    gallery = [
        {**p, "images": p["images"][:2]} for p in eligible if not roles[p["product_id"]]["absent"]
    ]
    gallery_fps = [
        (i["image_id"], fingerprint(data / i["file"])) for p in gallery for i in p["images"]
    ]
    queries = []
    for product in eligible:
        image = product["images"][2]
        fp = fingerprint(data / image["file"])
        matches = [image_id for image_id, gfp in gallery_fps if near_duplicate(fp, gfp)]
        if matches:
            excluded.append(
                {
                    "product_id": product["product_id"],
                    "reason": "cross_gallery_crop_near_duplicate",
                    "image_ids": matches,
                }
            )
            continue
        role = roles[product["product_id"]]
        queries.append(
            {
                **image,
                "file": str((data / image["file"]).resolve()),
                "product_id": product["product_id"],
                "label": product["label"],
                "split": role["split"],
                "true_product_in_catalog": not role["absent"],
            }
        )
    subset = []
    for split in ["development", "heldout"]:
        for label in sorted({q["label"] for q in queries}):
            subset.extend(
                q["image_id"]
                for q in [
                    q
                    for q in queries
                    if q["split"] == split and q["label"] == label and q["true_product_in_catalog"]
                ][:2]
            )
    protocol = {
        "seed": 6227,
        "scope": "Fresh catalog listing identities only; no verified household examples",
        "products_downloaded": len(products),
        "eligible_products": len(eligible),
        "gallery_products": len(gallery),
        "gallery_images": sum(len(p["images"]) for p in gallery),
        "queries": queries,
        "identity_roles": roles,
        "exclusions": excluded,
        "counts": {
            s: dict(
                Counter(
                    q["label"] + ("_present" if q["true_product_in_catalog"] else "_absent")
                    for q in queries
                    if q["split"] == s
                )
            )
            for s in ["development", "heldout"]
        },
        "detector_subset_image_ids": subset,
        "retrieval": "Frozen production original resize224 RGB mean/std0.5; L2-normalized SigLIP2, max image cosine per distinct listing; no quality filtering or new ranking",
        "category_policy": "Global, actual OWLv2 strict diagnostic, soft0.08, catalog-category oracle simulated confirmation, deliberately wrong confirmed category, missing-confirmation global fallback. At most two present queries per category/split, selected before inference. Not actual user study or detection accuracy.",
        "unknown_selection": "Development only: maximize mean(known_accept, unknown_reject), ties higher known_accept then simpler threshold order. Compare max cosine thresholds [0.70,0.75,0.80,0.85,0.90,0.95] and product top1-top2 margins [0,0.005,0.01,0.02,0.04,0.08]. One locked threshold per family then heldout once. No deployment.",
        "quality_analysis": "Existing examined development data only; provisional nearest visual-reference annotation, not human ground truth. No quality-based selection on heldout.",
        "bootstrap": {"seed": 6228, "samples": 5000},
        "commercial_rights": "Official ABO CC BY4.0; license and attribution retained. No customer review scraping.",
    }
    write_json(ROOT / "fresh-protocol.json", protocol)
    write_json(ROOT / "fresh-gallery.json", {"products": gallery, "data_root": str(data.resolve())})
    write_json(
        ROOT / "detector-input.json",
        {
            "schema_version": 1,
            "images": [
                {
                    "image_id": q["image_id"],
                    "file": q["file"],
                    "rights": {
                        "allowed_for_benchmark": True,
                        "source": "Amazon.com / official ABO",
                        "license": "CC BY4.0",
                    },
                }
                for q in queries
                if q["image_id"] in subset
            ],
        },
    )
    write_json(
        ROOT / "protocol-lock.json",
        {
            "protocol_sha256": digest(ROOT / "fresh-protocol.json"),
            "gallery_sha256": digest(ROOT / "fresh-gallery.json"),
            "ranking_results_seen": False,
        },
    )
    print(
        {
            k: protocol[k]
            for k in ["eligible_products", "gallery_products", "gallery_images", "counts"]
        }
    )


if __name__ == "__main__":
    main()
