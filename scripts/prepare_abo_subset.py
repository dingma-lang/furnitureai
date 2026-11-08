"""Bounded official ABO furniture subset; no retailer scraping or full archives."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from furniture_bench.data import digest, write_json

BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
TYPES = {"SOFA": "sofa", "CHAIR": "chair", "TABLE": "table", "DESK": "desk", "BED": "bed", "CABINET": "cabinet", "SHELF": "cabinet", "LAMP": "lamp"}


def english(values):
    if not values:
        return None
    return next((v["value"] for v in values if v.get("language_tag") == "en_US"), next((v["value"] for v in values if v.get("language_tag", "").startswith("en")), values[0].get("value")))


def fingerprint(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image.resize((32, 32)), dtype=np.int16)
        gray = np.asarray(image.convert("L").resize((9, 8)))
    bits = gray[:, 1:] > gray[:, :-1]
    return {"sha256": digest(path), "rgb": rgb, "dhash": bits}


def near_duplicate(a, b):
    return a["sha256"] == b["sha256"] or (np.count_nonzero(a["dhash"] != b["dhash"]) <= 6 and float(np.mean(np.abs(a["rgb"] - b["rgb"]))) < 18)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listings", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=140)
    p.add_argument("--exclude-manifest", help="Exclude identities and source image IDs from a prior subset")
    args = p.parse_args()
    excluded_products, excluded_images = set(), set()
    if args.exclude_manifest:
        previous = json.loads(Path(args.exclude_manifest).read_text())["products"]
        excluded_products = {p["product_id"] for p in previous}
        excluded_images = {i["image_id"] for p in previous for i in p["images"]}
    output = Path(args.output)
    if (output / "catalog.json").exists():
        raise ValueError("Preserve existing selection; output catalog already exists")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    with gzip.open(args.images, "rt") as stream:
        mapping = {r["image_id"]: r for r in csv.DictReader(stream)}
    groups = defaultdict(list)
    with gzip.open(args.listings, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            category = TYPES.get(row.get("product_type", [{}])[0].get("value"))
            name = english(row.get("item_name")) or ""
            ids = list(dict.fromkeys([row.get("main_image_id")] + row.get("other_image_id", [])))
            ids = [i for i in ids if i in mapping and min(int(mapping[i]["width"]), int(mapping[i]["height"])) >= 128]
            ids = [i for i in ids if i not in excluded_images]
            if row["item_id"] not in excluded_products and category and len(ids) >= 3 and not any(s in name.lower() for s in ("swatch", "replacement cover", "cover only", "slipcover")):
                groups[category].append((row, ids))
    # Deterministic category interleaving, with no embedding/model-based selection.
    ordered = [r for chunk in itertools.zip_longest(*(sorted(v, key=lambda r: r[0]["item_id"]) for _, v in sorted(groups.items()))) for r in chunk if r]
    selected, rejected, fingerprints, used_image_ids = [], [], [], set()
    downloaded = 0
    for row, ids in ordered:
        chosen, local = [], []
        for image_id in ids[:6]:
            if image_id in used_image_ids:
                continue
            record = mapping[image_id]
            source = BASE + "images/small/" + record["path"]
            path = output / "images" / (hashlib.sha256(image_id.encode()).hexdigest()[:20] + Path(record["path"]).suffix)
            try:
                if not path.exists():
                    if downloaded >= 30_000_000:
                        raise ValueError("30 MB image download budget exhausted")
                    subprocess.run(["curl", "-fsSL", "--max-time", "30", "--max-filesize", "1000000", "-o", str(path), source], check=True)
                    downloaded += path.stat().st_size
                mark = fingerprint(path)
                if any(near_duplicate(mark, prior) for prior in fingerprints + local):
                    rejected.append({"image_id": image_id, "product_id": row["item_id"], "reason": "exact_or_conservative_near_duplicate"})
                    continue
                chosen.append({"image_id": image_id, "file": str(path.relative_to(output)), "source_url": source, "sha256": mark["sha256"], "original_dimensions": [int(record["width"]), int(record["height"])], "variant": "official ABO max256px image"})
                local.append(mark)
                if len(chosen) == 3:
                    break
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                rejected.append({"image_id": image_id, "reason": str(exc)})
        if len(chosen) != 3:
            continue
        fingerprints.extend(local)
        used_image_ids.update(x["image_id"] for x in chosen)
        category = TYPES[row["product_type"][0]["value"]]
        product = {"product_id": row["item_id"], "domain_name": row.get("domain_name"), "label": category, "name": english(row.get("item_name")), "brand": english(row.get("brand")), "manufacturer": english(row.get("manufacturer")), "model_number": english(row.get("model_number")), "retailer": "Amazon", "price": None, "current_availability": None, "specifications": {k: row.get(k) for k in ("item_dimensions", "material", "color", "style", "model_number")}, "source_record": row, "images": chosen, "rights": {"source": "Amazon.com / Amazon Berkeley Objects; Guillaumin et al. and Collins et al.", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "allowed_for_benchmark": True, "modifications": "Official downscaled images; no synthetic modification to source files"}}
        selected.append(product)
        print(len(selected), category, row["item_id"], flush=True)
        if len(selected) == args.count:
            break
    if len(selected) < 100:
        raise ValueError(f"Only {len(selected)} independent products; inspect diagnostics")
    # Every seventh identity is withheld entirely; no absent-product photo in gallery.
    gallery, queries = [], []
    for index, product in enumerate(selected):
        absent = index % 7 == 0
        if not absent:
            gallery.append({**product, "images": product["images"][:2]})
        queries.append({"product_id": product["product_id"], "label": product["label"], "file": product["images"][2]["file"], "image_id": product["images"][2]["image_id"], "sha256": product["images"][2]["sha256"], "true_product_in_catalog": not absent})
    write_json(output / "catalog.json", {"schema_version": 2, "dataset": "ABO", "products": gallery, "retail_inventory_snapshot": False})
    write_json(output / "queries.json", {"schema_version": 1, "queries": queries})
    write_json(output / "selected-products.json", {"products": selected})
    write_json(output / "selection.json", {"source_listings": BASE + "listings/metadata/listings_0.json.gz", "source_listings_sha256": digest(args.listings), "source_image_metadata_sha256": digest(args.images), "selected_products": len(selected), "gallery_products": len(gallery), "query_images": len(queries), "absent_query_products": sum(not q["true_product_in_catalog"] for q in queries), "categories": dict(Counter(p["label"] for p in selected)), "downloaded_image_bytes_this_run": downloaded, "selection": "First valid product IDs interleaved by category from official shard 0; maximum six candidate photos inspected per identity; no retrieval tuning", "deduplication": "Across all selected images: identical bytes OR 64-bit dHash Hamming <=6 AND 32x32 RGB mean absolute difference <18. Conservative heuristic, not proof of different physical capture.", "rejections": rejected, "benchmark_scope": "Different catalog photographs of dataset listing identities; not household used-furniture recognition", "license": "CC BY 4.0; all assets per official 2023 update"})
    shutil.copyfile(args.listings, output / "listings_0.json.gz")
    shutil.copyfile(args.images, output / "images.csv.gz")


if __name__ == "__main__":
    main()
