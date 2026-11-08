"""Conservative crop-normalized duplicate exclusion using cached inference only."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from furniture_bench.data import read_json, write_json


def fingerprint(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    mask = np.any(np.asarray(image) < 240, axis=2)
    y, x = np.nonzero(mask)
    if len(x):
        image = image.crop((int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)))
    rgb = np.asarray(image.resize((32, 32)), float)
    coefficients = dctn(np.asarray(image.convert("L").resize((32, 32)), float), norm="ortho")[:8, :8].ravel()[1:]
    return coefficients > np.median(coefficients), rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/catalogs/abo-phase6d")
    p.add_argument("--report", default="reports/phase6d-product-identification")
    args = p.parse_args()
    data, root = Path(args.data), Path(args.report)
    catalog = read_json(data / "catalog.json")
    queries = read_json(data / "queries.json")["queries"]
    gallery = [(p["product_id"], image) for p in catalog["products"] for image in p["images"]]
    marks = [fingerprint(data / image["file"]) for _, image in gallery]
    flags = []
    for query in queries:
        h, rgb = fingerprint(data / query["file"])
        for (pid, image), (other, rr) in zip(gallery, marks):
            distance = int(np.count_nonzero(h != other))
            mae = float(np.mean(np.abs(rgb - rr)))
            if distance <= 6 and mae < 14:
                flags.append({"query_image_id": query["image_id"], "query_product_id": query["product_id"], "gallery_image_id": image["image_id"], "gallery_product_id": pid, "trimmed_phash_hamming": distance, "trimmed_rgb32_mae": mae})
    excluded = {f["query_image_id"] for f in flags}
    write_json(root / "crop-leakage-audit.json", {"method": "Remove near-white margins (<240 foreground); 63-bit DCT pHash Hamming <=6 AND RGB32 MAE<14", "flags": flags, "excluded_query_images": sorted(excluded), "selection_used_similarity_rankings": False, "note": "Conservative exclusion; no proof that remaining photos are independent household captures"})
    retained = [q for q in queries if q["image_id"] not in excluded]
    write_json(data / "queries-audited.json", {"schema_version": 1, "queries": retained, "excluded_for_near_duplicate_exports": sorted(excluded)})
    rows = read_json(root / "benchmark-validated/rankings.json")["queries"]
    present = [r for r in rows if r["true_product_in_catalog"] and r["image_id"] not in excluded]
    ranks = [r["correct_rank"] for r in present]
    result = read_json(root / "benchmark-validated/evaluation.json")
    result["unaudited_metrics"] = result["metrics"]
    result["metrics"] = {"queries": len(ranks), "recall_at_1": float(np.mean([r == 1 for r in ranks])), "recall_at_5": float(np.mean([r is not None and r <= 5 for r in ranks])), "mean_reciprocal_rank": float(np.mean([1/r if r is not None else 0 for r in ranks])), "incorrect_top1": sum(r != 1 for r in ranks), "ranks": ranks}
    result["query_images"] = len(retained)
    result["excluded_queries_after_crop_audit"] = sorted(excluded)
    sample = [q for q in queries if q["true_product_in_catalog"]][:24]
    if any(q["image_id"] in excluded for q in sample):
        raise ValueError("Robustness subset contains excluded queries; recompute its summary from saved ranks")
    result["robustness_subset_duplicate_audited"] = True
    result["new_model_inference_for_audit"] = False
    write_json(root / "evaluation-audited.json", result)
    print({k: v for k, v in result["metrics"].items() if k != "ranks"})


if __name__ == "__main__":
    main()
