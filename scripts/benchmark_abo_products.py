"""Real offline SigLIP 2 / FAISS catalog-to-catalog benchmark and robustness audit."""
import argparse
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from furniture_bench.data import digest, load_image, read_json, write_json
from furniture_bench.embedding.siglip2 import Siglip2Embedder
from furniture_bench.product_catalog import MultiImageCatalog, load_catalog_images


def transform(image, kind):
    image = image.copy()
    if kind == "lighting":
        return ImageEnhance.Brightness(image).enhance(.55)
    if kind == "color":
        return ImageEnhance.Color(image).enhance(.2)
    if kind == "shadow":
        layer = Image.new("RGBA", image.size)
        ImageDraw.Draw(layer).polygon([(0, 0), (image.width * .6, 0), (image.width * .3, image.height), (0, image.height)], fill=(0, 0, 0, 115))
        return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    if kind == "occlusion":
        ImageDraw.Draw(image).rectangle([image.width * .35, image.height * .35, image.width * .65, image.height * .85], fill=(115, 111, 102))
    elif kind == "synthetic_stain":
        layer = Image.new("RGBA", image.size)
        ImageDraw.Draw(layer).ellipse([image.width * .25, image.height * .35, image.width * .65, image.height * .65], fill=(75, 40, 15, 145))
        image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    elif kind == "synthetic_wear":
        draw = ImageDraw.Draw(image)
        for i in range(12):
            x, y = int(image.width * (.2 + i * .05)), int(image.height * (.4 + (i % 3) * .04))
            draw.line((x, y, x + 13, y + 3), fill=(215, 205, 180), width=2)
    return image


def metrics(rankings, queries):
    present = [(r, q) for r, q in zip(rankings, queries) if q["true_product_in_catalog"]]
    ranks = [next((i + 1 for i, item in enumerate(r) if item["product_id"] == q["product_id"]), None) for r, q in present]
    return {"queries": len(present), "recall_at_1": float(np.mean([r == 1 for r in ranks])) if ranks else None, "recall_at_5": float(np.mean([r is not None and r <= 5 for r in ranks])) if ranks else None, "mean_reciprocal_rank": float(np.mean([1 / r if r is not None else 0 for r in ranks])) if ranks else None, "incorrect_top1": sum(r != 1 for r in ranks), "ranks": ranks}


def encode_paths(embedder, paths):
    batches, seconds = [], 0.0
    for i in range(0, len(paths), 8):
        vectors, time = embedder.encode([load_image(path) for path in paths[i:i+8]])
        batches.append(vectors)
        seconds += time
    return np.concatenate(batches), seconds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--index-output", help="Optional trusted application index path")
    a = p.parse_args()
    data, output = Path(a.data), Path(a.output)
    if output.exists():
        raise ValueError("Use a fresh output, preserving prior benchmarks")
    output.mkdir(parents=True)
    from furniture_bench.runtime import get_torch
    torch = get_torch()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    embedder = Siglip2Embedder(str(Path(a.model).resolve()), device="cpu", allow_download=False, revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", batch_size=8)
    catalog, paths = load_catalog_images(data / "catalog.json")
    started = perf_counter()
    gallery_vectors, gallery_seconds = encode_paths(embedder, paths)
    np.save(output / "gallery-embeddings.npy", gallery_vectors)
    index = MultiImageCatalog.build(gallery_vectors, catalog, embedder.metadata(), data)
    index.save(Path(a.index_output) if a.index_output else output / "index")
    try:
        queries = read_json(data / "queries.json")["queries"]
        # Byte-disjointness independently enforced before inference.
        hashes = {digest(path) for path in paths}
        if any(q["sha256"] in hashes for q in queries):
            raise ValueError("Query/gallery image leakage")
        query_vectors, query_seconds = encode_paths(embedder, [data / q["file"] for q in queries])
        np.save(output / "query-embeddings.npy", query_vectors)
        full_rankings, search_seconds = index.search(query_vectors, k=len(catalog["products"]))
        result = metrics(full_rankings, queries)
        single = []
        for vector in query_vectors:
            t = perf_counter()
            index.search(vector[None])
            single.append(perf_counter() - t)
        write_json(output / "rankings.json", {"queries": [{**q, "candidates": ranking[:5], "correct_rank": next((i+1 for i, r in enumerate(ranking) if r["product_id"] == q["product_id"]), None)} for q, ranking in zip(queries, full_rankings)]})
        robustness = []
        sample = [(i, q) for i, q in enumerate(queries) if q["true_product_in_catalog"]][:24]
        for kind in ("lighting", "shadow", "occlusion", "color", "synthetic_stain", "synthetic_wear"):
            all_vectors, seconds = [], 0.0
            for offset in range(0, len(sample), 8):
                images = [transform(load_image(data / q["file"]), kind) for _, q in sample[offset:offset+8]]
                vectors, t = embedder.encode(images)
                all_vectors.append(vectors)
                seconds += t
                for (_, q), image in zip(sample[offset:offset+8], images):
                    folder = output / "synthetic" / kind
                    folder.mkdir(parents=True, exist_ok=True)
                    image.save(folder / (q["product_id"] + ".png"))
            rankings, _ = index.search(np.concatenate(all_vectors), k=len(catalog["products"]))
            base = [full_rankings[i] for i, _ in sample]
            robustness.append({"transformation": kind, "synthetic": True, "metrics": metrics(rankings, [q for _, q in sample]), "top1_changed_fraction": float(np.mean([r[0]["product_id"] != b[0]["product_id"] for r, b in zip(rankings, base)])), "inference_seconds": seconds, "baseline_same_subset": metrics(base, [q for _, q in sample])})
        absent = [(q, r) for q, r in zip(queries, full_rankings) if not q["true_product_in_catalog"]]
        write_json(output / "evaluation.json", {"benchmark": "catalog-to-catalog same dataset listing identity; NOT used furniture in homes", "gallery_products": len(catalog["products"]), "gallery_images": len(paths), "query_images": len(queries), "metrics": result, "robustness": robustness, "absent_products": {"queries": len(absent), "verified_exact_claims": 0, "no_verified_exact_match_messages": len(absent), "abstention_detection_accuracy": None, "note": "No calibrated open-set detector. Similar alternatives may still rank highly; every result explicitly lacks verified exact identity."}, "timing": {"load_seconds": embedder.load_seconds, "gallery_embedding_seconds": gallery_seconds, "query_embedding_seconds": query_seconds, "batch_search_seconds": search_seconds, "per_query_embedding_mean_seconds": query_seconds / len(queries), "search_single_query_median_seconds_including_IPC_and_aggregation": float(np.median(single)), "total_seconds": perf_counter()-started}, "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024), "embedding": embedder.metadata(), "catalog_sha256": digest(data / "catalog.json"), "queries_sha256": digest(data / "queries.json"), "gallery_query_duplicate_policy": read_json(data / "selection.json")["deduplication"], "camera_angle_test": "Distinct dataset catalog views only; view angles not annotated or independently quantified", "physical_dimensions_verified": False})
    finally:
        index.close()


if __name__ == "__main__":
    main()
