"""Reuse identity-aligned cached SigLIP features; infer only genuinely missing images."""

import resource
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, load_image, read_json, write_json
from furniture_bench.embedding.siglip2 import Siglip2Embedder

ROOT = Path("reports/phase6d2-data-quality")


def main():
    if (ROOT / "image-vectors.npy").exists():
        raise ValueError("Saved features already exist; reuse them")
    protocol_lock = read_json(ROOT / "protocol-lock.json")
    assert digest(ROOT / "fresh-protocol.json") == protocol_lock["protocol_sha256"]
    cache = {}
    old = Path("reports/phase6d-product-identification/benchmark-validated")
    gallery = read_json("data/catalogs/abo-phase6d/catalog.json")["products"]
    gallery_images = [i for p in gallery for i in p["images"]]
    for row, vector in zip(gallery_images, np.load(old / "gallery-embeddings.npy")):
        cache[row["image_id"]] = (row["sha256"], vector)
    for row, vector in zip(
        read_json(old / "rankings.json")["queries"], np.load(old / "query-embeddings.npy")
    ):
        cache[row["image_id"]] = (row["sha256"], vector)
    prior = Path("reports/phase6d1-retrieval-optimization")
    records = [i for p in read_json(prior / "gallery.json")["products"] for i in p["images"]]
    records += read_json(prior / "protocol.json")["queries"]
    for row, vector in zip(records, np.load(prior / "embeddings/original.npy")):
        cache[row["image_id"]] = (row["sha256"], vector)
    records = []
    for base in [
        Path("data/catalogs/abo-phase6d"),
        Path("data/catalogs/abo-phase6d1"),
        ROOT / "fresh-catalog",
    ]:
        for product in read_json(base / "selected-products.json")["products"]:
            for image in product["images"]:
                path = (base / image["file"]).resolve()
                assert digest(path) == image["sha256"]
                records.append(
                    {
                        **image,
                        "file": str(path),
                        "product_id": product["product_id"],
                        "label": product["label"],
                        "dataset": base.name,
                    }
                )
    assert len({r["image_id"] for r in records}) == len(records)
    missing = [r for r in records if r["image_id"] not in cache]
    from furniture_bench.runtime import get_torch

    torch = get_torch()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = Siglip2Embedder(
        str(Path(".cache/models/siglip2-base-75de2d55").resolve()),
        device="cpu",
        allow_download=False,
        revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        batch_size=8,
    )
    started = perf_counter()
    inference_seconds = 0
    for start in range(0, len(missing), 8):
        batch = missing[start : start + 8]
        vectors, elapsed = model.encode([load_image(r["file"]) for r in batch])
        inference_seconds += elapsed
        for row, vector in zip(batch, vectors):
            cache[row["image_id"]] = (row["sha256"], vector)
    for row in records:
        assert cache[row["image_id"]][0] == row["sha256"]
    vectors = np.stack([cache[r["image_id"]][1] for r in records])
    assert np.isfinite(vectors).all()
    np.save(ROOT / "image-vectors.npy", vectors)
    write_json(ROOT / "vector-image-rows.json", {"images": records})
    write_json(
        ROOT / "inference-report.json",
        {
            "images": len(records),
            "reused_embeddings": len(records) - len(missing),
            "new_encodings": len(missing),
            "inference_seconds": inference_seconds,
            "wall_seconds": perf_counter() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "model": model.metadata(),
            "protocol_sha256": protocol_lock["protocol_sha256"],
            "rankings_computed": False,
        },
    )
    print(
        {
            "images": len(records),
            "new_encodings": len(missing),
            "inference_seconds": inference_seconds,
        }
    )


if __name__ == "__main__":
    main()
