"""On-demand SigLIP 2 query worker using the existing embedding and FAISS adapters."""
import argparse
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import load_image, read_json, write_json
from furniture_bench.embedding.siglip2 import Siglip2Embedder
from furniture_bench.product_catalog import MultiImageCatalog


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    request = read_json(a.request)
    from furniture_bench.runtime import get_torch
    torch = get_torch()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    embedder = Siglip2Embedder(str(Path(request["model"]).resolve()), device="cpu", allow_download=False, revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2")
    embedding_cache = Path(request["embedding_cache"])
    if (embedding_cache / "vector.npy").exists() and (embedding_cache / "metadata.json").exists():
        vectors = np.load(embedding_cache / "vector.npy", allow_pickle=False)
        embedding_metadata = read_json(embedding_cache / "metadata.json")
        seconds, embedding_cached = 0.0, True
    else:
        vectors, seconds = embedder.encode([load_image(request["image"])])
        embedding_metadata = embedder.metadata()
        embedding_cache.mkdir(parents=True, exist_ok=True)
        np.save(embedding_cache / "vector.npy", vectors)
        write_json(embedding_cache / "metadata.json", embedding_metadata)
        embedding_cached = False
    catalog = MultiImageCatalog.load(request["index"])
    try:
        catalog.index.assert_compatible(embedding_metadata)
        candidates, search_seconds = catalog.search(vectors, request.get("preferred_retailers", []), request.get("retailer_only", False), request.get("category"), 5)
        write_json(a.output, {"candidates": candidates[0], "exact_identification": "No verified exact match.", "embedding_seconds": seconds, "query_embedding_cached": embedding_cached, "search_seconds": search_seconds, "total_seconds": perf_counter()-start, "model": {"name": "google/siglip2-base-patch16-224", "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", "feature": embedding_metadata["feature"]}, "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024), "confidence_note": "Cosine similarity is not an exact-identity probability; static ABO catalog, no current prices or stock"})
    finally:
        catalog.close()


if __name__ == "__main__":
    main()
