"""Real cached SigLIP2 inference on frozen fresh data; no ranking evaluation."""

import hashlib
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from furniture_bench.data import digest, load_image, read_json, write_json
from furniture_bench.embedding.siglip2 import Siglip2Embedder
from furniture_bench.retrieval_experiments import prepare_image


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    output = root / "embeddings"
    if output.exists():
        raise ValueError("Preserve inference; choose new output")
    output.mkdir()
    protocol = read_json(root / "protocol.json")
    lock = read_json(root / "protocol-lock.json")
    assert digest(root / "protocol.json") == lock["protocol_sha256"]
    catalog = read_json(root / "gallery.json")
    base = Path(catalog["data_root"])
    images = [base / i["file"] for p in catalog["products"] for i in p["images"]] + [
        Path(q["file"]) for q in protocol["queries"]
    ]
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
    cache = {}
    report = {
        "variants": {},
        "gallery_images": sum(len(p["images"]) for p in catalog["products"]),
        "queries": len(protocol["queries"]),
        "protocol_sha256": digest(root / "protocol.json"),
    }
    start = perf_counter()
    for mode in ["original", "white_margin_trim", "margin10", "letterbox"]:
        values = [prepare_image(load_image(p), mode) for p in images]
        keys = [hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest() for im in values]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))
        lookup = {k: im for k, im in zip(keys, values)}
        seconds = 0
        for offset in range(0, len(missing), 8):
            batch = missing[offset : offset + 8]
            vectors, t = model.encode([lookup[k] for k in batch])
            seconds += t
            for k, v in zip(batch, vectors):
                cache[k] = v
        vectors = np.stack([cache[k] for k in keys])
        np.save(output / (mode + ".npy"), vectors)
        report["variants"][mode] = {
            "images": len(values),
            "new_image_encodings": len(missing),
            "reused_identical_pixel_encodings": len(values) - len(missing),
            "inference_seconds": seconds,
            "embedding_norm_range": [
                float(np.linalg.norm(vectors, axis=1).min()),
                float(np.linalg.norm(vectors, axis=1).max()),
            ],
        }
        for i in range(min(12, len(values))):
            folder = output / "preprocessing-examples" / mode
            folder.mkdir(parents=True, exist_ok=True)
            values[i].save(folder / f"{i:03d}.png")
        print(mode, report["variants"][mode], flush=True)
    report.update(
        model=model.metadata(),
        total_seconds=perf_counter() - start,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        ranking_results_computed=False,
    )
    write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
