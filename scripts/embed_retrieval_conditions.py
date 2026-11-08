"""Actual SigLIP inference for actual OWLv2 crops, marked synthetic perturbations and room-photo crops."""

import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from benchmark_abo_products import transform
from PIL import Image

from furniture_bench.data import load_image, read_json, write_json
from furniture_bench.embedding.siglip2 import Siglip2Embedder


def main():
    root = Path("reports/phase6d1-retrieval-optimization")
    folder = root / "conditions"
    if folder.exists():
        raise ValueError("Preserve cached condition inference")
    folder.mkdir()
    protocol = read_json(root / "protocol.json")
    queries = {q["image_id"]: q for q in protocol["queries"]}
    detector = read_json(root / "detector-predictions/predictions.json")
    records = {r["image_id"]: r for r in detector["images"]}
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
    report = {"inference_seconds": {}, "detector_crops": []}
    start = perf_counter()

    def encode(images, name):
        vectors = []
        seconds = 0
        for i in range(0, len(images), 8):
            v, t = model.encode(images[i : i + 8])
            vectors.append(v)
            seconds += t
        np.save(folder / (name + ".npy"), np.concatenate(vectors))
        report["inference_seconds"][name] = seconds

    for mode, margin in [("detector_bbox", 0), ("detector_margin10", 0.10)]:
        images = []
        rows = []
        for key in protocol["predicted_category_query_image_ids"]:
            q = queries[key]
            image = load_image(q["file"])
            detections = records[key]["detections"]
            det = max(detections, key=lambda d: d["score"]) if detections else None
            if det:
                x1, y1, x2, y2 = det["box"]
                dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
                box = [
                    max(0, x1 - dx),
                    max(0, y1 - dy),
                    min(image.width, x2 + dx),
                    min(image.height, y2 + dy),
                ]
                image = image.crop(box)
            else:
                box = None
            images.append(image)
            rows.append(
                {
                    "image_id": key,
                    "source_box": det["box"] if det else None,
                    "applied_box": box,
                    "predicted_label": det["label"] if det else None,
                    "score": det["score"] if det else 0,
                    "fallback": det is None,
                }
            )
        encode(images, mode)
        if mode == "detector_bbox":
            report["detector_crops"] = rows
    for kind in [
        "lighting",
        "shadow",
        "occlusion",
        "color",
        "synthetic_stain",
        "synthetic_wear",
        "background_clutter",
    ]:
        images = []
        for key in protocol["predicted_category_query_image_ids"]:
            image = load_image(queries[key]["file"])
            if kind == "background_clutter":
                # Synthetic contextual surround; no real damage or room geometry claim.
                canvas = Image.new("RGB", (image.width + 60, image.height + 60), (110, 130, 90))
                canvas.paste(image, (30, 30))
                image = canvas
            else:
                image = transform(image, kind)
            images.append(image)
        encode(images, "synthetic_" + kind)
        examples = folder / "synthetic-examples" / kind
        examples.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images[:4]):
            image.save(examples / f"{i:02d}.png")
    # The actual licensed household image has no verified product identity.
    raw = read_json("reports/phase6d-product-identification/first-real-room-query.json")
    cache = Path(".local-projects/product-cache") / raw["cache_key"]
    request = read_json(cache / "request.json")
    crop = load_image(request["image"])
    images = [crop]
    report["real_room"] = {
        "source": "Phillip Goldsberry/Unsplash real room; actual confirmed OWLv2 crop; product identity unknown",
        "crop_source": "preserved Phase6D actual crop",
        "crop_sizes": [list(crop.size)],
    }
    # Use the saved normalized original photograph and exact existing reviewed box.
    project_id = read_json("reports/phase6d-product-identification/browser-first-real-tests.json")[
        "project_id"
    ]
    project = read_json(Path(".local-projects") / project_id / "project.json")
    photo = project["photos"][0]
    detection = photo["predictions"]["detections"][0]
    source = load_image(
        Path(".local-projects") / project_id / "photos" / photo["id"] / "processing.png"
    )
    x1, y1, x2, y2 = detection["box"]
    for margin in [-0.05, 0.10]:
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        images.append(
            source.crop(
                [
                    max(0, x1 - dx),
                    max(0, y1 - dy),
                    min(source.width, x2 + dx),
                    min(source.height, y2 + dy),
                ]
            )
        )
    encode(images, "real_room_crops")
    report.update(
        query_image_ids=protocol["predicted_category_query_image_ids"],
        model=model.metadata(),
        total_seconds=perf_counter() - start,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        synthetic_conditions_not_physical_damage_ground_truth=True,
    )
    write_json(folder / "report.json", report)


if __name__ == "__main__":
    main()
