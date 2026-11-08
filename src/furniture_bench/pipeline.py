from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

from .data import digest, load_image, load_manifest, write_json
from .geometry import crop_bounds
from .runtime import environment
from .visualization import annotate


def run_experiment(manifest, output, detector, embedder=None, catalog=None, segmenter=None):
    """One record per input, including failures. No substitute/fallback detections."""
    if (embedder is None) != (catalog is None):
        raise ValueError("retrieval requires both embedder and catalog")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("experiment output already exists; choose a new directory")
    rows = load_manifest(manifest, "rooms")
    output.mkdir(parents=True)
    records = []
    for number, row in enumerate(rows):
        started = perf_counter()
        token = hashlib.sha256(row["image_id"].encode()).hexdigest()[:12]
        folder = output / f"{number:04d}_{token}"
        folder.mkdir()
        record = {
            "image_id": row["image_id"],
            "file": row["resolved_file"],
            "status": "error",
            "detections": [],
            "errors": [],
            "timing_seconds": {},
        }
        stage = "image_load"
        detection_completed = False
        try:
            image = load_image(row["resolved_file"])
            record.update(
                {
                    "image_sha256": digest(row["resolved_file"]),
                    "width": image.width,
                    "height": image.height,
                }
            )
            # Ground-truth coordinates must refer to the displayed (EXIF-corrected) image.
            for obj in row.get("objects") or []:
                if obj["box"][2] > image.width or obj["box"][3] > image.height:
                    raise ValueError("ground-truth box extends outside oriented image")
            stage = "owlv2"
            detections, seconds = detector.detect(image)
            record["timing_seconds"]["owlv2"] = seconds
            record["detections"] = detections
            if getattr(detector, "last_candidates", None) is not None:
                record["raw_candidates"] = detector.last_candidates
                record["filter_audit"] = detector.last_filter_report
            detection_completed = True
            stage = "artifacts"
            crops = []
            for index, detection in enumerate(detections):
                detection["detection_id"] = f"d{index:04d}"
                bounds = crop_bounds(detection["box"], image.width, image.height)
                crop = image.crop(bounds)
                crop_file = folder / f"{detection['detection_id']}.png"
                crop.save(crop_file)
                detection.update(
                    {
                        "crop_file": str(crop_file.relative_to(output)),
                        "crop_box": list(bounds),
                        "matches": [],
                    }
                )
                crops.append(crop)
            annotated = folder / "annotated.png"
            annotate(image, detections).save(annotated)
            record["annotated_file"] = str(annotated.relative_to(output))
            if segmenter is not None and detections:
                stage = "sam3.1"
                try:
                    masks, seconds = segmenter.segment(image, detections)
                    record["timing_seconds"]["sam3.1"] = seconds
                    for index, mask in masks.items():
                        mask_file = folder / f"d{index:04d}_mask.png"
                        mask.save(mask_file)
                        detections[index]["mask_file"] = str(mask_file.relative_to(output))
                except Exception as exc:
                    # Optional segmentation failure must not prevent crop-based retrieval.
                    record["errors"].append(
                        {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
                    )
            if embedder is not None and crops:
                stage = "siglip2"
                vectors, seconds = embedder.encode(crops)
                record["timing_seconds"]["siglip2"] = seconds
                catalog.assert_compatible(embedder.metadata())
                stage = "faiss"
                matches, seconds = catalog.search(vectors, k=5)
                record["timing_seconds"]["faiss"] = seconds
                for detection, candidates in zip(detections, matches):
                    detection["matches"] = candidates
            record["status"] = "partial" if record["errors"] else "ok"
        except Exception as exc:
            record["status"] = "partial" if detection_completed else "error"
            record["errors"].append(
                {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
            )
        record["detection_completed"] = detection_completed
        record["timing_seconds"]["total"] = perf_counter() - started
        records.append(record)
        # Persist after each photograph so earlier results survive an interrupted run.
        write_json(output / "predictions.json", {"schema_version": 1, "images": records})
    run = {
        "schema_version": 1,
        "kind": "model_experiment",
        "environment": environment(),
        "room_manifest_sha256": digest(manifest),
        "detector": detector.metadata(),
        "embedding": embedder.metadata() if embedder else None,
        "segmentation": segmenter.metadata() if segmenter else None,
        "retrieval_enabled": catalog is not None,
        "catalog_manifest_sha256": catalog.metadata.get("catalog_manifest_sha256")
        if catalog
        else None,
        "catalog_product_ids": [p["product_id"] for p in catalog.metadata["products"]]
        if catalog
        else None,
        "images": records,
    }
    write_json(output / "predictions.json", run)
    return run
