"""Metrics only against explicitly human-reviewed layout annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from furniture_bench.data import (
    annotation_is_human_reviewed,
    digest,
    load_manifest,
    read_json,
    write_json,
)


def mask_counts(prediction, reference, code):
    valid = reference != 255
    p, r = (prediction == code) & valid, (reference == code) & valid
    return int((p & r).sum()), int((p | r).sum())


def boundary_metrics(predicted, reference, size, tolerance=5):
    masks = []
    for lines in (predicted, reference):
        mask = Image.new("1", size)
        draw = ImageDraw.Draw(mask)
        for points in lines:
            draw.line([tuple(point) for point in points], fill=1, width=1)
        masks.append(np.asarray(mask, dtype=bool))
    p, r = masks
    if not p.any() or not r.any():
        return {
            "symmetric_alignment_error_pixels": None,
            "precision_within_tolerance": 0.0 if p.any() else None,
            "recall_within_tolerance": 0.0 if r.any() else None,
            "predicted_pixels": int(p.sum()),
            "reference_pixels": int(r.sum()),
        }
    to_reference = ndimage.distance_transform_edt(~r)[p]
    to_prediction = ndimage.distance_transform_edt(~p)[r]
    return {
        "symmetric_alignment_error_pixels": float((to_reference.mean() + to_prediction.mean()) / 2),
        "precision_within_tolerance": float(np.mean(to_reference <= tolerance)),
        "recall_within_tolerance": float(np.mean(to_prediction <= tolerance)),
        "predicted_pixels": int(p.sum()),
        "reference_pixels": int(r.sum()),
        "tolerance_pixels": tolerance,
    }


def corner_metrics(predicted, reference, tolerance=20):
    matched = []
    if predicted and reference:
        distance = np.linalg.norm(
            np.asarray(predicted)[:, None, :] - np.asarray(reference)[None, :, :], axis=2
        )
        # Maximize valid one-to-one matches before minimizing their distance.
        cost = np.where(
            distance <= tolerance, distance, (max(distance.shape) + 1) * (tolerance + 1)
        )
        rows, columns = linear_sum_assignment(cost)
        matched = [
            float(distance[a, b]) for a, b in zip(rows, columns) if distance[a, b] <= tolerance
        ]
    return {
        "matched": len(matched),
        "false_positives": len(predicted) - len(matched),
        "missed_reference_corners": len(reference) - len(matched),
        "matched_mean_error_pixels": float(np.mean(matched)) if matched else None,
        "recall": len(matched) / len(reference) if reference else None,
        "tolerance_pixels": tolerance,
    }


def evaluate_layout(manifest, run_path, output, require_human_review=False):
    rows = load_manifest(manifest, "rooms")
    run_path = Path(run_path).resolve()
    run = read_json(run_path)
    if run["manifest_sha256"] != digest(manifest):
        raise ValueError("evaluation manifest differs from the layout experiment")
    records = {r["image_id"]: r for r in run["images"]}
    report = {
        "schema_version": 1,
        "kind": "room_structure_evaluation",
        "images": [],
        "human_reviewed_images": 0,
        "unreviewed_images": 0,
        "processing_failures": [],
        "partial_processing": [],
        "reviewed_mask_images": 0,
        "reviewed_reference_processing_failures": 0,
        "floor_iou": None,
        "wall_iou": None,
        "physical_dimension_accuracy": None,
        "note": "No physical accuracy is calculated without measured references and independent estimates.",
    }
    totals = {"floor": [0, 0], "wall": [0, 0]}
    for row in rows:
        record = records.get(row["image_id"])
        per_image = {
            "image_id": row["image_id"],
            "floor_iou": None,
            "wall_iou": None,
            "boundary_alignment": None,
            "corners": None,
            "topology_consistency": None,
            "physical_dimension_error": None,
        }
        if record is None or record["status"] == "error":
            report["processing_failures"].append(
                {
                    "image_id": row["image_id"],
                    "errors": record.get("errors") if record else "missing record",
                }
            )
        elif record["status"] == "partial":
            report["partial_processing"].append(
                {"image_id": row["image_id"], "errors": record["errors"]}
            )
        annotation = row.get("layout_annotations")
        if not annotation or not annotation_is_human_reviewed({"annotation": annotation}):
            report["unreviewed_images"] += 1
            per_image["metric_status"] = "unavailable_no_human_reviewed_layout_reference"
            if require_human_review:
                raise ValueError("human-reviewed room-layout references are required")
            report["images"].append(per_image)
            continue
        report["human_reviewed_images"] += 1
        per_image["metric_status"] = "against_human_reviewed_image_annotations"
        failed = record is None or record["status"] == "error"
        if failed:
            per_image["metric_status"] = "processing_failed_reference_available"
            report["reviewed_reference_processing_failures"] += 1
            structure = {"boundaries": [], "corner_candidates": []}
            from furniture_bench.data import load_image

            size = load_image(row["resolved_file"]).size
        else:
            folder = run_path.parent / record["folder"]
            structure = read_json(folder / "structure.json")
            scene = read_json(folder / "room-estimate.json")
            size = tuple(scene["views"][0]["image_size_pixels"])
        if annotation.get("mask_file"):
            reference = np.asarray(
                Image.open(Path(manifest).resolve().parent / annotation["mask_file"])
            )
            prediction = (
                np.zeros((size[1], size[0]), dtype=np.uint8)
                if failed
                else np.asarray(Image.open(folder / "structure-labels.png"))
            )
            report["reviewed_mask_images"] += 1
            if reference.shape != prediction.shape or reference.ndim != 2:
                raise ValueError("layout reference mask dimensions differ from source prediction")
            if not set(np.unique(reference)).issubset({0, 1, 2, 3, 4, 5, 6, 255}):
                raise ValueError("unknown reference-mask code")
            for name, code in (("floor", 2), ("wall", 1)):
                intersection, union = mask_counts(prediction, reference, code)
                per_image[name + "_iou"] = intersection / union if union else None
                totals[name][0] += intersection
                totals[name][1] += union
        if "boundaries" in annotation:
            per_image["boundary_alignment"] = {}
            for kind in ("wall_floor", "wall_ceiling", "wall_wall"):
                reference = [b["points"] for b in annotation["boundaries"] if b["kind"] == kind]
                predicted = [
                    b["endpoints"]
                    for b in structure["boundaries"]
                    if b["kind"] == kind and b["visibility"] == "visible_evidence"
                ]
                per_image["boundary_alignment"][kind] = boundary_metrics(predicted, reference, size)
        if "corners" in annotation:
            per_image["corners"] = corner_metrics(
                [c["pixel"] for c in structure["corner_candidates"]],
                [c["pixel"] for c in annotation["corners"]],
            )
        per_image["topology_status"] = (
            "unresolved_world_topology_not_compared_to_an_invented_rectangle"
        )
        per_image["dimension_status"] = "unknown_no_automatic_metric_dimension_estimate"
        report["images"].append(per_image)
    for name, (intersection, union) in totals.items():
        report[name + "_iou"] = intersection / union if union else None
    report["processing_failure_count"] = len(report["processing_failures"])
    write_json(output, report)
    return report
