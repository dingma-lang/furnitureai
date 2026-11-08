from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .data import (
    CATEGORIES,
    annotation_is_human_reviewed,
    digest,
    load_manifest,
    read_json,
    validate_prediction,
    write_json,
)
from .geometry import iou


def in_ignored_region(prediction, regions):
    box = prediction["box"]
    area = (box[2] - box[0]) * (box[3] - box[1])
    for region in regions:
        other = region["box"]
        intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
            0, min(box[3], other[3]) - max(box[1], other[1])
        )
        if intersection / area >= 0.5:
            return True
    return False


def match_objects(predictions, objects, threshold):
    """Confidence-ordered, one-to-one, same-category matching at one IoU threshold."""
    used, matches, unmatched = set(), [], []
    for pred_index in sorted(
        range(len(predictions)), key=lambda index: predictions[index]["score"], reverse=True
    ):
        pred = predictions[pred_index]
        candidates = [
            (iou(pred["box"], obj["box"]), gt_index)
            for gt_index, obj in enumerate(objects)
            if gt_index not in used and obj["label"] == pred["label"]
        ]
        best = max(candidates, default=(0.0, -1))
        if best[1] >= 0 and best[0] >= threshold:
            used.add(best[1])
            matches.append((pred_index, best[1], best[0]))
        else:
            unmatched.append(pred_index)
    return matches, unmatched, [index for index in range(len(objects)) if index not in used]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def latency_summary(values):
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def evaluate(manifest, predictions, output=None, iou_threshold=0.5, require_human_review=False):
    if not 0 < iou_threshold <= 1:
        raise ValueError("evaluation IoU threshold must be in (0, 1]")
    rows = load_manifest(manifest, "rooms")
    if require_human_review and any(
        row.get("objects") is not None and not annotation_is_human_reviewed(row) for row in rows
    ):
        raise ValueError(
            "human-reviewed annotations required; provisional/unspecified labels remain"
        )
    run = read_json(predictions)
    if run.get("schema_version") != 1:
        raise ValueError("unsupported prediction schema")
    if run.get("room_manifest_sha256") and run["room_manifest_sha256"] != digest(manifest):
        raise ValueError("ground-truth manifest changed since inference; use the original manifest")
    records = {}
    valid_ids = {row["image_id"] for row in rows}
    for record in run.get("images", []):
        if record["image_id"] in records or record["image_id"] not in valid_ids:
            raise ValueError("duplicate or unknown prediction image_id")
        if record.get("status") not in ("ok", "partial", "error"):
            raise ValueError("unknown prediction status")
        for prediction in record.get("detections", []):
            validate_prediction(prediction)
        records[record["image_id"]] = record
    totals = Counter()
    per_category = {label: Counter() for label in CATEGORIES}
    errors = []
    latencies = {}
    retrieval_hits = retrieval_total = conditional_hits = conditional_total = 0
    absent_catalog = unannotated = exact_images = annotated = 0
    retrieval_enabled = run.get("retrieval_enabled") is True
    per_image = []
    ignored_count = 0
    annotation_status = Counter()
    catalog_ids = set(run.get("catalog_product_ids") or [])
    for row in rows:
        record = records.get(row["image_id"])
        if record is None:
            errors.append({"image_id": row["image_id"], "stage": "missing_prediction"})
            predictions_for_image = []
        else:
            if record["status"] != "ok":
                errors.append(
                    {
                        "image_id": row["image_id"],
                        "status": record["status"],
                        "errors": record.get("errors", []),
                    }
                )
            predictions_for_image = (
                record.get("detections", []) if record.get("detection_completed", False) else []
            )
            for name, seconds in record.get("timing_seconds", {}).items():
                if (
                    not isinstance(seconds, (float, int))
                    or not math.isfinite(seconds)
                    or seconds < 0
                ):
                    raise ValueError("latencies must be finite nonnegative seconds")
                latencies.setdefault(name, []).append(seconds)
        objects = row.get("objects")
        if objects is None:
            unannotated += 1
            continue
        annotated += 1
        annotation_status[row.get("annotation", {}).get("status", "unspecified")] += 1
        matches, false_positives, missed = match_objects(
            predictions_for_image, objects, iou_threshold
        )
        ignored = [
            i
            for i in false_positives
            if in_ignored_region(predictions_for_image[i], row.get("ignore_regions", []))
        ]
        ignored_count += len(ignored)
        false_positives = [i for i in false_positives if i not in ignored]
        confusions = []
        for index in false_positives:
            pred = predictions_for_image[index]
            for obj in objects:
                if pred["label"] != obj["label"] and iou(pred["box"], obj["box"]) >= iou_threshold:
                    confusions.append(
                        {
                            "prediction_index": index,
                            "predicted_label": pred["label"],
                            "object_id": obj["object_id"],
                            "reference_label": obj["label"],
                        }
                    )
        per_image.append(
            {
                "image_id": row["image_id"],
                "true_positives": len(matches),
                "false_positives": len(false_positives),
                "false_negatives": len(missed),
                "matched_pairs": [
                    {"prediction_index": p, "object_id": objects[g]["object_id"], "iou": overlap}
                    for p, g, overlap in matches
                ],
                "false_positive_indices": false_positives,
                "ignored_indices": ignored,
                "missed_objects": [objects[i] for i in missed],
                "label_confusions": confusions,
            }
        )
        totals.update(tp=len(matches), fp=len(false_positives), fn=len(missed))
        if (
            not false_positives
            and not missed
            and record is not None
            and record.get("detection_completed", False)
        ):
            exact_images += 1
        for pred_index, _, _ in matches:
            per_category[predictions_for_image[pred_index]["label"]]["tp"] += 1
        for index in false_positives:
            per_category[predictions_for_image[index]["label"]]["fp"] += 1
        for index in missed:
            per_category[objects[index]["label"]]["fn"] += 1
        if retrieval_enabled:
            matched_gt = {gt_index: pred_index for pred_index, gt_index, _ in matches}
            for gt_index, obj in enumerate(objects):
                if obj.get("identity_verified") is not True:
                    continue
                if obj["product_id"] not in catalog_ids:
                    absent_catalog += 1
                    continue
                retrieval_total += 1  # Includes missed detections and processing failures.
                if gt_index not in matched_gt:
                    continue
                conditional_total += 1
                pred = predictions_for_image[matched_gt[gt_index]]
                top5 = [candidate["product_id"] for candidate in pred.get("matches", [])[:5]]
                hit = obj["product_id"] in top5
                retrieval_hits += int(hit)
                conditional_hits += int(hit)

    def metrics(counts):
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "detection_accuracy_jaccard": ratio(tp, tp + fp + fn),
        }

    report = {
        "schema_version": 1,
        "kind": run.get("kind", "offline_predictions"),
        "accuracy_claim": "Metrics against supplied annotations only; no general model accuracy claim.",
        "annotation_status_counts": dict(annotation_status),
        "human_review_required": require_human_review,
        "ignored_predictions": ignored_count,
        "per_image": per_image,
        "input_images": len(rows),
        "annotated_images": annotated,
        "unannotated_images": unannotated,
        "iou_threshold": iou_threshold,
        "matching": "confidence-ordered, one-to-one, same-category",
        "detection": metrics(totals),
        "image_exact_match_rate": ratio(exact_images, annotated),
        "per_category": {label: metrics(counts) for label, counts in per_category.items()},
        "retrieval": {
            "enabled": retrieval_enabled,
            "verified_in_catalog_objects": retrieval_total,
            "hits_at_5": retrieval_hits,
            "recall_at_5_end_to_end": ratio(retrieval_hits, retrieval_total),
            "matched_verified_objects": conditional_total,
            "recall_at_5_given_detection": ratio(conditional_hits, conditional_total),
            "verified_objects_absent_from_catalog": absent_catalog,
        },
        "latency_seconds": {key: latency_summary(values) for key, values in latencies.items()},
        "processing_failures": {
            "count": len(errors),
            "rate": ratio(len(errors), len(rows)),
            "details": errors,
        },
        "manifest_sha256": digest(manifest),
        "predictions_sha256": digest(predictions),
        "environment": run.get("environment"),
        "detector": run.get("detector"),
        "embedding": run.get("embedding"),
        "segmentation": run.get("segmentation"),
    }
    if output:
        write_json(output, report)
    return report
