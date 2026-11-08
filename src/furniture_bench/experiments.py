"""Model-free comparisons of filter settings over the same captured candidates."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

from .data import annotation_is_human_reviewed, digest, load_manifest, read_json, write_json
from .evaluation import evaluate
from .postprocessing import FilterConfig, filter_detections
from .visualization import visualize


def rebind_capture(manifest, capture_path, output, reason):
    """Explicitly evaluate new labels on the same image bytes without rerunning models."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("annotation rebind requires a reason")
    output = Path(output)
    if output.exists():
        raise ValueError("rebound capture output already exists")
    rows = load_manifest(manifest, "rooms")
    capture = read_json(capture_path)
    records = capture.get("images", [])
    by_id = {record["image_id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {row["image_id"] for row in rows}:
        raise ValueError("annotation update must preserve the exact image ID set")
    for row in rows:
        record = by_id[row["image_id"]]
        if record.get("image_sha256") != digest(row["resolved_file"]):
            raise ValueError("annotation update cannot substitute changed image bytes")
        record["file"] = row["resolved_file"]
    capture["annotation_rebind"] = {
        "from_manifest_sha256": capture.get("room_manifest_sha256"),
        "to_manifest_sha256": digest(manifest),
        "source_capture_sha256": digest(capture_path),
        "reason": reason,
        "image_bytes_verified": True,
        "inference_repeated": False,
    }
    capture["room_manifest_sha256"] = digest(manifest)
    write_json(output, capture)
    return capture


def compare_filters(manifest, capture_path, configurations, output, require_human_review=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("comparison output exists; choose a new directory")
    capture = read_json(capture_path)
    if capture.get("room_manifest_sha256") != digest(manifest):
        raise ValueError("capture manifest fingerprint differs")
    rows = load_manifest(manifest, "rooms")
    if require_human_review and any(
        r.get("objects") is not None and not annotation_is_human_reviewed(r) for r in rows
    ):
        raise ValueError("human-reviewed annotations are required")
    original_config = FilterConfig.from_dict(capture["detector"]["filter_config"])
    floor = original_config.candidate_floor
    configs = []
    for name, values in configurations.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("configuration names must be simple directory names")
        config = FilterConfig.from_dict(values)
        if config.candidate_floor < floor:
            raise ValueError("requested threshold is below the captured candidate floor; recapture")
        configs.append((name, config))
    if not configs:
        raise ValueError("at least one filter configuration is required")
    output.mkdir(parents=True)
    summaries = []
    source_rows = {r["image_id"]: r for r in rows}
    for name, config in configs:
        run = copy.deepcopy(capture)
        run["filter_experiment"] = {
            "capture_sha256": digest(capture_path),
            "filter_config": config.to_dict(),
            "inference_repeated": False,
        }
        run["detector"]["filter_config"] = config.to_dict()
        run["detector"]["threshold"] = config.threshold
        run["detector"]["nms_iou"] = config.nms_iou
        folder = output / name
        folder.mkdir()
        for record in run["images"]:
            if not record.get("detection_completed"):
                continue
            if "raw_candidates" not in record:
                raise ValueError(
                    "capture lacks pre-filter candidates; rerun detection with current adapter"
                )
            filtered, audit = filter_detections(record["raw_candidates"], config)
            for index, detection in enumerate(filtered):
                detection["detection_id"] = f"d{index:04d}"
            record["detections"] = filtered
            record["filter_audit"] = audit
            # Old capture crops belong to a different detection list. Export annotations only.
            record.pop("annotated_file", None)
            relative = Path(f"image-{list(source_rows).index(record['image_id']):03d}.png")
            visualize(source_rows[record["image_id"]]["resolved_file"], filtered, folder / relative)
            record["annotated_file"] = str(relative)
        predictions = folder / "predictions.json"
        write_json(predictions, run)
        report = evaluate(
            manifest,
            predictions,
            folder / "evaluation.json",
            require_human_review=require_human_review,
        )
        report["timing_note"] = (
            "Forward latency is inherited from the one shared capture, not remeasured per filter setting."
        )
        write_json(folder / "evaluation.json", report)
        summaries.append(
            {
                "configuration": name,
                **report["detection"],
                "ignored_predictions": report["ignored_predictions"],
                "processing_failures": report["processing_failures"]["count"],
            }
        )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(
        output / "comparison.json",
        {
            "kind": "exploratory_filter_comparison",
            "manifest_sha256": digest(manifest),
            "capture_sha256": digest(capture_path),
            "human_review_required": require_human_review,
            "note": "Small convenience sample; filter comparisons are not an independent test-set accuracy claim.",
            "configurations": summaries,
        },
    )
    return summaries
