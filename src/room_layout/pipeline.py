from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import ExifTags, Image

from furniture_bench.data import digest, load_image, load_manifest, read_json, write_json

from .geometry import extract_structure
from .scene import photo_scene
from .visualization import depth_image, floorplan_status, structural_overlay


def _cached_records(folders, rows, stage, model):
    records, reports = {}, []
    source_rows = {r["image_id"]: r for r in rows}
    for folder in folders:
        folder = Path(folder).resolve()
        report = read_json(folder / "report.json")
        if (
            report.get("stage") != stage
            or Path(report["model"]["path"]).resolve() != Path(model).resolve()
        ):
            raise ValueError("cache stage or checkpoint differs from the requested model")
        reports.append({"cache": str(folder), **report})
        for record in report["images"]:
            if record["image_id"] not in source_rows or record["status"] != "ok":
                continue
            if record["image_sha256"] != digest(source_rows[record["image_id"]]["resolved_file"]):
                raise ValueError("cached prediction image bytes changed")
            records[record["image_id"]] = {**record, "array_path": str(folder / record["file"])}
    return records, reports


def _stage(stage, model, manifest, rows, output, reused, threads):
    records, reports = _cached_records(reused, rows, stage, model)
    missing = [r for r in rows if r["image_id"] not in records]
    if missing:
        stage_manifest = output / f"{stage}-input.json"
        clean = [
            {k: v for k, v in r.items() if k != "resolved_file"} | {"file": r["resolved_file"]}
            for r in missing
        ]
        write_json(stage_manifest, {"schema_version": 1, "images": clean})
        cache = output / f"{stage}-cache"
        command = [
            sys.executable,
            "-u",
            "-m",
            "room_layout.worker",
            "--stage",
            stage,
            "--model",
            str(model),
            "--manifest",
            str(stage_manifest),
            "--output",
            str(cache),
            "--threads",
            str(threads),
        ]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (output / f"{stage}.log").open("w") as log:
            completed = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if (cache / "report.json").exists():
            fresh, fresh_reports = _cached_records([cache], rows, stage, model)
            records.update(fresh)
            reports.extend(fresh_reports)
        else:
            reports.append(
                {
                    "stage": stage,
                    "process_failure": completed.returncode,
                    "error": "Worker ended without a report; inspect native-crash log. No artificial geometry substituted.",
                }
            )
    return records, reports


def camera_metadata(path):
    with Image.open(path) as image:
        raw = image.getexif()
        wanted = {
            "Make",
            "Model",
            "FocalLength",
            "FocalLengthIn35mmFilm",
            "Orientation",
            "ExifImageWidth",
            "ExifImageHeight",
        }
        metadata = {
            ExifTags.TAGS.get(k, str(k)): str(v)
            for k, v in raw.items()
            if ExifTags.TAGS.get(k) in wanted
        }
        if 34665 in raw:
            for k, v in raw.get_ifd(34665).items():
                if ExifTags.TAGS.get(k) in wanted:
                    metadata[ExifTags.TAGS[k]] = str(v)
    return {
        "exif": metadata,
        "source": "original file metadata; may be absent/stripped",
        "intrinsics_inferred_from_exif": False,
        "reason": "Focal length alone, unknown sensor/crop and EXIF orientation do not establish calibrated intrinsics",
    }


def _depth_boundary_diagnostics(structure, depth):
    span = max(float(np.percentile(depth, 95) - np.percentile(depth, 5)), 1e-9)
    for line in structure["boundaries"]:
        p, q = np.asarray(line["endpoints"])
        direction = q - p
        normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-9)
        centres = np.linspace(p, q, 24)
        samples = []
        for side in (-1, 1):
            points = (centres + normal * side * 5).astype(int)
            points[:, 0] = np.clip(points[:, 0], 0, depth.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, depth.shape[0] - 1)
            samples.append(depth[points[:, 1], points[:, 0]])
        jump = float(np.median(np.abs(samples[0] - samples[1])) / span)
        line["relative_depth_diagnostic"] = {
            "normalized_local_jump": jump,
            "possible_occluding_edge_or_model_error": jump > 0.15,
            "heuristic_not_calibrated": True,
            "used_as_metric_depth": False,
        }


def run_layout(
    manifest,
    output,
    segmentation_model,
    depth_model,
    furniture_predictions=None,
    reuse_segmentation=(),
    reuse_depth=(),
    threads=4,
    min_confidence=0.55,
):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("layout experiment output exists; choose a new directory")
    if threads < 1 or not 0 < min_confidence < 1:
        raise ValueError("invalid thread count or segmentation confidence threshold")
    rows = load_manifest(manifest, "rooms")
    output.mkdir(parents=True)
    # Strictly sequential processes: segmentation exits before relative depth loads.
    segmented, seg_reports = _stage(
        "segmentation", segmentation_model, manifest, rows, output, reuse_segmentation, threads
    )
    depths, depth_reports = _stage(
        "depth", depth_model, manifest, rows, output, reuse_depth, threads
    )
    furniture = (
        {r["image_id"]: r for r in read_json(furniture_predictions)["images"]}
        if furniture_predictions
        else {}
    )
    run = {
        "schema_version": 1,
        "manifest_sha256": digest(manifest),
        "model_stages": {"segmentation": seg_reports, "depth": depth_reports},
        "images": [],
    }
    for index, row in enumerate(rows):
        folder = output / f"image-{index:03d}"
        folder.mkdir()
        record = {
            "image_id": row["image_id"],
            "status": "error",
            "role": row.get("role"),
            "errors": [],
        }
        try:
            postprocess_start = perf_counter()
            if row["image_id"] not in segmented:
                raise RuntimeError("No successful semantic model output; layout not substituted")
            image = load_image(row["resolved_file"])
            with np.load(segmented[row["image_id"]]["array_path"]) as values:
                labels, confidence = values["labels"], values["confidence"]
            if labels.shape != (image.height, image.width) or confidence.shape != labels.shape:
                raise ValueError("semantic arrays differ from source image size")
            if not np.isfinite(confidence).all():
                raise ValueError("nonfinite semantic confidence")
            id2label = next(
                report["model"]["id2label"]
                for report in seg_reports
                if report.get("model", {}).get("id2label")
            )
            prior = furniture.get(row["image_id"])
            if prior and prior.get("image_sha256") != digest(row["resolved_file"]):
                raise ValueError("furniture predictions reference different image bytes")
            hints = prior.get("detections", []) if prior else []
            structure, structural_labels = extract_structure(
                image, labels, confidence, id2label, hints, min_confidence
            )
            if row["image_id"] in depths:
                with np.load(depths[row["image_id"]]["array_path"]) as values:
                    depth = values["relative_depth"]
                np.save(folder / "relative-depth.npy", depth)
                depth_metadata = depth_image(depth, folder / "relative-depth.png")
                _depth_boundary_diagnostics(structure, depth)
            else:
                depth_metadata = {"status": "failed", "physical_units": None}
                record["errors"].append(
                    {"stage": "depth", "message": "Relative-depth inference unavailable"}
                )
            Image.fromarray(structural_labels).save(folder / "structure-labels.png")
            np.save(folder / "semantic-confidence.npy", confidence.astype(np.float16))
            structural_overlay(
                image, structural_labels, confidence, structure, folder / "layout-overlay.png"
            )
            scene = photo_scene(
                row["image_id"],
                digest(row["resolved_file"]),
                image.width,
                image.height,
                structure,
                camera_metadata(row["resolved_file"]),
                hints,
            )
            scene["views"][0]["relative_depth"] = {
                **depth_metadata,
                "file": "relative-depth.npy" if row["image_id"] in depths else None,
            }
            write_json(folder / "room-estimate.json", scene)
            write_json(folder / "structure.json", structure)
            floorplan_status(folder / "floorplan-status.png")
            record.update(
                status="partial" if record["errors"] else "ok",
                folder=str(folder.relative_to(output)),
                image_sha256=scene["views"][0]["image_sha256"],
                boundary_count=len(structure["boundaries"]),
                corner_candidate_count=len(structure["corner_candidates"]),
                inferred_boundary_count=sum(
                    s["visibility"] == "inferred_occluded" for s in structure["boundaries"]
                ),
                world_floor_plan_available=False,
                physical_accuracy_verified=False,
                structure_status=(
                    "partial_visible_structure"
                    if all(
                        any(r["surface"] == name for r in structure["regions"])
                        for name in ("wall", "floor")
                    )
                    else "insufficient_floor_or_wall_evidence"
                ),
                postprocessing_seconds=perf_counter() - postprocess_start,
            )
        except Exception as exc:
            record["errors"].append(
                {"stage": "structure", "type": type(exc).__name__, "message": str(exc)}
            )
        run["images"].append(record)
        write_json(output / "run.json", run)
    return run
