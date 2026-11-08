"""Serial offline model queue. Successful stages survive errors and reuse identity caches."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.data import digest, read_json
from room_layout.geometry import extract_structure
from room_layout.scene import photo_scene
from room_layout.visualization import COLORS, depth_image

from .photo_projects import atomic_json, now
from .service import ROOT

MODELS = {"detection": ROOT / ".cache/models/owlv2-base-cfd3195", "segmentation": ROOT / ".cache/models/room-layout/segmentation-876ffc56", "depth": ROOT / ".cache/models/room-layout/depth-5426e4f0"}
MODEL_NAMES = {"detection": "google/owlv2-base-patch16-ensemble", "segmentation": "OpenMMLab UPerNet ConvNeXt Tiny ADE20K", "depth": "depth-anything/Depth-Anything-V2-Small-hf"}


def configuration():
    prompts = read_json(ROOT / "examples/furniture-prompts-v2.json")
    prompts["cabinet"] += ["a photo of a bookshelf", "a photo of a bookcase"]
    checkpoints = {}
    for stage, folder in MODELS.items():
        files = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns, "sha256": digest(p) if p.suffix == ".json" else None} for p in folder.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors")} if folder.is_dir() else {}
        checkpoints[stage] = {"model": MODEL_NAMES[stage], "checkpoint": folder.name, "files": files}
    return {"adapter_version": 1, "normalization": "EXIF-RGB-PNG-longside1600-v1", "threshold": .2, "nms_iou": .5, "prompts": prompts, "checkpoints": checkpoints, "threads": 4}


class PhotoJobs:
    def __init__(self, store, runner=None):
        self.store = store
        self.cache = store.root / "inference-cache"
        self.cache.mkdir(exist_ok=True)
        self.settings = configuration()
        self.runner = runner or self.run_stage
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="room-model-queue")
        self.lock = threading.Lock()
        self.pending = set()
        # Queued work belongs to the old process; require an explicit retry after restart.
        with store.lock:
            for summary in store.listing():
                project = store.read(summary["id"])
                changed = False
                for photo in project["photos"]:
                    if photo.get("processing_state") in ("queued", "running"):
                        photo["processing_state"] = "partial" if any(s.get("state") == "complete" for s in photo.get("stages", {}).values()) else "failed"
                        photo["processing_error"] = "Previous session interrupted; retry failed stages explicitly"
                        for value in photo.get("stages", {}).values():
                            if value.get("state") in ("queued", "running"):
                                value["state"] = "interrupted"
                        changed = True
                if changed:
                    store.write(project)

    def key(self, photo):
        return hashlib.sha256(json.dumps({"image": photo["sha256"], "settings": self.settings}, sort_keys=True).encode()).hexdigest()

    def submit(self, project_id, photo_id, retry=False):
        with self.lock, self.store.lock:
            project = self.store.read(project_id)
            photo = self.store.photo(project, photo_id)
            if photo.get("deleted"):
                raise ValueError("This photograph was removed")
            token = (project_id, photo_id)
            if token in self.pending:
                return self.store.public(project)
            if photo.get("processing_state") == "complete" and photo.get("cache_key") == self.key(photo):
                return self.store.public(project)
            if photo.get("processing_state") in ("partial", "failed") and not retry:
                raise ValueError("Some stages failed. Use Retry failed stages after reviewing the error")
            photo["processing_state"] = "queued"
            self.store.write(project)
            self.pending.add(token)
            self.pool.submit(self.execute, project_id, photo_id, retry)
            return self.store.public(project)

    def status(self, project_id, photo_id, stage, value):
        with self.store.lock:
            project = self.store.read(project_id)
            photo = self.store.photo(project, photo_id)
            photo["stages"][stage] = value
            photo["processing_state"] = "running"
            self.store.write(project)

    def execute(self, project_id, photo_id, retry=False):
        try:
            with self.store.lock:
                photo = copy.deepcopy(self.store.photo(self.store.read(project_id), photo_id))
            key = self.key(photo)
            folder = self.cache / key
            folder.mkdir(exist_ok=True)
            source = self.store.asset(project_id, photo_id, "processing.png")
            if digest(source) != photo["sha256"]:
                raise ValueError("Processing image changed after upload")
            manifest = {"schema_version": 1, "images": [{"image_id": "content-" + photo["sha256"][:16], "file": str(source), "rights": photo["rights"]}]}
            # A shared cache manifest is immutable for this image/settings identity.
            if not (folder / "input.json").exists():
                atomic_json(folder / "input.json", manifest)
                atomic_json(folder / "settings.json", self.settings)
                atomic_json(folder / "prompts.json", self.settings["prompts"])
            for stage in ("detection", "segmentation", "depth"):
                result_path = folder / f"{stage}-result.json"
                previous = read_json(result_path) if result_path.exists() else None
                if previous and previous["state"] == "error" and not retry:
                    result = previous
                elif previous and previous["state"] == "complete":
                    result = {**previous, "cached": True}
                else:
                    self.status(project_id, photo_id, stage, {"state": "running", "message": f"Running {stage} locally on CPU"})
                    started = perf_counter()
                    try:
                        result = self.runner(stage, folder, photo)
                        result.update(state="complete", cached=False, completed_at=now(), processing_seconds=perf_counter() - started)
                    except Exception as exc:
                        result = {"state": "error", "message": f"{stage} could not finish. Saved outputs from other stages are preserved.", "error_type": type(exc).__name__, "technical_error": str(exc), "completed_at": now()}
                    atomic_json(result_path, result)
                self.apply(project_id, photo_id, stage, result, folder, key)
            with self.store.lock:
                project = self.store.read(project_id)
                photo = self.store.photo(project, photo_id)
                complete = sum(s.get("state") == "complete" for s in photo["stages"].values())
                photo["processing_state"] = "complete" if complete == 3 else "partial" if complete else "failed"
                self.store.write(project)
        except Exception as exc:
            with self.store.lock:
                project = self.store.read(project_id)
                photo = self.store.photo(project, photo_id)
                photo.update(processing_state="failed", processing_error="Processing stopped; your photograph and completed results remain saved", technical_error=str(exc))
                self.store.write(project)
        finally:
            with self.lock:
                self.pending.discard((project_id, photo_id))

    def run_stage(self, stage, folder, photo):
        if not MODELS[stage].is_dir():
            raise ValueError("Required local checkpoint missing. Downloads are disabled")
        attempt = sum(1 for _ in folder.glob(stage + "-attempt-*"))
        output = folder / f"{stage}-attempt-{attempt:03d}"
        base = [str(Path(sys.executable).absolute()), "-X", "faulthandler", "-u", "-m"]
        if stage == "detection":
            command = base + ["room_viewer.photo_worker", "--manifest", str(folder / "input.json"), "--model", str(MODELS[stage]), "--prompts", str(folder / "prompts.json"), "--output", str(output)]
        else:
            command = base + ["room_layout.worker", "--stage", stage, "--model", str(MODELS[stage]), "--manifest", str(folder / "input.json"), "--output", str(output), "--threads", "4"]
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"))
        with (folder / f"{stage}-{attempt:03d}.log").open("w") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"Worker exit {process.returncode}; inspect local stage log before retrying")
        metadata = {"name": MODEL_NAMES[stage], "checkpoint": MODELS[stage].name, "checkpoint_fingerprint": hashlib.sha256(json.dumps(self.settings["checkpoints"][stage], sort_keys=True).encode()).hexdigest(), "device": "cpu", "timestamp": now()}
        if stage == "detection":
            report = read_json(output / "predictions.json")
            row = report["images"][0]
            if row["status"] == "error":
                raise RuntimeError("No successful furniture prediction")
            return {"data": {"detections": row["detections"], "raw_candidates": row.get("raw_candidates", []), "filter_audit": row.get("filter_audit"), "settings": {k: self.settings[k] for k in ("threshold", "nms_iou", "prompts")}, "model": metadata, "inference_seconds": row["timing_seconds"]["owlv2"]}}
        report = read_json(output / "report.json")
        row = report["images"][0]
        if row["status"] != "ok":
            raise RuntimeError("No successful room-understanding output")
        with np.load(output / row["file"], allow_pickle=False) as values:
            arrays = {k: values[k] for k in values.files}
        if stage == "segmentation":
            with Image.open(read_json(folder / "input.json")["images"][0]["file"]) as source:
                image = source.convert("RGB")
            labels, confidence = arrays["labels"], arrays["confidence"]
            structure, mapped = extract_structure(image, labels, confidence, report["model"]["id2label"], [], .55)
            rgba = np.zeros((*mapped.shape, 4), np.uint8)
            for label, color in COLORS.items():
                rgba[(mapped == label) & (confidence >= .55)] = [*color, 110]
            Image.fromarray(rgba).save(folder / "semantic.png")
            return {"data": {"model": metadata, "structure": structure, "classes": ["wall", "floor", "ceiling", "window", "door", "floor_covering"], "inference_seconds": row["inference_seconds"], "peak_process_rss_bytes": report.get("peak_process_rss_bytes"), "confidence_note": "Semantic model confidence is not physical reconstruction accuracy"}}
        depth = arrays["relative_depth"]
        info = depth_image(depth, folder / "depth-with-legend.png")
        with Image.open(folder / "depth-with-legend.png") as preview:
            preview.crop((0, 0, depth.shape[1], depth.shape[0])).save(folder / "depth.png")
        np.savez_compressed(folder / "depth-values.npz", relative_depth=depth)
        return {"data": {"model": metadata, "inference_seconds": row["inference_seconds"], "peak_process_rss_bytes": report.get("peak_process_rss_bytes"), "relative_depth": info, "relative_depth_reference": "depth-values.npz", "physical_units": None, "calibration": None}}

    def apply(self, project_id, photo_id, stage, result, folder, key):
        with self.store.lock:
            project = self.store.read(project_id)
            photo = self.store.photo(project, photo_id)
            photo["cache_key"] = key
            photo["stages"][stage] = {k: v for k, v in result.items() if k != "data"}
            if result["state"] == "complete":
                data = result["data"]
                if stage == "detection":
                    previous = photo.get("predictions")
                    if previous is not None and previous != data:
                        photo.setdefault("prediction_history", []).append(copy.deepcopy(previous))
                        if photo["corrections"]:
                            photo["review_requires_reconciliation"] = True
                    photo["predictions"] = copy.deepcopy(data)
                    incoming = [{**d, "display_label": "coffee_table" if "coffee table" in d.get("matched_prompt", "") else "bookshelf" if any(w in d.get("matched_prompt", "") for w in ("bookshelf", "bookcase")) else d["label"], "source": "ai", "review_status": "unreviewed", "confirmed": False, "image_id": photo_id, "model": data["model"], "detection_settings": data["settings"]} for d in data["detections"]]
                    if not photo["corrections"]:
                        photo["reviewed"] = incoming
                    else:
                        reviewed_ids = {d["detection_id"] for d in photo["reviewed"]}
                        photo["reviewed"].extend(d for d in incoming if d["detection_id"] not in reviewed_ids)
                elif stage == "segmentation":
                    photo["surfaces"] = data
                    shutil.copyfile(folder / "semantic.png", self.store.folder(project_id) / "photos" / photo_id / "semantic.png")
                    if project["room"] and project["room"].get("schema_version") == 2:
                        incoming = photo_scene(photo_id, photo["sha256"], photo["width"], photo["height"], data["structure"], {}, [])
                        for view in project["room"]["views"]:
                            if view["view_id"] == photo_id:
                                view["visible_structure"] = copy.deepcopy(incoming["views"][0]["visible_structure"])
                        project["room"].setdefault("view_documents", {})[photo_id] = incoming
                else:
                    photo["depth"] = data
                    shutil.copyfile(folder / "depth.png", self.store.folder(project_id) / "photos" / photo_id / "depth.png")
                    if (folder / "depth-values.npz").is_file():
                        shutil.copyfile(folder / "depth-values.npz", self.store.folder(project_id) / "photos" / photo_id / "depth-values.npz")
            photo["processing_state"] = "running"
            self.store.write(project)
