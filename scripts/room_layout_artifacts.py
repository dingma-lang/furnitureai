"""Assemble comparison, provenance and explicitly synthetic integration artifacts."""

import argparse
import importlib.metadata
import re
import shutil
from pathlib import Path

from PIL import Image

from furniture_bench.data import digest, read_json, write_json
from room_layout.scene import require_metric_scene
from room_layout.visualization import compare_previous
from room_reconstruction.collision import check_scene
from room_reconstruction.demo import demo_scene, replacement_sofa
from room_reconstruction.export import floorplan_svg
from room_reconstruction.model import propose_replacement
from room_reconstruction.photo_estimates import attach_photo_estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="reports/room-layout-experiment")
    args = parser.parse_args()
    root = Path(args.experiment)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=False)
    run = read_json(root / "final/run.json")
    compare_previous(
        "reports/integration-v2/visuals/assumed-floor-perspective.png",
        root / "final/image-002/layout-overlay.png",
        artifacts / "previous-vs-automatic.png",
    )
    contact = Image.new("RGB", (1600, 1260), "white")
    for i, row in enumerate(run["images"]):
        with Image.open(root / "final" / row["folder"] / "layout-overlay.png") as image:
            image.thumbnail((800, 620))
            contact.paste(image, ((i % 2) * 800, (i // 2) * 630))
    contact.save(artifacts / "all-room-overlays.png")
    partial = read_json(root / "final/image-002/room-estimate.json")
    try:
        require_metric_scene(partial)
    except ValueError as exc:
        gate = {"blocked": True, "reason": str(exc)}
    else:
        raise AssertionError("uncalibrated photo must not become a metric room")
    before = attach_photo_estimate(
        demo_scene(),
        partial,
        "automated demonstration",
        "Compatibility test only; this metric room is not the photographed room.",
        demo_only=True,
    )
    after, replacement = propose_replacement(before, "sofa-original", replacement_sofa())
    geometry_preserved = all(
        before[k] == after[k] for k in ("floor", "walls", "openings", "dimensions")
    )
    for name, scene in (("before", before), ("after", after)):
        write_json(artifacts / f"integration-demo-{name}.json", scene)
        floorplan_svg(
            scene,
            artifacts / f"integration-demo-{name}.svg",
            f"Independent dimensioned demo: {name}",
            check_scene(scene),
        )
    write_json(
        artifacts / "integration-check.json",
        {
            "photo_metric_fit_gate": gate,
            "replacement": replacement,
            "room_geometry_preserved": geometry_preserved,
            "photo_evidence_preserved": before["photo_structure_estimates"]
            == after["photo_structure_estimates"],
            "demonstration_only": True,
            "photo_dimensions_estimated": False,
        },
    )
    view = partial["views"][0]
    write_json(
        artifacts / "correction-template.json",
        {
            "view_id": view["view_id"],
            "image_sha256": view["image_sha256"],
            "reviewer": "REPLACE_WITH_ACTUAL_REVIEWER",
            "reason": "REPLACE_WITH_OBSERVED_EVIDENCE",
            "geometry_verdict": "unreviewed",
            "corner_corrections": [],
            "boundary_corrections": [],
            "measurements": {},
            "wall_measurements": [],
            "reference_objects": [],
            "camera": {},
        },
    )
    hardware = {"architecture": "arm64", "physical_ram_gib": 16, "cpu_brand": "not collected"}
    stages = []
    for stage, reports in run["model_stages"].items():
        for report in reports:
            stages.append(
                {
                    "stage": stage,
                    "cache": report["cache"],
                    "model_load_seconds": report["model"]["load_seconds"],
                    "peak_process_rss_bytes": report["peak_process_rss_bytes"],
                    "images": [
                        {k: row[k] for k in ("image_id", "status", "inference_seconds")}
                        for row in report["images"]
                    ],
                }
            )
    write_json(
        artifacts / "runtime-summary.json",
        {
            "hardware": hardware,
            "environment": run["model_stages"]["depth"][0]["environment"],
            "additional_packages": {
                p: importlib.metadata.version(p) for p in ("scipy", "matplotlib", "safetensors")
            },
            "device": "cpu",
            "intraop_threads": 4,
            "interop_threads": 1,
            "dtype": "float32",
            "stages": stages,
            "runtime_processing_failures": sum(row["status"] != "ok" for row in run["images"]),
            "model_processes_sequential": True,
            "latency_repeated_benchmark": False,
            "timing_scope": "processor + forward + output resize; excludes load, file I/O, structural extraction and plotting",
            "memory_scope": "OS peak RSS per worker including model loading; not machine-wide or allocator-only",
        },
    )
    declarations = []
    for stage, repo, revision, license_name in (
        (
            "segmentation",
            "openmmlab/upernet-convnext-tiny",
            "876ffc56b819a829448f7e81e9f8606deef6fb65",
            "MIT",
        ),
        (
            "depth",
            "depth-anything/Depth-Anything-V2-Small-hf",
            "5426e4f0f36572d16453bbda7a8389317b1bef99",
            "Apache-2.0",
        ),
    ):
        checkpoint = Path(run["model_stages"][stage][0]["model"]["path"])
        destination = artifacts / "sources" / stage
        destination.mkdir(parents=True)
        for name in ("README.md", "config.json", "preprocessor_config.json"):
            shutil.copy2(checkpoint / name, destination / name)
        declarations.append(
            {
                "stage": stage,
                "repo_id": repo,
                "revision": revision,
                "declared_checkpoint_license": license_name,
                "model_card_url": f"https://huggingface.co/{repo}/blob/{revision}/README.md",
                "checkpoint": str(checkpoint),
                "size_bytes": (checkpoint / "model.safetensors").stat().st_size,
                "sha256": digest(checkpoint / "model.safetensors"),
                "license_scope": "checkpoint declaration; input-data rights remain separate",
            }
        )
    write_json(artifacts / "sources/checkpoints.json", declarations)
    tests = (root / "tests.log").read_text()
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", tests)
    write_json(
        artifacts / "validation.json",
        {
            "tests": int(match[1]) if match else None,
            "seconds": float(match[2]) if match else None,
            "passed": "\nOK\n" in tests and "FAILED (" not in tests,
            "model_downloads_in_tests": False,
            "new_offline_tests": 18,
            "real_photo_inferences": 4,
            "human_verified_structure_references": 0,
        },
    )
    print(artifacts.resolve())


if __name__ == "__main__":
    main()
