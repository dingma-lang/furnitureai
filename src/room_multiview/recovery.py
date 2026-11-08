"""Source-bound adoption of experimental odometry on original reconstruction views."""

import copy
import os
import subprocess
from pathlib import Path

from furniture_bench.data import digest, read_json, write_json

from .numeric import rigid_pose
from .sensor_pipeline import fuse_sensor_state


def run_odometry_worker(manifest, worker_python, output, anchor_mode="first"):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new odometry experiment directory")
    if not Path(worker_python).is_file():
        raise ValueError("explicit isolated Open3D worker Python is required")
    output.mkdir(parents=True)
    env = dict(
        os.environ,
        OPEN3D_CPU_RENDERING="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    with (output / "worker.log").open("w") as log:
        result = subprocess.run(
            [
                str(Path(worker_python).absolute()),
                "-X",
                "faulthandler",
                "-u",
                "-m",
                "room_multiview.odometry_worker",
                "--manifest",
                str(Path(manifest).resolve()),
                "--output",
                str(output / "worker"),
                "--anchor-mode",
                anchor_mode,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Open3D worker failed with code {result.returncode}; inspect {output}/worker.log before retrying"
        )
    return read_json(output / "worker/report.json")


def fuse_recovered_cameras(baseline_project, worker_directory, variant, heldout_manifest, output):
    if variant not in ("hybrid", "hybrid-icp", "pose-graph"):
        raise ValueError("unknown odometry variant")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new recovery fusion output directory")
    worker_directory = Path(worker_directory).resolve()
    tracking = read_json(worker_directory / "report.json")
    if tracking.get("reference_poses_used") is not False:
        raise ValueError("tracking provenance is not reference-free")
    result = read_json(worker_directory / (variant + ".json"))
    heldout = read_json(heldout_manifest)["images"]
    heldout_ids = {r["image_id"] for r in heldout}
    heldout_hashes = {r["image_sha256"] for r in heldout}
    heldout_depth_hashes = {r["sensor_depth"]["sha256"] for r in heldout if r.get("sensor_depth")}
    if set(tracking["sources"]) & heldout_ids or any(
        s["image_sha256"] in heldout_hashes or s["depth_sha256"] in heldout_depth_hashes
        for s in tracking["sources"].values()
    ):
        raise ValueError("tracking used held-out references; fusion refused")
    state = copy.deepcopy(read_json(baseline_project))
    sensor = state["sensor_reconstruction"]
    sources = sensor["sources"]
    if result["anchor_view_id"] != sensor["anchor_view_id"]:
        raise ValueError("recovery frame differs from preserved sensor anchor")
    adopted = {}
    for key, camera in result["cameras"].items():
        rigid_pose(camera["camera_from_sensor_world"])
        if key not in sources:
            continue
        if (
            sources[key]["image_sha256"] != tracking["sources"][key]["image_sha256"]
            or sources[key]["depth_sha256"] != tracking["sources"][key]["depth_sha256"]
        ):
            raise ValueError("odometry and source observation identities differ")
        adopted[key] = copy.deepcopy(camera)
    if len(adopted) < 2:
        raise ValueError("no connected recovered multi-view component to fuse")
    sensor["previous_registration"] = {
        "baseline_project": str(Path(baseline_project).resolve()),
        "sha256": digest(baseline_project),
        "cameras": copy.deepcopy(sensor["cameras"]),
        "reference_geometry_unchanged": True,
    }
    sensor["cameras"] = adopted
    sensor["registration_method"] = "Open3D RGB-D " + variant
    sensor["recovery_tracking"] = {
        "directory": str(worker_directory),
        "variant": variant,
        "variant_sha256": digest(worker_directory / (variant + ".json")),
        "registered_tracking_frames": len(result["cameras"]),
        "primary_frames_fused": len(adopted),
        "intermediate_frames_fused": False,
        "reference_poses_used_for_tracking": False,
        "policy": tracking["policy"],
        "heldout_manifest_sha256": digest(heldout_manifest),
        "interpretation": "Intermediate photographs support tracking only. Only original Phase 5 input depth/semantics are fused for comparable coverage.",
    }
    sensor["registration_failures"] = [
        {"view_id": key, "status": "rejected", "reason": "no_connected_validated_odometry_path"}
        for key in sources
        if key not in adopted
    ]
    state["fit_readiness"]["ready"] = False
    state["physical_accuracy_verified"] = False
    output.mkdir(parents=True)
    fused = fuse_sensor_state(state, output)
    write_json(output / "adoption.json", sensor["recovery_tracking"])
    return fused
