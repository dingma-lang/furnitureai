"""ARKitScenes format adapter; supplied poses are written only to evaluation references."""

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from furniture_bench.data import digest, read_json, write_json


def orient_calibration(k, size, sky_direction):
    k = np.asarray(k, float)
    w, h = size
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("invalid source intrinsics")
    settings = {"Up": 0, "Left": 3, "Down": 2, "Right": 1}
    if sky_direction not in settings:
        raise ValueError("unknown image sky direction")
    turns = settings[sky_direction]
    if turns == 0:
        return k.copy(), np.eye(3), (w, h), turns
    if turns == 3:
        result = np.array([[k[1, 1], 0, h - 1 - k[1, 2]], [0, k[0, 0], k[0, 2]], [0, 0, 1.0]])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        return result, rotation, (h, w), turns
    if turns == 1:
        result = np.array([[k[1, 1], 0, k[1, 2]], [0, k[0, 0], w - 1 - k[0, 2]], [0, 0, 1.0]])
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
        return result, rotation, (h, w), turns
    result = k.copy()
    result[0, 2] = w - 1 - k[0, 2]
    result[1, 2] = h - 1 - k[1, 2]
    return result, np.diag([-1.0, -1.0, 1.0]), (w, h), turns


def resample_intrinsics(array, source_k, target_k, size, is_depth=False):
    """Same-camera ray reprojection, with no pose/geometry reference input."""
    source_k, target_k = np.asarray(source_k), np.asarray(target_k)
    sx, sy = source_k[0, 0] / target_k[0, 0], source_k[1, 1] / target_k[1, 1]
    affine = (
        sx,
        0,
        source_k[0, 2] - sx * target_k[0, 2],
        0,
        sy,
        source_k[1, 2] - sy * target_k[1, 2],
    )
    source = Image.fromarray(np.ascontiguousarray(array))
    return np.asarray(
        source.transform(
            size,
            Image.Transform.AFFINE,
            affine,
            resample=Image.Resampling.NEAREST if is_depth else Image.Resampling.BILINEAR,
        )
    )


def trajectory_references(path, timestamps, camera_rotation):
    """Match official loader convention: inverse of stored world-to-camera [R|t]."""
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("invalid ARKit trajectory")
    order = np.argsort(values[:, 0])
    values = values[order]
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("trajectory timestamps must increase")
    stored = np.tile(np.eye(4), (len(values), 1, 1))
    stored[:, :3, :3] = Rotation.from_rotvec(values[:, 1:4]).as_matrix()
    stored[:, :3, 3] = values[:, 4:7]
    world = np.linalg.inv(stored)
    world[:, :3, :3] = np.einsum("nij,jk->nik", world[:, :3, :3], np.asarray(camera_rotation).T)
    slerp = Slerp(values[:, 0], Rotation.from_matrix(world[:, :3, :3]))
    result = []
    for key, timestamp in timestamps:
        if timestamp < values[0, 0] or timestamp > values[-1, 0]:
            continue
        left = max(0, min(len(values) - 2, np.searchsorted(values[:, 0], timestamp) - 1))
        gap = values[left + 1, 0] - values[left, 0]
        if gap > 0.2:
            continue
        centre = [float(np.interp(timestamp, values[:, 0], world[:, i, 3])) for i in range(3)]
        result.append(
            {
                "image_id": key,
                "rgb_timestamp": timestamp,
                "pose_timestamp": timestamp,
                "timestamp_delta_seconds": 0.0,
                "position_metres": centre,
                "quaternion_xyzw": slerp([timestamp]).as_quat()[0].tolist(),
                "reference_kind": "supplied_ARKit_sensor_trajectory_interpolated_not_external_motion_capture",
                "interpolation_source_gap_seconds": float(gap),
            }
        )
    return result


def prepare_scene(
    assets,
    output,
    scene_id,
    sky_direction,
    approval_record,
    start_index=60,
    stop_index=5580,
    tracking_step=6,
    primary_step=180,
):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    approval = read_json(approval_record)
    if approval.get("approved") is not True or approval.get("scene_id") != scene_id:
        raise ValueError("recorded one-scene dataset approval required")
    if output.exists():
        raise ValueError("choose a new research subset directory")
    if min(tracking_step, primary_step) <= 0 or primary_step % tracking_step:
        raise ValueError("primary interval must be divisible by tracking interval")

    def index(archive, extension):
        result = {}
        for name in archive.namelist():
            if name.endswith(extension):
                match = re.search(r"_(\d+\.\d+)" + re.escape(extension) + r"$", Path(name).name)
                if match:
                    result[float(match[1])] = name
        return result

    with (
        zipfile.ZipFile(assets / "lowres_wide.zip") as rgb_zip,
        zipfile.ZipFile(assets / "lowres_depth.zip") as depth_zip,
        zipfile.ZipFile(assets / "confidence.zip") as conf_zip,
        zipfile.ZipFile(assets / "lowres_wide_intrinsics.zip") as k_zip,
    ):
        rgb_index = index(rgb_zip, ".png")
        depth_index = index(depth_zip, ".png")
        conf_index = index(conf_zip, ".png")
        k_index = index(k_zip, ".pincam")
        times = sorted(rgb_index)
        stop_index = min(stop_index, len(times) - 1)
        selected = list(range(start_index, stop_index + 1, tracking_step))
        heldout = set(range(start_index + primary_step // 2, stop_index + 1, primary_step))
        selected = sorted(set(selected) | heldout)
        first_values = np.fromstring(k_zip.read(k_index[times[start_index]]).decode(), sep=" ")
        if len(first_values) != 6:
            raise ValueError("pincam must have six finite values")
        w, h, fx, fy, cx, cy = first_values
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        canonical, camera_rotation, size, turns = orient_calibration(
            k, (int(w), int(h)), sky_direction
        )
        output.mkdir(parents=True)
        rows = []
        skipped = []
        for raw_index in selected:
            timestamp = times[raw_index]
            if (
                timestamp not in depth_index
                or timestamp not in conf_index
                or timestamp not in k_index
            ):
                skipped.append({"raw_index": raw_index, "reason": "missing_synchronized_asset"})
                continue
            raw_rgb = rgb_zip.read(rgb_index[timestamp])
            raw_depth = depth_zip.read(depth_index[timestamp])
            raw_conf = conf_zip.read(conf_index[timestamp])
            rgb = np.asarray(Image.open(io.BytesIO(raw_rgb)).convert("RGB"))
            depth = np.asarray(Image.open(io.BytesIO(raw_depth)))
            confidence = np.asarray(Image.open(io.BytesIO(raw_conf)))
            values = np.fromstring(k_zip.read(k_index[timestamp]).decode(), sep=" ")
            current = np.array([[values[2], 0, values[4]], [0, values[3], values[5]], [0, 0, 1.0]])
            current, _, _, _ = orient_calibration(
                current, (int(values[0]), int(values[1])), sky_direction
            )
            rgb = resample_intrinsics(np.rot90(rgb, turns), current, canonical, size)
            depth = resample_intrinsics(
                np.rot90(depth, turns), current, canonical, size, True
            ).copy()
            confidence = resample_intrinsics(
                np.rot90(confidence, turns), current, canonical, size, True
            )
            depth[confidence < 2] = 0
            filename = f"rgb-{raw_index:05d}.png"
            depthname = f"depth-{raw_index:05d}.png"
            confname = f"confidence-{raw_index:05d}.png"
            Image.fromarray(rgb).save(output / filename)
            Image.fromarray(depth).save(output / depthname)
            Image.fromarray(confidence).save(output / confname)
            identifier = f"arkit-{scene_id}-{timestamp:.3f}"
            cal = {
                "intrinsics": canonical.tolist(),
                "distortion": [0.0] * 5,
                "status": "dataset_calibrated",
                "source": "Official per-frame .pincam; upright rotation and same-camera intrinsic resampling; no trajectory poses used",
            }
            row = {
                "image_id": identifier,
                "file": filename,
                "image_sha256": digest(output / filename),
                "source_rgb_index": raw_index,
                "timestamp": timestamp,
                "objects": None,
                "layout_annotations": None,
                "role": "heldout_evaluation" if raw_index in heldout else "tracking_input",
                "rights": {
                    "allowed_for_benchmark": True,
                    "source": f"Apple ARKitScenes Validation {scene_id}; Baruch et al., NeurIPS Datasets and Benchmarks 2021",
                    "license": "Apple ARKitScenes published conditional license; approved research experiment only",
                    "research_only": True,
                    "commercial_training_authorized": False,
                    "approval_record_sha256": digest(approval_record),
                },
                "calibration": cal,
                "sensor_depth": {
                    "file": depthname,
                    "sha256": digest(output / depthname),
                    "encoding": "uint16_depth_z",
                    "units_per_metre": 1000,
                    "status": "sensor_measured",
                    "source": "ARKitScenes synchronized low-resolution LiDAR depth in millimetres; confidence==2 retained",
                    "rgb_depth_delta_seconds": 0.0,
                    "calibration": cal,
                    "confidence_file": confname,
                    "uncertainty": {
                        "validated_error_bound_metres": None,
                        "shared_sensor_limitations": "LiDAR depth, ARKit trajectory and ARKit mesh are correlated sensor-derived references.",
                    },
                },
                "preprocessing": {
                    "sky_direction": sky_direction,
                    "image_quarter_turns_ccw": turns,
                    "upright_camera_from_original_camera": camera_rotation.tolist(),
                    "per_frame_intrinsics_resampled_to": canonical.tolist(),
                    "raw_rgb_sha256": hashlib.sha256(raw_rgb).hexdigest(),
                    "raw_depth_sha256": hashlib.sha256(raw_depth).hexdigest(),
                    "depth_interpolation": "nearest, no depth averaging or hole filling",
                },
            }
            rows.append(row)
    header = {"schema_version": 1, "room_id": "arkitscenes-" + scene_id}
    tracking = [r for r in rows if r["role"] != "heldout_evaluation"]
    primary = [r for r in tracking if (r["source_rgb_index"] - start_index) % primary_step == 0]
    primary_ids = {r["image_id"] for r in primary}
    for row in rows:
        row["primary_scene_view"] = row["image_id"] in primary_ids
    for name, items in [
        ("tracking", tracking),
        ("primary", primary),
        ("heldout", [r for r in rows if r["role"] == "heldout_evaluation"]),
    ]:
        write_json(output / (name + ".json"), {**header, "images": items})
    rgb_only = [{k: v for k, v in r.items() if k != "sensor_depth"} for r in primary]
    write_json(output / "rgb-only.json", {**header, "images": rgb_only})
    # Reference conversion occurs only after reconstruction manifests are complete.
    references = trajectory_references(
        assets / "lowres_wide.traj",
        [(r["image_id"], r["timestamp"]) for r in rows],
        camera_rotation,
    )
    write_json(
        output / "reference-poses.json",
        {
            "source": "Supplied ARKit sensor trajectory, evaluation only; shares capture sensor limitations",
            "used_for_inference": False,
            "poses": references,
        },
    )
    write_json(
        output / "receipt.json",
        {
            "scene_id": scene_id,
            "source_assets": {
                p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in assets.iterdir()
                if p.is_file()
            },
            "approval_record_sha256": digest(approval_record),
            "tracking_frames": len(tracking),
            "primary_frames": len(primary),
            "heldout_frames": len(rows) - len(tracking),
            "raw_rgb_count": len(times),
            "skipped": skipped,
            "selection": {
                "start_index": start_index,
                "stop_index": stop_index,
                "tracking_step": tracking_step,
                "primary_step": primary_step,
            },
            "data_use": "Research-restricted experimental evaluation; no training or commercial rights assertion",
        },
    )
    return output
