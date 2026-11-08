"""Experimental metric RGB-D path. Ground-truth poses are not accepted as inputs."""

import copy
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.data import digest, load_image, load_manifest, read_json, write_json
from room_multiview.features import extract_features, match_features
from room_multiview.geometry import calibration, rotation_error
from room_multiview.project import merge_document, photo_guidance

from .gltf import export_gltf
from .rgbd import dense_samples, register_rgbd, sample_depth, transform, voxel_fuse
from .structured import fit_observed_planes, intersections, supported_mesh


def load_sensor(row, manifest):
    value = row.get("sensor_depth")
    if (
        not value
        or value.get("status") != "sensor_measured"
        or value.get("encoding") != "uint16_depth_z"
        or not value.get("source")
    ):
        raise ValueError(
            "explicit genuine registered sensor-depth metadata required; relative depth is not accepted"
        )
    path = Path(manifest).resolve().parent / value["file"]
    if digest(path) != value["sha256"]:
        raise ValueError("sensor-depth bytes changed")
    factor = value.get("units_per_metre")
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not np.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError("positive sensor depth units_per_metre required")
    delta = value.get("rgb_depth_delta_seconds", float("inf"))
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not np.isfinite(delta)
        or not 0 <= delta <= 0.02
    ):
        raise ValueError("RGB/depth temporal association exceeds 20 ms")
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype.kind not in ("u", "i") or raw.max() > 65535:
        raise ValueError("expected single-channel integer sensor depth")
    depth = raw.astype(np.float32) / factor
    if np.count_nonzero((depth >= 0.4) & (depth <= 5)) < min(100, depth.size):
        raise ValueError(
            "sensor frame has insufficient valid depth; no measured geometry substituted"
        )
    return depth, calibration(value["calibration"])


def run_sensor(manifest, layout_run, output, previous=None, stride=4):
    start = perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new sensor output directory")
    rows = load_manifest(manifest, "rooms")
    header = read_json(manifest)
    if stride < 2:
        raise ValueError("CPU experiment requires sample stride >=2")
    old = read_json(previous) if previous else None
    if old and old["room_id"] != header.get("room_id"):
        raise ValueError("progressive sensor room identity mismatch")
    if old and old.get("sensor_reconstruction", {}).get("stride") != stride:
        raise ValueError("preserve sensor sampling policy on resume")
    layout_run = Path(layout_run).resolve()
    layout = {
        r["image_id"]: (layout_run / r["folder"], r)
        for r in read_json(layout_run / "run.json")["images"]
        if r["status"] != "error"
    }
    state = copy.deepcopy(old)
    sources = copy.deepcopy(old.get("sensor_reconstruction", {}).get("sources", {})) if old else {}
    output.mkdir(parents=True)
    for row in rows:
        view = row["image_id"]
        if view in sources:
            if (
                digest(row["resolved_file"]) != sources[view]["image_sha256"]
                or row["sensor_depth"] != sources[view]["sensor_metadata"]
            ):
                raise ValueError("existing sensor observation changed; previous data preserved")
            continue
        if view not in layout:
            raise ValueError(f"no semantic single-photo estimate for {view}")
        folder, record = layout[view]
        if record["image_sha256"] != digest(row["resolved_file"]):
            raise ValueError("single-photo estimates reference different bytes")
        incoming = read_json(folder / "room-estimate.json")
        state = merge_document(state, incoming)
        state["room_id"] = header.get("room_id", state["room_id"])
        image = load_image(row["resolved_file"])
        depth, cal = load_sensor(row, manifest)
        if depth.shape != (image.height, image.width):
            raise ValueError("registered depth and source image sizes differ")
        features = extract_features(image)
        feature_path = output / f"sensor-features-{len(sources):03d}.npz"
        np.savez_compressed(feature_path, **features)
        sources[view] = {
            "image_path": row["resolved_file"],
            "image_sha256": digest(row["resolved_file"]),
            "depth_path": str(
                (Path(manifest).resolve().parent / row["sensor_depth"]["file"]).resolve()
            ),
            "depth_sha256": row["sensor_depth"]["sha256"],
            "sensor_metadata": copy.deepcopy(row["sensor_depth"]),
            "calibration": cal,
            "size": list(image.size),
            "feature_path": str(feature_path),
            "feature_sha256": digest(feature_path),
            "labels_path": str(folder / "structure-labels.png"),
            "confidence_path": str(folder / "semantic-confidence.npy"),
            "role": "reconstruction_input",
            "rights": row["rights"],
        }
    sensor = copy.deepcopy(old.get("sensor_reconstruction")) if old else None
    sensor = sensor or {
        "mode": "sensor_assisted_rgbd",
        "units": "m",
        "stride": stride,
        "cameras": {},
        "pair_reports": [],
        "anchor_view_id": next(iter(sources)),
        "physical_accuracy_verified": False,
        "metric_calibration": {
            "status": "sensor_depth_metric_with_approximate_pinhole_alignment",
            "source": "genuine Kinect depth integer units / 5000 per metre; no GT camera poses",
            "scale_factor": 1.0,
            "depth_uncertainty_metres": None,
            "camera_alignment_uncertainty": None,
            "note": "Estimated camera poses and approximate registered-depth intrinsics limit physical accuracy; no certified bound.",
        },
        "gravity_alignment": None,
        "closed_room_envelope": None,
    }
    sensor["sources"] = sources
    cameras = sensor["cameras"]
    if not cameras:
        cameras[sensor["anchor_view_id"]] = {
            "camera_from_sensor_world": np.eye(4).tolist(),
            "status": "gauge_anchor",
            "units": "m",
        }
    features = {}
    depths = {}
    for view, source in sources.items():
        if (
            digest(source["image_path"]) != source["image_sha256"]
            or digest(source["depth_path"]) != source["depth_sha256"]
            or digest(source["feature_path"]) != source["feature_sha256"]
        ):
            raise ValueError("progressive source observation/cache hash mismatch")
        with np.load(source["feature_path"]) as values:
            features[view] = {k: values[k] for k in ("pixels", "descriptors")}
        with Image.open(source["depth_path"]) as im:
            depths[view] = (
                np.asarray(im, dtype=np.float32) / source["sensor_metadata"]["units_per_metre"]
            )
    already = {tuple(p["views"]): p for p in sensor["pair_reports"]}
    registration = []
    remaining = [v for v in sources if v not in cameras]
    for _ in range(len(remaining)):
        progress = False
        for view in list(remaining):
            candidates = []
            # Nearby preceding/recently recovered views; bounded cost for the CPU experiment.
            for other in list(cameras)[-6:]:
                key = (other, view)
                if key in already:
                    result = already[key]
                else:
                    matches = match_features(features[other], features[view])
                    pa = features[other]["pixels"][matches[:, 0]]
                    pb = features[view]["pixels"][matches[:, 1]]
                    xyz_a, va = sample_depth(
                        depths[other], pa, sources[other]["calibration"]["intrinsics"]
                    )
                    xyz_b, vb = sample_depth(
                        depths[view], pb, sources[view]["calibration"]["intrinsics"]
                    )
                    valid = va & vb
                    result = register_rgbd(
                        xyz_a[valid], xyz_b[valid], pa[valid], pb[valid], sources[view]["size"]
                    )
                    result.update(
                        views=[other, view],
                        mutual_matches=len(matches),
                        pixels_a=pa[valid].tolist(),
                        pixels_b=pb[valid].tolist(),
                    )
                    already[key] = result
                if result["status"] == "accepted":
                    pose = np.array(result["camera_b_from_a"]) @ np.array(
                        cameras[other]["camera_from_sensor_world"]
                    )
                    candidates.append((result["inliers"], pose, result))
            if not candidates:
                registration.append(
                    {
                        "view_id": view,
                        "status": "rejected",
                        "reason": "no_consistent_rgbd_overlap_with_registered_views",
                    }
                )
                continue
            candidates.sort(key=lambda r: r[0], reverse=True)
            _, pose, chosen = candidates[0]
            disagreements = []
            for _, alternative, result in candidates[1:]:
                r = rotation_error(pose[:3, :3], alternative[:3, :3])
                c = -pose[:3, :3].T @ pose[:3, 3]
                d = -alternative[:3, :3].T @ alternative[:3, 3]
                disagreements.append(
                    {
                        "other_view": result["views"][0],
                        "rotation_degrees": r,
                        "centre_distance_metres": float(np.linalg.norm(c - d)),
                    }
                )
            if any(
                c["rotation_degrees"] > 5 or c["centre_distance_metres"] > 0.15
                for c in disagreements
            ):
                registration.append(
                    {
                        "view_id": view,
                        "status": "rejected",
                        "reason": "inconsistent_independent_rgbd_pose_paths",
                        "checks": disagreements,
                    }
                )
                continue
            cameras[view] = {
                "camera_from_sensor_world": pose.tolist(),
                "status": "estimated_from_measured_depth_correspondences",
                "units": "m",
                "chosen_parent": chosen["views"][0],
                "inliers": chosen["inliers"],
                "path_consistency": disagreements,
            }
            remaining.remove(view)
            progress = True
        if not progress:
            break
    sensor["pair_reports"] = list(already.values())
    sensor["registration_attempts"] = registration
    sensor["registration_failures"] = [r for r in registration if r["view_id"] not in cameras]
    state["sensor_reconstruction"] = sensor
    return fuse_sensor_state(state, output, depths, start)


def fuse_sensor_state(state, output, depths=None, start=None):
    """Reuse observed fusion after a separate, validated camera-registration stage."""
    start = perf_counter() if start is None else start
    output = Path(output).resolve()
    sensor = state["sensor_reconstruction"]
    sources, cameras, stride = sensor["sources"], sensor["cameras"], sensor["stride"]
    if not cameras:
        raise ValueError("no validated camera poses for depth fusion")
    if depths is None:
        depths = {}
        for view in cameras:
            source = sources[view]
            if (
                digest(source["image_path"]) != source["image_sha256"]
                or digest(source["depth_path"]) != source["depth_sha256"]
            ):
                raise ValueError("source bytes changed before depth fusion")
            with Image.open(source["depth_path"]) as image:
                depths[view] = (
                    np.asarray(image, dtype=np.float32)
                    / source["sensor_metadata"]["units_per_metre"]
                )
    point_rows = []
    color_rows = []
    label_rows = []
    view_rows = []
    samples_by_view = {}
    for index, (view, camera) in enumerate(cameras.items()):
        source = sources[view]
        labels = np.asarray(Image.open(source["labels_path"]))
        confidence = np.load(source["confidence_path"])
        rgb = load_image(source["image_path"])
        samples = dense_samples(
            depths[view], rgb, source["calibration"]["intrinsics"], labels, confidence, stride
        )
        samples_by_view[view] = samples
        world = transform(samples["points"], np.linalg.inv(camera["camera_from_sensor_world"]))
        point_rows.append(world)
        color_rows.append(samples["colors"])
        label_rows.append(samples["labels"])
        view_rows.append(np.full(len(world), index))
    raw = {
        "points": np.concatenate(point_rows),
        "colors": np.concatenate(color_rows),
        "labels": np.concatenate(label_rows),
        "view_indices": np.concatenate(view_rows),
    }
    np.savez_compressed(output / "observed-depth-points.npz", **raw)
    fused = voxel_fuse(raw["points"], raw["colors"], raw["labels"], raw["view_indices"])
    np.savez_compressed(output / "fused-observed-cloud.npz", **fused)
    planes = fit_observed_planes(fused)
    vertices = []
    colors = []
    triangles = []
    offset = 0
    for view, samples in samples_by_view.items():
        xyz, faces, _ = supported_mesh(
            samples, np.linalg.inv(cameras[view]["camera_from_sensor_world"]), planes, stride
        )
        if not len(faces):
            continue
        used = np.unique(faces)
        remap = np.full(len(xyz), -1, int)
        remap[used] = np.arange(len(used))
        vertices.append(xyz[used])
        colors.append(samples["colors"][used])
        triangles.append(remap[faces] + offset)
        offset += len(used)
    mesh = {
        "points": np.concatenate(vertices) if vertices else np.empty((0, 3)),
        "colors": np.concatenate(colors) if colors else np.empty((0, 3), np.uint8),
        "triangles": np.concatenate(triangles) if triangles else np.empty((0, 3), np.uint32),
    }
    np.savez_compressed(output / "observed-surfaces.npz", **mesh)
    export_gltf(
        output / "observed-room.gltf",
        mesh["points"],
        mesh["colors"],
        mesh["triangles"],
        {
            "input_mode": "sensor_assisted_rgbd",
            "units": "m",
            "physical_accuracy_verified": False,
            "geometry": "only measured-depth grid triangles on supported semantic planes; hidden room unknown",
            "frame_conversion": "camera x-right,y-down,z-forward to glTF x-right,y-up,z-back",
        },
    )
    export_gltf(
        output / "dense-cloud.gltf",
        fused["points"],
        fused["colors"],
        extras={"input_mode": "sensor_assisted_rgbd", "units": "m", "closed_room": False},
    )
    sensor.update(
        surface_planes=planes,
        plane_intersection_hypotheses=intersections(planes),
        point_cloud={
            "file": str(output / "fused-observed-cloud.npz"),
            "sha256": digest(output / "fused-observed-cloud.npz"),
            "points": len(fused["points"]),
            "voxel_size_metres": 0.025,
            "raw_observed_samples": len(raw["points"]),
        },
        observed_partial_mesh={
            "file": str(output / "observed-surfaces.npz"),
            "triangles": len(mesh["triangles"]),
            "gltf": str(output / "observed-room.gltf"),
        },
        unresolved=[
            "hidden surfaces",
            "closed room boundary",
            "gravity",
            "verified dimensions",
            "doors/windows in world coordinates",
            "collision-ready envelope",
        ],
    )
    state["sensor_reconstruction"] = sensor
    state["coordinate_frames"]["sensor_world"] = {
        "available": True,
        "units": "m",
        "frame_id": "sensor-anchor:" + sensor["anchor_view_id"],
        "orientation": "anchor camera x-right, y-down, z-forward; gravity unknown",
    }
    state["coordinate_frames"]["transforms"]["sensor_camera_to_sensor_world"] = {
        v: {
            "matrix": np.linalg.inv(c["camera_from_sensor_world"]).tolist(),
            "status": "estimated_from_sensor_measurements",
        }
        for v, c in cameras.items()
    }
    state["coordinate_frames"]["transforms"]["sensor_world_to_editable_metric_room"] = None
    state["physical_accuracy_verified"] = False
    state["fit_readiness"]["ready"] = False
    state["photo_guidance"] = photo_guidance(state)
    state["sensor_runtime"] = {
        "seconds_excluding_visualization": perf_counter() - start,
        "cpu": True,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "reference_poses_used": False,
        "depth_mode": "genuine_sensor_measurements",
    }
    write_json(output / "room.json", state)
    return state
