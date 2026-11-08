"""Progressive CPU SfM around the existing independent single-image pipeline."""

import copy
import itertools
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.data import digest, load_image, load_manifest, read_json, write_json
from room_layout.pipeline import run_layout
from room_layout.scene import validate_photo_scene

from .features import extract_features, match_features, opencv
from .geometry import (
    Policy,
    calibration,
    centre,
    estimate_pair,
    project,
    register_camera,
    rotation_error,
    triangulate,
)
from .project import establish_scale, merge_document, photo_guidance
from .surfaces import plane_patches


def _label(asset, pixel):
    labels = asset.get("_labels")
    if labels is None:
        return "unknown"
    x, y = np.floor(pixel).astype(int)
    if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
        return "unknown"
    confidence = asset.get("_confidence")
    if confidence is not None and confidence[y, x] < 0.55:
        return "unknown"
    return {1: "wall", 2: "floor", 3: "ceiling", 4: "window", 5: "door", 6: "floor_covering"}.get(
        int(labels[y, x]), "other"
    )


def _add_landmark(geometry, point, observations, assets, features, errors, angle):
    votes = [
        _label(assets[o["view_id"]], features[o["view_id"]]["pixels"][o["feature_index"]])
        for o in observations
    ]
    semantic = votes[0] if len(set(votes)) == 1 and votes[0] != "unknown" else "unknown"
    landmark = {
        "id": f"landmark-{len(geometry['landmarks']):06d}",
        "position": point.tolist(),
        "coordinate_frame": geometry["frame_id"],
        "units": "arbitrary",
        "status": "estimated",
        "observations": observations,
        "semantic_surface": semantic,
        "semantic_votes": votes,
        "max_initial_reprojection_pixels": float(errors),
        "initial_parallax_degrees": float(angle),
        "physically_verified": False,
    }
    geometry["landmarks"].append(landmark)
    return landmark["id"]


def _attach_tracks(state, a, b, pairs, features, assets, policy):
    geometry = state["shared_geometry"]
    points = {p["id"]: p for p in geometry["landmarks"]}
    tracks = {
        (o["view_id"], o["feature_index"]): p["id"]
        for p in geometry["landmarks"]
        for o in p["observations"]
    }
    cameras = geometry["cameras"]
    new = []
    conflicts = 0
    for i, j in pairs:
        i, j = int(i), int(j)
        left, right = tracks.get((a, i)), tracks.get((b, j))
        if left and right and left != right:
            conflicts += 1
            continue
        if left or right:
            landmark = points[left or right]
            valid = True
            for v, k in ((a, i), (b, j)):
                pose = np.array(cameras[v]["camera_from_world"])
                p = np.array(landmark["position"])
                error = np.linalg.norm(
                    project([p], pose, assets[v]["calibration"])[0] - features[v]["pixels"][k]
                )
                if (
                    error > policy.max_reprojection_pixels
                    or (pose[:3, :3] @ p + pose[:3, 3])[2] <= 0
                ):
                    valid = False
            if not valid:
                conflicts += 1
                continue
            for v, k in ((a, i), (b, j)):
                if (v, k) not in tracks:
                    landmark["observations"].append(
                        {
                            "view_id": v,
                            "feature_index": k,
                            "pixel": features[v]["pixels"][k].tolist(),
                        }
                    )
                    tracks[v, k] = landmark["id"]
        else:
            new.append((i, j))
    if new:
        ids = np.asarray(new)
        xyz, valid, angles, errors = triangulate(
            features[a]["pixels"][ids[:, 0]],
            features[b]["pixels"][ids[:, 1]],
            cameras[a]["camera_from_world"],
            cameras[b]["camera_from_world"],
            assets[a]["calibration"],
            assets[b]["calibration"],
            policy,
        )
        for (i, j), p, keep, angle, error in zip(ids, xyz, valid, angles, errors):
            if keep:
                observations = [
                    {
                        "view_id": v,
                        "feature_index": int(k),
                        "pixel": features[v]["pixels"][k].tolist(),
                    }
                    for v, k in ((a, i), (b, j))
                ]
                _add_landmark(geometry, p, observations, assets, features, error, angle)
    if conflicts:
        state["geometric_conflicts"].append(
            {
                "views": [a, b],
                "rejected_track_associations": conflicts,
                "reason": "existing_3d_track_or_reprojection_disagreement",
                "action": "kept_existing_landmarks",
            }
        )


def _registration_candidates(view_id, pair_rows, state, features):
    geometry = state["shared_geometry"]
    tracks = {
        (o["view_id"], o["feature_index"]): p
        for p in geometry["landmarks"]
        for o in p["observations"]
    }
    candidates = {}
    for pair in pair_rows:
        a, b = pair["views"]
        if view_id not in (a, b):
            continue
        other = b if a == view_id else a
        if other not in geometry["cameras"]:
            continue
        for i, j in pair["matches"]:
            old, new = (int(j), int(i)) if a == view_id else (int(i), int(j))
            point = tracks.get((other, old))
            if point:
                candidates.setdefault(new, {})[point["id"]] = point
    # Conflicting 2D-to-3D identity hypotheses are excluded, never silently merged.
    ambiguous = sum(len(v) > 1 for v in candidates.values())
    selected, used = [], set()
    for key, values in sorted(candidates.items()):
        if len(values) == 1:
            point = next(iter(values.values()))
            if point["id"] not in used:
                selected.append((key, point))
                used.add(point["id"])
    return selected, ambiguous


def _check_relative_rotation(view_id, pose, pairs, cameras, policy):
    diagnostics = []
    for pair in pairs:
        a, b = pair["views"]
        if view_id not in (a, b) or pair["geometry"]["status"] != "accepted":
            continue
        if (b if a == view_id else a) not in cameras:
            continue
        pa = np.array(pose if a == view_id else cameras[a]["camera_from_world"])
        pb = np.array(pose if b == view_id else cameras[b]["camera_from_world"])
        estimated = np.array(pair["geometry"]["camera_b_from_a"])
        error = rotation_error(pb[:3, :3] @ pa[:3, :3].T, estimated[:3, :3])
        translation = pb[:3, 3] - (pb[:3, :3] @ pa[:3, :3].T) @ pa[:3, 3]
        denominator = np.linalg.norm(translation) * np.linalg.norm(estimated[:3, 3])
        direction_error = (
            float(
                np.degrees(
                    np.arccos(np.clip(np.dot(translation, estimated[:3, 3]) / denominator, -1, 1))
                )
            )
            if denominator > 1e-10
            else None
        )
        diagnostics.append(
            {
                "views": [a, b],
                "rotation_disagreement_degrees": error,
                "translation_direction_disagreement_degrees": direction_error,
                "status": "consistent"
                if error <= policy.max_rotation_disagreement_degrees
                and (
                    direction_error is None
                    or direction_error <= policy.max_translation_direction_disagreement_degrees
                )
                else "flagged_inconsistent",
            }
        )
    return all(
        d["rotation_disagreement_degrees"] <= policy.max_rotation_disagreement_degrees
        and (
            d["translation_direction_disagreement_degrees"] is None
            or d["translation_direction_disagreement_degrees"]
            <= policy.max_translation_direction_disagreement_degrees
        )
        for d in diagnostics
    ), diagnostics


def reconstruct(state, assets, features, pairs, policy):
    state.setdefault("geometric_conflicts", [])
    geometry = state.setdefault(
        "shared_geometry",
        {
            "frame_id": None,
            "units": "arbitrary",
            "cameras": {},
            "landmarks": [],
            "metric_scale": None,
            "surface_patches": [],
        },
    )
    cameras = geometry["cameras"]
    if not geometry["landmarks"]:
        for pair in pairs:
            if pair["geometry"]["status"] != "accepted":
                continue
            a, b = pair["views"]
            geometry["frame_id"] = "camera-anchor:" + assets[a]["image_sha256"][:16]
            geometry["anchor_view_id"] = a
            for v, pose in ((a, np.eye(4)), (b, np.array(pair["geometry"]["camera_b_from_a"]))):
                cameras[v] = {
                    "camera_from_world": pose.tolist(),
                    "centre": centre(pose).tolist(),
                    "status": "gauge_anchor" if v == a else "estimated_two_view",
                    "units": "arbitrary",
                    "calibration_source": copy.deepcopy(assets[v]["calibration"]),
                }
            _attach_tracks(state, a, b, pair["matches"], features, assets, policy)
            break
    registrations = {}
    for v, camera in cameras.items():
        valid, diagnostics = _check_relative_rotation(
            v, camera["camera_from_world"], pairs, cameras, policy
        )
        if not valid:
            state["geometric_conflicts"].append(
                {
                    "reason": "existing_map_pairwise_pose_inconsistency",
                    "checks": diagnostics,
                    "action": "preserved_existing_map_and_stopped_fusion",
                }
            )
            state["reconstruction_status"] = "inconsistent_geometry_requires_review"
            state["fit_readiness"]["ready"] = False
            state["physical_accuracy_verified"] = False
            return state
    # Repeat only if a new camera opens further shared tracks; never chain independent baselines.
    remaining = [v for v in assets if v not in cameras]
    for _ in range(len(remaining)):
        progress = False
        for v in list(remaining):
            selected, ambiguous = _registration_candidates(v, pairs, state, features)
            result = register_camera(
                [p["position"] for _, p in selected],
                np.array([features[v]["pixels"][k] for k, _ in selected]),
                assets[v]["calibration"],
                assets[v]["size"],
                policy,
            )
            result["ambiguous_2d_3d_hypotheses_excluded"] = ambiguous
            result["shared_3d_candidates"] = len(selected)
            if result["status"] == "accepted":
                consistent, diagnostics = _check_relative_rotation(
                    v, result["camera_from_world"], pairs, cameras, policy
                )
                result["relative_pose_checks"] = diagnostics
                if not consistent:
                    result.update(status="rejected", reason="inconsistent_pairwise_and_pnp_pose")
                else:
                    pose = np.array(result["camera_from_world"])
                    cameras[v] = {
                        "camera_from_world": pose.tolist(),
                        "centre": centre(pose).tolist(),
                        "status": "estimated_pnp_in_existing_frame",
                        "units": "arbitrary",
                        "calibration_source": copy.deepcopy(assets[v]["calibration"]),
                    }
                    for pair in pairs:
                        a, b = pair["views"]
                        if v in (a, b) and a in cameras and b in cameras:
                            _attach_tracks(state, a, b, pair["matches"], features, assets, policy)
                    remaining.remove(v)
                    progress = True
            registrations[v] = result
        if not progress:
            break
    geometry["surface_patches"] = plane_patches(geometry["landmarks"])
    state["registration_attempts"] = registrations
    state["future_views"]["registration_status"] = (
        "partial_sparse_shared_frame" if cameras else "unaligned"
    )
    state["reconstruction_status"] = (
        "estimated_sparse_multiview" if cameras else "single_or_unaligned_image_estimates"
    )
    if cameras:
        state["coordinate_frames"]["camera_relative_3d"] = {"units": "arbitrary", "available": True}
        state["coordinate_frames"]["transforms"]["camera_to_world"] = {
            v: {
                "matrix": np.linalg.inv(np.asarray(c["camera_from_world"])).tolist(),
                "status": "estimated",
            }
            for v, c in cameras.items()
        }
        state["coordinate_frames"]["world_room"] = {
            "units": "arbitrary",
            "available": True,
            "frame_id": geometry["frame_id"],
            "orientation": "anchor camera x-right, y-down, z-forward; not gravity-aligned room axes",
        }
    state["fit_readiness"]["ready"] = False
    state["physical_accuracy_verified"] = False
    return state


def run_project(
    manifest,
    output,
    previous=None,
    layout_run=None,
    segmentation_model=None,
    depth_model=None,
    furniture_predictions=None,
    scale=None,
    policy=Policy(),
):
    start = perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose a new multiview output directory")
    rows = load_manifest(manifest, "rooms")
    header = read_json(manifest)
    state = read_json(previous) if previous else None
    if state:
        validate_photo_scene(state)
    if state and header.get("room_id") and header["room_id"] != state["room_id"]:
        raise ValueError("project room ID differs; unrelated rooms cannot be merged")
    if state and (policy.to_dict() | state.get("matching_policy", {})) != policy.to_dict():
        raise ValueError("resume with the same feature/geometry policy")
    assets = copy.deepcopy(state.get("view_assets", {})) if state else {}
    layouts = {}
    if layout_run:
        layout_run = Path(layout_run).resolve()
        layouts = {
            r["image_id"]: (layout_run / r["folder"], r)
            for r in read_json(layout_run / "run.json")["images"]
            if r["status"] != "error"
        }
    needed = [
        r
        for r in rows
        if r["image_id"] not in assets
        and r["image_id"] not in layouts
        and not r.get("photo_scene_file")
    ]
    if needed and (not segmentation_model or not depth_model):
        raise ValueError(
            "new photos need --layout-run, a photo_scene_file, or both local structure/depth model paths"
        )
    output.mkdir(parents=True)
    if needed:
        pending = output / "pending-photos.json"
        write_json(
            pending,
            {
                "schema_version": 1,
                "images": [
                    {
                        **{k: v for k, v in r.items() if k != "resolved_file"},
                        "file": r["resolved_file"],
                    }
                    for r in needed
                ],
            },
        )
        run = run_layout(
            pending,
            output / "single-photo",
            segmentation_model,
            depth_model,
            furniture_predictions=furniture_predictions,
        )
        layouts.update(
            {
                r["image_id"]: (output / "single-photo" / r["folder"], r)
                for r in run["images"]
                if r["status"] != "error"
            }
        )
    for row in rows:
        v = row["image_id"]
        checksum = digest(row["resolved_file"])
        if v in assets:
            if assets[v]["image_sha256"] != checksum:
                raise ValueError("resumed image bytes changed")
            if (
                row.get("calibration")
                and calibration(row["calibration"]) != assets[v]["calibration"]
            ):
                raise ValueError(
                    "resumed camera calibration changed; explicit reprocessing required"
                )
            continue
        if row.get("photo_scene_file"):
            scene_path = Path(manifest).resolve().parent / row["photo_scene_file"]
            folder = scene_path.parent
            incoming = read_json(scene_path)
        else:
            if v not in layouts:
                raise RuntimeError(f"No structure output for {v}; no artificial room substituted")
            folder, record = layouts[v]
            if record["image_sha256"] != checksum:
                raise ValueError("layout estimates reference changed image bytes")
            incoming = read_json(folder / "room-estimate.json")
        if (
            len(incoming["views"]) != 1
            or incoming["views"][0]["view_id"] != v
            or incoming["views"][0]["image_sha256"] != checksum
        ):
            raise ValueError("photo estimate must be source-bound to its manifest image")
        camera = incoming["views"][0]["camera"]
        supplied = camera.get("provided_calibration_source", {})
        calibrated = calibration(row.get("calibration"))
        if camera.get("intrinsics") is not None:
            if not supplied.get("source"):
                raise ValueError("supplied photo intrinsics need explicit source")
            calibrated = calibration(
                {
                    "intrinsics": camera["intrinsics"],
                    "distortion": [0] * 5,
                    "source": supplied["source"] + "; distortion assumed zero, unverified",
                    "status": "estimated",
                }
            )
        if state is None:
            state = merge_document(None, incoming)
            state["room_id"] = header.get("room_id", "progressive:" + checksum[:16])
        else:
            state = merge_document(state, incoming)
        image = load_image(row["resolved_file"])
        cache = output / f"features-{len(assets):03d}.npz"
        feature_start = perf_counter()
        features = extract_features(image)
        np.savez_compressed(cache, **features)
        assets[v] = {
            "image_path": row["resolved_file"],
            "image_sha256": checksum,
            "size": list(image.size),
            "features_path": str(cache),
            "features_sha256": digest(cache),
            "feature_count": len(features["pixels"]),
            "feature_seconds": perf_counter() - feature_start,
            "calibration": calibrated,
            "labels_path": str(folder / "structure-labels.png")
            if (folder / "structure-labels.png").exists()
            else None,
            "confidence_path": str(folder / "semantic-confidence.npy")
            if (folder / "semantic-confidence.npy").exists()
            else None,
            "rights": row["rights"],
        }
    features = {}
    for v, asset in assets.items():
        view = next(view for view in state["views"] if view["view_id"] == v)
        depth = view.get("relative_depth")
        if depth and depth.get("file") and asset.get("labels_path"):
            original_depth = Path(asset["labels_path"]).parent / Path(depth["file"]).name
            if original_depth.exists():
                depth["file"] = str(original_depth.resolve())
                asset["relative_depth_path"] = str(original_depth.resolve())
        corrected_k = view["camera"].get("intrinsics")
        if corrected_k is not None and (
            asset["calibration"] is None
            or not np.allclose(corrected_k, asset["calibration"]["intrinsics"])
        ):
            raise ValueError(
                "corrected calibration differs from cached map; preserve corrections and explicitly reprocess"
            )
        if (
            digest(asset["image_path"]) != asset["image_sha256"]
            or digest(asset["features_path"]) != asset["features_sha256"]
        ):
            raise ValueError("previous source image or feature cache changed")
        with np.load(asset["features_path"]) as values:
            features[v] = {k: values[k] for k in ("pixels", "descriptors")}
        asset["_labels"] = (
            np.asarray(Image.open(asset["labels_path"])) if asset["labels_path"] else None
        )
        asset["_confidence"] = (
            np.load(asset["confidence_path"]) if asset["confidence_path"] else None
        )
    old_pairs = {tuple(p["views"]): p for p in state.get("pair_reports", [])}
    pairs = []
    for a, b in itertools.combinations(assets, 2):
        if (a, b) in old_pairs:
            pair = copy.deepcopy(old_pairs[a, b])
            pair["reused"] = True
        else:
            pair_start = perf_counter()
            matches = match_features(features[a], features[b])
            if assets[a]["image_sha256"] == assets[b]["image_sha256"]:
                geometry = {
                    "status": "rejected",
                    "reason": "duplicate_image_no_new_parallax",
                    "matches": len(matches),
                }
            else:
                geometry = estimate_pair(
                    features[a]["pixels"][matches[:, 0]],
                    features[b]["pixels"][matches[:, 1]],
                    assets[a]["calibration"],
                    assets[b]["calibration"],
                    assets[a]["size"],
                    assets[b]["size"],
                    policy,
                )
            geometry.pop("points", None)
            pair = {
                "views": [a, b],
                "matches": matches.tolist(),
                "geometry": geometry,
                "seconds": perf_counter() - pair_start,
                "reused": False,
            }
        pairs.append(pair)
    state = reconstruct(state, assets, features, pairs, policy)
    checks = {}
    for v, camera in state["shared_geometry"]["cameras"].items():
        _, diagnostics = _check_relative_rotation(
            v, camera["camera_from_world"], pairs, state["shared_geometry"]["cameras"], policy
        )
        for check in diagnostics:
            checks[tuple(check["views"])] = check
    state["pose_consistency_checks"] = list(checks.values())
    state["view_assets"] = {
        v: {k: val for k, val in asset.items() if not k.startswith("_")}
        for v, asset in assets.items()
    }
    state["pair_reports"] = pairs
    state["matching_policy"] = policy.to_dict()
    state["multiview_schema_version"] = 1
    state.setdefault("progress_history", []).append(
        {
            "manifest_sha256": digest(manifest),
            "input_views": [r["image_id"] for r in rows],
            "total_views": len(assets),
            "registered_views": len(state["shared_geometry"]["cameras"]),
            "landmarks": len(state["shared_geometry"]["landmarks"]),
            "previous_project_sha256": digest(previous) if previous else None,
        }
    )
    if scale:
        state = establish_scale(state, read_json(scale))
    state["photo_guidance"] = photo_guidance(state)
    state["runtime"] = {
        "seconds_excluding_visualization": perf_counter() - start,
        "opencv": opencv().__version__,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "models_loaded_in_this_process": False,
        "feature_method": "CPU SIFT + mutual Lowe ratio 0.75",
    }
    write_json(output / "room.json", state)
    from .visualization import write_outputs

    write_outputs(state, output)
    return state
