"""Isolated Open3D CPU experiment. No Torch, FAISS, models or reference pose input."""

import argparse
import faulthandler
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.data import digest, load_image, load_manifest, write_json

from .numeric import rigid_pose
from .odometry_quality import OdometryPolicy, alignment_quality, motion_size, validate_candidate


def run(manifest, output, anchor_mode="first"):
    start = perf_counter()
    import open3d as o3d

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("choose new odometry output")
    output.mkdir(parents=True)
    rows = load_manifest(manifest, "rooms")
    policy = OdometryPolicy()
    sources = {}
    frames = {}
    for row in rows:
        spec = row["sensor_depth"]
        path = (Path(manifest).resolve().parent / spec["file"]).resolve()
        if (
            spec["status"] != "sensor_measured"
            or spec["encoding"] != "uint16_depth_z"
            or digest(path) != spec["sha256"]
        ):
            raise ValueError("genuine hash-verified measured depth required")
        scale = float(spec["units_per_metre"])
        if not np.isfinite(scale) or scale <= 0 or not 0 <= spec["rgb_depth_delta_seconds"] <= 0.02:
            raise ValueError("invalid depth units or temporal calibration")
        rgb = np.asarray(load_image(row["resolved_file"]))
        with Image.open(path) as image:
            depth = np.asarray(image, dtype=np.float32) / scale
        depth[(depth < 0.4) | (depth > 5)] = 0
        cal = spec["calibration"]
        k = np.array(cal["intrinsics"], float)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            rgb.shape[1], rgb.shape[0], k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.copy()),
            o3d.geometry.Image(depth.copy()),
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=True,
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic).voxel_down_sample(
            0.02
        )
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
        key = row["image_id"]
        sources[key] = {
            "image_path": row["resolved_file"],
            "image_sha256": digest(row["resolved_file"]),
            "depth_path": str(path),
            "depth_sha256": spec["sha256"],
            "source_rgb_index": row.get("source_rgb_index"),
            "timestamp": row.get("timestamp"),
            "calibration": cal,
        }
        frames[key] = {
            "rgbd": rgbd,
            "cloud": cloud,
            "depth": depth,
            "gray": np.asarray(rgbd.color),
            "intrinsic": intrinsic,
            "k": k,
        }
    keys = list(frames)
    pairs = []
    options = o3d.pipelines.odometry.OdometryOption()
    options.depth_min = 0.4
    options.depth_max = 5.0
    options.depth_diff_max = 0.03
    options.iteration_number_per_pyramid_level = o3d.utility.IntVector([20, 10, 5])

    def compute(a, b):
        first, second = frames[a], frames[b]
        if not np.allclose(first["k"], second["k"]):
            raise ValueError("Open3D experiment requires same intrinsics for paired images")
        success, pose, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            first["rgbd"],
            second["rgbd"],
            first["intrinsic"],
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            options,
        )
        return bool(success), pose, info

    def quality(a, b, pose):
        first, second = frames[a], frames[b]
        return alignment_quality(
            first["depth"], second["depth"], first["gray"], second["gray"], first["k"], pose, policy
        )

    for j in range(1, len(keys)):
        for i in range(max(0, j - 3), j):
            a, b = keys[i], keys[j]
            pair_start = perf_counter()
            record = {"views": [a, b], "sample_gap": j - i, "status": "rejected"}
            try:
                ok, pose, info = compute(a, b)
                back_ok, reverse, _ = compute(b, a)
                if not ok or not back_ok:
                    record["reason"] = "open3d_odometry_returned_failure"
                else:
                    rigid_pose(pose)
                    rigid_pose(reverse)
                    qf, qr = quality(a, b, pose), quality(b, a, reverse)
                    check = validate_candidate(pose, reverse, qf, qr, policy)
                    record.update(
                        status="accepted" if check["accepted"] else "rejected",
                        reason=check["reason"],
                        camera_b_from_a=pose.tolist(),
                        information=np.asarray(info).tolist(),
                        forward=qf,
                        reverse=qr,
                        checks=check,
                    )
                    if check["accepted"]:
                        estimation = (
                            o3d.pipelines.registration.TransformationEstimationPointToPlane()
                        )
                        criterion = o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=30
                        )
                        icp = o3d.pipelines.registration.registration_icp(
                            frames[a]["cloud"],
                            frames[b]["cloud"],
                            0.04,
                            pose,
                            estimation,
                            criterion,
                        )
                        back = o3d.pipelines.registration.registration_icp(
                            frames[b]["cloud"],
                            frames[a]["cloud"],
                            0.04,
                            reverse,
                            estimation,
                            criterion,
                        )
                        iqf, iqr = (
                            quality(a, b, icp.transformation),
                            quality(b, a, back.transformation),
                        )
                        icheck = validate_candidate(
                            icp.transformation, back.transformation, iqf, iqr, policy
                        )
                        delta_t, delta_r = motion_size(np.linalg.inv(pose) @ icp.transformation)
                        adopted = (
                            icheck["accepted"]
                            and delta_t <= 0.03
                            and delta_r <= 2
                            and icp.fitness >= 0.6
                            and icp.inlier_rmse <= 0.025
                            and iqf["median_depth_error_metres"]
                            <= qf["median_depth_error_metres"] + 0.001
                            and iqf["mean_intensity_error"] <= qf["mean_intensity_error"] + 0.005
                        )
                        record["icp"] = {
                            "adopted": adopted,
                            "camera_b_from_a": np.asarray(icp.transformation).tolist(),
                            "fitness": float(icp.fitness),
                            "inlier_rmse_metres": float(icp.inlier_rmse),
                            "delta_translation_metres": delta_t,
                            "delta_rotation_degrees": delta_r,
                            "forward": iqf,
                            "reverse": iqr,
                            "checks": icheck,
                            "reason": "passed_conservative_refinement_checks"
                            if adopted
                            else "retained_odometry_icp_failed_or_worsened_local_checks",
                        }
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                record.update(
                    status="rejected",
                    reason="invalid_or_numerically_failed_geometry",
                    error=str(exc),
                )
            record["seconds"] = perf_counter() - pair_start
            pairs.append(record)
            write_json(
                output / "pairs.json",
                {"sources": sources, "policy": policy.to_dict(), "pairs": pairs},
            )
            print(i, j, record["status"], record["reason"], flush=True)

    def build(use_icp):
        if anchor_mode == "largest-component":
            from .registration_graph import largest_component_cameras
            preferred = [row["image_id"] for row in rows if row.get("primary_scene_view")]
            return largest_component_cameras(keys, pairs, preferred, use_icp)[:3]
        cameras = {keys[0]: np.eye(4)}
        parents = {}
        used = []
        for key in keys[1:]:
            available = [
                p
                for p in pairs
                if p["views"][1] == key and p["views"][0] in cameras and p["status"] == "accepted"
            ]
            if not available:
                continue
            chosen = min(available, key=lambda p: p["sample_gap"])
            pose = np.array(
                chosen["icp"]["camera_b_from_a"]
                if use_icp and chosen.get("icp", {}).get("adopted")
                else chosen["camera_b_from_a"]
            )
            cameras[key] = pose @ cameras[chosen["views"][0]]
            parents[key] = chosen["views"][0]
            used.append(chosen)
        return cameras, parents, used

    variants = {}
    for name, use_icp in [("hybrid", False), ("hybrid-icp", True)]:
        cameras, parents, used = build(use_icp)
        variants[name] = {
            "cameras": {
                k: {"camera_from_sensor_world": v.tolist(), "status": name, "units": "m"}
                for k, v in cameras.items()
            },
            "parents": parents,
            "anchor_view_id": next(iter(cameras)),
        }
    cameras, parents, used = build(True)
    ids = list(cameras)
    graph = o3d.pipelines.registration.PoseGraph()
    for key in ids:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(cameras[key])))
    graph_edges = []
    for pair in pairs:
        a, b = pair["views"]
        if pair["status"] != "accepted" or a not in cameras or b not in cameras:
            continue
        transform = np.array(
            pair["icp"]["camera_b_from_a"]
            if pair.get("icp", {}).get("adopted")
            else pair["camera_b_from_a"]
        )
        predicted = cameras[b] @ np.linalg.inv(cameras[a])
        dt, dr = motion_size(np.linalg.inv(predicted) @ transform)
        if dt > 0.06 or dr > 3:
            graph_edges.append(
                {
                    "views": [a, b],
                    "accepted": False,
                    "reason": "inconsistent_with_existing_pose_paths",
                    "translation_disagreement_metres": dt,
                    "rotation_disagreement_degrees": dr,
                }
            )
            continue
        certain = parents.get(b) == a or parents.get(a) == b
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                ids.index(a),
                ids.index(b),
                transform,
                np.array(pair["information"]),
                uncertain=not certain,
            )
        )
        graph_edges.append({"views": [a, b], "accepted": True, "uncertain": not certain})
    graph_start = perf_counter()
    if len(ids) >= 3 and len(graph.edges) > len(ids) - 1:
        o3d.pipelines.registration.global_optimization(
            graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=0.04, edge_prune_threshold=0.25, reference_node=0
            ),
        )
        optimized = {
            key: np.linalg.inv(np.asarray(graph.nodes[i].pose)) for i, key in enumerate(ids)
        }
        deltas = [motion_size(np.linalg.inv(cameras[key]) @ optimized[key]) for key in ids]
        adopted = all(t <= 0.05 and r <= 2 for t, r in deltas)
        for pair in used:
            a, b = pair["views"]
            candidate = optimized[b] @ np.linalg.inv(optimized[a])
            q = quality(a, b, candidate)
            adopted &= (
                q["accepted"]
                and q["median_depth_error_metres"]
                <= pair["forward"]["median_depth_error_metres"] + 0.003
            )
        variants["pose-graph"] = {
            "cameras": {
                k: {
                    "camera_from_sensor_world": v.tolist(),
                    "status": "pose_graph_estimate",
                    "units": "m",
                }
                for k, v in (optimized if adopted else cameras).items()
            },
            "parents": parents,
            "anchor_view_id": next(iter(cameras)),
            "optimizer_adopted": bool(adopted),
            "deltas": deltas,
            "edge_checks": graph_edges,
            "seconds": perf_counter() - graph_start,
        }
    else:
        variants["pose-graph"] = {
            **variants["hybrid-icp"],
            "optimizer_adopted": False,
            "reason": "no_redundant_accepted_constraints",
            "edge_checks": graph_edges,
        }
    for name, value in variants.items():
        write_json(output / (name + ".json"), value)
    report = {
        "sources": sources,
        "manifest_sha256": digest(manifest),
        "policy": policy.to_dict(),
        "open3d": o3d.__version__,
        "numpy": np.__version__,
        "variants": {k: {"registered": len(v["cameras"])} for k, v in variants.items()},
        "pair_count": len(pairs),
        "accepted_pairs": sum(p["status"] == "accepted" for p in pairs),
        "seconds": perf_counter() - start,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "reference_poses_used": False,
        "models_loaded": False,
        "anchor_mode": anchor_mode,
    }
    write_json(output / "report.json", report)
    return report


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-mode", choices=["first", "largest-component"], default="first")
    args = parser.parse_args()
    try:
        report = run(args.manifest, args.output, args.anchor_mode)
        print({k: v for k, v in report.items() if k not in ("sources", "policy")})
    except Exception as exc:
        output = Path(args.output)
        if output.is_dir():
            write_json(
                output / "failure.json",
                {"type": type(exc).__name__, "message": str(exc), "geometry_substituted": False},
            )
        raise


if __name__ == "__main__":
    main()
