"""Build bounded diagnostic artifacts from preserved Phase 5 and recovery outputs."""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from furniture_bench.data import read_json, write_json
from room_layout.visualization import font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/phase5-registration-recovery")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    comparison = read_json(root / "registration-comparison.json")
    original = read_json("reports/phase5-room-reconstruction/final/room.json")[
        "sensor_reconstruction"
    ]
    recovered = read_json(root / "fused-hybrid/room.json")["sensor_reconstruction"]
    workers = read_json(root / "short-baseline-odometry/report.json")
    baseline_eval = read_json("reports/phase5-room-reconstruction/evaluation.json")
    recovery_eval = read_json(root / "fused-hybrid-evaluation.json")
    diagnosis = read_json(root / "failure-diagnosis-raw.json")["views"]
    nearest = read_json(root / "consecutive-feature-diagnosis.json")["pairs"]
    before_by_id = {r["view_id"]: r for r in diagnosis}
    pairs_by_target = {r["views"][1]: r for r in nearest}
    diagnoses = []
    for row in diagnosis:
        if row["registered"]:
            continue
        best = row["best_mutual_matches"]
        depth = row["best_matches_with_depth"]
        local = pairs_by_target.get(row["view_id"], {})
        category = "insufficient_shared_correspondences_after_tracking_loss"
        if best is not None and best >= 30 and depth < 30:
            category = "valid_depth_filter_reduced_correspondences_below_minimum"
        flags = []
        if row["sift_keypoints"] < 300:
            flags.append("limited_local_texture_heuristic")
        if row["laplacian_variance"] < 60:
            flags.append("low_sharpness_or_smooth_surface_heuristic_not_verified_blur")
        if local.get("status") == "accepted":
            flags.append("local_pair_can_align_but_parent_was_not_registered")
        diagnoses.append(
            {
                **row,
                "primary_classification": category,
                "contributing_flags": flags,
                "nearest_predecessor_mutual_matches": local.get("mutual_matches"),
                "nearest_predecessor_valid_depth_matches": local.get("matches_with_depth"),
                "nearest_predecessor_status": local.get("status"),
                "camera_geometry": "overlap cascade supported by stale-parent versus consecutive-pair match counts; true texture/blur labels unavailable",
                "inconsistent_pose_estimate_accepted": False,
            }
        )
    write_json(
        output / "failure-classification.json",
        {
            "failed_views": len(diagnoses),
            "primary_counts": dict(Counter(r["primary_classification"] for r in diagnoses)),
            "views": diagnoses,
            "notes": [
                "Depth validity is 81–84% globally; invalid/edge depth can still remove key feature correspondences.",
                "Low sharpness does not prove motion blur; a successfully registered view also has low Laplacian variance.",
                "Many downstream pairs have enough local features but cannot join the disconnected original map.",
            ],
        },
    )

    def surface_summary(sensor):
        return {
            kind: {
                "plane_candidates": sum(
                    p["surface_hypothesis"] == kind for p in sensor["surface_planes"]
                ),
                "observed_support_area_estimate_m2": sum(
                    p["observed_support_area_estimate_m2"]
                    for p in sensor["surface_planes"]
                    if p["surface_hypothesis"] == kind
                ),
                "multiview_support_voxels": sum(
                    p["support_points"]
                    for p in sensor["surface_planes"]
                    if p["surface_hypothesis"] == kind
                ),
            }
            for kind in ("wall", "floor", "ceiling")
        }

    write_json(
        output / "coverage-comparison.json",
        {
            "baseline": surface_summary(original),
            "recovery": surface_summary(recovered),
            "baseline_heldout_points_within_5cm": baseline_eval[
                "heldout_sensor_geometry_to_predicted"
            ]["fraction_within_5cm"],
            "recovery_heldout_points_within_5cm": recovery_eval[
                "heldout_sensor_geometry_to_predicted"
            ]["fraction_within_5cm"],
            "same_heldout_samples": baseline_eval["heldout_sensor_geometry_to_predicted"]["samples"]
            == recovery_eval["heldout_sensor_geometry_to_predicted"]["samples"],
            "complete_room_coverage": None,
            "structural_semantic_accuracy": None,
            "interpretation": "Plane occupancy is observed model-classified support area, not a verified complete floor plan. Held-out coverage uses separate sensor observations in reference camera poses.",
        },
    )
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    rows = comparison["variants"]["hybrid"]["per_camera"]
    reference = np.array([r["reference_centre_metres"] for r in rows])
    ax.plot(
        *reference.T, color="black", label="Motion-capture reference (evaluation only)", linewidth=2
    )
    for name, color in [
        ("hybrid", "#2678c4"),
        ("hybrid-icp", "#dc8640"),
        ("pose-graph", "#48a074"),
    ]:
        points = np.array(
            [r["estimated_centre_metres"] for r in comparison["variants"][name]["per_camera"]]
        )
        ax.plot(*points.T, label=name, color=color)
    points = np.array([r["estimated_centre_metres"] for r in comparison["baseline"]["per_camera"]])
    ax.scatter(*points.T, label="Phase 5 feature baseline (3 views)", marker="x", s=90, c="#c83158")
    ax.set_xlabel("Anchor camera X (m)")
    ax.set_ylabel("Anchor camera Y (m)")
    ax.set_zlabel("Anchor camera Z (m)")
    ax.set_title(
        "53 connected tracking frames; 9 original views available for fusion\nNo reference scale or reference pose used in tracking; trajectory stops after failed edges"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "camera-trajectories.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name, color in [
        ("hybrid", "#2678c4"),
        ("hybrid-icp", "#dc8640"),
        ("pose-graph", "#48a074"),
    ]:
        rows = comparison["variants"][name]["per_camera"]
        indices = [workers["sources"][r["view_id"]]["source_rgb_index"] for r in rows]
        axes[0].plot(
            indices, [r["translation_error_metres"] * 100 for r in rows], color=color, label=name
        )
        axes[1].plot(indices, [r["rotation_error_degrees"] for r in rows], color=color, label=name)
    rows = comparison["baseline"]["per_camera"]
    indices = [before_by_id[r["view_id"]]["rgb_index"] for r in rows]
    axes[0].scatter(
        indices,
        [r["translation_error_metres"] * 100 for r in rows],
        color="#c83158",
        marker="x",
        label="Phase 5 baseline",
    )
    axes[1].scatter(
        indices, [r["rotation_error_degrees"] for r in rows], color="#c83158", marker="x"
    )
    axes[0].set_ylabel("Position error (cm)")
    axes[1].set_ylabel("Rotation error (degrees)")
    axes[1].set_xlabel("Original RGB frame index")
    for ax in axes:
        ax.set_xlim(480, 600)
        ax.grid(alpha=0.2)
    axes[0].legend()
    axes[0].set_title("Independent camera-reference errors; unregistered intervals remain blank")
    fig.tight_layout()
    fig.savefig(output / "camera-error-curves.png", dpi=140)
    plt.close(fig)
    width = 5 * 256
    sheet = Image.new("RGB", (width, 4 * 242), "white")
    for i, (view, source) in enumerate(original["sources"].items()):
        x, y = (i % 5) * 256, (i // 5) * 242
        with Image.open(source["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 192))
            sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        index = before_by_id[view]["rgb_index"]
        draw.text(
            (x + 5, y + 194),
            f"Frame {index}: baseline {'yes' if view in original['cameras'] else 'no'}",
            fill="black",
            font=font(14),
        )
        draw.text(
            (x + 5, y + 215),
            "Recovery " + ("registered" if view in recovered["cameras"] else "REJECTED"),
            fill="green" if view in recovered["cameras"] else "red",
            font=font(15),
        )
    sheet.save(output / "original-view-registration.png")
    clouds = []
    for sensor in (original, recovered):
        with np.load(sensor["point_cloud"]["file"]) as cloud:
            clouds.append({k: cloud[k] for k in cloud.files})
    combined = np.concatenate([c["points"] for c in clouds])
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    fig = plt.figure(figsize=(15, 7))
    for i, (sensor, cloud) in enumerate(zip((original, recovered), clouds)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ids = np.arange(0, len(cloud["points"]), 3)
        ax.scatter(*cloud["points"][ids].T, s=1, c="#bbbbbb", alpha=0.2)
        first = True
        second = True
        for plane in sensor["surface_planes"]:
            ids = np.array(plane["support_voxel_indices"])[::3]
            kind = plane["surface_hypothesis"]
            color = {"wall": "#3986d5", "floor": "#20a765", "ceiling": "#d9aa34"}[kind]
            label = kind if (kind == "wall" and first) or (kind == "floor" and second) else None
            ax.scatter(*cloud["points"][ids].T, s=3, color=color, label=label)
            if kind == "wall":
                first = False
            if kind == "floor":
                second = False
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.legend()
        ax.set_title(
            "Phase 5: 3 original views"
            if i == 0
            else "Recovery: 9 original views; floor support added"
        )
        ax.set_xlabel("Anchor X (m)")
        ax.set_ylabel("Anchor Y (m)")
        ax.set_zlabel("Anchor Z (m)")
    fig.suptitle(
        "Observed depth only; blue wall / green floor / grey other or unknown\nSemantic classes unverified; no hidden geometry, complete room, or gravity alignment assumed"
    )
    fig.tight_layout()
    fig.savefig(output / "observed-surfaces-before-after.png", dpi=140)
    plt.close(fig)
    pair_records = read_json(root / "short-baseline-odometry/pairs.json")["pairs"]
    accepted = [p for p in pair_records if p["status"] == "accepted"]
    write_json(
        output / "odometry-edge-summary.json",
        {
            "tested_edges": len(pair_records),
            "accepted_edges": len(accepted),
            "rejection_counts": dict(
                Counter(p["reason"] for p in pair_records if p["status"] != "accepted")
            ),
            "icp_proposals": len(accepted),
            "icp_adopted_locally": sum(p.get("icp", {}).get("adopted", False) for p in accepted),
            "connected_tracking_frames": workers["variants"]["hybrid"]["registered"],
            "unregistered_tracking_frames": len(workers["sources"])
            - workers["variants"]["hybrid"]["registered"],
            "inference_seconds": workers["seconds"],
            "peak_worker_rss_bytes": workers["peak_process_rss_bytes"],
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([p["forward"]["overlap"] for p in accepted], bins=20)
    axes[0].set_xlabel("Accepted directed depth overlap fraction")
    axes[1].hist([p["checks"]["cycle_translation_metres"] * 100 for p in accepted], bins=20)
    axes[1].set_xlabel("Forward/reverse cycle position error (cm)")
    fig.suptitle(
        "Internal validation of accepted edges; independent camera errors shown separately"
    )
    fig.tight_layout()
    fig.savefig(output / "edge-consistency.png", dpi=140)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
