"""Summarize saved controlled RGB and RGB-D results without rerunning models."""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from furniture_bench.data import read_json, write_json

ROOT = Path("reports/phase5-public-room")


def main():
    comparison = read_json(ROOT / "rgb-controlled/comparison.json")
    audit = read_json(ROOT / "cached-rgbd-audit/components.json")
    variants = read_json(ROOT / "cached-rgbd-audit/variant-pose-audit.json")
    rows = []
    for result in comparison["configurations"]:
        counts = {}
        for pair in result["pairs"]:
            for index, count in zip(pair["raw_indices"], pair["features"]):
                counts[index] = count
        rows.append({"resolution": result["resolution"], "features": result["configuration"], "gap_seconds": result["gap_seconds_approximately"], "tested_pairs": len(result["pairs"]), "geometry_accepted": result["geometrically_accepted"], "pose_audit_passed": result["reference_audit_passed"], "median_keypoints_in_tested_views": float(np.median(list(counts.values())))})
    with (ROOT / "rgb-controlled/summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, gap in zip(axes, (0.2, 0.5, 3.0)):
        selected = [r for r in rows if r["gap_seconds"] == gap and r["resolution"] != "vga-downsampled"]
        labels = [r["resolution"] + "\n" + r["features"] for r in selected]
        x = np.arange(len(selected))
        ax.bar(x - .17, [r["geometry_accepted"] for r in selected], .34, label="Geometric acceptance")
        ax.bar(x + .17, [r["pose_audit_passed"] for r in selected], .34, label="Also passes pose audit")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 8)
        ax.set_title(f"{gap:g} s spacing: eight pairs/config")
    axes[0].set_ylabel("Pairs; not connected room cameras")
    axes[-1].legend(fontsize=7)
    fig.suptitle("RGB-only controlled comparison; 40 matches / 25 inliers unchanged")
    fig.tight_layout()
    fig.savefig(ROOT / "rgb-controlled/comparison.png", dpi=160)
    plt.close(fig)

    approved = [c for c in audit["components"] if c["fusion_approved_by_audit"]]
    fragment = approved[0]
    fig = plt.figure(figsize=(11, 5))
    for index, (title, poses) in enumerate((("Largest component: rejected for fusion", variants["hybrid-icp"]["per_camera"]), ("Audited fragment: incomplete surface support", fragment["pose_audit"]["per_camera"]))):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        for field, label in (("estimated_centre_metres", "Estimated RGB-D"), ("reference_centre_metres", "ARKit reference (correlated sensor)")):
            points = np.array([p[field] for p in poses])
            ax.plot(*points.T, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Anchor x (m)")
        ax.set_ylabel("Anchor y (m)")
        ax.set_zlabel("Anchor z (m)")
        ax.legend(fontsize=7)
    fig.suptitle("Separately anchored camera trajectories; axes are not gravity-aligned")
    fig.tight_layout()
    fig.savefig(ROOT / "camera-trajectories.png", dpi=160)
    plt.close(fig)

    folder = Path(fragment["fusion"]["directory"])
    state = read_json(folder / "room.json")
    first = next(iter(state["sensor_reconstruction"]["cameras"]))
    source = state["sensor_reconstruction"]["sources"][first]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(Image.open(source["image_path"]))
    ax.set_title("Single photo: image evidence only")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(Image.open(source["image_path"]))
    labels = np.array(Image.open(source["labels_path"]))
    wall = np.zeros((*labels.shape, 4))
    wall[labels == 1] = [0.2, 0.5, 0.9, 0.4]
    ax.imshow(wall)
    ax.set_title("Blue: cached model wall candidate")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    with np.load(folder / "fused-observed-cloud.npz") as cloud:
        points, colors = cloud["points"], cloud["colors"] / 255
    ax.scatter(*points.T, c=colors, s=1)
    ax.set_xlabel("Anchor x (m)")
    ax.set_ylabel("Anchor y (m)")
    ax.set_zlabel("Anchor z (m)")
    ax.set_title("Two primary depth views: partial cloud\nNo supported floor/wall mesh")
    fig.suptitle("Same scene: unknown dimensions and hidden surfaces remain unresolved")
    fig.tight_layout()
    fig.savefig(ROOT / "single-versus-multiview.png", dpi=160)
    plt.close(fig)

    with np.load(folder / "fused-observed-cloud.npz") as cloud:
        support = {str(int(label)): {"voxels": int(np.sum(cloud["labels"] == label)), "multi_view_voxels": int(np.sum((cloud["labels"] == label) & (cloud["view_count"] >= 2)))} for label in np.unique(cloud["labels"])}
    report = {"rgb_comparison": rows, "rgb_comparison_seconds": comparison["seconds"], "vga_download_bytes": read_json("data/research-restricted/arkitscenes/42444946/vga-controlled-v1/receipt.json")["downloaded_bytes"], "rgbd_largest_component_variants": {k: v["all_registered"] for k, v in variants.items()}, "component_count": len(audit["components"]), "audited_accepted_fragments": len(approved), "fragment": {"id": fragment["id"], "pose_errors": fragment["pose_audit"]["all_registered"], "cloud_voxels": fragment["fusion"]["points"], "surface_triangles": fragment["fusion"]["triangles"], "semantic_support": support, "floor_supported_area_m2": 0.0, "wall_supported_area_m2": 0.0, "externally_verified_surface_area_m2": None, "global_heldout": fragment["fusion"]["heldout_evaluation"], "local_heldout": read_json(folder / "local-heldout-evaluation.json")}, "physical_scale": read_json(ROOT / "scale-audit.json"), "historical_different_dataset_baseline": read_json(ROOT / "phase51-baseline.json"), "physical_placement_enabled": False, "new_full_odometry_runs": 0, "interpretation": "Some validated RGB two-view pairs and one accurate local RGB-D camera component. No meaningful supported room-wide floor/wall reconstruction. ARKit reference consistency is not externally verified accuracy."}
    write_json(ROOT / "final-evaluation.json", report)


if __name__ == "__main__":
    main()
