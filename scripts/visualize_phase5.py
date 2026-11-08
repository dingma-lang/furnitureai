"""Diagnostic views, sparse/dense comparison and supported surfaces from saved outputs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from furniture_bench.data import read_json, write_json
from room_layout.visualization import font
from room_multiview.geometry import project


def rgb_residual_plot(state, output):
    values = []
    geometry = state["shared_geometry"]
    for landmark in geometry["landmarks"]:
        for observation in landmark["observations"]:
            view = observation["view_id"]
            pixel = project(
                [landmark["position"]],
                geometry["cameras"][view]["camera_from_world"],
                state["view_assets"][view]["calibration"],
            )[0]
            values.append(float(np.linalg.norm(pixel - observation["pixel"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20)
    ax.set_xlabel("RGB reprojection residual (source pixels)")
    ax.set_ylabel("Image observations")
    ax.set_title("Internal consistency of accepted RGB tracks; not independent room accuracy")
    fig.tight_layout()
    fig.savefig(Path(output) / "rgb-reprojection-distribution.png", dpi=140)
    plt.close(fig)
    np.save(Path(output) / "rgb-reprojection-residuals.npy", values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-project", required=True)
    parser.add_argument("--sensor-project", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rgb = read_json(args.rgb_project)
    state = read_json(args.sensor_project)
    sensor = state["sensor_reconstruction"]
    rgb_residual_plot(rgb, output)
    sources = sensor["sources"]
    count = len(sources)
    sheet = Image.new("RGB", (5 * 320, ((count + 4) // 5) * 280), "white")
    for i, (view, source) in enumerate(sources.items()):
        x, y = (i % 5) * 320, (i // 5) * 280
        with Image.open(source["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((320, 240))
            sheet.paste(image, (x, y))
        draw = ImageDraw.Draw(sheet)
        status = "RGB-D registered" if view in sensor["cameras"] else "RGB-D REJECTED"
        draw.text(
            (x + 5, y + 242),
            f"{i}: {status}",
            fill="green" if view in sensor["cameras"] else "red",
            font=font(15),
        )
        draw.text(
            (x + 5, y + 260),
            "RGB-only "
            + ("registered" if view in rgb["shared_geometry"]["cameras"] else "unregistered"),
            fill="black",
            font=font(13),
        )
    sheet.save(output / "registered-views.png")
    with np.load(sensor["point_cloud"]["file"]) as z:
        cloud = {k: z[k] for k in z.files}
    fig = plt.figure(figsize=(15, 7))
    for i, title in enumerate(
        ["RGB-only sparse map (arbitrary units)", "RGB-D observed cloud (metres, estimated poses)"]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if i == 0:
            g = rgb["shared_geometry"]
            points = np.array([p["position"] for p in g["landmarks"]]).reshape(-1, 3)
            if len(points):
                ax.scatter(*points.T, s=2, c="#777777")
            cams = {k: np.linalg.inv(v["camera_from_world"]) for k, v in g["cameras"].items()}
        else:
            ids = np.arange(0, len(cloud["points"]), max(1, len(cloud["points"]) // 20000))
            ax.scatter(*cloud["points"][ids].T, s=1, c=cloud["colors"][ids] / 255)
            cams = {
                k: np.linalg.inv(v["camera_from_sensor_world"])
                for k, v in sensor["cameras"].items()
            }
        for cam in cams.values():
            ax.scatter(*cam[:3, 3], marker="^", color="#cd2244", s=12)
        ax.set_title(title)
        ax.set_xlabel("anchor X")
        ax.set_ylabel("anchor Y")
        ax.set_zlabel("anchor Z")
    fig.suptitle(
        "Different reconstruction modes/frames; no evaluation scale fitted; neither is a complete room"
    )
    fig.tight_layout()
    fig.savefig(output / "sparse-vs-dense.png", dpi=140)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ids = np.arange(0, len(cloud["points"]), max(1, len(cloud["points"]) // 12000))
    ax.scatter(*cloud["points"][ids].T, s=1, c="#bbbbbb", alpha=0.2)
    palette = {"floor": "#20a765", "wall": "#3986d5", "ceiling": "#d6a428"}
    summary = []
    for plane in sensor["surface_planes"]:
        ids = np.array(plane["support_voxel_indices"])[::3]
        ax.scatter(
            *cloud["points"][ids].T,
            s=3,
            c=palette[plane["surface_hypothesis"]],
            label=plane["id"] + " " + plane["surface_hypothesis"],
        )
        xyz = cloud["points"][plane["support_voxel_indices"]]
        uv = np.einsum(
            "nj,ij->ni",
            xyz - plane["origin_metres"],
            np.array(plane["tangent_axes"]),
            optimize=False,
        )
        span = np.ptp(uv, axis=0)
        summary.append(
            {
                "plane_id": plane["id"],
                "surface_hypothesis": plane["surface_hypothesis"],
                "observed_tangent_span_metres": span.tolist(),
                "is_complete_wall_or_room_dimension": False,
                "metric_source": sensor["metric_calibration"],
                "verified_physical_measurement": False,
            }
        )
    if sensor["surface_planes"]:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        "Supported semantic plane samples only; grey = other/unknown observed depth\nNo inferred surfaces filled; gravity/hidden room boundary unknown"
    )
    ax.set_xlabel("sensor anchor X (m)")
    ax.set_ylabel("sensor anchor Y (m)")
    ax.set_zlabel("sensor anchor Z (m)")
    fig.tight_layout()
    fig.savefig(output / "observed-room-surfaces.png", dpi=140)
    plt.close(fig)
    write_json(
        output / "physical-scale-observed-spans.json",
        {
            "sensor_units_per_metre": 5000,
            "sensor_depth_scale_applied": True,
            "evaluation_reference_scale_used": False,
            "observed_plane_support_spans": summary,
            "note": "Spans describe observed samples transformed by estimated camera poses, not verified wall lengths or room dimensions.",
        },
    )
    residuals = [
        p["median_3d_residual_metres"] for p in sensor["pair_reports"] if p["status"] == "accepted"
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=15)
    ax.set_xlabel("Accepted RGB-D pair median 3D match residual (m)")
    ax.set_ylabel("Pairs")
    ax.set_title("Internal metric registration consistency, not independent accuracy")
    fig.tight_layout()
    fig.savefig(output / "rgbd-residual-distribution.png", dpi=140)
    plt.close(fig)
    # Preserve accepted and rejected correspondences, each drawn from measured source observations.
    pairs = sensor["pair_reports"]
    chosen = []
    for status in ["accepted", "rejected"]:
        candidates = [p for p in pairs if p["status"] == status and p.get("pixels_a")]
        if candidates:
            chosen.append(max(candidates, key=lambda p: len(p["pixels_a"])))
    for pair in chosen:
        a, b = pair["views"]
        canvas = Image.new("RGB", (1280, 520), "white")
        for x, view in ((0, a), (640, b)):
            with Image.open(sources[view]["image_path"]) as im:
                canvas.paste(im.convert("RGB"), (x, 40))
        draw = ImageDraw.Draw(canvas)
        accepted = set(pair.get("inlier_indices", []))
        step = max(1, len(pair["pixels_a"]) // 70)
        for i in range(0, len(pair["pixels_a"]), step):
            x, y = pair["pixels_a"][i]
            u, v = pair["pixels_b"][i]
            draw.line(
                [(x, y + 40), (u + 640, v + 40)],
                fill="#20b56a" if i in accepted else "#df6570",
                width=1,
            )
        draw.text((10, 10), pair["status"] + ": " + pair["reason"], fill="black", font=font(17))
        canvas.save(output / f"{pair['status']}-rgbd-matches.png")
    print(output.resolve())


if __name__ == "__main__":
    main()
