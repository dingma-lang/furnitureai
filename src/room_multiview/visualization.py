"""Diagnostic images and editable point-cloud export; no fabricated room mesh."""

import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from furniture_bench.data import load_image, read_json, write_json
from room_layout.visualization import font

COLORS = {
    "floor": "#25ae65",
    "wall": "#4596ed",
    "ceiling": "#e3b52b",
    "other": "#999999",
    "unknown": "#666666",
    "window": "#14b7bf",
    "door": "#d58c33",
}


def write_ply(geometry, path, scale=1.0, units="arbitrary"):
    points = geometry["landmarks"]
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment units {units}; estimated sparse geometry, hidden extent unknown",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p in points:
        color = COLORS.get(p["semantic_surface"], "#999999")
        rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
        xyz = np.asarray(p["position"]) * scale
        lines.append(" ".join([*(f"{v:.8f}" for v in xyz), *(str(v) for v in rgb)]))
    Path(path).write_text("\n".join(lines) + "\n")


def write_outputs(state, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    geometry = state["shared_geometry"]
    write_ply(geometry, output / "sparse-room.ply")
    if geometry.get("metric_scale"):
        write_ply(
            geometry,
            output / "sparse-room-scaled.ply",
            geometry["metric_scale"]["metres_per_unit"],
            "metres_from_supplied_reference",
        )
    write_json(
        output / "cameras.json",
        {
            "frame_id": geometry["frame_id"],
            "units": "arbitrary",
            "world_to_metres": state["coordinate_frames"]["transforms"]["world_to_metres"],
            "registered": geometry["cameras"],
            "unregistered": [v for v in state["view_assets"] if v not in geometry["cameras"]],
        },
    )
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for surface in sorted({p["semantic_surface"] for p in geometry["landmarks"]}):
        points = np.array(
            [p["position"] for p in geometry["landmarks"] if p["semantic_surface"] == surface]
        )
        ax.scatter(*points.T, s=3, color=COLORS.get(surface, "#999999"), label=surface, alpha=0.6)
    if geometry["landmarks"]:
        cloud = np.array([p["position"] for p in geometry["landmarks"]])
        arrow_size = max(0.1, float(np.median(np.linalg.norm(cloud, axis=1))) * 0.08)
        for i, (view_id, camera) in enumerate(geometry["cameras"].items()):
            p = np.array(camera["centre"])
            direction = np.array(camera["camera_from_world"])[:3, :3].T @ [0, 0, 1]
            ax.scatter(*p, marker="^", s=80, color="#c12744")
            ax.quiver(*p, *direction, length=arrow_size, color="#c12744")
            ax.text(*p, f" camera {i}", fontsize=8)
        ax.legend(loc="upper right", markerscale=3)
    else:
        ax.text2D(
            0.1,
            0.5,
            "No aligned 3D points/camera positions.\nVisible single-photo structure remains available.\nDo not interpret these empty axes as room dimensions.",
            transform=ax.transAxes,
        )
    ax.set_xlabel("anchor-camera X (arbitrary units)")
    ax.set_ylabel("anchor-camera Y (arbitrary units)")
    ax.set_zlabel("anchor-camera Z (arbitrary units)")
    ax.set_title(
        f"{len(geometry['cameras'])} aligned cameras; {len(geometry['landmarks'])} sparse landmarks\nEstimated geometry; no closed room, gravity alignment or physical scale"
    )
    fig.tight_layout()
    fig.savefig(output / "reconstruction.png", dpi=140)
    plt.close(fig)
    accepted = [p for p in state["pair_reports"] if p["geometry"]["status"] == "accepted"]
    if accepted:
        pair = max(accepted, key=lambda p: p["geometry"].get("triangulated_inliers", 0))
        a, b = pair["views"]
        first, second = [load_image(state["view_assets"][v]["image_path"]) for v in (a, b)]
        canvas = Image.new(
            "RGB", (first.width + second.width, max(first.height, second.height) + 45), "white"
        )
        canvas.paste(first, (0, 45))
        canvas.paste(second, (first.width, 45))
        draw = ImageDraw.Draw(canvas)
        features = []
        for v in (a, b):
            with np.load(state["view_assets"][v]["features_path"]) as data:
                features.append(data["pixels"])
        indices = pair["geometry"]["match_indices"]
        for index in indices[:: max(1, len(indices) // 80)][:80]:
            i, j = pair["matches"][index]
            x, y = features[0][i]
            u, v = features[1][j]
            draw.line([(x, y + 45), (u + first.width, v + 45)], fill="#20df75", width=1)
        draw.text(
            (10, 12),
            f"Two-view geometric inlier hypotheses: {len(indices)}; sample shown. Unverified physical geometry.",
            fill="black",
        )
        canvas.save(output / "feature-correspondences.png")


def compare(single_project, multi_project, output):
    """Same scene: source-image structure versus actual shared sparse geometry."""
    single, multi = read_json(single_project), read_json(multi_project)
    if "view_assets" not in single:
        # A genuine existing single-photo estimate can be compared directly, without
        # extracting features or rerunning neural models solely for visualization.
        references = {}
        for view in single["views"]:
            key = view["view_id"]
            asset = multi["view_assets"].get(key)
            if asset and view["image_sha256"] == asset["image_sha256"]:
                references[key] = asset
        single["view_assets"] = references
    common = set(single["view_assets"]) & set(multi["view_assets"])
    if not common:
        raise ValueError("comparison needs a common scene view")
    first = next(v for v in single["view_assets"] if v in common)
    if single["view_assets"][first]["image_sha256"] != multi["view_assets"][first]["image_sha256"]:
        raise ValueError("comparison source bytes differ")
    assets = single["view_assets"][first]
    source_folder = Path(assets["labels_path"]).parent if assets["labels_path"] else None
    panels = []
    for path in (
        [source_folder / "layout-overlay.png", source_folder / "relative-depth.png"]
        if source_folder
        else [Path(assets["image_path"])]
    ):
        with Image.open(path) as image:
            # Earlier legends were sized for 1600px sources. Re-caption this
            # comparison at its own display width without changing old outputs.
            image = image.convert("RGB").crop((0, 0, *assets["size"]))
            image.thumbnail((640, 600))
            caption = (
                "Relative depth model scores; per-image color stretch. Unknown physical scale/offset; not metres."
                if path.name == "relative-depth.png"
                else "Estimated semantic regions: blue wall, green floor, yellow ceiling. Hidden surfaces unknown. Semantic scores are uncalibrated; no single-photo metric room."
            )
            lines = textwrap.wrap(caption, width=75)
            panel = Image.new("RGB", (640, image.height + len(lines) * 22 + 12), "white")
            panel.paste(image)
            ImageDraw.Draw(panel).multiline_text(
                (10, image.height + 6), "\n".join(lines), fill="black", font=font(15), spacing=4
            )
            panels.append(panel)
    with Image.open(Path(multi_project).parent / "reconstruction.png") as image:
        right = image.convert("RGB")
        right.thumbnail((1050, 1050))
    canvas = Image.new(
        "RGB", (640 + right.width, max(sum(p.height for p in panels), right.height) + 75), "white"
    )
    y = 65
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    canvas.paste(right, (640, 65))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 12),
        "SINGLE PHOTO: estimated visible structure + unitless relative depth; no metric 3D",
        fill="black",
    )
    draw.text(
        (655, 12),
        "MULTI PHOTO: shared sparse geometry/cameras where alignment succeeds",
        fill="black",
    )
    draw.text(
        (655, 35), "Incomplete room; no physical scale or accuracy gain assumed", fill="#852342"
    )
    canvas.save(output)
