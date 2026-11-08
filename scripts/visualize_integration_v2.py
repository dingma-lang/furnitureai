"""Export schematics and model before/after comparisons from saved artifacts only."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from furniture_bench.data import load_image, read_json
from room_reconstruction.calibration import transform_points
from room_reconstruction.collision import footprint
from room_reconstruction.export import COLORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/integration-v2")
    args = parser.parse_args()
    root = Path(args.report)
    output = root / "visuals"
    output.mkdir(parents=True, exist_ok=False)
    try:
        font = ImageFont.load_default(size=27)
    except TypeError:
        font = ImageFont.load_default()
    for index in range(3):
        before = load_image(root / f"original-prompts/nms-050/image-{index:03d}.png")
        after = load_image(root / f"expanded-prompts/nms-030-contained/image-{index:03d}.png")
        panel = Image.new(
            "RGB", (before.width + after.width, max(before.height, after.height) + 100), "white"
        )
        panel.paste(before, (0, 100))
        panel.paste(after, (before.width, 100))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (15, 12),
            "BEFORE: original prompts; confidence 0.15; class NMS 0.5",
            font=font,
            fill="black",
        )
        draw.text(
            (before.width + 15, 12),
            "AFTER: expanded prompts; confidence 0.15; NMS 0.3; IoS 0.9",
            font=font,
            fill="black",
        )
        draw.text(
            (15, 56),
            "Actual model outputs. Experimental comparison against provisional assistant-marked reference boxes.",
            font=font,
            fill="#824123",
        )
        panel.save(output / f"before-after-{index:03d}.png")

    calibration = read_json(root / "floor-calibration.json")
    image = load_image("data/detection-pilot/photos/tosolini-ykK7DS6vueg.jpg")
    draw = ImageDraw.Draw(image)
    width, depth = calibration["dimensions"]["width"], calibration["dimensions"]["depth"]
    matrix = calibration["floor_to_image"]
    for x in range(int(width) + 1):
        points = transform_points(matrix, [[x, 0], [x, depth]])
        draw.line([tuple(p) for p in points], fill="#fa32b3", width=3)
    for y in [0, 1, 2, 3, 4, depth]:
        points = transform_points(matrix, [[0, y], [width, y]])
        draw.line([tuple(p) for p in points], fill="#fa32b3", width=3)
    for point in calibration["image_points_pixels"]:
        x, y = point
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#ffcc00", outline="black")
    a, b, c = calibration["camera_perspective"]["horizon_line"]
    if abs(b) > 1e-9:
        draw.line([(0, -c / b), (image.width, -(a * image.width + c) / b)], fill="#00ffee", width=3)
    draw.rectangle([0, 0, image.width, 120], fill="white")
    draw.text(
        (12, 10),
        "DEMONSTRATION ONLY: assumed floor rectangle and example 6 x 4.5 m scale",
        font=font,
        fill="#8b173f",
    )
    draw.text(
        (12, 48),
        "Magenta: planar grid. Cyan: assumed horizon. Yellow: supplied controls.",
        font=font,
        fill="black",
    )
    draw.text(
        (12, 84),
        "No physical verification; camera intrinsics/3D pose and furniture centres remain unknown.",
        font=font,
        fill="#8b173f",
    )
    image.save(output / "assumed-floor-perspective.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    room = read_json("reports/room-reconstruction-demo/after.json")
    assessment = read_json(root / "arrangement-after.json")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.add_patch(
        Polygon(room["floor"]["boundary"], facecolor="#f5f7fa", edgecolor="#283b53", linewidth=3)
    )
    for item in room["furniture"]:
        axis.add_patch(
            Polygon(
                footprint(item), facecolor=COLORS.get(item["category"], "#ddd"), edgecolor="#39475b"
            )
        )
        x, y, _ = item["position"]
        axis.text(x, y, item["category"], ha="center", va="center", fontsize=10)
    for access in assessment["front_access_zones"]:
        axis.add_patch(
            Polygon(
                access["footprint"],
                facecolor="#f6a54c",
                alpha=0.25,
                edgecolor="#ad4b00",
                hatch="//" if access["obstacles"] else None,
            )
        )
    sofa = next(o for o in room["furniture"] if o["id"] == "sofa-replacement")
    lamp = next(o for o in room["furniture"] if o["id"] == "floor-lamp")
    left = max(x for x, _ in footprint(sofa))
    right = min(x for x, _ in footprint(lamp))
    y = lamp["position"][1]
    axis.plot([left, right], [y, y], "r-", linewidth=4)
    gap = assessment["passage_checks"][0]["local_gap_metres"]
    axis.annotate(
        f"Restrictive side gap: {gap:.3f} m",
        ((left + right) / 2, y),
        xytext=(3.6, 4.38),
        arrowprops={"arrowstyle": "->", "color": "#ad2400"},
        color="#ad2400",
    )
    axis.annotate(
        "Chair-front target extends beyond room",
        (6, 2.2),
        xytext=(3.0, 1.15),
        arrowprops={"arrowstyle": "->", "color": "#ad4b00"},
        color="#ad4b00",
    )
    axis.set(
        xlim=(-0.25, 6.35),
        ylim=(-0.25, 4.8),
        xlabel="x (metres)",
        ylabel="y (metres)",
        title="Geometric fit: PASS | configured arrangement targets: RESTRICTED",
    )
    axis.set_aspect("equal")
    figure.text(
        0.5,
        0.035,
        "Specified demo dimensions. Orange: requested front access. Comfort and real-world dimensions are not verified.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "arrangement-restrictions.png", dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
