from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    1: (70, 150, 245),
    2: (45, 200, 100),
    3: (245, 200, 55),
    4: (60, 220, 225),
    5: (240, 120, 50),
    6: (165, 100, 220),
}


def font(size=20):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def dashed(draw, endpoints, fill, width=4):
    a, b = np.asarray(endpoints, dtype=float)
    length = np.linalg.norm(b - a)
    count = max(2, int(length / 10))
    for i in range(0, count, 2):
        draw.line(
            [tuple(a + (b - a) * i / count), tuple(a + (b - a) * min(i + 1, count) / count)],
            fill=fill,
            width=width,
        )


def structural_overlay(image, labels, confidence, structure, output):
    base = np.asarray(image).astype(float)
    tint = base.copy()
    for code, color in COLORS.items():
        mask = (labels == code) & (confidence >= structure["analysis"]["min_confidence"])
        tint[mask] = base[mask] * 0.72 + np.array(color) * 0.28
    result = Image.fromarray(tint.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for line in structure["boundaries"]:
        if line["visibility"] == "inferred_occluded":
            dashed(draw, line["endpoints"], "#ee22bb")
        else:
            color = {
                "wall_floor": "#22ff77",
                "wall_ceiling": "#ffdc35",
                "wall_window": "#22dddd",
                "wall_door": "#ee8833",
            }[line["kind"]]
            draw.line([tuple(p) for p in line["endpoints"]], fill=color, width=4)
    for line in structure["wall_wall_candidates"]:
        dashed(draw, line["endpoints"], "#ff7766", width=3)
    for corner in structure["corner_candidates"]:
        x, y = corner["pixel"]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="red", width=3)
        draw.text(
            (x + 8, y + 2), f"candidate {corner['confidence']:.2f}", fill="red", font=font(15)
        )
    # Legend is outside the source image; it does not cover evidence in the photo.
    canvas = Image.new("RGB", (image.width, image.height + 130), "white")
    canvas.paste(result)
    legend = ImageDraw.Draw(canvas)
    legend.text(
        (12, image.height + 8),
        "MODEL ESTIMATES: blue wall / green floor / yellow ceiling / cyan window / orange door / purple covering",
        fill="black",
        font=font(18),
    )
    legend.text(
        (12, image.height + 36),
        "Solid: visible-pixel boundary support. Dashed magenta: inferred occlusion bridge. Red: unverified corner/edge candidates.",
        fill="black",
        font=font(17),
    )
    legend.text(
        (12, image.height + 65),
        "No assumed room rectangle, fixed corners, physical dimensions or calibrated horizon.",
        fill="#852342",
        font=font(20),
    )
    scores = "; ".join(f"{r['surface']} {r['confidence_mean']:.2f}" for r in structure["regions"])
    legend.text(
        (12, image.height + 98),
        "Mean semantic scores (uncalibrated): " + (scores or "none"),
        fill="black",
        font=font(17),
    )
    canvas.save(output)


def depth_image(depth, output):
    from matplotlib import colormaps

    lower, upper = np.percentile(depth, [2, 98])
    normalized = np.clip((depth - lower) / max(float(upper - lower), 1e-9), 0, 1)
    rgb = (colormaps["viridis"](normalized)[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    canvas = Image.new("RGB", (image.width, image.height + 70), "white")
    canvas.paste(image)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, image.height + 8),
        "Relative depth-model score: per-image 2nd–98th percentile color stretch. NO METRES.",
        fill="black",
        font=font(20),
    )
    draw.text(
        (12, image.height + 38),
        "Raw numeric scores are saved separately. Unknown scale/offset; not projected into physical 3D.",
        fill="#852342",
        font=font(18),
    )
    canvas.save(output)
    return {
        "display_percentiles": [2, 98],
        "display_range_raw": [float(lower), float(upper)],
        "representation": "raw relative depth-model score",
        "physical_units": None,
        "scale_and_offset": "unknown",
        "metric_transform": None,
    }


def floorplan_status(output):
    image = Image.new("RGB", (1100, 350), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    lines = [
        "TOP-DOWN METRIC FLOOR PLAN: UNAVAILABLE",
        "Visible image regions do not establish a closed world-space room.",
        "Camera pose/intrinsics, hidden walls and metric scale remain unresolved.",
        "No rectangle or dimensions were substituted.",
        "Use independently measured room geometry, calibrated multi-view imagery or a compatible scan.",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (30, 30 + index * 58),
            line,
            fill="#852342" if index == 0 else "#283b53",
            font=font(25 if index == 0 else 20),
        )
    image.save(output)


def compare_previous(previous, current, output):
    with Image.open(previous) as source, Image.open(current) as replacement:
        width = max(source.width, replacement.width)
        left = source.convert("RGB")
        right = replacement.convert("RGB")
        canvas = Image.new("RGB", (2 * width, max(left.height, right.height) + 55), "white")
        canvas.paste(left, (0, 55))
        canvas.paste(right, (width, 55))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (15, 10),
            "PREVIOUS: manually assumed grid; not physical reconstruction",
            font=font(24),
            fill="#852342",
        )
        draw.text(
            (width + 15, 10),
            "NEW: automatic visible-structure estimates; unknown geometry retained",
            font=font(24),
            fill="#283b53",
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
