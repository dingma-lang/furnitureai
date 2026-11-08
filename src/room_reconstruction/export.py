from __future__ import annotations

import html
import re
from pathlib import Path

from .collision import footprint, opening_clearance, solid_wall_parts, wall_body
from .model import validate_scene, wall_length

COLORS = {
    "sofa": "#c5daef",
    "table": "#e8d5b5",
    "chair": "#d8c9eb",
    "cabinet": "#d1ded5",
    "lamp": "#f6dfa1",
}


def floorplan_svg(scene, path, title, check):
    validate_scene(scene)
    width, depth, _ = scene["dimensions"]
    scale, margin, top = 85, 75, 110
    canvas_width = margin * 2 + width * scale + 360
    canvas_height = top + depth * scale + 135

    def xy(point):
        return margin + point[0] * scale, top + (depth - point[1]) * scale

    def points(polygon):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, polygon))

    def text(x, y, value, size=14, extra=""):
        return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" {extra}>{html.escape(str(value))}</text>'

    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="Arial, sans-serif" fill="#243246">',
        text(35, 35, title, 24),
        text(
            35,
            62,
            "Known demonstration dimensions; geometry was not recovered from the photograph.",
        ),
        f'<polygon points="{points(scene["floor"]["boundary"])}" fill="#f5f7fa" stroke="#aaa"/>',
    ]
    for wall in scene["walls"]:
        body = wall_body(wall, 0, wall_length(wall), 0, wall["height"])
        fragments.append(f'<polygon points="{points(footprint(body))}" fill="#39475b"/>')
    wall_map = {wall["id"]: wall for wall in scene["walls"]}
    for opening in scene["openings"]:
        wall = wall_map[opening["wall_id"]]
        opening_body = wall_body(
            wall,
            opening["offset"],
            opening["offset"] + opening["width"],
            0,
            opening["height"],
            thickness=wall["thickness"] + 0.03,
        )
        color = "#92c5de" if opening["kind"] == "window" else "#8ac7ab"
        fragments.append(f'<polygon points="{points(footprint(opening_body))}" fill="{color}"/>')
        if opening.get("interior_clearance_depth", 0) > 0:
            area = footprint(opening_clearance(wall, opening))
            fragments.append(
                f'<polygon points="{points(area)}" fill="none" stroke="#39815f" stroke-dasharray="5 4"/>'
            )
    legend_x = margin + width * scale + 40
    fragments.append(text(legend_x, top, "Independent furniture objects", 17))
    for i, item in enumerate(scene["furniture"]):
        color = COLORS.get(item["category"], "#ddd")
        fragments.append(
            f'<polygon points="{points(footprint(item))}" fill="{color}" stroke="#35455a" stroke-width="1.5"/>'
        )
        x, y = xy(item["position"][:2])
        fragments.append(text(x, y + 4, i + 1, 14, 'text-anchor="middle"'))
        y = top + 30 + i * 58
        fragments.append(text(legend_x, y, f"{i + 1}. {item['id']}", 14))
        size = " × ".join(f"{v:g}" for v in item["size"])
        fragments.append(text(legend_x, y + 21, f"{size} m; yaw {item['rotation_degrees']:g}°", 12))
    fragments.extend(
        [
            text(
                margin + width * scale / 2,
                top - 15,
                f"{width:g} m interior clear width",
                14,
                'text-anchor="middle"',
            ),
            text(
                margin,
                top + depth * scale + 32,
                f"Depth {depth:g} m · Height {scene['dimensions'][2]:g} m",
            ),
            text(
                margin,
                top + depth * scale + 59,
                "Blue: window · Green: door · Dashed: reserved doorway clearance",
                13,
            ),
            text(
                margin,
                top + depth * scale + 86,
                f"Static placement: {'FITS' if check['fits'] else 'REJECTED'}; {len(check['violations'])} violations",
                16,
            ),
            "</g></svg>",
        ]
    )
    Path(path).write_text("\n".join(fragments) + "\n", encoding="utf-8")


def export_obj(scene, path):
    """Simple, separate named cuboids. No photo textures, inferred surfaces, or rendering."""
    validate_scene(scene)
    lines = [
        "# Metric room demonstration; units: metres; x/y floor plane, z up.",
        "# Furniture sizes are explicit cuboid envelopes; JSON preserves editable transforms.",
    ]
    vertex_count = 0

    def cuboid(name, body):
        nonlocal vertex_count
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        lines.append(f"o {safe_name}")
        polygon = footprint(body)
        bottom = body["position"][2]
        top = bottom + body["size"][2]
        for z in (bottom, top):
            lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y in polygon)
        for face in (
            (1, 4, 3, 2),
            (5, 6, 7, 8),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 4, 8, 7),
            (4, 1, 5, 8),
        ):
            lines.append("f " + " ".join(str(vertex_count + index) for index in face))
        vertex_count += 8

    width, depth, _ = scene["dimensions"]
    thickness = scene["floor"]["thickness"]
    cuboid(
        "floor",
        {
            "position": [width / 2, depth / 2, -thickness],
            "size": [width, depth, thickness],
            "rotation_degrees": 0,
        },
    )
    for wall in scene["walls"]:
        for i, part in enumerate(solid_wall_parts(wall, scene["openings"])):
            cuboid(f"wall_{wall['id']}_{i}", part)
    for item in scene["furniture"]:
        cuboid(f"furniture_{item['id']}", item)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
