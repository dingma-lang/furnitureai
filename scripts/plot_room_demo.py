"""Export a schematic comparison from saved room JSON using optional Matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

from room_reconstruction.collision import footprint, opening_clearance
from room_reconstruction.export import COLORS
from room_reconstruction.model import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for axis, name, title in zip(
        axes,
        ("before", "sofa-removed", "after"),
        ("1. Original sofa", "2. Sofa removed", "3. Larger sofa fits"),
    ):
        scene = load_scene(Path(args.demo) / f"{name}.json")
        width, depth, _ = scene["dimensions"]
        axis.add_patch(Polygon(scene["floor"]["boundary"], facecolor="#f5f7fa", edgecolor="#999"))
        walls = {wall["id"]: wall for wall in scene["walls"]}
        for wall in walls.values():
            axis.plot(
                [wall["start"][0], wall["end"][0]],
                [wall["start"][1], wall["end"][1]],
                color="#39475b",
                linewidth=5,
                solid_capstyle="butt",
            )
        for opening in scene["openings"]:
            wall = walls[opening["wall_id"]]
            dx, dy = wall["end"][0] - wall["start"][0], wall["end"][1] - wall["start"][1]
            length = (dx * dx + dy * dy) ** 0.5
            values = [opening["offset"], opening["offset"] + opening["width"]]
            color = "#4696c0" if opening["kind"] == "window" else "#3a9964"
            axis.plot(
                [wall["start"][0] + dx * u / length for u in values],
                [wall["start"][1] + dy * u / length for u in values],
                color=color,
                linewidth=6,
                solid_capstyle="butt",
            )
            if opening.get("interior_clearance_depth", 0):
                axis.add_patch(
                    Polygon(
                        footprint(opening_clearance(wall, opening)),
                        fill=False,
                        edgecolor=color,
                        linestyle="--",
                    )
                )
        for item in scene["furniture"]:
            axis.add_patch(
                Polygon(
                    footprint(item),
                    facecolor=COLORS.get(item["category"], "#ddd"),
                    edgecolor="#35455a",
                    linewidth=1.2,
                )
            )
            label = item["category"]
            if item["category"] == "sofa":
                label = f"sofa\n{item['size'][0]:.2f} × {item['size'][1]:.2f} m"
            x, y, _ = item["position"]
            axis.text(x, y, label, ha="center", va="center", fontsize=8)
        axis.set(
            xlim=(-0.3, width + 0.3),
            ylim=(-0.3, depth + 0.3),
            xlabel="x (metres)",
            ylabel="y (metres)",
            title=title,
        )
        axis.set_aspect("equal")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Editable room: 6.0 × 4.5 × 2.8 m", fontsize=18)
    figure.text(
        0.5,
        0.055,
        "Specified demonstration geometry, independent of the photograph. Blue: window; green: door; dashed: reserved clearance.",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.015,
        "Replacement gaps: north wall 0.255 m · coffee table 0.445 m · floor lamp 0.075 m. Static fit only.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.94))
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
