"""Minimal glTF 2.0 export of supported point/triangle geometry, with provenance."""

from pathlib import Path

import numpy as np

from furniture_bench.data import write_json


def export_gltf(path, points, colors, triangles=None, extras=None):
    path = Path(path)
    if path.exists() or path.with_suffix(".bin").exists():
        raise ValueError("glTF output exists")
    points = np.asarray(points, dtype="<f4").reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors) or not np.isfinite(points).all():
        raise ValueError("invalid finite vertex/color arrays")
    triangles = np.asarray(triangles if triangles is not None else [], dtype="<u4").reshape(-1, 3)
    if len(triangles) and triangles.max() >= len(points):
        raise ValueError("triangle index outside vertices")
    # Sensor camera x-right/y-down/z-forward -> glTF right-handed x-right/y-up/z-back.
    vertices = points * np.array([1, -1, -1], dtype="<f4")
    parts = []
    views = []
    accessors = []

    def append(
        data, component, kind, count, minimum=None, maximum=None, target=34962, normalized=False
    ):
        offset = sum(len(p) for p in parts)
        if offset % 4:
            parts.append(b"\0" * (4 - offset % 4))
            offset = sum(len(p) for p in parts)
        content = data.tobytes()
        parts.append(content)
        views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(content), "target": target}
        )
        acc = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": count,
            "type": kind,
        }
        if normalized:
            acc["normalized"] = True
        if minimum is not None:
            acc["min"] = minimum
            acc["max"] = maximum
        accessors.append(acc)
        return len(accessors) - 1

    document = {
        "asset": {"version": "2.0", "generator": "furniture-ai observed-geometry exporter"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "extras": extras or {},
    }
    if len(points):
        pos = append(
            vertices,
            5126,
            "VEC3",
            len(vertices),
            vertices.min(axis=0).tolist(),
            vertices.max(axis=0).tolist(),
        )
        rgb = append(colors, 5121, "VEC3", len(colors), normalized=True)
        primitive = {
            "attributes": {"POSITION": pos, "COLOR_0": rgb},
            "mode": 4 if len(triangles) else 0,
        }
        if len(triangles):
            primitive["indices"] = append(triangles, 5125, "SCALAR", triangles.size, target=34963)
        document.update(
            meshes=[{"primitives": [primitive]}],
            nodes=[{"mesh": 0, "name": "observed partial geometry"}],
            buffers=[
                {"uri": path.with_suffix(".bin").name, "byteLength": sum(len(p) for p in parts)}
            ],
            bufferViews=views,
            accessors=accessors,
        )
        document["scenes"][0]["nodes"] = [0]
        path.with_suffix(".bin").write_bytes(b"".join(parts))
    write_json(path, document)
    return document
