"""Image-space structure from model masks. No rectangular room completion."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

STRUCTURE_NAMES = {
    0: "other",
    1: "wall",
    2: "floor",
    3: "ceiling",
    4: "window",
    5: "door",
    6: "floor_covering",
}
NAME_CODES = {
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "windowpane": 4,
    "door": 5,
    "screen door": 5,
    "rug": 6,
}


def cross2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def structure_labels(labels, id2label):
    result = np.zeros_like(labels, dtype=np.uint8)
    for key, name in id2label.items():
        if name in NAME_CODES:
            result[labels == int(key)] = NAME_CODES[name]
    return result


def rdp(points, epsilon):
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points
    direction = points[-1] - points[0]
    norm = np.linalg.norm(direction)
    distances = (
        np.linalg.norm(points - points[0], axis=1)
        if norm < 1e-12
        else np.abs(cross2(direction, points - points[0])) / norm
    )
    index = int(np.argmax(distances))
    if distances[index] <= epsilon:
        return points[[0, -1]]
    return np.concatenate((rdp(points[: index + 1], epsilon)[:-1], rdp(points[index:], epsilon)))


def mask_rings(mask, epsilon=1.5):
    """Trace pixel edges; concavities and holes remain, unlike a convex hull."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1)
    transitions = [
        (mask & ~padded[:-2, 1:-1], (0, 0), (1, 0)),
        (mask & ~padded[1:-1, 2:], (1, 0), (1, 1)),
        (mask & ~padded[2:, 1:-1], (1, 1), (0, 1)),
        (mask & ~padded[1:-1, :-2], (0, 1), (0, 0)),
    ]
    edges = defaultdict(set)
    for selector, a, b in transitions:
        for y, x in zip(*np.where(selector)):
            edges[(int(x + a[0]), int(y + a[1]))].add((int(x + b[0]), int(y + b[1])))
    rings = []
    while edges:
        start = min(edges)
        current = start
        ring = [start]
        for _ in range(sum(len(v) for v in edges.values()) + 1):
            choices = edges.get(current)
            if not choices:
                break
            next_point = min(choices)
            choices.remove(next_point)
            if not choices:
                del edges[current]
            ring.append(next_point)
            current = next_point
            if current == start:
                break
        if len(ring) < 4 or ring[-1] != start:
            continue
        raw = np.asarray(ring, dtype=float)
        signed_area = float(np.sum(raw[:-1, 0] * raw[1:, 1] - raw[1:, 0] * raw[:-1, 1]) / 2)
        middle = len(raw) // 2
        simplified = np.concatenate(
            (rdp(raw[: middle + 1], epsilon)[:-1], rdp(raw[middle:], epsilon))
        )
        if len(simplified) < 4:
            simplified = raw
        rings.append(
            {
                "points": simplified[:-1].tolist(),
                "ring_role": "outer" if signed_area > 0 else "hole",
                "pixel_area_signed": signed_area,
            }
        )
    return rings


def boundary_mask(labels, first, second, confidence, min_confidence, occlusion):
    result = np.zeros_like(labels, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, labels.shape[0] - dy or None), slice(0, labels.shape[1] - dx or None))
        b = (slice(dy, None), slice(dx, None))
        adjacent = ((labels[a] == first) & (labels[b] == second)) | (
            (labels[a] == second) & (labels[b] == first)
        )
        valid = (
            (confidence[a] >= min_confidence)
            & (confidence[b] >= min_confidence)
            & ~occlusion[a]
            & ~occlusion[b]
        )
        result[a] |= adjacent & valid
    return result


def fit_segments(mask, confidence, min_length=24, distance=1.8, min_points=12):
    """Deterministic RANSAC then PCA, bounded by actual support extents."""
    if (
        np.shape(mask) != np.shape(confidence)
        or np.ndim(mask) != 2
        or not np.isfinite(confidence).all()
    ):
        raise ValueError("line extraction requires matching 2D mask and finite confidence")
    if (
        min_points < 2
        or not np.isfinite(min_length)
        or min_length <= 0
        or not np.isfinite(distance)
        or distance <= 0
    ):
        raise ValueError("line fitting needs positive lengths and at least two support points")
    components, count = ndimage.label(mask, np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    components_to_try = sorted(range(1, count + 1), key=lambda k: sizes[k], reverse=True)[:40]
    rng = np.random.default_rng(0)
    segments = []
    for component in components_to_try:
        y, x = np.where(components == component)
        points = np.column_stack((x + 0.5, y + 0.5))
        for _ in range(4):
            if len(points) < min_points:
                break
            best = np.zeros(len(points), dtype=bool)
            for _ in range(96):
                a, b = points[rng.choice(len(points), 2, replace=False)]
                direction = b - a
                norm = np.linalg.norm(direction)
                if norm < min_length / 3:
                    continue
                inliers = np.abs(cross2(direction, points - a)) / norm <= distance
                if inliers.sum() > best.sum():
                    best = inliers
            selected = points[best]
            if len(selected) < min_points:
                break
            centre = selected.mean(axis=0)
            _, _, vh = np.linalg.svd(selected - centre, full_matrices=False)
            direction = vh[0]
            projected = np.sum((selected - centre) * direction, axis=1)
            endpoints = np.array(
                [centre + direction * projected.min(), centre + direction * projected.max()]
            )
            points = points[~best]
            if np.linalg.norm(endpoints[1] - endpoints[0]) < min_length:
                continue
            coords = np.floor(selected).astype(int)
            residual = np.abs(cross2(direction, selected - centre))
            segments.append(
                {
                    "endpoints": endpoints.tolist(),
                    "support_pixels": len(selected),
                    "confidence": float(confidence[coords[:, 1], coords[:, 0]].mean()),
                    "fit_residual_pixels": float(np.sqrt(np.mean(residual**2))),
                }
            )
    return segments


def line_intersection(first, second):
    a, b = np.asarray(first), np.asarray(second)
    if (
        a.shape != (2, 2)
        or b.shape != (2, 2)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or max(np.max(abs(a)), np.max(abs(b))) > 1e8
    ):
        raise ValueError("line endpoints must be finite bounded 2x2 coordinates")
    la = np.cross([*a[0], 1.0], [*a[1], 1.0])
    lb = np.cross([*b[0], 1.0], [*b[1], 1.0])
    point = np.cross(la, lb)
    if abs(point[2]) < 1e-9:
        return None
    return point[:2] / point[2]


def corner_candidates(segments, width, height, tolerance=16):
    candidates = []
    for index, a in enumerate(segments):
        if a["kind"] not in ("wall_floor", "wall_ceiling"):
            continue
        for b in segments[index + 1 :]:
            if (
                a["kind"] != b["kind"]
                or a["visibility"] != "visible_evidence"
                or b["visibility"] != "visible_evidence"
            ):
                continue
            pa, pb = np.array(a["endpoints"]), np.array(b["endpoints"])
            va, vb = pa[1] - pa[0], pb[1] - pb[0]
            if min(np.linalg.norm(va), np.linalg.norm(vb)) < 1e-9:
                continue
            cosine = abs(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            if cosine > math.cos(math.radians(20)):
                continue
            point = line_intersection(pa, pb)
            if point is None or not (0 <= point[0] <= width and 0 <= point[1] <= height):
                continue
            if (
                min(np.linalg.norm(point - p) for p in pa) > tolerance
                or min(np.linalg.norm(point - p) for p in pb) > tolerance
            ):
                continue
            if any(np.linalg.norm(point - np.array(c["pixel"])) < tolerance for c in candidates):
                continue
            candidates.append(
                {
                    "id": f"corner-{len(candidates)}",
                    "pixel": point.tolist(),
                    "type": a["kind"] + "_junction_candidate",
                    "status": "estimated_visible_candidate",
                    "supporting_boundaries": [a["id"], b["id"]],
                    "confidence": min(a["confidence"], b["confidence"]),
                    "uncertainty_pixels": tolerance,
                    "human_verified": False,
                }
            )
    return candidates


def infer_occlusion_bridges(segments, occlusion, max_gap=100):
    inferred = []
    for index, first in enumerate(segments):
        if first["kind"] not in ("wall_floor", "wall_ceiling"):
            continue
        for second in segments[index + 1 :]:
            if first["kind"] != second["kind"]:
                continue
            a, b = np.asarray(first["endpoints"]), np.asarray(second["endpoints"])
            va, vb = a[1] - a[0], b[1] - b[0]
            if min(np.linalg.norm(va), np.linalg.norm(vb)) < 1e-9:
                continue
            if abs(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))) < math.cos(
                math.radians(8)
            ):
                continue
            pairs = [(np.linalg.norm(p - q), p, q) for p in a for q in b]
            gap, p, q = min(pairs, key=lambda item: item[0])
            if not 5 < gap <= max_gap or abs(cross2(va, q - p)) / np.linalg.norm(va) > 4:
                continue
            sample = np.linspace(p, q, 40).astype(int)
            sample[:, 0] = np.clip(sample[:, 0], 0, occlusion.shape[1] - 1)
            sample[:, 1] = np.clip(sample[:, 1], 0, occlusion.shape[0] - 1)
            if np.mean(occlusion[sample[:, 1], sample[:, 0]]) < 0.6:
                continue
            inferred.append(
                {
                    "id": f"inferred-{len(inferred)}",
                    "kind": first["kind"],
                    "endpoints": [p.tolist(), q.tolist()],
                    "visibility": "inferred_occluded",
                    "supporting_boundaries": [first["id"], second["id"]],
                    "confidence": min(first["confidence"], second["confidence"]) * 0.5,
                    "source": "collinear support on both sides plus conservative furniture-box overlap; not observed background",
                }
            )
    return inferred


def extract_structure(
    image, labels, confidence, id2label, furniture=None, min_confidence=0.55, analysis_width=640
):
    full = structure_labels(labels, id2label)
    width, height = image.size
    scale = min(1.0, analysis_width / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(Image.fromarray(full).resize(size, Image.Resampling.NEAREST))
    conf = np.asarray(
        Image.fromarray(confidence.astype(np.float32)).resize(size, Image.Resampling.BILINEAR)
    )
    sx, sy = width / size[0], height / size[1]
    hints = np.zeros(small.shape, dtype=bool)
    hint_records = []
    for detection in furniture or []:
        if detection["score"] < 0.2 or detection["label"] not in (
            "sofa",
            "chair",
            "table",
            "bed",
            "desk",
            "cabinet",
        ):
            continue
        box = detection["box"]
        x1, y1 = max(0, int(box[0] / sx)), max(0, int(box[1] / sy))
        x2, y2 = min(size[0], math.ceil(box[2] / sx)), min(size[1], math.ceil(box[3] / sy))
        hints[y1:y2, x1:x2] = True
        hint_records.append(
            {
                "box": box,
                "label": detection["label"],
                "score": detection["score"],
                "status": "conservative_occlusion_hint_not_object_mask",
            }
        )
    regions = []
    for code, name in STRUCTURE_NAMES.items():
        if not code:
            continue
        raw = (small == code) & (conf >= min_confidence)
        connected, _ = ndimage.label(raw)
        counts = np.bincount(connected.ravel())
        keep = counts >= 48
        keep[0] = False
        cleaned = keep[connected]
        if not cleaned.any():
            continue
        rings = mask_rings(cleaned)
        for ring in rings:
            ring["points"] = [[x * sx, y * sy] for x, y in ring["points"]]
        regions.append(
            {
                "id": name + "-visible-mask",
                "surface": name,
                "coordinate_frame": "image_pixels",
                "rings": rings,
                "source": "pretrained_semantic_estimate_on_visible_image",
                "confidence_mean": float(conf[cleaned].mean()),
                "pixel_fraction": float(raw.mean()),
                "uncertain_occlusion_fraction": float(np.mean(hints[cleaned])),
                "hidden_extent": "unknown",
                "physical_polygon": None,
            }
        )
    segments = []
    for name, a, b in (
        ("wall_floor", 1, 2),
        ("wall_ceiling", 1, 3),
        ("wall_window", 1, 4),
        ("wall_door", 1, 5),
    ):
        mask = boundary_mask(small, a, b, conf, min_confidence, hints)
        for line in fit_segments(mask, conf):
            line.update(
                id=f"boundary-{len(segments)}",
                kind=name,
                visibility="visible_evidence",
                source="adjacent high-confidence semantic classes outside conservative occlusion boxes",
            )
            segments.append(line)
    inferred = infer_occlusion_bridges(segments, hints, max_gap=size[0] * 0.25)
    all_segments = segments + inferred
    for line in all_segments:
        line["endpoints"] = [[x * sx, y * sy] for x, y in line["endpoints"]]
        line["confidence_calibrated"] = False
        if "fit_residual_pixels" in line:
            line["fit_residual_pixels"] *= max(sx, sy)
    corners = corner_candidates(all_segments, width, height, tolerance=16 * max(sx, sy))
    # Wall-wall lines need additional image-edge support; do not split wall masks arbitrarily.
    gray = (
        np.asarray(image.resize(size).convert("L").filter(ImageFilter.GaussianBlur(1)), dtype=float)
        / 255
    )
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    wall_pixels = (small == 1) & (conf >= min_confidence) & ~hints
    wall_candidates = []
    if wall_pixels.sum() > 100:
        cutoff = max(0.025, float(np.quantile(magnitude[wall_pixels], 0.94)))
        edges = wall_pixels & (magnitude >= cutoff)
        for line in fit_segments(edges, conf, min_length=45, min_points=25):
            p, q = np.array(line["endpoints"]) * [sx, sy]
            if abs(q[0] - p[0]) > 0.4 * np.linalg.norm(q - p):
                continue
            if not any(
                min(
                    np.linalg.norm(np.array(c["pixel"]) - p),
                    np.linalg.norm(np.array(c["pixel"]) - q),
                )
                < 50
                for c in corners
            ):
                continue
            wall_candidates.append(
                {
                    "endpoints": [p.tolist(), q.tolist()],
                    "status": "unverified_wall_wall_candidate",
                    "source": "long image edge within wall mask near a semantic junction; may be another object edge",
                    "confidence": line["confidence"],
                    "confidence_calibrated": False,
                }
            )
    perspective = []
    for a in segments:
        if a["kind"] != "wall_floor":
            continue
        for b in segments:
            if b["kind"] != "wall_ceiling":
                continue
            va = np.diff(np.array(a["endpoints"]), axis=0)[0]
            vb = np.diff(np.array(b["endpoints"]), axis=0)[0]
            cosine = abs(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            if cosine < math.cos(math.radians(35)):
                continue
            point = line_intersection(a["endpoints"], b["endpoints"])
            if (
                point is not None
                and min(
                    np.linalg.norm(point - np.array(p)) for p in a["endpoints"] + b["endpoints"]
                )
                > 50
            ):
                perspective.append(
                    {
                        "pixel_homogeneous": [*point.tolist(), 1.0],
                        "status": "direction_hypothesis_only",
                        "requires": "these floor/ceiling segments must be physically parallel on the same wall",
                        "supporting_boundaries": [a["id"], b["id"]],
                    }
                )
    return {
        "regions": regions,
        "boundaries": all_segments,
        "corner_candidates": corners,
        "wall_wall_candidates": wall_candidates,
        "furniture_occlusion_hints": hint_records,
        "camera_perspective": {
            "direction_candidates": perspective[:12],
            "horizon_line": None,
            "intrinsics": None,
            "camera_pose": None,
            "status": "partial_unverified_direction_hypotheses" if perspective else "unknown",
            "reason": "No verified parallel-direction assignment or camera calibration; no assumed horizon.",
        },
        "topology": {
            "room_shape": "unknown",
            "wall_plane_count": None,
            "hidden_corners": "unknown",
            "closed_world_floor_polygon": None,
            "visible_surface_types": [r["surface"] for r in regions],
            "manhattan_or_rectangle_assumed": False,
        },
        "analysis": {
            "width": size[0],
            "height": size[1],
            "min_confidence": min_confidence,
            "confidence_is_calibrated": False,
            "unknown_hidden_structure": True,
        },
    }, full
