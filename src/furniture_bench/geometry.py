from __future__ import annotations

import math
from typing import Sequence


def validate_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be finite")
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("box must have nonnegative origin and positive area")
    return x1, y1, x2, y2


def clip_box(box: Sequence[float], width: int, height: int):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("model returned a malformed box")
    clipped = tuple(
        max(0.0, min(float(v), float(limit)))
        for v, limit in zip(box, (width, height, width, height))
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: list[dict], threshold: float) -> list[dict]:
    """Greedy, category-aware NMS. Different labels may overlap."""
    if not 0 <= threshold <= 1:
        raise ValueError("NMS IoU must be in [0, 1]")
    kept = []
    for candidate in sorted(detections, key=lambda d: d["score"], reverse=True):
        if not any(
            candidate["label"] == old["label"] and iou(candidate["box"], old["box"]) > threshold
            for old in kept
        ):
            kept.append(candidate)
    return kept


def crop_bounds(box, width: int, height: int) -> tuple[int, int, int, int]:
    clipped = clip_box(box, width, height)
    if clipped is None:
        raise ValueError("cannot crop an empty box")
    return (
        math.floor(clipped[0]),
        math.floor(clipped[1]),
        math.ceil(clipped[2]),
        math.ceil(clipped[3]),
    )
