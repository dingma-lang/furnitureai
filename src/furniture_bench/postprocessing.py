"""Explicit, reusable filtering of saved or live detection candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .data import validate_label, validate_prediction
from .geometry import iou


def containment(a, b):
    """Intersection divided by smaller area; not IoU."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / min(area_a, area_b)


@dataclass
class FilterConfig:
    threshold: float = 0.15
    nms_iou: float = 0.5
    category_thresholds: dict = field(default_factory=dict)
    containment_threshold: float | None = None
    cross_class_iou: float | None = None

    def __post_init__(self):
        for name in ("threshold", "nms_iou", "containment_threshold", "cross_class_iou"):
            value = getattr(self, name)
            if value is None and name in ("containment_threshold", "cross_class_iou"):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be in [0, 1]")
            if name == "containment_threshold" and value == 0:
                raise ValueError("containment_threshold must be positive")
        if not isinstance(self.category_thresholds, dict):
            raise ValueError("category_thresholds must be an object")
        for label, threshold in self.category_thresholds.items():
            validate_label(label)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0 <= threshold <= 1
            ):
                raise ValueError("category confidence thresholds must be in [0, 1]")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("unknown filter configuration fields")
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @property
    def candidate_floor(self):
        return min([self.threshold, *self.category_thresholds.values()])


def filter_detections(candidates, config):
    """Suppress, never merge or relabel. Return decisions for reproducible audit."""
    kept, removed = [], []
    for index in sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True):
        candidate = candidates[index]
        validate_prediction(candidate)
        threshold = config.category_thresholds.get(candidate["label"], config.threshold)
        if candidate["score"] < threshold:
            removed.append(
                {"candidate_index": index, "reason": "confidence", "threshold": threshold}
            )
            continue
        reason = None
        suppressor = None
        for old_index in kept:
            old = candidates[old_index]
            same = candidate["label"] == old["label"]
            if same and iou(candidate["box"], old["box"]) > config.nms_iou:
                reason = "same_class_iou"
            elif (
                same
                and config.containment_threshold is not None
                and containment(candidate["box"], old["box"]) >= config.containment_threshold
            ):
                reason = "same_class_containment"
            elif (
                not same
                and config.cross_class_iou is not None
                and iou(candidate["box"], old["box"]) > config.cross_class_iou
            ):
                reason = "cross_class_iou"
            if reason:
                suppressor = old_index
                break
        if reason:
            removed.append(
                {"candidate_index": index, "reason": reason, "suppressed_by": suppressor}
            )
        else:
            kept.append(index)
    return [dict(candidates[i]) for i in kept], {
        "config": config.to_dict(),
        "input_count": len(candidates),
        "kept_indices": kept,
        "removed": removed,
    }
