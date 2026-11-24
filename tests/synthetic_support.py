"""Deterministic unit-only fixtures, never real photos, catalogues or model outputs."""
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from furniture_bench.data import CATEGORIES, write_json
from room_layout.scene import photo_scene


def synthetic_model_configuration():
    return {"adapter_version": "synthetic-test-only", "normalization": "synthetic-EXIF-RGB",
            "threshold": .2, "nms_iou": .5, "threads": 1,
            "prompts": {c: ["synthetic fixture " + c] for c in CATEGORIES},
            "checkpoints": {s: {"model": "synthetic-no-weights", "files": {}} for s in ("detection", "segmentation", "depth")}}


def synthetic_product():
    return {"product_id": "SYNTHETIC-NOT-A-REAL-LISTING", "label": "sofa", "name": "Synthetic unit fixture",
            "images": [{"image_id": "SYNTHETIC-IMAGE-NO-BINARY"}], "rights": {"source": "synthetic test only", "license": "CC0", "allowed_for_benchmark": True},
            "source_record": {"item_dimensions": {"width": {"value": 210, "unit": "centimeters"},
                                                    "height": {"value": 90, "unit": "centimeters"},
                                                    "length": {"value": 220, "unit": "centimeters"}}}}


@contextmanager
def synthetic_partial_preset():
    scene = photo_scene("synthetic-room", "b" * 64, 100, 80,
                        {"regions": [], "camera_perspective": {"status": "unknown"}}, {}, [])
    with tempfile.TemporaryDirectory(prefix="synthetic-preset-") as tmp:
        root = Path(tmp)
        write_json(root / "single-photo.json", scene)
        with patch("room_viewer.service.ROOT", root), patch("room_viewer.service.PRESETS", {"single-photo": "single-photo.json"}):
            yield
