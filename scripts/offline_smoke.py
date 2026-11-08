"""Generate an explicit software fixture. Never runs a pretrained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from furniture_bench.data import write_json
from furniture_bench.evaluation import evaluate
from furniture_bench.pipeline import run_experiment


class GeometryFixtureDetector:
    def detect(self, image):
        # Deliberately scripted geometry, not image recognition.
        return [{"label": "sofa", "score": 1.0, "box": [30, 40, 170, 100]}], 0.0

    def metadata(self):
        return {"model_id": "scripted-geometry-fixture-NOT-OWLv2", "load_seconds": 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (220, 140), "#eeeeee")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 40, 169, 99], fill="#437ba1")
    draw.text((5, 115), "SYNTHETIC SOFTWARE FIXTURE", fill="black")
    image.save(root / "geometry.png")
    manifest = root / "rooms.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "images": [
                {
                    "image_id": "synthetic-geometry",
                    "file": "geometry.png",
                    "rights": {
                        "source": "programmatically generated geometry; no external images",
                        "license": "CC0-1.0",
                        "allowed_for_benchmark": True,
                    },
                    "objects": [
                        {
                            "object_id": "rectangle-1",
                            "label": "sofa",
                            "box": [30, 40, 170, 100],
                            "identity_verified": False,
                        }
                    ],
                }
            ],
        },
    )
    run = run_experiment(manifest, root / "run", GeometryFixtureDetector())
    run["kind"] = "synthetic_fixture_NOT_model_inference"
    write_json(root / "run" / "predictions.json", run)
    report = evaluate(manifest, root / "run" / "predictions.json")
    report["accuracy_claim"] = "Software fixture only. No pretrained models evaluated."
    write_json(root / "evaluation.json", report)
    print(root / "evaluation.json")


if __name__ == "__main__":
    main()
