from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw

from .data import load_image, validate_prediction


def annotate(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for detection in detections:
        validate_prediction(detection)
        box = detection["box"]
        draw.rectangle(box, outline="#ff6b35", width=3)
        text = f"{detection['label']} {detection['score']:.3f}"
        x, y = box[0], max(0, box[1] - 15)
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill="black")
        draw.text((x, y), text, fill="white")
    return result


def visualize(image_path, detections, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotate(load_image(image_path), detections).save(output)
