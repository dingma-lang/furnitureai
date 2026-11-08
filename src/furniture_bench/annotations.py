"""Review artifacts for manually marked image-space reference boxes."""

from __future__ import annotations

from pathlib import Path

from PIL import ImageDraw, ImageFont

from .data import digest, load_image, load_manifest, write_json


def annotation_sheets(manifest, output):
    output = Path(output)
    if output.exists():
        raise ValueError("annotation sheet directory already exists")
    rows = load_manifest(manifest, "rooms")
    output.mkdir(parents=True)
    for index, row in enumerate(rows):
        if row.get("objects") is None:
            continue
        image = load_image(row["resolved_file"])
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()
        for obj in row["objects"]:
            box = obj["box"]
            if box[2] > image.width or box[3] > image.height:
                raise ValueError("reference annotation is outside the oriented image")
            draw.rectangle(box, outline="#1665ca", width=4)
            text = f"REF {obj['object_id']}: {obj['label']}"
            xy = (box[0], max(0, box[1] - 23))
            draw.rectangle(draw.textbbox(xy, text, font=font), fill="white")
            draw.text(xy, text, fill="#104a9b", font=font)
        for region in row.get("ignore_regions", []):
            draw.rectangle(region["box"], outline="#a12d92", width=5)
        image.save(output / f"reference-{index:03d}.png")
    write_json(
        output / "human-review.template.json",
        {
            "manifest_sha256_before_review": digest(manifest),
            "reviewer": None,
            "reviewed_at": None,
            "reviewed_image_ids": [],
            "evidence_or_notes": None,
            "instructions": "Inspect every image and supported object, edit boxes/labels/ignore regions in a NEW manifest version, then record human_reviewed=true, status=human_reviewed, reviewer, reviewed_at, and evidence for each reviewed image. This template alone never promotes labels.",
        },
    )
