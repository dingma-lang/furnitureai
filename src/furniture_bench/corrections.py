from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .data import require_string, validate_label
from .geometry import validate_box


def save_correction(path, correction):
    """Append human feedback; never apply it to predictions or ground truth automatically."""
    correction = dict(correction)
    for key in ("image_id", "image_sha256", "reviewer", "action"):
        require_string(correction.get(key), key)
    image_hash = correction["image_sha256"]
    if len(image_hash) != 64 or any(char not in "0123456789abcdef" for char in image_hash):
        raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
    action = correction["action"]
    if action not in ("add", "update", "delete"):
        raise ValueError("action must be add, update, or delete")
    if action in ("update", "delete"):
        require_string(correction.get("detection_id"), "detection_id")
    if action in ("add", "update"):
        validate_label(correction.get("label"))
        correction["box"] = list(validate_box(correction.get("box", [])))
    if correction.get("identity_verified") is True:
        require_string(correction.get("product_id"), "verified product_id")
        require_string(correction.get("verification_evidence"), "verification_evidence")
    else:
        correction["identity_verified"] = False
    correction.update(
        {
            "schema_version": 1,
            "correction_id": str(uuid4()),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "used_for_training": False,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(correction, ensure_ascii=False, allow_nan=False) + "\n")
    return correction
