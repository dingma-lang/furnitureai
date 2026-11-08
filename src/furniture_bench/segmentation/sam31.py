from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from ..data import digest
from ..geometry import iou
from ..runtime import ModelTimer, get_torch, resolve_device


class Sam31Segmenter:
    """Experimental single-frame bridge to Meta's SAM 3.1 multiplex predictor.

    Uses category text prompts, then associates masks to OWLv2 boxes by IoU.
    It does not silently substitute SAM 3 or use a hosted API.
    """

    def __init__(self, checkpoint, min_iou=0.3):
        if not 0 < min_iou <= 1:
            raise ValueError("mask association IoU must be in (0, 1]")
        self.checkpoint = Path(checkpoint).resolve()
        self.min_iou = min_iou
        self.predictor = self.device = None
        self.load_seconds = 0.0
        self.checkpoint_sha256 = None

    def load(self):
        if self.predictor is not None:
            return
        if not self.checkpoint.is_file():
            raise ValueError("SAM 3.1 requires an existing local multiplex checkpoint")
        if sys.version_info < (3, 12):
            raise RuntimeError("Meta SAM 3.1 requires a separate Python >=3.12 environment")
        self.device = resolve_device("cuda")
        start = perf_counter()
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "Install Meta's current sam3 repository in the optional GPU environment"
            ) from exc
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(self.checkpoint), use_fa3=False, compile=False, warm_up=False
        )
        self.checkpoint_sha256 = digest(self.checkpoint)
        self.load_seconds = perf_counter() - start

    def metadata(self):
        return {
            "model": "Meta SAM 3.1 multiplex",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": "cuda",
            "load_seconds": self.load_seconds,
            "association_iou": self.min_iou,
            "mode": "single-frame category-text prompts; rectangular crops retained",
        }

    def segment(self, image, detections):
        self.load()
        torch = get_torch()
        masks_by_detection = {}
        with (
            ModelTimer(self.device) as timer,
            torch.inference_mode(),
            tempfile.TemporaryDirectory(prefix="furniture-sam31-") as directory,
        ):
            # Meta's video loader accepts a directory containing numbered JPEG frames.
            image.save(Path(directory) / "00000.jpg", quality=100)
            response = self.predictor.handle_request(
                request={"type": "start_session", "resource_path": directory}
            )
            session_id = response["session_id"]
            try:
                for label in dict.fromkeys(d["label"] for d in detections):
                    self.predictor.handle_request(
                        request={"type": "reset_session", "session_id": session_id}
                    )
                    response = self.predictor.handle_request(
                        request={
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": label,
                        }
                    )
                    masks = response["outputs"]["out_binary_masks"]
                    if hasattr(masks, "detach"):
                        masks = masks.detach().cpu().numpy()
                    masks = np.asarray(masks)
                    candidates = []
                    for mask in masks:
                        mask = np.asarray(mask).reshape(image.height, image.width).astype(bool)
                        ys, xs = np.nonzero(mask)
                        if xs.size:
                            candidates.append(
                                (
                                    [
                                        int(xs.min()),
                                        int(ys.min()),
                                        int(xs.max()) + 1,
                                        int(ys.max()) + 1,
                                    ],
                                    mask,
                                )
                            )
                    used = set()
                    for index in sorted(
                        range(len(detections)), key=lambda k: detections[k]["score"], reverse=True
                    ):
                        detection = detections[index]
                        if detection["label"] != label:
                            continue
                        options = [
                            (iou(detection["box"], box), k)
                            for k, (box, _) in enumerate(candidates)
                            if k not in used
                        ]
                        overlap, k = max(options, default=(0.0, -1))
                        if k >= 0 and overlap >= self.min_iou:
                            used.add(k)
                            masks_by_detection[index] = Image.fromarray(
                                candidates[k][1].astype(np.uint8) * 255
                            )
            finally:
                self.predictor.handle_request(
                    request={"type": "close_session", "session_id": session_id}
                )
        return masks_by_detection, timer.seconds
