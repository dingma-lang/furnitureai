from __future__ import annotations

from time import perf_counter

import numpy as np
from PIL import Image

from furniture_bench.runtime import get_torch


class LocalModel:
    def __init__(self, path, stage):
        self.path, self.stage = str(path), stage
        self.model = self.processor = None
        self.load_seconds = 0.0

    def load(self):
        start = perf_counter()
        get_torch()
        from transformers import (
            AutoImageProcessor,
            DepthAnythingForDepthEstimation,
            UperNetForSemanticSegmentation,
        )

        self.processor = AutoImageProcessor.from_pretrained(
            self.path, local_files_only=True, use_fast=False, trust_remote_code=False
        )
        model_type = (
            UperNetForSemanticSegmentation
            if self.stage == "segmentation"
            else DepthAnythingForDepthEstimation
        )
        self.model = (
            model_type.from_pretrained(self.path, local_files_only=True, use_safetensors=True)
            .to("cpu")
            .eval()
        )
        self.load_seconds = perf_counter() - start

    def predict(self, image):
        self.load() if self.model is None else None
        torch = get_torch()
        start = perf_counter()
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            output = self.model(**inputs)
            if self.stage == "segmentation":
                probabilities = output.logits.softmax(dim=1)
                confidence, labels = probabilities.max(dim=1)
                # Avoid >1 GB allocation for full-resolution 150-channel logits.
                labels = np.asarray(
                    Image.fromarray(labels[0].cpu().numpy().astype(np.uint8)).resize(
                        image.size, Image.Resampling.NEAREST
                    )
                )
                confidence = np.asarray(
                    Image.fromarray(confidence[0].cpu().numpy()).resize(
                        image.size, Image.Resampling.BILINEAR
                    )
                )
                result = {"labels": labels, "confidence": confidence.astype(np.float32)}
            else:
                depth = torch.nn.functional.interpolate(
                    output.predicted_depth[:, None],
                    size=(image.height, image.width),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                result = {"relative_depth": depth.cpu().numpy().astype(np.float32)}
        return result, perf_counter() - start

    def metadata(self):
        return {
            "path": self.path,
            "stage": self.stage,
            "load_seconds": self.load_seconds,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else None,
            "preprocessing": self.processor.to_dict() if self.processor else None,
            "id2label": self.model.config.id2label
            if self.model and self.stage == "segmentation"
            else None,
            "resolution_policy": "checkpoint processor; resize argmax/confidence to source"
            if self.stage == "segmentation"
            else "checkpoint processor; bicubic scalar relative score to source",
            "device": "cpu",
            "dtype": "float32",
        }
