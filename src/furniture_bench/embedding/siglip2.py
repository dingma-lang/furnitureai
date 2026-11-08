from __future__ import annotations

from time import perf_counter

import numpy as np

from ..runtime import ModelTimer, get_torch, resolve_device

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


class Siglip2Embedder:
    def __init__(
        self,
        model_id=DEFAULT_MODEL,
        device="cpu",
        allow_download=False,
        revision="main",
        batch_size=8,
    ):
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.model_id, self.device_name = model_id, device
        self.allow_download, self.revision = allow_download, revision
        self.batch_size = batch_size
        self.model = self.processor = self.device = None
        self.load_seconds = 0.0

    def load(self):
        if self.model is not None:
            return
        start = perf_counter()
        get_torch()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError('SigLIP 2 needs Transformers; install ".[models]"') from exc
        device = resolve_device(self.device_name)
        options = dict(local_files_only=not self.allow_download, revision=self.revision)
        processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=False, **options)
        model = (
            AutoModel.from_pretrained(self.model_id, use_safetensors=True, **options)
            .to(device)
            .eval()
        )
        self.device, self.processor, self.model = device, processor, model
        self.load_seconds = perf_counter() - start

    def metadata(self):
        return {
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
            "device": str(self.device or self.device_name),
            "batch_size": self.batch_size,
            "load_seconds": self.load_seconds,
            "feature": "get_image_features:l2-normalized",
            "preprocessing": "model-default",
        }

    def encode(self, images):
        if not images:
            raise ValueError("encode requires at least one image")
        self.load()
        torch = get_torch()
        batches = []
        with ModelTimer(self.device) as timer, torch.inference_mode():
            for offset in range(0, len(images), self.batch_size):
                inputs = self.processor(
                    images=images[offset : offset + self.batch_size], return_tensors="pt"
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                batches.append(features.float().cpu().numpy())
        vectors = np.concatenate(batches)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("SigLIP 2 returned nonfinite or zero embeddings")
        return vectors, timer.seconds
