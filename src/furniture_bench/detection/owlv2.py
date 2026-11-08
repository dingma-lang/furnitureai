from __future__ import annotations

import math
from time import perf_counter

from ..data import CATEGORIES, validate_label
from ..geometry import clip_box
from ..postprocessing import FilterConfig, filter_detections
from ..runtime import ModelTimer, get_torch, resolve_device

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


class Owlv2Detector:
    def __init__(
        self,
        model_id=DEFAULT_MODEL,
        device="cpu",
        threshold=0.15,
        nms_iou=0.5,
        categories=CATEGORIES,
        allow_download=False,
        revision="main",
        category_thresholds=None,
        containment_threshold=None,
        cross_class_iou=None,
        prompt_bank=None,
    ):
        if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError("confidence and NMS thresholds must be in [0, 1]")
        if not categories or len(set(categories)) != len(categories):
            raise ValueError("categories must be nonempty and unique")
        for label in categories:
            validate_label(label)
        self.model_id = model_id
        self.device_name = device
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.categories = tuple(categories)
        self.allow_download = allow_download
        self.revision = revision
        self.model = self.processor = self.device = None
        self.load_seconds = 0.0
        self.filter_config = FilterConfig(
            threshold, nms_iou, category_thresholds or {}, containment_threshold, cross_class_iou
        )
        self.last_candidates = None
        self.last_filter_report = None
        if prompt_bank is not None and (
            not isinstance(prompt_bank, dict) or set(prompt_bank) - set(self.categories)
        ):
            raise ValueError("prompt bank keys must be supported detector categories")
        self.query_texts, self.query_labels = [], []
        self.custom_prompts = prompt_bank is not None
        for label in self.categories:
            queries = (prompt_bank or {}).get(label, [f"a photo of a {label}"])
            if (
                not isinstance(queries, list)
                or not queries
                or any(not isinstance(q, str) or not q.strip() for q in queries)
            ):
                raise ValueError("each prompt category needs a nonempty list of strings")
            self.query_texts.extend(queries)
            self.query_labels.extend([label] * len(queries))

    def load(self):
        if self.model is not None:
            return
        start = perf_counter()
        get_torch()
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise RuntimeError('OWLv2 needs Transformers; install ".[models]"') from exc
        device = resolve_device(self.device_name)
        options = dict(local_files_only=not self.allow_download, revision=self.revision)
        try:
            processor = Owlv2Processor.from_pretrained(self.model_id, **options)
            model = (
                Owlv2ForObjectDetection.from_pretrained(
                    self.model_id, use_safetensors=True, **options
                )
                .to(device)
                .eval()
            )
        except OSError as exc:
            raise RuntimeError(
                "OWLv2 could not load. Supply a local checkpoint or use "
                "--allow-download when ready to fetch model files."
            ) from exc
        self.device, self.processor, self.model = device, processor, model
        self.load_seconds = perf_counter() - start

    def metadata(self):
        return {
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
            "device": str(self.device or self.device_name),
            "threshold": self.threshold,
            "nms_iou": self.nms_iou,
            "categories": list(self.categories),
            "load_seconds": self.load_seconds,
            "filter_config": self.filter_config.to_dict(),
            "query_texts": self.query_texts,
            "query_categories": self.query_labels,
        }

    def detect(self, image):
        self.load()
        torch = get_torch()
        prompts = [self.query_texts]
        with ModelTimer(self.device) as timer, torch.inference_mode():
            inputs = self.processor(text=prompts, images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            # Stable indexed-label interface; target size is (height, width), not xy.
            result = self.processor.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.filter_config.candidate_floor,
                target_sizes=[(image.height, image.width)],
            )[0]
            detections = self.decode(result, image.width, image.height)
        return detections, timer.seconds

    def decode(self, result, width, height):
        boxes, scores, labels = (
            result[key].detach().cpu().tolist() for key in ("boxes", "scores", "labels")
        )
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("OWLv2 output arrays have different lengths")
        candidates = []
        for box, score, label in zip(boxes, scores, labels):
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("OWLv2 returned an invalid confidence score")
            if not isinstance(label, int) or not 0 <= label < len(self.query_labels):
                raise ValueError("OWLv2 returned an invalid label index")
            clipped = clip_box(box, width, height)
            if clipped is not None and score >= self.filter_config.candidate_floor:
                candidates.append(
                    {"label": self.query_labels[label], "score": float(score), "box": list(clipped)}
                )
                if self.custom_prompts:
                    candidates[-1]["matched_prompt"] = self.query_texts[label]
        self.last_candidates = [dict(d) for d in candidates]
        filtered, self.last_filter_report = filter_detections(candidates, self.filter_config)
        return filtered
