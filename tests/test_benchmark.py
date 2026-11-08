from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from furniture_bench.cli import main
from furniture_bench.corrections import save_correction
from furniture_bench.data import digest, load_image, load_manifest, read_json, write_json
from furniture_bench.detection.owlv2 import Owlv2Detector
from furniture_bench.embedding.siglip2 import Siglip2Embedder
from furniture_bench.evaluation import evaluate, match_objects
from furniture_bench.geometry import clip_box, crop_bounds, iou, nms, validate_box
from furniture_bench.pipeline import run_experiment
from furniture_bench.retrieval import CatalogIndex, unit_vectors
from furniture_bench.segmentation.sam31 import Sam31Segmenter

RIGHTS = {"source": "created by test", "license": "CC0-1.0", "allowed_for_benchmark": True}
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def detection(label="sofa", score=0.9, box=None, matches=None):
    return {
        "label": label,
        "score": score,
        "box": box or [10, 10, 50, 40],
        "matches": matches or [],
    }


def room(image_id="r1", file="room.png", objects=None):
    return {"image_id": image_id, "file": file, "rights": dict(RIGHTS), "objects": objects}


def obj(object_id="o1", label="sofa", box=None, product_id=None, verified=False):
    return {
        "object_id": object_id,
        "label": label,
        "box": box or [10, 10, 50, 40],
        "product_id": product_id,
        "identity_verified": verified,
    }


class TensorList:
    """Only the postprocessing interface is faked; no model files are involved."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FixtureDetector:
    def detect(self, image):
        return [detection()], 0.01

    def metadata(self):
        return {"model_id": "test-double-not-owlv2"}


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        Image.new("RGB", (100, 60), "white").save(self.root / "room.png")

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, rows=None):
        path = self.root / "rooms.json"
        write_json(path, {"schema_version": 1, "images": rows or [room(objects=[obj()])]})
        return path

    def predictions(self, records, **extra):
        path = self.root / "predictions.json"
        write_json(
            path, {"schema_version": 1, "kind": "synthetic_fixture", "images": records, **extra}
        )
        return path


class GeometryTests(unittest.TestCase):
    def test_iou_clipping_and_rounding(self):
        self.assertEqual(iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(clip_box([-2, -3, 120, 90], 100, 60), (0, 0, 100, 60))
        self.assertIsNone(clip_box([120, 0, 130, 10], 100, 60))
        self.assertEqual(crop_bounds([0.2, 1.5, 10.1, 11.2], 100, 60), (0, 1, 11, 12))

    def test_invalid_boxes(self):
        for box in ([0, 0, 0, 1], [0, 0, float("nan"), 3], [-1, 0, 3, 3], [1, 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box)

    def test_nms_preserves_distinct_categories(self):
        kept = nms([detection(score=0.7), detection(score=0.9), detection("chair")], 0.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["score"], 0.9)


class DetectionTests(unittest.TestCase):
    def test_construction_is_lazy_and_categories_are_complete(self):
        detector = Owlv2Detector()
        self.assertIsNone(detector.model)
        self.assertFalse(detector.allow_download)
        self.assertEqual(
            set(detector.categories), {"sofa", "chair", "table", "bed", "desk", "cabinet", "lamp"}
        )

    def test_decode_clips_maps_filters_and_suppresses(self):
        detector = Owlv2Detector(threshold=0.2)
        result = {
            "boxes": TensorList(
                [[-5, 0, 110, 60], [0, 0, 100, 60], [10, 10, 20, 20], [120, 0, 130, 10]]
            ),
            "scores": TensorList([0.9, 0.8, 0.1, 0.8]),
            "labels": TensorList([0, 0, 1, 1]),
        }
        decoded = detector.decode(result, 100, 60)
        self.assertEqual(decoded, [{"label": "sofa", "score": 0.9, "box": [0, 0, 100, 60]}])

    def test_decode_rejects_invalid_outputs(self):
        for labels, scores in (([99], [0.8]), ([0], [float("nan")]), ([], [0.8])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                Owlv2Detector().decode(
                    {
                        "boxes": TensorList([[0, 0, 10, 10]]),
                        "scores": TensorList(scores),
                        "labels": TensorList(labels),
                    },
                    100,
                    60,
                )

    def test_configuration_validation(self):
        for options in (
            {"threshold": -1},
            {"nms_iou": 2},
            {"categories": ["couch"]},
            {"categories": []},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Owlv2Detector(**options)

    @unittest.skipUnless(HAS_TORCH, "optional torch not installed")
    def test_forward_uses_inference_mode_and_height_width(self):
        import torch

        detector = Owlv2Detector(categories=("sofa", "chair"))
        detector.device = torch.device("cpu")
        processor = Mock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 2, 2)}
        processor.image_processor.post_process_object_detection.return_value = [
            {
                "boxes": torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
            }
        ]

        def forward(**inputs):
            self.assertFalse(torch.is_grad_enabled())
            return "raw-output"

        detector.processor, detector.model = processor, Mock(side_effect=forward)
        decoded, seconds = detector.detect(Image.new("RGB", (100, 60)))
        self.assertEqual(decoded[0]["label"], "chair")
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual(
            processor.call_args.kwargs["text"], [["a photo of a sofa", "a photo of a chair"]]
        )
        self.assertEqual(
            processor.image_processor.post_process_object_detection.call_args.kwargs[
                "target_sizes"
            ],
            [(60, 100)],
        )

    @unittest.skipUnless(HAS_TORCH, "optional torch not installed")
    def test_load_disallows_downloads_by_default(self):
        processor_type, model_type = Mock(), Mock()
        module = SimpleNamespace(Owlv2Processor=processor_type, Owlv2ForObjectDetection=model_type)
        with patch.dict(sys.modules, {"transformers": module}):
            detector = Owlv2Detector()
            detector.load()
        for factory in (processor_type, model_type):
            self.assertTrue(factory.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertTrue(model_type.from_pretrained.call_args.kwargs["use_safetensors"])
        model_type.from_pretrained.return_value.to.return_value.eval.assert_called_once()


class DataTests(WorkspaceTest):
    def test_relative_paths_and_annotation_states(self):
        rows = load_manifest(self.manifest([room(objects=[])]), "rooms")
        self.assertEqual(rows[0]["resolved_file"], str(self.root / "room.png"))
        self.assertEqual(rows[0]["objects"], [])

    def test_rights_duplicate_ids_and_verification_validation(self):
        invalid = [room(), room()]
        with self.assertRaises(ValueError):
            load_manifest(self.manifest(invalid), "rooms")
        invalid = room()
        invalid["rights"]["allowed_for_benchmark"] = False
        with self.assertRaises(ValueError):
            load_manifest(self.manifest([invalid]), "rooms")
        with self.assertRaises(ValueError):
            load_manifest(self.manifest([room(objects=[obj(verified=True)])]), "rooms")

    def test_exif_orientation(self):
        image = Image.new("RGB", (20, 10))
        exif = image.getexif()
        exif[274] = 6
        image.save(self.root / "rotated.jpg", exif=exif)
        self.assertEqual(load_image(self.root / "rotated.jpg").size, (10, 20))


class PipelineTests(WorkspaceTest):
    def test_artifacts_and_readable_saved_prediction(self):
        output = self.root / "run"
        run = run_experiment(self.manifest(), output, FixtureDetector())
        record = run["images"][0]
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["image_sha256"], digest(self.root / "room.png"))
        with Image.open(output / record["detections"][0]["crop_file"]) as crop:
            self.assertEqual(crop.size, (40, 30))
        with Image.open(output / record["annotated_file"]) as annotated:
            self.assertEqual(annotated.size, (100, 60))
            self.assertNotEqual(annotated.getpixel((10, 20)), (255, 255, 255))
        self.assertEqual(read_json(output / "predictions.json")["images"][0]["status"], "ok")

    def test_failure_is_recorded_and_next_image_runs(self):
        manifest = self.manifest([room("bad", "missing.png", [obj()]), room("good", objects=[])])
        run = run_experiment(manifest, self.root / "run", FixtureDetector())
        self.assertEqual([r["status"] for r in run["images"]], ["error", "ok"])
        self.assertEqual(run["images"][0]["errors"][0]["stage"], "image_load")

    def test_detector_failure_is_not_an_empty_success(self):
        detector = FixtureDetector()
        detector.detect = Mock(side_effect=RuntimeError("checkpoint absent"))
        run = run_experiment(self.manifest(), self.root / "run", detector)
        self.assertFalse(run["images"][0]["detection_completed"])
        self.assertEqual(run["images"][0]["errors"][0]["stage"], "owlv2")

    def test_unsafe_ids_cannot_escape_artifact_directory(self):
        output = self.root / "run"
        run = run_experiment(
            self.manifest([room("../../outside", objects=[])]), output, FixtureDetector()
        )
        artifact = (output / run["images"][0]["annotated_file"]).resolve()
        self.assertIn(output, artifact.parents)

    def test_run_directory_is_not_overwritten(self):
        with self.assertRaises(ValueError):
            run_experiment(self.manifest(), self.root, FixtureDetector())

    def test_unlabeled_image_runs_without_ground_truth(self):
        manifest = self.manifest([room(objects=None)])
        output = self.root / "run"
        run_experiment(manifest, output, FixtureDetector())
        report = evaluate(manifest, output / "predictions.json")
        self.assertIsNone(report["detection"]["precision"])
        self.assertEqual(report["unannotated_images"], 1)

    def test_optional_segmentation_failure_still_retrieves(self):
        embedder = Mock()
        embedder.encode.return_value = (np.array([[1.0, 0.0]], dtype=np.float32), 0.02)
        embedder.metadata.return_value = {"model_id": "fixture"}
        catalog = Mock()
        catalog.assert_compatible = Mock()
        catalog.search.return_value = ([[{"product_id": "p1", "identity_verified": False}]], 0.001)
        catalog.metadata = {"products": [{"product_id": "p1"}]}
        segmenter = Mock()
        segmenter.segment.side_effect = RuntimeError("no optional GPU")
        segmenter.metadata.return_value = {"model": "fixture"}
        manifest = self.manifest()
        output = self.root / "run"
        run = run_experiment(manifest, output, FixtureDetector(), embedder, catalog, segmenter)
        record = run["images"][0]
        self.assertEqual(record["status"], "partial")
        self.assertEqual(record["errors"][0]["stage"], "sam3.1")
        self.assertEqual(record["detections"][0]["matches"][0]["product_id"], "p1")
        self.assertEqual(
            evaluate(manifest, output / "predictions.json")["image_exact_match_rate"], 1
        )


class EvaluationTests(WorkspaceTest):
    def test_duplicate_predictions_are_false_positives(self):
        matches, extras, missed = match_objects([detection(), detection(score=0.7)], [obj()], 0.5)
        self.assertEqual((len(matches), len(extras), len(missed)), (1, 1, 0))

    def test_wrong_label_and_missing_predictions_count_as_misses(self):
        manifest = self.manifest([room("one", objects=[obj()]), room("two", objects=[obj()])])
        predictions = self.predictions(
            [
                {
                    "image_id": "one",
                    "status": "ok",
                    "detection_completed": True,
                    "detections": [detection("chair")],
                }
            ]
        )
        report = evaluate(manifest, predictions)
        self.assertEqual(report["detection"]["false_negatives"], 2)
        self.assertEqual(report["detection"]["false_positives"], 1)
        self.assertEqual(report["processing_failures"]["count"], 1)

    def test_retrieval_uses_verified_identity_and_includes_failed_rooms(self):
        manifest = self.manifest(
            [
                room(
                    "one",
                    objects=[
                        obj(product_id="p1", verified=True),
                        obj("o2", "lamp", [60, 0, 70, 40], "unverified", False),
                    ],
                ),
                room("two", objects=[obj(product_id="p2", verified=True)]),
                room("three", objects=[obj(product_id="absent", verified=True)]),
            ]
        )
        predictions = self.predictions(
            [
                {
                    "image_id": "one",
                    "status": "ok",
                    "detection_completed": True,
                    "detections": [detection(matches=[{"product_id": "p1"}])],
                    "timing_seconds": {"owlv2": 0.1, "total": 0.2},
                },
                {
                    "image_id": "two",
                    "status": "error",
                    "detection_completed": False,
                    "detections": [],
                    "errors": [{"stage": "owlv2"}],
                },
                {
                    "image_id": "three",
                    "status": "ok",
                    "detection_completed": True,
                    "detections": [detection()],
                },
            ],
            retrieval_enabled=True,
            catalog_product_ids=["p1", "p2"],
        )
        report = evaluate(manifest, predictions)
        self.assertEqual(report["retrieval"]["verified_in_catalog_objects"], 2)
        self.assertEqual(report["retrieval"]["recall_at_5_end_to_end"], 0.5)
        self.assertEqual(report["retrieval"]["recall_at_5_given_detection"], 1)
        self.assertEqual(report["retrieval"]["verified_objects_absent_from_catalog"], 1)
        self.assertEqual(report["latency_seconds"]["owlv2"]["count"], 1)

    def test_unverified_and_disabled_retrieval_return_null(self):
        manifest = self.manifest([room(objects=[obj(product_id="guess")])])
        predictions = self.predictions(
            [
                {
                    "image_id": "r1",
                    "status": "ok",
                    "detection_completed": True,
                    "detections": [detection(matches=[{"product_id": "guess"}])],
                }
            ]
        )
        report = evaluate(manifest, predictions)
        self.assertIsNone(report["retrieval"]["recall_at_5_end_to_end"])

    def test_empty_ground_truth_and_zero_denominators(self):
        manifest = self.manifest([room(objects=[])])
        predictions = self.predictions(
            [{"image_id": "r1", "status": "ok", "detection_completed": True, "detections": []}]
        )
        report = evaluate(manifest, predictions)
        self.assertEqual(report["image_exact_match_rate"], 1)
        self.assertIsNone(report["detection"]["recall"])

    def test_duplicate_records_and_nonfinite_latency_are_rejected(self):
        manifest = self.manifest()
        record = {"image_id": "r1", "status": "ok", "detections": []}
        with self.assertRaises(ValueError):
            evaluate(manifest, self.predictions([record, record]))
        record["timing_seconds"] = {"total": -1}
        with self.assertRaises(ValueError):
            evaluate(manifest, self.predictions([record]))

    def test_manifest_drift_is_rejected(self):
        manifest = self.manifest()
        predictions = self.predictions([], room_manifest_sha256="wrong")
        with self.assertRaises(ValueError):
            evaluate(manifest, predictions)


class RetrievalTests(unittest.TestCase):
    def test_normalization_and_invalid_vectors(self):
        np.testing.assert_allclose(unit_vectors([[3, 4]]), [[0.6, 0.8]])
        for vectors in ([[0, 0]], [[float("nan"), 1]], [], [1, 2]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                unit_vectors(vectors)

    def test_embedder_is_lazy(self):
        self.assertIsNone(Siglip2Embedder().model)

    @unittest.skipUnless(HAS_TORCH, "optional torch not installed")
    def test_siglip_feature_extraction_batches_and_normalizes(self):
        import torch

        embedder = Siglip2Embedder(batch_size=2)
        embedder.device = torch.device("cpu")
        embedder.processor = Mock(
            side_effect=lambda images, **_: {"pixel_values": torch.ones(len(images), 3, 2, 2)}
        )
        embedder.model = Mock()
        embedder.model.get_image_features.side_effect = lambda pixel_values: torch.tensor(
            [[3.0, 4.0]] * len(pixel_values)
        )
        vectors, seconds = embedder.encode([Image.new("RGB", (8, 8)) for _ in range(3)])
        np.testing.assert_allclose(vectors, [[0.6, 0.8]] * 3)
        self.assertEqual(embedder.model.get_image_features.call_count, 2)
        self.assertGreaterEqual(seconds, 0)

    def test_embedding_revision_mismatch_is_rejected(self):
        catalog = CatalogIndex(None, {"embedding": {"model_id": "a"}})
        with self.assertRaises(ValueError):
            catalog.assert_compatible({"model_id": "b"})

    @unittest.skipUnless(HAS_FAISS, "optional faiss-cpu not installed")
    def test_real_faiss_top5_roundtrip_and_small_catalog(self):
        products = [{"product_id": f"p{k}", "label": "chair", "file": f"{k}.png"} for k in range(6)]
        vectors = [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0, 1]]
        index = CatalogIndex.build(vectors, products, {"model_id": "test"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index"
            index.save(output)
            loaded = CatalogIndex.load(output)
            matches, _ = loaded.search([[1, 0]])
            self.assertEqual([m["product_id"] for m in matches[0]], [f"p{k}" for k in range(5)])
            self.assertFalse(any(m["identity_verified"] for m in matches[0]))
        small = CatalogIndex.build([[1, 0]], products[:1], {})
        self.assertEqual(len(small.search([[1, 0]])[0][0]), 1)
        index.close()
        loaded.close()
        small.close()


class CorrectionsAndCliTests(WorkspaceTest):
    def correction(self):
        return {
            "image_id": "r1",
            "image_sha256": digest(self.root / "room.png"),
            "reviewer": "test",
            "action": "update",
            "detection_id": "d0000",
            "label": "chair",
            "box": [10, 10, 50, 40],
        }

    def test_corrections_append_and_require_evidence_for_identity(self):
        path = self.root / "corrections.jsonl"
        correction = self.correction()
        first = save_correction(path, correction)
        second = save_correction(path, correction)
        self.assertNotEqual(first["correction_id"], second["correction_id"])
        self.assertFalse(first["used_for_training"])
        self.assertEqual(len(path.read_text().splitlines()), 2)
        correction.update(identity_verified=True, product_id="p1")
        with self.assertRaises(ValueError):
            save_correction(path, correction)

    def test_cli_evaluation_and_visualization(self):
        manifest = self.manifest()
        output = self.root / "run"
        run_experiment(manifest, output, FixtureDetector())
        self.assertEqual(
            main(
                [
                    "evaluate",
                    "--manifest",
                    str(manifest),
                    "--predictions",
                    str(output / "predictions.json"),
                    "--output",
                    str(self.root / "report.json"),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "visualize",
                    "--image",
                    str(self.root / "room.png"),
                    "--predictions",
                    str(output / "predictions.json"),
                    "--image-id",
                    "r1",
                    "--output",
                    str(self.root / "visual.png"),
                ]
            ),
            0,
        )
        self.assertTrue((self.root / "visual.png").exists())

    def test_cli_reports_failure_with_nonzero_exit(self):
        manifest = self.manifest()
        with patch.object(Owlv2Detector, "detect", side_effect=RuntimeError("no weights")):
            self.assertEqual(
                main(["detect", "--manifest", str(manifest), "--output", str(self.root / "run")]), 2
            )

    def test_cli_help_does_not_import_optional_libraries(self):
        code = (
            "import sys; from furniture_bench.cli import parser; parser(); "
            "assert not any(k in sys.modules for k in ['torch','transformers','faiss','sam3'])"
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sam_checkpoint_is_optional_and_lazy(self):
        segmenter = Sam31Segmenter(self.root / "missing.pt")
        self.assertIsNone(segmenter.predictor)
        with self.assertRaises(ValueError):
            segmenter.load()

    @unittest.skipUnless(HAS_TORCH, "optional torch not installed")
    def test_sam_session_associates_mask_and_closes_on_failure(self):
        import torch

        segmenter = Sam31Segmenter(self.root / "not-needed.pt")
        segmenter.device = torch.device("cpu")
        mask = np.zeros((1, 60, 100), dtype=bool)
        mask[0, 10:40, 10:50] = True

        def request(request):
            if request["type"] == "start_session":
                self.assertTrue((Path(request["resource_path"]) / "00000.jpg").exists())
                return {"session_id": "test"}
            if request["type"] == "add_prompt":
                return {"outputs": {"out_binary_masks": mask}}
            return {}

        segmenter.predictor = Mock()
        segmenter.predictor.handle_request.side_effect = request
        masks, seconds = segmenter.segment(load_image(self.root / "room.png"), [detection()])
        self.assertEqual(masks[0].size, (100, 60))
        self.assertEqual(masks[0].getpixel((20, 20)), 255)
        self.assertEqual(
            segmenter.predictor.handle_request.call_args.kwargs["request"]["type"], "close_session"
        )

        def failing_request(request):
            if request["type"] == "add_prompt":
                raise RuntimeError("fixture failure")
            return {"session_id": "test"}

        segmenter.predictor.handle_request.side_effect = failing_request
        with self.assertRaises(RuntimeError):
            segmenter.segment(load_image(self.root / "room.png"), [detection()])
        self.assertEqual(
            segmenter.predictor.handle_request.call_args.kwargs["request"]["type"], "close_session"
        )


@unittest.skipUnless(HAS_TRANSFORMERS and HAS_TORCH, "optional transformers/torch not installed")
class TinyTransformersTests(unittest.TestCase):
    """Exercise actual library graphs with small random models, never pretrained weights."""

    def test_real_owlv2_forward_and_postprocessing(self):
        import torch
        from transformers import Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor

        config = Owlv2Config(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "max_position_embeddings": 8,
                "eos_token_id": 2,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 32,
                "patch_size": 16,
            },
            projection_dim=32,
        )
        detector = Owlv2Detector(threshold=0, categories=("sofa", "chair"))
        detector.device = torch.device("cpu")
        detector.model = Owlv2ForObjectDetection(config).eval()
        image_processor = Owlv2ImageProcessor(size={"height": 32, "width": 32})

        class LocalProcessor:
            def __init__(self):
                self.image_processor = image_processor

            def __call__(self, text, images, return_tensors):
                batch = image_processor(images=images, return_tensors=return_tensors)
                batch["input_ids"] = torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0]])
                batch["attention_mask"] = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
                return batch

        detector.processor = LocalProcessor()
        detections, seconds = detector.detect(Image.new("RGB", (100, 60), "white"))
        self.assertIsInstance(detections, list)
        self.assertTrue(all(0 <= d["box"][2] <= 100 for d in detections))
        self.assertTrue(all(0 <= d["box"][3] <= 60 for d in detections))
        self.assertGreaterEqual(seconds, 0)

    def test_real_siglip2_image_features(self):
        import torch
        from transformers import Siglip2Config, Siglip2ImageProcessor, Siglip2Model

        config = Siglip2Config(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "max_position_embeddings": 8,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "image_size": 32,
                "patch_size": 16,
            },
        )
        embedder = Siglip2Embedder(batch_size=2)
        embedder.device = torch.device("cpu")
        embedder.model = Siglip2Model(config).eval()
        embedder.processor = Siglip2ImageProcessor(size={"height": 32, "width": 32})
        vectors, seconds = embedder.encode([Image.new("RGB", (48, 32), "blue")])
        self.assertEqual(vectors.shape, (1, 32))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1], atol=1e-5)
        self.assertGreaterEqual(seconds, 0)


if __name__ == "__main__":
    unittest.main()
