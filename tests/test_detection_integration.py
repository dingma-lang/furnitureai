from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from furniture_bench.annotations import annotation_sheets
from furniture_bench.data import digest, read_json, write_json
from furniture_bench.detection.owlv2 import Owlv2Detector
from furniture_bench.evaluation import evaluate
from furniture_bench.experiments import compare_filters, rebind_capture
from furniture_bench.postprocessing import FilterConfig, filter_detections
from furniture_bench.room_links import link_matches_room, link_observation, make_observation
from room_reconstruction.demo import demo_scene, replacement_sofa
from room_reconstruction.model import propose_replacement


def pred(label="sofa", score=0.8, box=None, detection_id="d0"):
    return {
        "label": label,
        "score": score,
        "box": box or [10, 10, 50, 40],
        "detection_id": detection_id,
    }


class FilterTests(unittest.TestCase):
    def test_containment_is_optional_and_category_aware(self):
        candidates = [
            pred(box=[0, 0, 100, 100]),
            pred(score=0.6, box=[10, 10, 20, 20]),
            pred("chair", 0.5, [10, 10, 20, 20]),
        ]
        self.assertEqual(len(filter_detections(candidates, FilterConfig())[0]), 3)
        filtered, audit = filter_detections(candidates, FilterConfig(containment_threshold=0.9))
        self.assertEqual(len(filtered), 2)
        self.assertEqual(audit["removed"][0]["reason"], "same_class_containment")

    def test_category_thresholds_can_be_lower_than_global_threshold(self):
        policy = FilterConfig(threshold=0.4, category_thresholds={"chair": 0.1})
        kept, _ = filter_detections([pred("chair", 0.15), pred("table", 0.3)], policy)
        self.assertEqual([p["label"] for p in kept], ["chair"])
        self.assertEqual(policy.candidate_floor, 0.1)
        self.assertEqual(
            Owlv2Detector(
                threshold=0.4, category_thresholds={"chair": 0.1}
            ).filter_config.candidate_floor,
            0.1,
        )

    def test_cross_class_suppression_requires_opt_in(self):
        candidates = [pred(), pred("chair", 0.6)]
        self.assertEqual(len(filter_detections(candidates, FilterConfig())[0]), 2)
        kept, audit = filter_detections(candidates, FilterConfig(cross_class_iou=0.95))
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit["removed"][0]["reason"], "cross_class_iou")

    def test_fragments_are_not_silently_joined(self):
        candidates = [pred(box=[0, 0, 40, 40]), pred(box=[41, 0, 80, 40])]
        self.assertEqual(
            len(filter_detections(candidates, FilterConfig(containment_threshold=0.9))[0]), 2
        )

    def test_invalid_configuration_is_rejected(self):
        for value in (
            {"threshold": float("nan")},
            {"category_thresholds": {"chair": -1}},
            {"category_thresholds": {"couch": 0.2}},
            {"cross_class_iou": True},
            {"unknown": 1},
            {"containment_threshold": 0},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FilterConfig.from_dict(value)


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (100, 60), "white").save(self.root / "room.png")
        self.row = {
            "image_id": "r",
            "file": "room.png",
            "image_sha256": digest(self.root / "room.png"),
            "rights": {"source": "test-generated", "license": "CC0", "allowed_for_benchmark": True},
            "objects": [{"object_id": "sofa", "label": "sofa", "box": [10, 10, 50, 40]}],
            "annotation": {"status": "assistant_provisional", "human_reviewed": False},
        }
        self.manifest = self.root / "rooms.json"

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        write_json(self.manifest, {"schema_version": 1, "images": [self.row]})

    def capture(self, candidates):
        self.save_manifest()
        path = self.root / "capture.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "room_manifest_sha256": digest(self.manifest),
                "detector": {"filter_config": FilterConfig(threshold=0.05).to_dict()},
                "images": [
                    {
                        "image_id": "r",
                        "status": "ok",
                        "detection_completed": True,
                        "image_sha256": self.row["image_sha256"],
                        "detections": candidates,
                        "raw_candidates": candidates,
                    }
                ],
            },
        )
        return path

    def test_ignored_regions_only_remove_unmatched_false_positives(self):
        self.row["ignore_regions"] = [{"box": [0, 0, 100, 60], "reason": "test exclusion"}]
        capture = self.capture([pred(), pred("lamp", 0.5, [60, 0, 90, 30])])
        report = evaluate(self.manifest, capture)
        self.assertEqual(report["detection"]["true_positives"], 1)
        self.assertEqual(report["detection"]["false_positives"], 0)
        self.assertEqual(report["ignored_predictions"], 1)

    def test_human_review_guard_and_missing_attestation(self):
        capture = self.capture([pred()])
        with self.assertRaises(ValueError):
            evaluate(self.manifest, capture, require_human_review=True)
        self.row["annotation"] = {"status": "human_reviewed"}
        self.save_manifest()
        with self.assertRaises(ValueError):
            evaluate(self.manifest, capture)

    def test_report_marks_review_status_confusion_and_missed_objects(self):
        report = evaluate(self.manifest, self.capture([pred("chair")]))
        self.assertEqual(report["annotation_status_counts"], {"assistant_provisional": 1})
        self.assertEqual(report["per_image"][0]["missed_objects"][0]["object_id"], "sofa")
        self.assertEqual(report["per_image"][0]["label_confusions"][0]["reference_label"], "sofa")

    def test_capture_reused_for_multiple_configurations(self):
        capture = self.capture([pred(), pred(score=0.2, box=[60, 0, 90, 30])])
        rows = compare_filters(
            self.manifest,
            capture,
            {"low": {"threshold": 0.15}, "high": {"threshold": 0.3}},
            self.root / "comparison",
        )
        self.assertEqual([r["false_positives"] for r in rows], [1, 0])
        self.assertFalse(
            read_json(self.root / "comparison/high/predictions.json")["filter_experiment"][
                "inference_repeated"
            ]
        )
        self.assertTrue((self.root / "comparison/comparison.csv").exists())

    def test_threshold_below_capture_floor_fails_before_output(self):
        capture = self.capture([pred()])
        output = self.root / "comparison"
        with self.assertRaises(ValueError):
            compare_filters(self.manifest, capture, {"too-low": {"threshold": 0.01}}, output)
        self.assertFalse(output.exists())

    def test_image_fingerprint_change_is_rejected(self):
        capture = self.capture([pred()])
        Image.new("RGB", (100, 60), "black").save(self.root / "room.png")
        with self.assertRaises(ValueError):
            evaluate(self.manifest, capture)

    def test_reference_sheets_do_not_promote_labels(self):
        self.save_manifest()
        annotation_sheets(self.manifest, self.root / "review")
        self.assertTrue((self.root / "review/reference-000.png").exists())
        self.assertIsNone(read_json(self.root / "review/human-review.template.json")["reviewer"])
        self.assertFalse(read_json(self.manifest)["images"][0]["annotation"]["human_reviewed"])

    def test_reviewed_annotation_update_reuses_identical_image_capture(self):
        capture = self.capture([pred()])
        self.row["annotation"] = {
            "status": "human_reviewed",
            "human_reviewed": True,
            "reviewer": "synthetic test reviewer",
            "reviewed_at": "2026-11-01",
            "evidence": "unit test only",
        }
        self.save_manifest()
        rebound = self.root / "reviewed-capture.json"
        rebind_capture(self.manifest, capture, rebound, "annotation review test")
        report = evaluate(self.manifest, rebound, require_human_review=True)
        self.assertEqual(report["detection"]["recall"], 1)
        self.assertFalse(read_json(rebound)["annotation_rebind"]["inference_repeated"])

    def test_annotation_rebind_rejects_different_image_bytes(self):
        capture = self.capture([pred()])
        Image.new("RGB", (100, 60), "black").save(self.root / "different.png")
        self.row["file"] = "different.png"
        self.row["image_sha256"] = digest(self.root / "different.png")
        self.save_manifest()
        with self.assertRaises(ValueError):
            rebind_capture(self.manifest, capture, self.root / "rebound.json", "test")


class RoomLinkTests(unittest.TestCase):
    def observation(self):
        return make_observation(
            {
                "image_id": "r",
                "image_sha256": "a" * 64,
                "width": 100,
                "height": 60,
                "detections": [pred(), pred(score=0.6, detection_id="d1")],
            },
            ["d0"],
            "obs",
        )

    def test_observation_has_pixels_but_no_invented_metric_geometry(self):
        obs = self.observation()
        self.assertIsNone(obs["estimated_pose"])
        self.assertIsNone(obs["physical_dimensions"])
        self.assertEqual(obs["image_frame"]["units"], "pixels")

    def test_demo_link_explicit_and_room_unmodified(self):
        room = demo_scene()
        snapshot = copy.deepcopy(room)
        with self.assertRaises(ValueError):
            link_observation(self.observation(), room, "sofa-original", "test", "illustrative")
        linked = link_observation(
            self.observation(), room, "sofa-original", "test", "illustrative", demo_only=True
        )
        self.assertEqual(linked["link_status"], "demonstration_only")
        self.assertEqual(room, snapshot)

    def test_estimates_require_method_and_uncertainty_and_measurements_evidence(self):
        for pose, dims in (
            ({"units": "m", "status": "estimated", "source": "test", "position": [1, 2, 0]}, None),
            (None, {"units": "m", "status": "measured", "source": "test", "size": [2, 1, 1]}),
        ):
            with self.subTest(pose=pose, dims=dims), self.assertRaises(ValueError):
                link_observation(
                    self.observation(),
                    demo_scene(),
                    "sofa-original",
                    "test",
                    "illustrative",
                    pose,
                    dims,
                    True,
                )

    def test_unreviewed_fragment_group_cannot_be_linked(self):
        obs = self.observation()
        obs["image_regions"].append(dict(obs["image_regions"][0], detection_id="d1"))
        obs["grouping_review"] = "pending"
        with self.assertRaises(ValueError):
            link_observation(
                obs, demo_scene(), "sofa-original", "test", "illustrative", demo_only=True
            )

    def test_replacement_invalidates_old_observation_link_without_changing_geometry(self):
        room = demo_scene()
        linked = link_observation(
            self.observation(), room, "sofa-original", "test", "illustrative", demo_only=True
        )
        self.assertTrue(link_matches_room(linked, room))
        updated, report = propose_replacement(room, "sofa-original", replacement_sofa())
        self.assertTrue(report["accepted"])
        self.assertFalse(link_matches_room(linked, updated))
        for key in ("floor", "walls", "openings", "dimensions"):
            self.assertEqual(room[key], updated[key])


if __name__ == "__main__":
    unittest.main()
