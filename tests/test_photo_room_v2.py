from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from furniture_bench.detection.owlv2 import Owlv2Detector
from furniture_bench.room_links import link_matches_room, make_observation
from room_reconstruction.calibration import (
    calibrate_floor,
    project_detection_contacts,
    project_floor_contact,
    room_from_dimensions,
    transform_points,
)
from room_reconstruction.clearance import assess_arrangement, polygon_gap
from room_reconstruction.cli import main
from room_reconstruction.collision import footprint
from room_reconstruction.demo import demo_scene, replacement_sofa
from room_reconstruction.integration import associate_observation
from room_reconstruction.model import propose_replacement, save_json


class Values:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def controls():
    matrix = np.array([[120.0, 20, 180], [10, -80, 750], [0.015, 0.035, 1]])
    corners = [[0, 0], [6, 0], [6, 4.5], [0, 4.5]]
    return {
        "schema_version": 1,
        "room_id": "known-dimensions-demo",
        "image": {"sha256": "a" * 64, "width": 1600, "height": 1067},
        "dimensions": {
            "width": 6,
            "depth": 4.5,
            "height": 2.8,
            "status": "demo_specified",
            "source": "synthetic test only",
        },
        "floor_corners_pixels": transform_points(matrix, corners).tolist(),
        "pixel_uncertainty": 3,
    }, matrix


def observation():
    record = {
        "image_id": "test",
        "image_sha256": "a" * 64,
        "width": 1600,
        "height": 1067,
        "detections": [
            {"detection_id": "d0", "label": "sofa", "score": 0.8, "box": [100, 100, 400, 500]}
        ],
    }
    return make_observation(record, ["d0"], "sofa-observation")


class PromptBankTests(unittest.TestCase):
    def test_default_queries_unchanged(self):
        detector = Owlv2Detector(categories=("sofa", "table"))
        self.assertEqual(detector.query_texts, ["a photo of a sofa", "a photo of a table"])
        self.assertEqual(detector.query_labels, ["sofa", "table"])

    def test_alias_index_maps_to_canonical_category(self):
        detector = Owlv2Detector(
            categories=("sofa", "table"), prompt_bank={"table": ["coffee table", "console table"]}
        )
        result = {"boxes": Values([[0, 0, 50, 40]]), "scores": Values([0.8]), "labels": Values([2])}
        predicted = detector.decode(result, 100, 60)
        self.assertEqual(predicted[0]["label"], "table")
        self.assertEqual(predicted[0]["matched_prompt"], "console table")
        self.assertEqual(detector.metadata()["query_categories"], ["sofa", "table", "table"])

    def test_invalid_prompt_bank(self):
        for bank in ({"couch": ["sofa"]}, {"sofa": []}, {"table": "coffee table"}, {"lamp": [""]}):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                Owlv2Detector(prompt_bank=bank)


class FloorCalibrationTests(unittest.TestCase):
    def test_known_synthetic_homography_and_held_out_point(self):
        config, matrix = controls()
        calibration = calibrate_floor(config)
        held_out = [[2.3, 1.7]]
        pixel = transform_points(matrix, held_out)[0]
        result = project_floor_contact(calibration, pixel)
        np.testing.assert_allclose(result["position"], [2.3, 1.7, 0], atol=1e-9)
        self.assertLess(calibration["reprojection_rmse_pixels"], 1e-8)
        self.assertFalse(calibration["physical_accuracy_verified"])
        self.assertIsNone(calibration["camera_perspective"]["camera_pose_3d"])
        self.assertIsNone(result["entity_centre_pose"])
        self.assertGreater(result["uncertainty_metres"], 0)

    def test_declared_room_scale_changes_projected_distance(self):
        config, matrix = controls()
        first = calibrate_floor(config)
        config["dimensions"]["width"] *= 2
        config["dimensions"]["depth"] *= 2
        second = calibrate_floor(config)
        pixel = transform_points(matrix, [[2, 1]])[0]
        np.testing.assert_allclose(
            project_floor_contact(second, pixel)["position"], [4, 2, 0], atol=1e-9
        )
        np.testing.assert_allclose(
            project_floor_contact(first, pixel)["position"], [2, 1, 0], atol=1e-9
        )

    def test_collinear_and_outside_control_points_rejected(self):
        for points in (
            [[100, 100], [200, 100], [300, 100], [400, 100]],
            [[-1, 100], [200, 100], [300, 200], [100, 200]],
        ):
            config, _ = controls()
            config["floor_corners_pixels"] = points
            with self.subTest(points=points), self.assertRaises(ValueError):
                calibrate_floor(config)

    def test_horizon_projection_rejected(self):
        config, _ = controls()
        calibration = calibrate_floor(config)
        calibration["image_to_floor"] = [[1, 0, 0], [0, 1, 0], [0, 1, -100]]
        with self.assertRaises(ValueError):
            project_floor_contact(calibration, [200, 100])

    def test_projection_binds_exact_image(self):
        config, _ = controls()
        calibration = calibrate_floor(config)
        with self.assertRaises(ValueError):
            project_detection_contacts(calibration, {"image_sha256": "b" * 64})

    def test_verified_dimensions_do_not_verify_calibration(self):
        config, _ = controls()
        config["dimensions"]["status"] = "measured"
        with self.assertRaises(ValueError):
            calibrate_floor(config)
        config["dimensions"]["measurement_evidence"] = "synthetic unit test record"
        self.assertFalse(calibrate_floor(config)["physical_accuracy_verified"])

    def test_room_from_provided_dimensions_retains_provenance(self):
        config, _ = controls()
        dimensions = dict(config["dimensions"], status="provided_unverified")
        room = room_from_dimensions(dimensions)
        self.assertEqual(room["dimensions"], [6, 4.5, 2.8])
        self.assertEqual(room["provenance"]["dimensions"]["status"], "provided_unverified")
        self.assertFalse(room["provenance"]["physical_accuracy_verified"])
        self.assertEqual(room["furniture"], [])


class PhotoAssociationTests(unittest.TestCase):
    def test_association_is_persistent_without_applying_pixels_as_geometry(self):
        room = demo_scene()
        original = copy.deepcopy(room)
        updated, link = associate_observation(
            room, observation(), "sofa-original", "test", "illustrative only", True
        )
        self.assertTrue(link_matches_room(link, updated))
        sofa = next(o for o in updated["furniture"] if o["id"] == "sofa-original")
        self.assertEqual(len(sofa["photo_observations"]), 1)
        self.assertFalse(sofa["photo_observations"][0]["metric_geometry_applied"])
        self.assertEqual(room, original)
        for key in ("floor", "walls", "openings", "dimensions"):
            self.assertEqual(room[key], updated[key])
        self.assertEqual(sofa["position"], original["furniture"][0]["position"])
        self.assertEqual(sofa["size"], original["furniture"][0]["size"])

    def test_contact_estimate_is_separate_and_image_bound(self):
        config, matrix = controls()
        estimate = project_floor_contact(
            calibrate_floor(config), transform_points(matrix, [[3, 3]])[0]
        )
        updated, _ = associate_observation(
            demo_scene(),
            observation(),
            "sofa-original",
            "test",
            "illustrative only",
            True,
            estimate,
        )
        self.assertEqual(
            updated["furniture"][0]["photo_observations"][0]["ground_contact_estimate"]["status"],
            "estimated",
        )
        estimate["source_image_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            associate_observation(
                demo_scene(),
                observation(),
                "sofa-original",
                "test",
                "illustrative only",
                True,
                estimate,
            )

    def test_replacement_does_not_inherit_original_photo(self):
        room, link = associate_observation(
            demo_scene(), observation(), "sofa-original", "test", "illustrative only", True
        )
        replaced, report = propose_replacement(room, "sofa-original", replacement_sofa())
        self.assertTrue(report["accepted"])
        sofa = next(o for o in replaced["furniture"] if o["id"] == "sofa-replacement")
        self.assertNotIn("photo_observations", sofa)
        self.assertFalse(link_matches_room(link, replaced))

    def test_duplicate_association_rejected(self):
        room, _ = associate_observation(
            demo_scene(), observation(), "sofa-original", "test", "illustrative only", True
        )
        with self.assertRaises(ValueError):
            associate_observation(
                room, observation(), "sofa-original", "test", "illustrative only", True
            )


class ClearanceTests(unittest.TestCase):
    def replaced_room(self):
        return propose_replacement(demo_scene(), "sofa-original", replacement_sofa())[0]

    def test_rotated_gap_distance_and_overlap(self):
        a = {"position": [0, 0, 0], "size": [2, 1, 1], "rotation_degrees": 0}
        b = {"position": [3, 0, 0], "size": [2, 1, 1], "rotation_degrees": 0}
        self.assertAlmostEqual(polygon_gap(footprint(a), footprint(b)), 1)
        a["rotation_degrees"] = b["rotation_degrees"] = 90
        b["position"] = [0, 3, 0]
        self.assertAlmostEqual(polygon_gap(footprint(a), footprint(b)), 1)
        b["position"] = [0, 0, 0]
        self.assertEqual(polygon_gap(footprint(a), footprint(b)), 0)

    def test_geometric_fit_is_distinct_from_clearance_targets(self):
        policy = {
            "minimum_furniture_gap": 0.1,
            "front_access_depths": {"chair": 0.6},
            "passages": [
                {"id": "side", "between": ["sofa-replacement", "floor-lamp"], "minimum_width": 0.8}
            ],
        }
        report = assess_arrangement(self.replaced_room(), policy)
        self.assertTrue(report["geometrically_valid"])
        self.assertFalse(report["clearance_targets_met"])
        self.assertFalse(report["comfort_verified"])
        self.assertAlmostEqual(report["passage_checks"][0]["local_gap_metres"], 0.075)
        self.assertEqual(
            {w["type"] for w in report["warnings"]},
            {
                "restrictive_furniture_gap",
                "potential_front_access_obstruction",
                "restrictive_nominated_passage",
            },
        )

    def test_no_implicit_walkway_claim(self):
        report = assess_arrangement(self.replaced_room(), {})
        self.assertTrue(report["clearance_targets_met"])
        self.assertFalse(report["comfort_verified"])
        self.assertEqual(report["passage_checks"], [])

    def test_wall_clearance_and_unknown_passages(self):
        report = assess_arrangement(self.replaced_room(), {"minimum_wall_gap": 0.3})
        self.assertTrue(
            any(
                w["object_id"] == "sofa-replacement" and w["wall_id"] == "north"
                for w in report["warnings"]
            )
        )
        with self.assertRaises(ValueError):
            assess_arrangement(
                self.replaced_room(),
                {"passages": [{"between": ["missing", "chair"], "minimum_width": 0.8}]},
            )

    def test_door_blockage_still_fails_geometry(self):
        room = demo_scene()
        room["furniture"] = [
            {
                "id": "blocker",
                "category": "chair",
                "position": [0.95, 0.45, 0],
                "size": [0.5, 0.5, 0.8],
                "rotation_degrees": 0,
            }
        ]
        report = assess_arrangement(room, {})
        self.assertFalse(report["geometrically_valid"])
        self.assertIn(
            "opening_clearance", {v["type"] for v in report["geometry_check"]["violations"]}
        )

    def test_cli_reports_restricted_fit_with_exit_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "room.json", self.replaced_room())
            save_json(root / "policy.json", {"minimum_furniture_gap": 0.1})
            self.assertEqual(
                main(
                    [
                        "assess",
                        "--room",
                        str(root / "room.json"),
                        "--policy",
                        str(root / "policy.json"),
                        "--output",
                        str(root / "assessment.json"),
                    ]
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main()
