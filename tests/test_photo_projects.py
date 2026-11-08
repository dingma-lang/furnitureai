import copy
import io
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from PIL import Image

from room_reconstruction.demo import demo_scene
from room_viewer.photo_jobs import PhotoJobs
from room_viewer.photo_projects import MAX_BYTES, ProjectStore, image_copies


def png():
    out = io.BytesIO()
    Image.new("RGB", (64, 48), "tan").save(out, "PNG")
    return out.getvalue()


class PhotoProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.tmp.name)
        self.project = self.store.create("Synthetic offline test")
        self.id = self.project["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def upload(self):
        result = self.store.upload(self.id, png(), "../../user/photo.png", True)
        return result, result["photos"][-1]["id"]

    def fake_runner(self, calls, fail_seg=False):
        def run(stage, folder, photo):
            calls.append(stage)
            if stage == "segmentation" and fail_seg:
                raise RuntimeError("Synthetic segmentation failure")
            if stage == "detection":
                return {"data": {"detections": [{"detection_id": "d0000", "label": "sofa", "score": .6, "box": [1, 2, 30, 40]}], "raw_candidates": [{"label": "sofa", "score": .6, "box": [1, 2, 30, 40]}], "model": {"name": "mock-test-only"}, "settings": {"threshold": .2}}}
            Image.new("RGBA", (photo["width"], photo["height"])).save(folder / ("semantic.png" if stage == "segmentation" else "depth.png"))
            return {"data": {"structure": {"regions": [], "camera_perspective": {"status": "unknown"}}, "physical_units": None}}
        return run

    def infer(self, photo_id, calls=None, fail_seg=False):
        calls = [] if calls is None else calls
        jobs = PhotoJobs(self.store, self.fake_runner(calls, fail_seg))
        jobs.execute(self.id, photo_id)
        jobs.pool.shutdown()
        return self.store.read(self.id), calls

    def test_image_content_limits_and_safe_identity_storage(self):
        project, photo_id = self.upload()
        self.assertEqual(project["photos"][0]["name"], "photo.png")
        self.assertEqual(self.store.asset(self.id, photo_id, "processing.png").parent.name, photo_id)
        self.assertFalse(project["consent"]["training_authorized"])
        self.assertNotIn(self.tmp.name, str(project))
        for data in (b"not an image", b"0" * (MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                image_copies(data)
        with self.assertRaises(ValueError):
            self.store.asset(self.id, "../outside", "processing.png")
        with self.assertRaises(ValueError):
            self.store.upload(self.id, png(), "photo.png", False)
        with patch("room_viewer.photo_projects.MAX_PIXELS", 10):
            with self.assertRaises(ValueError):
                image_copies(png())

    def test_actual_orientation_and_unsupported_format(self):
        out = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (32, 24)).save(out, "JPEG", exif=exif)
        _, normalized, size, original = image_copies(out.getvalue())
        self.assertEqual(size, (24, 32))
        self.assertEqual(Image.open(io.BytesIO(normalized)).size, (24, 32))
        out = io.BytesIO()
        Image.new("RGB", (32, 24)).save(out, "GIF")
        with self.assertRaisesRegex(ValueError, "JPEG and PNG"):
            image_copies(out.getvalue())

    def test_independent_stage_failure_and_cached_reuse(self):
        _, photo_id = self.upload()
        project, calls = self.infer(photo_id, fail_seg=True)
        photo = project["photos"][0]
        self.assertEqual(calls, ["detection", "segmentation", "depth"])
        self.assertEqual(photo["processing_state"], "partial")
        self.assertTrue(photo["predictions"]["detections"])
        self.assertIsNone(photo["depth"]["physical_units"])
        _, second = self.upload()
        project, new_calls = self.infer(second)
        self.assertEqual(new_calls, [])
        self.assertTrue(project["photos"][1]["stages"]["detection"]["cached"])

    def test_corrections_undo_and_persistence_do_not_change_predictions(self):
        _, photo = self.upload()
        project, _ = self.infer(photo)
        original = copy.deepcopy(project["photos"][0]["predictions"])
        for operation, extra in (("update", {"label": "chair", "box": [2, 3, 31, 41]}), ("confirm", {}), ("reject", {}), ("undo", {}), ("add", {"label": "bookshelf", "box": [32, 3, 60, 44]})):
            self.store.action(self.id, {"type": "correct", "photo_id": photo, "operation": operation, "detection_id": "d0000", **extra})
        project = self.store.read(self.id)
        self.assertEqual(project["photos"][0]["predictions"], original)
        self.assertEqual(project["photos"][0]["reviewed"][0]["label"], "chair")
        self.assertTrue(project["photos"][0]["reviewed"][0]["confirmed"])
        self.assertEqual(project["photos"][0]["reviewed"][1]["label"], "cabinet")
        self.assertIsNone(project["photos"][0]["reviewed"][1]["score"])
        self.assertIsNone(project["observations"][0]["estimated_pose"])
        self.assertEqual(next(iter(project["associations"].values()))["status"], "unresolved")
        with self.assertRaises(ValueError):
            self.store.action(self.id, {"type": "correct", "photo_id": photo, "operation": "update", "detection_id": "d0000", "box": [-1, 2, 8, 9]})

    def test_measurements_and_explicit_demo_association_never_calibrate_photo(self):
        _, photo = self.upload()
        self.infer(photo)
        self.store.action(self.id, {"type": "correct", "photo_id": photo, "operation": "confirm", "detection_id": "d0000"})
        project = self.store.read(self.id)
        observation = project["observations"][0]["observation_id"]
        with self.assertRaisesRegex(ValueError, "No dimensioned"):
            self.store.action(self.id, {"type": "associate", "observation_id": observation, "entity_id": "sofa-original", "evidence": "test"})
        self.store.action(self.id, {"type": "room", "room": demo_scene()})
        with self.assertRaisesRegex(ValueError, "demo_only"):
            self.store.action(self.id, {"type": "associate", "observation_id": observation, "entity_id": "sofa-original", "evidence": "test"})
        result = self.store.action(self.id, {"type": "associate", "observation_id": observation, "entity_id": "sofa-original", "evidence": "Synthetic demonstration only", "demo_only": True})
        self.assertEqual(result["associations"][observation]["status"], "demonstration_only")
        self.store.action(self.id, {"type": "measurement", "reference": "room width", "value": 6, "source": "Tape", "status": "measured", "confirmed": True, "evidence": "Synthetic test"})
        self.assertFalse(self.store.read(self.id)["geometry"]["physical_accuracy_verified"])
        self.assertFalse(self.store.read(self.id)["measurements"][0]["applied_to_geometry"])
        _, second = self.upload()
        self.assertEqual(len(self.store.read(self.id)["measurements"]), 1)
        self.assertEqual(len(self.store.read(self.id)["observations"]), 1)

    def test_archive_round_trip_preserves_extensions_and_rejects_traversal(self):
        self.upload()
        project = self.store.read(self.id)
        project["future_extension"] = {"preserved": True}
        self.store.write(project)
        imported = self.store.import_bundle(self.store.export(self.id))
        self.assertNotEqual(imported["id"], self.id)
        self.assertEqual(imported["future_extension"], {"preserved": True})
        self.assertEqual(imported["photos"][0]["sha256"], project["photos"][0]["sha256"])
        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("../outside.txt", "not permitted")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.store.import_bundle(bad.getvalue())

    def test_invalid_calibration_and_unconfirmed_measured_values_are_rejected(self):
        self.upload()
        room = copy.deepcopy(self.store.read(self.id)["room"])
        room["views"][0]["camera"].update(intrinsics=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]], calibration_source="synthetic invalid camera", calibration_status="estimated")
        with self.assertRaisesRegex(ValueError, "calibration"):
            self.store.action(self.id, {"type": "room", "room": room})
        with self.assertRaisesRegex(ValueError, "Confirm direct"):
            self.store.action(self.id, {"type": "measurement", "reference": "room width", "value": 5, "source": "Tape", "status": "measured"})

    def test_manual_observation_before_inference_survives_and_ai_rows_are_added(self):
        _, photo = self.upload()
        self.store.action(self.id, {"type": "correct", "photo_id": photo, "operation": "add", "label": "chair", "box": [2, 3, 20, 30]})
        project, _ = self.infer(photo)
        reviewed = project["photos"][0]["reviewed"]
        self.assertEqual(len(reviewed), 2)
        self.assertEqual(reviewed[0]["source"], "user")
        self.assertEqual(reviewed[0]["box"], [2, 3, 20, 30])
        self.assertEqual(reviewed[1]["source"], "ai")

    def test_private_provenance_is_hidden_but_survives_editor_roundtrip(self):
        self.upload()
        room = copy.deepcopy(self.store.read(self.id)["room"])
        room["future_geometry_extension"] = {"file": "/Users/ding/private-example/source.npz", "custom_field": 17}
        visible = self.store.action(self.id, {"type": "room", "room": room})
        self.assertNotIn("/Users/ding", str(visible))
        self.store.action(self.id, {"type": "room", "room": visible["room"]})
        self.assertEqual(self.store.read(self.id)["room"]["future_geometry_extension"], room["future_geometry_extension"])

    def test_interrupted_session_requires_retry_and_duplicate_submit_runs_once(self):
        _, photo = self.upload()
        project = self.store.read(self.id)
        project["photos"][0]["processing_state"] = "running"
        self.store.write(project)
        calls = []
        jobs = PhotoJobs(self.store, self.fake_runner(calls))
        self.assertEqual(self.store.read(self.id)["photos"][0]["processing_state"], "failed")
        with self.assertRaisesRegex(ValueError, "Retry"):
            jobs.submit(self.id, photo)
        jobs.submit(self.id, photo, retry=True)
        jobs.submit(self.id, photo, retry=True)
        jobs.pool.shutdown(wait=True)
        self.assertEqual(calls, ["detection", "segmentation", "depth"])
        self.assertEqual(self.store.read(self.id)["photos"][0]["processing_state"], "complete")


if __name__ == "__main__":
    unittest.main()
