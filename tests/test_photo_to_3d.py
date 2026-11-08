import copy
import io
import tempfile
import unittest

from PIL import Image

from room_viewer.photo_projects import ProjectStore
from room_viewer.service import edit_scene, presentation


class PhotoTo3DTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ProjectStore(self.tmp.name)
        self.pid = self.store.create("Synthetic design integration")["id"]
        out = io.BytesIO()
        Image.new("RGB", (64, 48), "tan").save(out, "PNG")
        p = self.store.upload(self.pid, out.getvalue(), "test.png", True)
        self.photo = p["photos"][0]["id"]
        p = self.act(type="correct", photo_id=self.photo, operation="add", label="sofa", box=[2, 3, 50, 40])
        self.detection = p["photos"][0]["reviewed"][0]["detection_id"]
        self.act(type="correct", photo_id=self.photo, operation="confirm", detection_id=self.detection)
        self.observation = self.photo + ":" + self.detection

    def act(self, **body):
        return self.store.action(self.pid, body)

    def setup(self, **extra):
        return self.act(type="setup_room", layout="rectangular", dimensions=[6, 4.5, 2.8], dimension_status="unknown", source="Synthetic approximate display", approximate_acknowledged=True, **extra)

    def create(self, **extra):
        return self.act(type="create_placeholder", observation_id=self.observation, size=[2.1, .85, .85], position=[3, 3, 0], rotation_degrees=0, dimension_status="unknown", source="Example proportions, not measurements", approximate_acknowledged=True, **extra)

    def test_creation_requires_confirmation_and_explicit_room_not_inferred_box(self):
        original = copy.deepcopy(self.store.read(self.pid))
        with self.assertRaisesRegex(ValueError, "Set up"):
            self.create()
        self.setup()
        result = self.create()
        entity = result["room"]["furniture"][0]
        self.assertEqual(entity["category"], "sofa")
        self.assertEqual(entity["position"], [3, 3, 0])
        self.assertIsNone(entity["design_evidence"]["position"]["physical_position"])
        self.assertIsNone(entity["design_evidence"]["dimensions"]["value"])
        self.assertEqual(self.store.read(self.pid)["photos"][0], original["photos"][0])
        self.assertTrue(result["room_history"])
        self.assertEqual(result["associations"][self.observation]["entity_id"], entity["id"])
        self.assertEqual(presentation(result["room"])["status"], "disabled")
        with self.assertRaisesRegex(ValueError, "already"):
            self.create()
        self.act(type="correct", photo_id=self.photo, operation="reject", detection_id=self.detection)
        self.assertEqual(self.store.read(self.pid)["associations"][self.observation]["status"], "observation_not_confirmed")

    def test_edit_replace_remove_restore_persist_stable_identity(self):
        self.setup()
        result = self.create()
        entity = result["room"]["furniture"][0]
        entity_id = entity["id"]
        room = edit_scene(result["room"], {"type": "update", "id": entity_id, "size": [2.5, 1, 1], "position": [2, 2, 0], "rotation_degrees": 45})["scene"]
        result = self.act(type="room", room=room)
        self.assertEqual(result["associations"][self.observation]["status"], "editable_placeholder")
        self.assertEqual(result["photo_objects"][entity_id]["entity"]["rotation_degrees"], 45)
        replaced = edit_scene(room, {"type": "replace", "id": entity_id, "size": [9, 2, 1]})
        self.assertIsNone(replaced["replacement_proposal"]["accepted"])
        self.assertIsNone(replaced["assessment"])
        result = self.act(type="room", room=replaced["scene"])
        result = self.act(type="room", room=edit_scene(result["room"], {"type": "remove", "id": entity_id})["scene"])
        self.assertEqual(result["photo_objects"][entity_id]["state"], "removed")
        self.assertEqual(result["associations"][self.observation]["status"], "placeholder_removed")
        result = self.act(type="restore_placeholder", entity_id=entity_id)
        self.assertEqual(result["room"]["furniture"][0]["id"], entity_id)
        self.assertEqual(result["room"]["furniture"][0]["size"], [9, 2, 1])
        reopened = self.store.import_bundle(self.store.export(self.pid))
        self.assertEqual(reopened["photo_objects"], result["photo_objects"])
        self.assertEqual(reopened["associations"], result["associations"])
        self.assertTrue(reopened["editing_history"])

    def test_measured_sources_preserved_but_edit_invalidates_current_measurement(self):
        self.act(type="setup_room", layout="rectangular", dimensions=[6, 4, 3], dimension_status="measured", source="Synthetic tape test", evidence="Test evidence only", confirmed=True, approximate_acknowledged=True)
        p = self.act(type="create_placeholder", observation_id=self.observation, size=[2, 1, 1], position=[3, 2, 0], dimension_status="measured", source="Synthetic tape test", evidence="Test furniture evidence", confirmed=True, approximate_acknowledged=True)
        item = p["room"]["furniture"][0]
        self.assertEqual(item["design_evidence"]["dimensions"]["status"], "measured")
        edited = edit_scene(p["room"], {"type": "update", "id": item["id"], "size": [3, 1, 1]})
        p = self.act(type="room", room=edited["scene"])
        self.assertEqual(p["room"]["furniture"][0]["design_evidence"]["dimensions"]["status"], "unknown")
        self.assertEqual(p["measurements"][-1]["status"], "measured")
        resized = edit_scene(p["room"], {"type": "resize", "dimensions": [7, 4, 3]})
        self.assertEqual(resized["scene"]["provenance"]["dimension_evidence"]["status"], "unknown")
        self.assertFalse(resized["physical_placement_verified"])
        self.assertIsNone(resized["assessment"])

    def test_saved_undo_restores_objects_and_leaves_ai_corrections_untouched(self):
        self.setup()
        p = self.create()
        before = copy.deepcopy(p["photos"])
        p = self.act(type="undo_design")
        self.assertEqual(p["room"]["furniture"], [])
        self.assertFalse(any(a.get("status") == "editable_placeholder" for a in p["associations"].values()))
        self.assertEqual(p["photos"], before)

    def test_unconfirmed_and_unsupported_setup_are_rejected_without_geometry(self):
        with self.assertRaises(ValueError):
            self.act(type="setup_room", layout="inferred", dimensions=[6, 4, 3], dimension_status="unknown", source="test", approximate_acknowledged=True)
        self.setup()
        self.act(type="correct", photo_id=self.photo, operation="update", detection_id=self.detection, label="chair")
        with self.assertRaisesRegex(ValueError, "Confirm"):
            self.create()
        self.assertEqual(self.store.read(self.pid)["room"]["furniture"], [])


    def test_other_room_never_receives_removed_placeholder_and_evidence_is_explicit(self):
        self.setup()
        p = self.create()
        entity_id = p["room"]["furniture"][0]["id"]
        self.act(type="room", room=edit_scene(p["room"], {"type": "remove", "id": entity_id})["scene"])
        self.setup()
        self.assertEqual(self.store.read(self.pid)["photo_objects"][entity_id]["state"], "other_room")
        with self.assertRaisesRegex(ValueError, "current room"):
            self.act(type="restore_placeholder", entity_id=entity_id)
        with self.assertRaisesRegex(ValueError, "Confirm"):
            self.act(type="setup_room", layout="rectangular", dimensions=[6, 4, 3], dimension_status="measured", source="Tape", approximate_acknowledged=True)
        with self.assertRaisesRegex(ValueError, "Acknowledge"):
            self.act(type="create_placeholder", observation_id=self.observation, size=[2, 1, 1], position=[2, 2, 0], source="Example")


if __name__ == "__main__":
    unittest.main()
