import io
import unittest
import zipfile

from room_multiview.public_room_audit import approve_fragment
from scripts.fetch_arkitscenes_vga_subset import RangeZip


class FragmentAuditTests(unittest.TestCase):
    def test_complete_accurate_fragment_and_insufficient_primary_support(self):
        stats = {"evaluated_non_anchor_views": 36, "translation_max_metres": 0.046, "rotation_max_degrees": 2.84}
        self.assertTrue(approve_fragment(stats, 37, 2))
        self.assertFalse(approve_fragment(stats, 37, 1))

    def test_unobserved_or_bad_tracking_views_prevent_whole_fragment_fusion(self):
        stats = {"evaluated_non_anchor_views": 35, "translation_max_metres": 0.01, "rotation_max_degrees": 0.1}
        self.assertFalse(approve_fragment(stats, 37, 2))
        stats.update(evaluated_non_anchor_views=36, translation_max_metres=0.11)
        self.assertFalse(approve_fragment(stats, 37, 2))

    def test_invalid_errors_are_never_accepted(self):
        for value in (None, float("nan"), float("inf"), -1, True):
            stats = {"evaluated_non_anchor_views": 2, "translation_max_metres": value, "rotation_max_degrees": 0.1}
            self.assertFalse(approve_fragment(stats, 3, 2))


class LocalRangeZip(RangeZip):
    def __init__(self, data):
        self.data = data
        super().__init__(len(data), "unused-test-cache")

    def fetch(self, first, last):
        count = last - first + 1
        if first < 0 or count <= 0 or self.bytes + count > self.limit:
            raise ValueError("bounded subset download limit exceeded")
        self.bytes += count
        return self.data[first:last + 1]


class RangeSubsetTests(unittest.TestCase):
    def test_selected_entries_support_stored_and_deflated_members(self):
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w") as archive:
            archive.writestr("padding.bin", bytes(range(256)) * 320, compress_type=zipfile.ZIP_STORED)
            archive.writestr("selected.png", b"image payload" * 20, compress_type=zipfile.ZIP_DEFLATED)
        archive = LocalRangeZip(memory.getvalue())
        self.assertEqual(archive.read("selected.png"), b"image payload" * 20)
        self.assertEqual(archive.read("padding.bin"), bytes(range(256)) * 320)
        archive.entries["selected.png"]["crc"] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            archive.read("selected.png")

    def test_missing_archive_end_is_rejected_without_network(self):
        with self.assertRaisesRegex(ValueError, "end record"):
            LocalRangeZip(b"0" * 70000)


if __name__ == "__main__":
    unittest.main()
