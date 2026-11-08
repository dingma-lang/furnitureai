import unittest

import numpy as np

from room_multiview.registration_graph import largest_component_cameras


def edge(a, b, shift, status="accepted"):
    pose = np.eye(4)
    pose[0, 3] = shift
    return {"views": [a, b], "status": status, "sample_gap": 1, "camera_b_from_a": pose.tolist()}


class RegistrationGraphTests(unittest.TestCase):
    def test_largest_supported_component_ignores_rejected_bridge(self):
        pairs = [edge("a", "b", 1), edge("b", "c", 1, "rejected"), edge("c", "d", 2), edge("d", "e", 3)]
        cameras, parents, used, anchor, sizes = largest_component_cameras(list("abcde"), pairs, ["d"])
        self.assertEqual(anchor, "d")
        self.assertEqual(sizes, [3, 2])
        self.assertEqual(set(cameras), set("cde"))
        self.assertAlmostEqual(cameras["c"][0, 3], -2)
        self.assertAlmostEqual(cameras["e"][0, 3], 3)
        self.assertEqual(len(used), 2)

    def test_tie_uses_input_order_and_disconnected_views_stay_unregistered(self):
        cameras, _, _, anchor, sizes = largest_component_cameras(list("abc"), [])
        self.assertEqual(anchor, "a")
        self.assertEqual(list(cameras), ["a"])
        self.assertEqual(sizes, [1, 1, 1])

    def test_optional_icp_transform_and_reverse_composition(self):
        pair = edge("a", "b", 1)
        pair["icp"] = {"adopted": True, "camera_b_from_a": edge("a", "b", 0.8)["camera_b_from_a"]}
        cameras, _, _, _, _ = largest_component_cameras(list("ab"), [pair], ["b"], True)
        self.assertAlmostEqual(cameras["a"][0, 3], -0.8)


if __name__ == "__main__":
    unittest.main()
