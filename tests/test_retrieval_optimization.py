import unittest

import numpy as np
from PIL import Image, ImageDraw

from furniture_bench.retrieval_experiments import (
    align_image_rows,
    category_scores,
    paired_interval,
    prepare_image,
    query_ranks,
    score_products,
)


class RetrievalOptimizationTests(unittest.TestCase):
    def test_unit_normalization_and_multiple_views_preserve_product_identity(self):
        q = np.array([[2.0, 0.0]])
        g = np.array([[4.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 3.0]])
        for method in ["max", "normalized_centroid", "top2_mean", "main_only"]:
            scores = score_products(q, g, [[0, 1], [2, 3]], method)
            np.testing.assert_allclose(scores, [[1, 0]])
            self.assertEqual(
                query_ranks(scores, ["sofa-a", "sofa-b"], [{"product_id": "sofa-b"}]), [2]
            )
        with self.assertRaises(ValueError):
            query_ranks(np.array([[1, 0]]), ["same", "same"], [{"product_id": "same"}])

    def test_unknown_identity_and_candidate_pool_do_not_invent_rank(self):
        scores = score_products(
            np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]), [[0], [1]], pool=1
        )
        self.assertIsNone(query_ranks(scores, ["a", "b"], [{"product_id": "unknown"}])[0])
        self.assertIsNone(query_ranks(scores, ["a", "b"], [{"product_id": "b"}])[0])

    def test_low_confidence_or_missing_category_falls_back_to_global(self):
        scores = np.array([[0.9, 0.8]])
        for prediction in [
            {"label": "chair", "score": 0.1},
            {"label": "missing", "score": 0.9},
            {"label": None, "score": 0},
        ]:
            adjusted, count = category_scores(scores, ["sofa", "chair"], [prediction], "strict")
            np.testing.assert_array_equal(adjusted, scores)
            self.assertEqual(count, 1)

    def test_wrong_confident_category_can_harm_strict_filter_but_soft_keeps_candidates(self):
        scores = np.array([[0.9, 0.7]])
        strict, _ = category_scores(
            scores, ["sofa", "chair"], [{"label": "chair", "score": 0.9}], "strict"
        )
        soft, _ = category_scores(
            scores, ["sofa", "chair"], [{"label": "chair", "score": 0.9}], "soft"
        )
        self.assertIsNone(query_ranks(strict, ["sofa-a", "chair-a"], [{"product_id": "sofa-a"}])[0])
        self.assertEqual(query_ranks(soft, ["sofa-a", "chair-a"], [{"product_id": "sofa-a"}]), [1])

    def test_condition_vectors_are_aligned_by_source_image_id(self):
        values = np.array([[1, 2], [3, 4], [5, 6]])
        result = align_image_rows(values, ["c", "a", "b"], ["a", "b", "c"])
        np.testing.assert_array_equal(result, [[3, 4], [5, 6], [1, 2]])
        with self.assertRaises(ValueError):
            align_image_rows(values, ["a", "a", "b"], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            align_image_rows(values, ["c", "a", "b"], ["d", "b", "c"])

    def test_blank_or_nonwhite_context_is_not_misrepresented_as_mask(self):
        white = Image.new("RGB", (100, 80), "white")
        self.assertEqual(prepare_image(white, "white_margin_trim").size, white.size)
        textured = Image.new("RGB", (100, 80), "brown")
        self.assertEqual(prepare_image(textured, "white_margin_trim").size, textured.size)
        ImageDraw.Draw(white).rectangle([30, 20, 70, 60], fill="navy")
        tight = prepare_image(white, "white_margin_trim")
        margin = prepare_image(white, "margin10")
        self.assertLess(tight.width, margin.width)
        self.assertEqual(prepare_image(white, "letterbox").size, (224, 224))

    def test_paired_bootstrap_uses_deterministic_identity_resampling(self):
        a = [1, 6, 3, 9]
        b = [1, 2, 3, 10]
        self.assertEqual(paired_interval(a, b, range(4)), paired_interval(a, b, range(4)))
        self.assertEqual(paired_interval(a, a, range(4)), [0, 0])

    def test_invalid_embedding_never_enters_ranking(self):
        for q in [np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]])]:
            with self.assertRaises(ValueError):
                score_products(q, np.array([[1.0, 0.0]]), [[0]])


if __name__ == "__main__":
    unittest.main()
