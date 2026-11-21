"""CPU regression checks for diagnostic selection and query alignment."""
import unittest
import numpy as np
from analyze11 import envelope,delta_stats,load_raw


class DiagnosticTests(unittest.TestCase):
    def test_unattainable_recall_is_not_extrapolated(self):
        self.assertIsNone(envelope([dict(recall=.8,precision=.9)],.95))

    def test_matched_recall_never_uses_lower_recall_operating_point(self):
        points=[dict(recall=.6,precision=1),dict(recall=.8,precision=.5),dict(recall=.9,precision=.4)]
        self.assertEqual(envelope(points,.7),points[1])

    def test_common_shift_does_not_claim_category_discrimination(self):
        x=np.zeros((5,7));r=delta_stats(x,x+3)
        self.assertEqual(r['fraction_queries_all_seven_logits_increase'],1)
        self.assertEqual(r['common_mean'],3)
        self.assertEqual(r['semantic_residual_RMS'],0)

    def test_raw_comparison_rejects_changed_photo_identity_before_arrays(self):
        import tempfile,json
        from pathlib import Path
        from common11 import digest
        rows=[dict(image_id='TRAIN-DIAGNOSTIC',sha256='a')]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'raw-complete.json').write_text(json.dumps(dict(rows_sha256=digest(rows),FP32=True,records=[dict(image_id='RESERVED',source_sha256='a',file='missing.npz',sha256='a')])))
            with self.assertRaisesRegex(ValueError,'identity/hash'):load_raw(p,rows)


if __name__=='__main__':unittest.main()
