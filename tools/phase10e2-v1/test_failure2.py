"""Post-stop CPU integrity checks,no model inference or backward replay."""
import unittest
import numpy as np
from common2 import *

class FailureIntegrityTests(unittest.TestCase):
    def setUp(self):self.failure=read(RUN/'failure.json');self.summary=read(REPORT/'failure-analysis.json')
    def test_exact_consumed_counters_and_no_retry(self):
        self.assertEqual(self.failure['state']['forwards'],6);self.assertEqual(self.failure['state']['actual_forwards'],6);self.assertEqual(self.failure['state']['gradient_calculations'],7)
        self.assertEqual(len(list((RUN/'forward-intents').glob('*.json'))),6);self.assertEqual(len(list((RUN/'gradient-intents').glob('*.json'))),7)
        self.assertFalse(read(RUN/'launch-result.json')['automatic_retry'])
    def test_original_and_adapted_weights_unchanged_within_each_state(self):
        self.assertEqual(read(RUN/'state-A-before.json'),read(RUN/'state-A-after.json'))
        self.assertEqual(read(RUN/'state-B-before.json'),self.failure['model_fingerprint'])
    def test_all_six_raw_replays_are_exact(self):
        for p in sorted((RUN/'original').glob('*.json'))+sorted((RUN/'adapted').glob('*.json')):
            self.assertTrue(all(v['bit_identical'] for v in read(p)['replay'].values()))
        self.assertTrue(all(v['bit_identical'] for v in self.summary['last_raw_replay'].values()))
    def test_terminal_derivative_finite_is_not_a_parameter_norm(self):
        self.assertTrue(self.summary['failure_scalar_and_terminal_derivative_finite'])
        self.assertEqual(self.summary['scaled_terminal_derivative_closed_form'],[1582.])
        self.assertIsNone(self.summary['projection_norm_for_failed_photo']);self.assertIsNone(self.summary['gradient_cosine_for_failed_photo'])
    def test_no_fabricated_adapted_nonzero_gradient_or_coefficient(self):
        self.assertEqual(self.summary['persisted_nonzero_KD_parameter_measurements'],0)
        self.assertEqual(self.summary['persisted_positive_parameter_measurements'],5)
        self.assertIsNone(read(REPORT/'decision.json')['coefficient'])
        self.assertFalse(read(REPORT/'decision.json')['training_authorized'])
    def test_original_positive_signal_and_zero_distill_are_recorded(self):
        values=[read(p)['measurement'] for p in sorted((RUN/'original').glob('*.json'))]
        self.assertEqual(len(values),4);self.assertEqual(sum(r['eligible_distillation_terms'] for r in values),10)
        self.assertTrue(all(r['positive_projection']['L2']>0 for r in values))
        self.assertTrue(all(r['unit_distill_projection']['L2']==0 for r in values))
    def test_failure_within_resource_bounds(self):
        self.assertLess(self.failure['seconds'],180);self.assertLess(read(RUN/'launch-result.json')['wall_seconds'],180)
        self.assertLessEqual(self.failure['peak_allocated_bytes'],LIMITS['VRAM_bytes']);self.assertLess(allocated(),LIMITS['output_bytes'])
    def test_finite_loss_can_have_unrepresentable_half_gradient_generic_fixture(self):
        # Generic IEEE example only,not a claim about the unidentified actual intermediate.
        residual=np.float32(256);loss=residual**2;scaled_gradient=2*residual*128
        self.assertTrue(np.isfinite(loss));self.assertGreater(scaled_gradient,np.finfo(np.float16).max)
        with np.errstate(over='ignore'):self.assertFalse(np.isfinite(np.float16(scaled_gradient)))

if __name__=='__main__':unittest.main()
