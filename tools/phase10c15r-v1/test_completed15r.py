"""CPU assertions on real recovery results;no additional GPU work."""
import unittest
from common15r import *


class CompletedTests(unittest.TestCase):
    def test_complete_exact_20_forward_recovery_and_25_cumulative(self):
        d=read(RUNR/'complete.json');self.assertEqual((d['new_forwards'],d['cumulative_forwards'],d['new_gradient_calculations'],d['cumulative_gradient_calculations']),(20,25,140,168));self.assertFalse(d['optimizer_constructed']);self.assertEqual(d['optimizer_steps'],0)
    def test_all20_terminal_and_per_photo_fingerprints_unchanged(self):
        d=read(RUNR/'complete.json');self.assertEqual(d['model_state_before'],d['model_state_after'])
        for r in d['records']:
            v=read(RUNR/r['file']);self.assertEqual(v['model_state_sha256'],d['model_state_before']);self.assertTrue(v['parameter_grads_all_none']);self.assertTrue(v['parameters_and_buffers_unchanged'])
    def test_combination_has_four_prior_twenty_new_and_no_duplicates(self):
        d=read(RUNR/'combined-index.json')['records'];self.assertEqual(len(d),24);self.assertEqual(len({r['image_id'] for r in d}),24);self.assertEqual([r['origin'] for r in d],['prior_reused']*4+['recovered']*20)
    def test_resource_limits_and_original_missing_measurements_separate(self):
        d=read(REPORTR/'results.json');r=d['resource'];self.assertLess(r['elapsed_seconds'],180);self.assertLessEqual(r['peak_allocated_bytes'],MAX_VRAM);self.assertLessEqual(r['peak_observed_device_bytes'],MAX_VRAM);self.assertIsNone(d['original_first_attempt_missing_measurements']['peak_VRAM']);self.assertLess(allocated_r(),MAX_BYTES)
    def test_no_new_labels_or_supervision_coverage_expansion(self):
        d=read(REPORTR/'results.json');self.assertEqual(d['counts']['positive'],79);self.assertEqual(d['counts']['source_negative'],281);self.assertEqual(d['counts']['protected_violations'],0);self.assertEqual(d['covered_regions'],24);self.assertEqual(d['uncovered_regions'],17);self.assertEqual(d['new_negative_labels'],0);self.assertFalse(d['precision_recall_or_AP_measured'])


if __name__=='__main__':unittest.main()
