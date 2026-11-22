"""CPU checks on actual completed artifacts;no model calls."""
import unittest
from common17 import *


class CompletedTests(unittest.TestCase):
    def test_exact59_forwards295_VJPs_zero_updates_and15groups(self):
        d=read(RUN17/'complete.json');self.assertEqual((d['forwards'],d['gradient_calculations'],d['photos'],d['known_source_groups'],d['accumulation_groups']),(59,295,59,56,15));self.assertFalse(d['optimizer_constructed']);self.assertEqual(d['model_updates'],0);self.assertEqual(d['optimizer_steps'],0)
    def test_all_group_fingerprints_and_cap_compliance(self):
        d=read(RUN17/'complete.json');self.assertEqual(d['model_state_before'],d['model_state_after'])
        for ref in d['groups']:
            r=read(RUN17/ref['file']);m=r['metrics'];self.assertEqual(r['model_state_sha256'],d['model_state_before']);self.assertEqual(m['cap_violations'],0);self.assertTrue(m['positive_buffers_unchanged']);self.assertTrue(m['box_gradients_unchanged']);self.assertLessEqual(m['negatives']['C_post_cap']['norm'],.25*m['positive']['norm']*(1+1e-6)+1e-12);self.assertEqual(m['cap']['amp_scale'],128);self.assertLessEqual(m['cap']['scale'],1)
    def test_source_and_positive_counts_and_unscored_status(self):
        r=read(REPORT17/'results.json');self.assertEqual(r['positive_boxes'],182);self.assertEqual(r['source_negative_queries'],292);self.assertEqual(r['covered_regions'],24);self.assertEqual(r['uncovered_regions'],17);self.assertFalse(r['evaluation_or_holdout_scored']);self.assertTrue(r['positive_and_box_buffers_bit_identical']);self.assertFalse(r['source_signal_silently_dropped'])
    def test_all24_original_cached_probes_bit_identical(self):
        d=read(RUN17/'complete.json');values=[read(RUN17/r['file'])['original_C8_cached_outputs_bit_identical'] for r in d['records']];self.assertEqual(values.count(True),24);self.assertEqual(values.count(None),35);self.assertNotIn(False,values)
    def test_caps_not_naturally_triggered_and_no_parameter_tensor_dominates(self):
        r=read(REPORT17/'results.json');self.assertEqual(r['cap_activated_groups'],0);self.assertTrue(r['cap_activation_not_exercised_on_real_data']);self.assertTrue(all(v['opposing_negative_larger_than_positive_groups']==0 for v in r['per_tensor'].values()))
    def test_measured_resources_and_default_disabled_future(self):
        r=read(REPORT17/'results.json');s=r['resource'];self.assertLess(s['elapsed_seconds'],180);self.assertLessEqual(s['peak_allocated_bytes'],MAX_VRAM17);self.assertLessEqual(s['peak_reserved_bytes'],MAX_VRAM17);self.assertLessEqual(s['peak_observed_device_bytes'],MAX_VRAM17);self.assertLess(allocation17(),MAX_OUTPUT17);cfg=read(REPORT17/'future-training-disabled.json');self.assertFalse(cfg['training_authorized']);self.assertFalse(cfg['GPU_authorized']);self.assertFalse(cfg['method']['enabled']);self.assertEqual(len(cfg['trainable_keys']),12);self.assertFalse(cfg['calibration_keys_frozen']);self.assertFalse(cfg['L2SP_enabled']);self.assertEqual(cfg['unchanged_C8_settings']['learning_rate'],1e-5);self.assertFalse((ROOT/cfg['destination']).exists())
    def test_CPU_GPU_reduction_deltas_only_with_bit_identical_weights(self):
        r=read(REPORT17/'results.json')
        for d in r['CPU_GPU_metadata_reduction_differences']:self.assertTrue(d['weights_bit_identical']);self.assertLess(abs(d['CPU_sum']-d['GPU_sum']),5e-7)


if __name__=='__main__':unittest.main()
