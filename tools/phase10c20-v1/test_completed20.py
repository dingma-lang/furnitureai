"""Read-only completed recovery integrity tests; no model inference."""
import unittest
import torch
from common20 import *


class Completed20Tests(unittest.TestCase):
    def test_exact_cumulative_and_additional_budgets(self):
        d=read(RUN20/'complete.json');self.assertEqual((d['updates'],d['optimizer_hook_calls'],d['additional_updates']),(75,75,47));self.assertEqual((d['forwards'],d['cumulative_forwards'],d['training_VJPs'],d['probe_VJPs']),(270,452,552,144));self.assertLess(d['seconds'],515)
        for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_device_bytes'):self.assertLessEqual(d[k],MAX_VRAM17)
        self.assertLess(allocation20(),MAX_OUTPUT20)
    def test_restoration_verified_before_any_forward(self):
        d=read(RUN20/'restoration.json');self.assertEqual(d['updates'],28);self.assertEqual(d['checkpoint_sha256'],CHECKPOINT28_SHA256);self.assertTrue(d['no_model_forward_before_restore']);self.assertTrue(d['rng']['CUDA_roundtrip_verified']);self.assertTrue(d['rng']['Python_restored']);self.assertFalse(d['rng']['NumPy_restored']);self.assertEqual((d['epoch_zero_based'],d['next_group_zero_based']),(1,13));self.assertFalse((RUN20/'probes/epoch-0').exists())
    def test_all47_groups_correct_order_caps_and_positive_box_preservation(self):
        cfg,*_=inputs20()
        for i in range(29,76):
            d=read(RUN20/'updates'/('update-'+str(i).zfill(3)+'.json'));m=d['metrics'];self.assertEqual(d['source_order'],cfg['orders'][(i-1)//15][(i-1)%15]);self.assertTrue(m['direct_guard']['cap_verified']);self.assertEqual(m['cap_violations'],0);self.assertTrue(m['positive_buffers_unchanged']);self.assertTrue(m['box_buffers_unchanged']);self.assertFalse(any(r['protected_violations'] for r in d['images']));self.assertEqual(d['scaler_scale'],128)
            self.assertLessEqual(m['source_post']['norm'],m['cap']['budget']+m['direct_guard']['numerical_tolerance']['total'])
    def test_saved_predictions_exact19_and_no_new_holdout_membership(self):
        from analyze18 import raw_outputs
        cfg,_,train,dev,_=inputs20()
        for epoch in (3,5):self.assertEqual(len(raw_outputs(RUN20/'evaluation'/('epoch-'+str(epoch)),dev)),19)
        r=read(REPORT20/'results.json');self.assertFalse(r['protocol']['reserved_or_frozen_scored']);self.assertFalse(r['checkpoint28_scored']);self.assertFalse(r['protocol']['independent_accuracy']);self.assertEqual(r['new_GPU_forwards_in_analysis'],0)
    def test_47_new_checkpoints_and_exact_final_optimizer_state(self):
        self.assertEqual(len(list((RUN20/'checkpoints').glob('*.pt'))),47);d=read(RUN20/'complete.json');p=RUN20/d['final_checkpoint']['file'];self.assertEqual(sha(p),d['final_checkpoint']['sha256']);s=torch.load(p,map_location='cpu',weights_only=True);self.assertEqual(s['updates'],75);self.assertEqual(set(s['weights']),HEAD_KEYS);self.assertTrue(all(int(v['step'])==75 for v in s['optimizer']['state'].values()));self.assertIn('numpy',s['rng'])
    def test_training_pool_counts_and_no_promoted_labels(self):
        r=read(REPORT20/'training-curves.json');self.assertEqual((r['updates'],r['recovery_updates'],r['photo_presentations'],r['unique_training_photos'],r['known_source_groups'],r['positive_targets']),(75,47,295,59,56,910));self.assertEqual(r['cap_violations'],0);self.assertTrue(r['positive_box_buffers_preserved'])
    def test_original_model_and_source_checkpoint_unchanged(self):
        cfg,*_=inputs20();self.assertEqual(sha(CHECKPOINT28),CHECKPOINT28_SHA256);self.assertEqual(sha(BASE/'model.safetensors'),cfg['initialization']['SHA256']);self.assertEqual(read(RUN20/'restoration.json')['frozen_sha256'],read(RUN20/'complete.json')['frozen_sha256'])
    def test_CPU_analysis_preserves_metric_implementation(self):
        r=read(RUN20/'CPU-analysis-v1/complete.json');self.assertTrue(r['identical_historical_metrics_verified']);self.assertEqual(r['new_GPU_forwards'],0);self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
