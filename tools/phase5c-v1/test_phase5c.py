import copy
import json
from pathlib import Path
import tempfile
import unittest
from common import *
import control as c
from worker import validate

class Tests(unittest.TestCase):
 def test_data_authorization_and_mapping(self):
  self.assertEqual(len(load_rows('train')),27);self.assertEqual(len(load_rows('development')),18)
  with self.assertRaises(ValueError):load_rows('frozen_evaluation')
  for name in c.FIT_IDS:self.assertEqual(validate(name)['mode'],'heads')
  with self.assertRaises(ValueError):validate('unapproved-third')
 def test_update_counter_and_uncertain_boundary(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c.ledger_init(root);c.reserve(c.FIT_IDS[0],0,root)
   with self.assertRaises(RuntimeError):c.check_resume(c.FIT_IDS[0],0,root)
   c.applied(c.FIT_IDS[0],1,root);c.check_resume(c.FIT_IDS[0],1,root)
   with self.assertRaises(RuntimeError):c.reserve(c.FIT_IDS[0],0,root)
 def test_caps_exactly_70_and_140(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c.ledger_init(root)
   for name in c.FIT_IDS:
    for n in range(70):c.reserve(name,n,root);c.applied(name,n+1,root)
    with self.assertRaises(RuntimeError):c.reserve(name,70,root)
   self.assertEqual(sum(r['applied_calls'] for r in read(root/'update-budget.json')['fits'].values()),140)
 def test_no_lora_or_unfreezing(self):
  from adaptation import configure
  with self.assertRaises(ValueError):configure(None,'last1')
  with self.assertRaises(ValueError):configure(None,'lora4')
 def test_partial_supervision_gradient(self):
  import torch
  from adaptation import detection_loss
  row=copy.deepcopy(load_rows('train')[0]);logits=torch.zeros(10,7,requires_grad=True);boxes=torch.full((10,4),.3,requires_grad=True)
  loss,parts=detection_loss(logits,boxes,row);loss.backward()
  for i,cat in enumerate(CLASSES):
   if not row['coverage'][cat]:self.assertTrue(torch.all(logits.grad[:,i]==0))
  self.assertFalse(parts['objectness_supervised']);self.assertTrue(torch.isfinite(loss))
 def test_primary_metric_code_is_identical(self):
  self.assertEqual(sha(HERE/'metrics.py'),sha(ROOT/'tools/phase5b-v1/metrics.py'))
  self.assertEqual(sha(HERE/'baseline.py'),sha(ROOT/'tools/phase5b-v1/baseline.py'))
 def test_partitions_and_licensing_exclusions(self):
  lock=read(OUT/'dataset-lock.json');rights=read(PILOT/'licensing-audit.json');bad={r['image_id'] for r in rights['records'] if r['status']!='verified_individual_current_source'}
  held={r['image_id'] for r in lock['frozen_references']}
  self.assertEqual(len(bad),32)
  for row in lock['rows']:self.assertNotIn(row['image_id'],bad|held)

 def test_completed_update_budget_and_gradient_scope(self):
  ledger=read(c.RUN_ROOT/'update-budget.json')
  self.assertEqual(sum(r['applied_calls'] for r in ledger['fits'].values()),140)
  for name in c.FIT_IDS:
   done=read(c.RUN_ROOT/name/'complete.json');scope=read(c.RUN_ROOT/name/'scope.json')
   self.assertEqual((done['epochs'],done['optimizer_updates'],done['optimizer_hook_calls']),(10,70,70))
   self.assertTrue(done['frozen_parameters_unchanged'])
   self.assertEqual(done['frozen_parameter_sha256'],scope['frozen_parameter_sha256'])
   self.assertTrue(all(n.startswith(('class_head.','box_head.')) for n in scope['trainable_names']))
   c.check_resume(name,70)

 def test_all_optimization_examples_are_approved_training_members(self):
  from collections import Counter
  allowed={r['image_id'] for r in load_rows('train')}
  for name in c.FIT_IDS:
   updates=[read(p) for p in sorted((c.RUN_ROOT/name).glob('update-*.json'))]
   self.assertEqual([r['step'] for r in updates],list(range(1,71)))
   counts=Counter(i for r in updates for i in r['images'])
   self.assertEqual(dict(counts),{i:10 for i in allowed})
   self.assertTrue(all(not x['objectness_supervised'] for r in updates for x in r['components']))

 def test_selected_checkpoints_independently_reproduce(self):
  comparison=read(c.RUN_ROOT/'comparison-v1/comparison.json')
  self.assertFalse(comparison['primary_protocol_changed'])
  self.assertEqual(comparison['frozen_photographs_used'],0)
  for name in c.FIT_IDS:
   replay=comparison['independent_replay'][name]
   self.assertTrue(replay['predictions_bitwise_json_equal']);self.assertTrue(replay['metrics_exactly_reproduced'])
   done=read(c.RUN_ROOT/name/'complete.json')
   self.assertEqual(sha(done['best']['checkpoint']),done['best']['sha256'])
   epochs=[read(p) for p in sorted((c.RUN_ROOT/name).glob('validation-epoch-*.json'))]
   best=max(epochs,key=lambda r:(r['metrics']['macro_ap50_supported'],-r['epoch']))
   self.assertEqual(best['epoch'],done['best']['epoch'])
   self.assertTrue(all(set(r['predictions'])=={x['image_id'] for x in load_rows('development')} for r in epochs))

 def test_seed_and_pre_update_numerics_reproduce_between_fits(self):
  a,b=[read(c.RUN_ROOT/name/'update-00001.json') for name in c.FIT_IDS]
  for key in ('images','loss','components','grad_norm','scaler_scale'):self.assertEqual(a[key],b[key])
  for name in c.FIT_IDS:
   self.assertEqual(read(c.RUN_ROOT/name/'binding.json')['config']['seed'],20261105)
   self.assertEqual(len(list((c.RUN_ROOT/name).glob('amp-overflow-*.json'))),1)
   updates=[read(p) for p in (c.RUN_ROOT/name).glob('update-*.json')]
   # The paused smoke receipt predates the bounded AMP-replay instrumentation.
   self.assertEqual(sum(r.get('numeric_replays',0) for r in updates),1)

 def test_sensitivity_does_not_replace_primary_or_hide_regressions(self):
  from compare import subset_summary
  comparison=read(c.RUN_ROOT/'comparison-v1/comparison.json')
  for name,metrics in comparison['primary_metrics'].items():
   self.assertEqual(subset_summary(metrics,set()),comparison['sensitivity'][name]['all_seven_primary'])
  original=comparison['primary_metrics']['original']['categories']
  selected=comparison['primary_metrics'][comparison['selected_experiment']]['categories']
  self.assertGreater(selected['table']['fp'],original['table']['fp'])
  self.assertLess(selected['bed']['ap75'],original['bed']['ap75'])

if __name__=='__main__':unittest.main()
