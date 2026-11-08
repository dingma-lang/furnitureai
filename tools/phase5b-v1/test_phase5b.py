import copy
import tempfile
import unittest
from pathlib import Path
from common import *
import metrics as m

def gt(c='chair',b=None,ignore=False):return dict(label=c,source_mid=MIDS[c],box=b or [.1,.1,.6,.6],ignore=ignore,flags=dict(IsTruncated=0,IsOccluded=0))
def row(annotations=None,known=('chair',)):
 return dict(image_id='x',split='development',annotations=annotations if annotations is not None else [gt()],coverage={c:c in known for c in CLASSES})
class MetricTests(unittest.TestCase):
 def test_iou_and_nms(self):
  self.assertEqual(m.iou([0,0,1,1],[0,0,1,1]),1);self.assertAlmostEqual(m.iou([0,0,.5,.5],[.25,.25,.75,.75]),1/7)
  preds=[dict(label='chair',box=[.1,.1,.6,.6],score=s) for s in (.9,.8)]
  self.assertEqual(len(m.nms(preds)),1)
 def test_perfect_ap_and_duplicate(self):
  preds=[dict(label='chair',box=[.1,.1,.6,.6],score=s) for s in (.9,.8)]
  out=m.evaluate([row()],{'x':preds});c=out['categories']['chair']
  self.assertEqual(c['ap50'],1);self.assertEqual(c['ap75'],1);self.assertEqual((c['tp'],c['fp'],c['duplicate_fp']),(1,1,1));self.assertEqual(c['precision'],.5)
 def test_ap75_is_stricter_than_ap50(self):
  p=dict(label='chair',box=[.1,.1,.9,.6],score=.9)
  result=m.evaluate([row()],{'x':[p]})['categories']['chair']
  self.assertEqual(result['ap50'],1);self.assertEqual(result['ap75'],0)

 def test_ap_recall_and_ranking(self):
  events=[dict(outcome='fp'),dict(outcome='tp')]
  self.assertAlmostEqual(m.ap101(events,1),.5)
  self.assertAlmostEqual(m.ap101([dict(outcome='tp')],2),51/101)
  self.assertIsNone(m.ap101([],0))
 def test_unknown_not_negative(self):
  pred=dict(label='table',box=[.1,.1,.6,.6],score=.9)
  out=m.evaluate([row()],{'x':[pred]});self.assertEqual(out['categories']['table']['fp'],0);self.assertEqual(out['categories']['table']['ignored_unknown_predictions'],1)
  out=m.evaluate([row(known=('chair','table'))],{'x':[pred]});self.assertEqual(out['categories']['table']['fp'],1)
 def test_ignore_region_and_match_precedence(self):
  preds=[dict(label='chair',box=[.1,.1,.6,.6],score=.9)]
  out=m.evaluate([row([gt(ignore=True)])],{'x':preds});self.assertEqual(out['categories']['chair']['fp'],0)
  out=m.evaluate([row([gt(ignore=True),gt()])],{'x':preds});self.assertEqual(out['categories']['chair']['tp'],1)
 def test_miss_and_wrong_category(self):
  pred=dict(label='table',box=[.1,.1,.6,.6],score=.9)
  out=m.evaluate([row(known=('chair','table'))],{'x':[pred]});self.assertEqual(out['categories']['chair']['fn'],1);self.assertEqual(len(out['diagnostics']['category_mismatch_candidates']),1)
 def test_frozen_metrics_and_loader_blocked(self):
  r=row();r['split']='frozen_evaluation'
  with self.assertRaises(ValueError):m.evaluate([r],{'x':[]})
  with self.assertRaises(ValueError):load_rows('frozen_evaluation')
  dev=load_rows('development')[0];spoof=copy.deepcopy(dev);spoof['file']='photos/not-locked.jpg'
  with self.assertRaises(ValueError):image_path(spoof,'development')
 def test_locked_partition_groups(self):
  lock=read(OUT/'dataset-lock.json');rows=lock['rows']+[{**r,'split':'frozen_evaluation'} for r in lock['frozen_references']]
  for key in ('image_id','sha256','identity_group','source_identity'):
   seen={}
   for r in rows:
    if not r.get(key):continue
    self.assertNotIn(r[key],seen);seen[r[key]]=r['split']
  self.assertEqual(lock['eligible_counts'],dict(train=27,development=18,frozen_evaluation=18))

class AdaptationTests(unittest.TestCase):
 def test_square_padding_roundtrip(self):
  import torch
  from adaptation import to_square,from_square,xyxy_to_cxcywh,cxcywh_to_xyxy
  b=torch.tensor([[.1,.2,.9,.8]])
  sq=to_square(b,1000,500);self.assertTrue(torch.allclose(sq,torch.tensor([[.1,.1,.9,.4]])))
  self.assertTrue(torch.allclose(from_square(sq,1000,500),b))
  self.assertTrue(torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)),b))
 def test_partial_label_loss_masks_unknown_gradients(self):
  import torch
  from adaptation import detection_loss
  r=row();r.update(width=1000,height=500)
  logits=torch.zeros((2,7),requires_grad=True);boxes=torch.tensor([[.35,.175,.5,.25],[.7,.3,.1,.1]],requires_grad=True)
  before=copy.deepcopy(r);loss,parts=detection_loss(logits,boxes,r);loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertEqual(r,before)
  self.assertTrue(torch.all(logits.grad[:,[i for i,c in enumerate(CLASSES) if c!='chair']]==0))
  self.assertGreater(float(logits.grad[:,CLASSES.index('chair')].abs().sum()),0);self.assertFalse(parts['objectness_supervised'])
 def test_zero_l1_for_exact_gt_and_giou(self):
  import torch
  from adaptation import detection_loss,giou
  r=row();r.update(width=100,height=100)
  logits=torch.zeros((1,7),requires_grad=True);b=torch.tensor([[.35,.35,.5,.5]],requires_grad=True)
  loss,parts=detection_loss(logits,b,r);self.assertAlmostEqual(parts['l1'],0,places=6);self.assertAlmostEqual(parts['giou'],0,places=6)
  self.assertAlmostEqual(float(giou(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([[0.,0.,1.,1.]])).item()),1.)
 def test_lora_zero_initial_delta_and_no_base_update(self):
  import torch
  from adaptation import LoRALinear
  base=torch.nn.Linear(4,3);x=torch.randn(2,4);original=base(x).detach();layer=LoRALinear(base,2)
  self.assertTrue(torch.equal(layer(x).detach(),original));self.assertFalse(layer.base.weight.requires_grad)
  layer(x).sum().backward();self.assertIsNone(layer.base.weight.grad)
 def test_exclusive_run_lease(self):
  from checkpoints import Lease
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'worker.lock'
   with Lease(path):
    with self.assertRaises(BlockingIOError):
     with Lease(path):pass
   with Lease(path):pass

 def test_checkpoint_receipts_recovery_corruption_and_orphans(self):
  import torch
  from checkpoints import snapshot,recover
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);binding=dict(dataset='frozen-hash');state=dict(weights={'w':torch.ones(2)},optimizer={},scaler={},rng={},step=1,epoch=0,next_group=1,best=None)
   path,h=snapshot(out,state,binding);(out/'checkpoints/orphan.tmp').write_bytes(b'partial uncommitted')
   got=recover(out,binding);self.assertEqual(got[1]['step'],1)
   with self.assertRaises(ValueError):recover(out,dict(dataset='changed'))
   state['best']={'score':.5};newpath,_=snapshot(out,state,binding);self.assertEqual(recover(out,binding)[2],newpath)
   Path(newpath).write_bytes(b'corrupted')
   with self.assertRaises(ValueError):recover(out,binding)
 def test_authorization_gate_before_outputs_or_model_load(self):
  import train
  import importlib.util
  spec=importlib.util.spec_from_file_location('phase5b_queue',HERE/'run_queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'must-not-exist'
   with self.assertRaises(PermissionError):train.train(Path(tmp)/'missing.json',out)
   with self.assertRaises(PermissionError):q.run(out)
   self.assertFalse(out.exists())
 def test_config_bounds_and_probe_no_optimizer(self):
  import train
  for path in (OUT/'configs').glob('*.json'):
   c=read(path);train.validate_config(c)
   for key,val in [('epochs',11),('max_updates',71),('fit_split','frozen_evaluation')]:
    bad=copy.deepcopy(c);bad[key]=val
    with self.assertRaises(ValueError):train.validate_config(bad)
  probe=read(OUT/'memory-probes-v1/summary.json');self.assertEqual(probe['optimizer_steps'],0)
  self.assertTrue(all(r['trainable_hash_unchanged'] and r['optimizer_steps']==0 for r in probe['results']))
 def test_actual_baseline_artifact_no_frozen_predictions(self):
  completed=read(OUT/'original-development-v1/complete.json');self.assertEqual(completed['images'],18);self.assertFalse(completed['frozen_evaluated'])
  ids={r['image_id'] for r in load_rows('development')}
  self.assertEqual({p.stem for p in (OUT/'original-development-v1').glob('*.checksum.json')},{i+'.checksum' for i in ids})

if __name__=='__main__':unittest.main()
