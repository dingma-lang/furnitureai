import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train as t
import compare as cp

class TrainingTests(unittest.TestCase):
 def rows(self,n=6):return [{'image_id':f'{c}-{i}','abo_product_id':f'{c}-{i}','leakage_group_id':f'{c}-{i}','sha256':f'h-{c}-{i}','weak_label':c,'furniture_class':c,'proposed_split':'train'} for c in t.CLASSES for i in range(n)]
 def test_grouping_deterministic_all_classes(self):
  rows=self.rows();folds,n=t.group_folds(rows,3);self.assertEqual(n,42);self.assertEqual(folds,t.group_folds(rows,3)[0])
  for f in folds:self.assertEqual({rows[i]['weak_label'] for i in f},set(t.CLASSES))
 def test_capture_and_family_remain_whole(self):
  rows=self.rows();rows[1]['capture_cluster_id']='capture';rows[2]['capture_cluster_id']='capture';rows[0]['leakage_group_id']=rows[1]['leakage_group_id']
  folds,n=t.group_folds(rows,3)
  self.assertEqual(len({j for j,f in enumerate(folds) if any(i in f for i in [0,1,2])}),1)
 def test_frozen_or_official_overlap_rejected(self):
  rows=self.rows()
  with self.assertRaises(ValueError):t.isolate(rows,[rows[0]])
  with self.assertRaises(ValueError):t.isolate([{**rows[0],'proposed_split':'test'}],[])
 def test_unknown_or_contradictory_labels_rejected(self):
  row=self.rows()[0]
  for rows in [[{**row,'weak_label':'other'}],[row,{**row,'weak_label':'lamp'}]]:
   with self.assertRaises(ValueError):t.isolate(rows,[])
 def test_balance_uses_only_training_indices(self):
  rows=self.rows();indices=list(range(21));indices=[i for i in range(len(rows)) if i%6<2]
  weights,counts=t.balance_weights(rows,indices,'inverse');self.assertEqual(sum(counts.values()),14);self.assertTrue(np.allclose(weights,1))
 def test_selection_deterministic_and_no_panel_field(self):
  rows=[{'name':'b','mean_macro_f1':.8,'mean_balanced_accuracy':.8},{'name':'a','mean_macro_f1':.8,'mean_balanced_accuracy':.8}]
  self.assertEqual(t.ranking(rows),t.ranking(list(reversed(rows))));self.assertEqual(t.ranking(rows)[0]['name'],'a');self.assertFalse(t.POLICY['frozen_panel_for_selection'])
 def test_gpu_required(self):
  with patch.object(t.w.torch.cuda,'is_available',return_value=False):
   with self.assertRaises(RuntimeError):t.gpu()
 def test_resume_equivalence_on_synthetic_cpu_fixture(self):
  rows=self.rows();fold=t.group_folds(rows,3)[0][0];array=np.zeros((len(rows),768),dtype=np.float32)
  for i,r in enumerate(rows):array[i,t.CLASSES.index(r['weak_label'])]=1
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);a=root/'interrupted';b=root/'whole'
   for out in (a,b):t.save(out/'features-v1/metadata.json',{'model_processor_fingerprint':'synthetic'})
   with patch.object(t,'gpu',return_value=torch.device('cpu')),patch.object(t.torch.cuda,'max_memory_allocated',return_value=0):
    original=t.STOP
    class StopAfterFirst:
     count=0
     def __call__(self):
      self.count+=1
      if self.count>2:t.STOP=True
      return time.monotonic()
    # Set stop immediately after first checkpoint write; its state must be resumable.
    save=t.w.save_tensor
    def save_then_stop(path,value):
     save(path,value)
     if Path(path).name=='progress.pt':t.STOP=True
    try:
     with patch.object(t.w,'save_tensor',side_effect=save_then_stop):
      with self.assertRaises(InterruptedError):t.fit(a,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],3,time.monotonic()+60)
     t.STOP=False;t.fit(a,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],3,time.monotonic()+60)
     t.fit(b,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],3,time.monotonic()+60)
    finally:t.STOP=original
   x=torch.load(a/'trial/best.pt',weights_only=True);y=torch.load(b/'trial/best.pt',weights_only=True)
   self.assertTrue(all(torch.equal(x['state_dict'][k],y['state_dict'][k]) for k in x['state_dict']))
   with patch.object(t,'gpu',return_value=torch.device('cpu')),patch.object(t.w,'network',side_effect=AssertionError('Completed trial must skip model construction')):
    t.fit(a,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],3,time.monotonic()+60)
    (a/'trial/best.pt').write_bytes(b'corrupt synthetic fixture')
    with self.assertRaises(ValueError):t.fit(a,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],3,time.monotonic()+60)
 def test_oom_rolls_back_partial_epoch_and_reduces_batch(self):
  rows=self.rows(12);fold=t.group_folds(rows,3)[0][0];array=np.zeros((len(rows),768),dtype=np.float32)
  for i,r in enumerate(rows):array[i,t.CLASSES.index(r['weak_label'])]=1
  original_factory=t.w.network
  def factory(cfg,dim):
   model=original_factory(cfg,dim);forward=model.forward;calls=[0]
   def injected(x):
    if model.training:
     calls[0]+=1
     if calls[0]==2:raise torch.cuda.OutOfMemoryError('synthetic OOM after first optimizer step')
    return forward(x)
   model.forward=injected;return model
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);a=root/'oom';b=root/'reference'
   for out in [a,b]:t.save(out/'features-v1/metadata.json',{'model_processor_fingerprint':'synthetic'})
   with patch.object(t,'gpu',return_value=torch.device('cpu')),patch.object(t.torch.cuda,'max_memory_allocated',return_value=0):
    with patch.object(t.w,'network',side_effect=factory):result=t.fit(a,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],1,time.monotonic()+60)
    self.assertEqual(result['oom_events'][0]['reduced_batch_size'],16)
    with patch.dict(t.POLICY,{'batch_size':16}):t.fit(b,'trial',array,rows,t.fold_indices(len(rows),fold),fold,t.CONFIGS[3],1,time.monotonic()+60)
   x=torch.load(a/'trial/best.pt',weights_only=True);y=torch.load(b/'trial/best.pt',weights_only=True)
   self.assertTrue(all(torch.equal(x['state_dict'][k],y['state_dict'][k]) for k in x['state_dict']))
 def test_completed_trial_skipped_and_changed_hash_rejected(self):
  # This guard checks actual real allowlist decisions without editing any review data.
  rows=t.read(t.IMAGES/'audit-v3/provisional-candidates.json')['images'];audit=t.read(t.IMAGES/'audit-v3/screening.json')['images'];risks=t.read(t.IMAGES/'audit-v3/product-identity-quarantine.json')['products']
  t.guard_allowlist(rows,audit,risks)
  changed=copy.deepcopy(rows);changed[0]['status']='unresolved_low_margin'
  with self.assertRaises(ValueError):t.guard_allowlist(changed,audit,risks)
 def test_comparison_requires_selection_before_panel_access(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(t,'validate'),patch.object(cp.np,'load',side_effect=AssertionError('No panel access')):
   with self.assertRaises(ValueError):cp.compare(Path(temp))
 def test_product_metrics_same_examples(self):
  rows=self.rows(1);logits=np.eye(7)
  records=cp.product_records(rows,logits);self.assertEqual(len(records),7)
  detail=cp.detailed([r['truth'] for r in records],[r['prediction'] for r in records]);self.assertEqual(detail['macro_f1'],1)
 def test_paired_bootstrap_deterministic(self):
  rec=[{'product_id':str(i),'group':str(i),'truth':i%7,'prediction':i%7} for i in range(21)]
  one=cp.paired_bootstrap({'baseline':rec,'selected_expanded':rec},repeats=200);two=cp.paired_bootstrap({'baseline':rec,'selected_expanded':rec},repeats=200)
  self.assertEqual(one,two);self.assertEqual(one['paired_delta_vs_baseline_intervals']['selected_expanded'],[0.,0.])

if __name__=='__main__':unittest.main(verbosity=2)
