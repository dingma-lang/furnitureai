import copy
import hashlib
import json
import pathlib
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
import pipeline as p

class PipelineTests(unittest.TestCase):
 def rows(self,n=6):
  return [{'image_id':f'{c}-{i}','abo_product_id':f'{c}-{i}','leakage_group_id':f'{c}-{i}','weak_label':c,'furniture_class':c,'proposed_split':'train','sha256':f'sha-{c}-{i}','decoded_pixel_sha256':f'pix-{c}-{i}'} for c in p.CLASSES for i in range(n)]
 def test_group_folds_deterministic_and_all_classes(self):
  rows=self.rows();folds=p.make_folds(rows);self.assertEqual(folds,p.make_folds(rows))
  for f in folds:self.assertEqual({rows[i]['weak_label'] for i in f},set(p.CLASSES))
  self.assertEqual(sorted(sum(folds,[])),list(range(len(rows))))
 def test_product_capture_leakage_blocked(self):
  rows=self.rows();rows[0]['capture_cluster_id']='same';rows[1]['capture_cluster_id']='same'
  folds=p.make_folds(self.rows());other=next(i for f in folds if 0 not in f for i in f);rows[other]['capture_cluster_id']='same'
  with self.assertRaises(ValueError):p.make_folds(rows)
 def test_conflicting_family_categories_blocked(self):
  rows=self.rows();rows[6]['leakage_group_id']=rows[0]['leakage_group_id']
  with self.assertRaises(ValueError):p.make_folds(rows)
 def test_metadata_family_reservation_transitive(self):
  rows=[{'abo_product_id':x} for x in ['a','b','c']];identity={'same_brand_model_candidates_not_verified_duplicates':[{'product_ids':['a','b']},{'product_ids':['b','c']}]}
  g=p.group_inventory(rows,identity)
  with self.assertRaises(ValueError):p.check_scope([{'abo_product_id':'a','proposed_split':'train'}],[{'abo_product_id':'c','proposed_split':'test'}],g)
 def test_generic_shared_stock_does_not_merge_products(self):
  g=p.group_inventory([{'abo_product_id':'a','image_id':'generic'},{'abo_product_id':'b','image_id':'generic'}],{'same_brand_model_candidates_not_verified_duplicates':[]})
  self.assertNotEqual(g.find('a'),g.find('b'))
 def test_holdout_image_hash_blocked(self):
  g=p.Groups()
  with self.assertRaises(ValueError):p.check_scope([{'abo_product_id':'a','proposed_split':'train','sha256':'x'}],[{'abo_product_id':'b','proposed_split':'validation','sha256':'x'}],g)
 def test_extra_requires_model_agreement_and_margin(self):
  row={'weak_label':'bed','original_cached_training_view':False};scores=np.zeros(len(p.PROMPTS));scores[3]=.4
  admit,conf=p.model_screen(row,scores);self.assertTrue(admit);self.assertIsNone(conf['calibrated_probability'])
  scores[0]=.5;self.assertFalse(p.model_screen(row,scores)[0])
 def test_strong_contradiction_rejects_original_cached(self):
  row={'weak_label':'bed','original_cached_training_view':True};scores=np.zeros(len(p.PROMPTS));scores[-1]=.8
  self.assertFalse(p.model_screen(row,scores)[0])
 def test_product_balanced_centroids(self):
  rows=self.rows(3);x=np.zeros((len(rows),7),dtype=np.float32)
  for i,r in enumerate(rows):x[i,p.CLASSES.index(r['weak_label'])]=1
  centers=p.centroid(x,rows,list(range(len(rows))));self.assertTrue(np.allclose(centers,np.eye(7)))
  self.assertEqual(p.metric(rows,list(range(len(rows))),x)['product']['macro_f1'],1)
 def test_dynamic_more_than_275_images(self):
  rows=self.rows(50);self.assertEqual(sum(map(len,p.make_folds(rows))),350)
 def test_atomic_failed_commit_keeps_previous(self):
  with tempfile.TemporaryDirectory() as t:
   path=pathlib.Path(t)/'record';p.atomic(path,b'old')
   with patch.object(p.os,'replace',side_effect=OSError('interrupted')):
    with self.assertRaises(OSError):p.atomic(path,b'new')
   self.assertEqual(path.read_bytes(),b'old')
 def test_stop_file_and_deadline(self):
  with tempfile.TemporaryDirectory() as t:
   out=pathlib.Path(t);self.assertFalse(p.stopped(out,time.monotonic()+10));(out/'STOP').touch();self.assertTrue(p.stopped(out,time.monotonic()+10))
 def test_network_shapes_and_gradients(self):
  for cfg in p.CONFIGS:
   if cfg['kind']=='centroid':continue
   m=p.network(cfg,768);x=torch.randn(3,768);z=m(x);self.assertEqual(z.shape,(3,7));z.sum().backward();self.assertTrue(all(v.grad is not None for v in m.parameters()))
 def test_checkpoint_resume_matches_uninterrupted_synthetic_fit(self):
  rows=self.rows();array=np.zeros((len(rows),7),dtype=np.float32)
  for i,r in enumerate(rows):array[i,p.CLASSES.index(r['weak_label'])]=1
  folds=p.make_folds(rows);cfg={**p.CONFIGS[0],'name':'synthetic-resume'}
  with tempfile.TemporaryDirectory() as t:
   root=pathlib.Path(t);a=root/'interrupted';b=root/'uninterrupted'
   for out in [a,b]:p.save(out/'features-v1/metadata.json',{'model_processor_fingerprint':'synthetic'})
   with patch.object(p,'gpu',return_value=torch.device('cpu')),patch.object(p.torch.cuda,'max_memory_allocated',return_value=0):
    with patch.object(p,'stopped',side_effect=[False,True]):
     with self.assertRaises(InterruptedError):p.fit(a,array,rows,folds,cfg,0,3,time.monotonic()+60)
    p.fit(a,array,rows,folds,cfg,0,3,time.monotonic()+60)
    p.fit(b,array,rows,folds,cfg,0,3,time.monotonic()+60)
   ra=torch.load(a/'trials/synthetic-resume/fold-0/best.pt',weights_only=True)
   rb=torch.load(b/'trials/synthetic-resume/fold-0/best.pt',weights_only=True)
   self.assertTrue(all(torch.equal(ra['state_dict'][k],rb['state_dict'][k]) for k in ra['state_dict']))
 def test_no_cuda_no_fallback(self):
  with patch.object(p.torch.cuda,'is_available',return_value=False):
   with self.assertRaises(RuntimeError):p.gpu()

if __name__=='__main__':unittest.main(verbosity=2)
