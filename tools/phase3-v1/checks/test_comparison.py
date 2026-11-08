import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import common as c
import runner as r

class ComparisonTests(unittest.TestCase):
 def test_frozen_vector_tampering_rejected_even_with_new_checksum(self):
  original=c.ROOT/'data/abo/experiments/phase3-baseline-v1/features-v1'
  with tempfile.TemporaryDirectory() as tmp:
   out=pathlib.Path(tmp);dest=out/'features-v1';dest.mkdir()
   meta=c.read(original/'metadata.json');x=np.load(original/'features.npy',allow_pickle=False).copy();i=meta['fit_count']
   x[[i,i+1]]=x[[i+1,i]];c.w.save_array(dest/'features.npy',x);meta['feature_sha256']=c.sha(dest/'features.npy');c.save(dest/'metadata.json',meta)
   with self.assertRaisesRegex(ValueError,'Frozen development vectors changed'):r.load_cache(c.OUT,out,'baseline')
 def test_experiment_ties_select_same_name_independent_of_queue_order(self):
  rows=[{'abo_product_id':str(i),'leakage_group_id':str(i),'weak_label':c.CLASSES[i%7]} for i in range(14)]
  x=np.zeros((14,768),dtype=np.float32);x[:,0]=1;folds=[list(range(7,14))]
  configs=[{'name':name,'kind':'linear','lr':.001,'wd':.001} for name in ['zeta','alpha']]
  state=torch.nn.Linear(768,7).state_dict()
  with tempfile.TemporaryDirectory() as tmp:
   base=pathlib.Path(tmp);prep=base/'prep';prep.mkdir();chosen=[]
   def fake_fit(out,array,rows,folds,cfg,*args):
    path=out/'trials'/cfg['name']/'fold-0/best.pt';path.parent.mkdir(parents=True);path.write_bytes(b'synthetic-checkpoint')
    return {'config':cfg,'metrics':{'product':{'macro_f1':.5}}}
   for n,queue in enumerate([configs,list(reversed(configs))]):
    out=base/str(n);out.mkdir();c.save(prep/'frozen-development-protocol.json',{'configs':queue,'queue_seconds_per_invocation':10,'max_epochs':3,'scope':'synthetic test only'})
    c.atomic(prep/'frozen-development-features.npy',b'synthetic-vector-fingerprint')
    with patch.object(r,'load_cache',return_value=(x,rows,folds)),patch.object(r.w,'fit',side_effect=fake_fit),patch.object(r.torch,'load',return_value={'state_dict':state}):r.run(prep,out,'baseline')
    chosen.append(c.read(out/'comparison-report.json')['selected_recipe']['name'])
   self.assertEqual(chosen,['alpha','alpha'])
