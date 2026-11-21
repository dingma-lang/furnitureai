"""Meaningful CPU loss/gradient equivalence and data-isolation tests."""
import unittest,copy,tempfile
import numpy as np
import torch
from common8 import *
from loss8 import objective,gradient_norm
from expanded import detection_loss,xyxy_to_cxcywh

class LossTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
        self.logits=torch.zeros((3,7),requires_grad=True);self.boxes=xyxy_to_cxcywh(torch.tensor([[.69,.69,.91,.91],[.1,.1,.2,.2],[.4,.4,.5,.5]])).requires_grad_();self.ev=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC-REGION')]
    def test_control_exact_legacy_value_and_gradients(self):
        old,_=detection_loss(self.logits,self.boxes,self.row);new,_,_=objective(self.logits,self.boxes,self.row,self.ev,[],False)
        self.assertTrue(torch.equal(old,new));og=torch.autograd.grad(old,(self.logits,self.boxes),retain_graph=True);ng=torch.autograd.grad(new,(self.logits,self.boxes));self.assertTrue(all(torch.equal(a,b) for a,b in zip(og,ng)))
    def test_only_negative_mask_differs(self):
        a,_,sa=objective(self.logits,self.boxes,self.row,self.ev,[],False);b,_,sb=objective(self.logits,self.boxes,self.row,self.ev,[],True)
        self.assertEqual(sa['positive_classification'],sb['positive_classification']);self.assertEqual(sa['box_loss'],sb['box_loss']);self.assertEqual(sb['added_active_Chair_terms'],1);self.assertGreater(float((b-a).detach()),0)
    def test_corrected_additional_gradient_is_real(self):
        _,p,s=objective(self.logits,self.boxes,self.row,self.ev,[],True);g=torch.autograd.grad(p['added'],self.logits)[0];self.assertGreater(g[1,1],0);self.assertEqual(int((g!=0).sum()),1)
    def test_no_regions_identical_arms(self):
        a,_,_=objective(self.logits,self.boxes,self.row,[],[],False);b,_,_=objective(self.logits,self.boxes,self.row,[],[],True);self.assertTrue(torch.equal(a,b))
    def test_protected_regions_cannot_receive_targets(self):
        _,_,s=objective(self.logits,self.boxes,self.row,self.ev,[[0,0,.3,.3]],True);self.assertEqual(s['added_active_Chair_terms'],0)
    def test_nonfinite_stops(self):
        x=self.logits.detach().clone();x[0,0]=float('nan')
        with self.assertRaises(ValueError):objective(x,self.boxes,self.row,self.ev,[],True)
    def test_gradient_probe_does_not_accumulate_parameter_grad(self):
        layer=torch.nn.Linear(2,7);x=layer(torch.ones((3,2)));_,parts,_=objective(x,self.boxes,self.row,self.ev,[],True);g=torch.autograd.grad(parts['added']*128,tuple(layer.parameters()),retain_graph=True);self.assertGreater(gradient_norm(g,128),0);self.assertTrue(all(p.grad is None for p in layer.parameters()))

class DataTests(unittest.TestCase):
    def test_standalone_worker_dependency_imports(self):
        import subprocess
        code='import worker8; from expanded import configure_heads,HEAD_KEYS,HEAD_PARAMETERS; from masks6 import validate_bundle,authorized; from loss8 import objective; from checkpoint8 import save_snapshot'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_partitions_counts(self):
        train,dev=subsets();self.assertEqual((len(train),len(dev)),(59,19))
        for key in ('sha256','identity_group','image_id'):self.assertFalse({r[key] for r in train}&{r[key] for r in dev})
    def test_same_sampling_for_arms(self):
        train,_=subsets()
        for ep in range(5):self.assertEqual(grouped_epoch(train,ep),grouped_epoch(train,ep));self.assertEqual(len(grouped_epoch(train,ep)),15)
    def test_last_group_and_budget(self):
        train,_=subsets();self.assertEqual(len(grouped_epoch(train,0)[-1]),3);self.assertEqual(sum(len(grouped_epoch(train,e)) for e in range(5)),75)
    def test_validation_rejected_for_grad_loader(self):
        p=manifest();_,dev=subsets(p)
        with self.assertRaises(PermissionError):image_path(dev[0],'optimization',p)
    def test_raw_outputs_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.npz';a=dict(logits=np.zeros((3600,7),np.float32),square_cxcywh=np.ones((3600,4),np.float32));save_npz(p,a);old=sha(p);self.assertEqual(load_npz(p)['logits'].shape,(3600,7))
            with self.assertRaises(FileExistsError):save_npz(p,a)
            self.assertEqual(sha(p),old)
    def test_group_leakage_rejected(self):
        p=copy.deepcopy(manifest());train,dev=subsets(p);dev[0]['identity_group']=train[0]['identity_group']
        with self.assertRaises(ValueError):subsets(p)
    def test_evaluation_cache_preserves_GPU_scores_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'eval.npz';a=dict(logits=np.zeros((3600,7),np.float32),square_cxcywh=np.ones((3600,4),np.float32),scores=np.zeros(3600,np.float32),labels=np.zeros(3600,np.int64),boxes=np.ones((3600,4),np.float32));save_npz(p,a);b=load_npz(p);self.assertEqual(set(a),set(b));self.assertEqual(b['labels'].dtype,np.int64)

class MetricTests(unittest.TestCase):
    def test_recovery_metric_and_legacy_metric_equivalent(self):
        import importlib.util
        sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));spec=importlib.util.spec_from_file_location('ab8_cpu_metric',ROOT/'tools/phase10c-v1/metric_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        _,dev=subsets();row=dev[0];a=next(a for a in row['annotations'] if not a['ignore']);pred={row['image_id']:[dict(label=a['label'],box=a['box'],score=.8)]}
        sys.path.insert(0,str(ROOT/'tools/phase10b-v1'));import internal
        old=internal.evaluate([row],pred);new=m.evaluate([m.scopes(row)],pred)
        for k in ('tp','fp','fn','precision','recall','ap50','ap75'):self.assertEqual(old[k],new[k])

if __name__=='__main__':unittest.main()
