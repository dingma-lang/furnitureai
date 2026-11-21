"""CPU proof that C11 observation leaves C8 loss and gradients unchanged."""
import unittest
import numpy as np
import torch
from common11 import *
from observe11 import objective,original_objective,compare_array_sets
from expanded import xyxy_to_cxcywh

class ObservationTests(unittest.TestCase):
    def data(self,dtype=torch.float32):
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False),dict(label='sofa',box=[.4,.4,.6,.6],ignore=False)],explicit_native_negative={})
        z=torch.zeros((3,7),dtype=dtype,requires_grad=True);b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.4,.4,.6,.6],[.1,.1,.2,.2]],dtype=dtype)).requires_grad_();ev=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC-SOURCE')];return row,z,b,ev
    def test_loss_and_all_input_gradients_bit_identical(self):
        r,z,b,e=self.data();old,parts,stats=original_objective(z,b,r,e,[],True);new,pp,ss=objective(z,b,r,e,[],True)
        self.assertTrue(torch.equal(old,new));a=torch.autograd.grad(old,(z,b),retain_graph=True);q=torch.autograd.grad(new,(z,b));self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,q)));self.assertTrue(all(torch.equal(parts[k],pp[k]) for k in parts));self.assertTrue(all(stats[k]==ss[k] for k in stats))
    def test_FP16_input_gradients_identical(self):
        r,z,b,e=self.data(torch.float16);a,_,_=original_objective(z,b,r,e,[],True);q,_,_=objective(z,b,r,e,[],True);ga=torch.autograd.grad(a,(z,b),retain_graph=True);gq=torch.autograd.grad(q,(z,b));self.assertTrue(torch.equal(a,q));self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gq)))
    def test_Chair_only_contribution_partition(self):
        r,z,b,e=self.data();_,_,s=objective(z,b,r,e,[],True);self.assertEqual(s['Chair_positive_target_terms'],1);self.assertEqual(s['Chair_negative_target_terms'],1);self.assertAlmostEqual(s['Chair_positive_classification']*2,s['positive_classification']);self.assertAlmostEqual(s['Chair_negative_classification'],s['added_Chair_negative_loss'])
    def test_unknowns_not_converted_to_negatives(self):
        r,z,b,e=self.data();_,_,s=objective(z,b,r,[],[],True);self.assertEqual(s['Chair_negative_target_terms'],0);self.assertEqual(s['Chair_negative_classification'],0)
    def test_protected_rectangle_has_no_added_negatives(self):
        r,z,b,e=self.data();_,_,s=objective(z,b,r,e,[[0,0,.3,.3]],True);self.assertEqual(s['Chair_negative_target_terms'],0)
    def test_legacy_observation_does_not_apply_counterfactual_negatives(self):
        r,z,b,e=self.data();_,_,s=objective(z,b,r,e,[],False);self.assertEqual(s['Chair_negative_target_terms'],0);self.assertEqual(s['Chair_source_negative_classification'],0)
    def test_initial_array_comparison_and_mismatch_stop(self):
        a=dict(logits=np.zeros((2,7),np.float32),square_cxcywh=np.ones((2,4),np.float32));r=compare_array_sets(a,a);self.assertTrue(all(v['bit_identical'] for v in r.values()));b={k:v.copy() for k,v in a.items()};b['logits'][0,0]=1
        with self.assertRaises(ValueError):compare_array_sets(a,b)
    def test_worker_only_instrumentation_and_initial_check_changes(self):
        src=(ROOT/'tools/phase10c10-v1/future_worker10.py').read_text();src=src.replace('Prepared single calibration-frozen experiment. NOT EXECUTED during C10.','Phase10C.11 authorized execution; prepared algorithm plus detached diagnostics.').replace('from loss8 import objective,gradient_norm','from loss8 import gradient_norm\n        from observe11 import objective,verify_initial_outputs11').replace('probe(0);evaluate_raw(0);last=snapshot(0,0)','probe(0);evaluate_raw(0);verify_initial_outputs11();last=snapshot(0,0)')
        src=src.replace('import time,threading,traceback,uuid,random','import time,threading,traceback,uuid,random,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]/\'phase10c10-v1\'))')
        self.assertEqual(src,(HERE11/'worker11.py').read_text())
    def test_standalone_launcher_and_worker_import_without_CUDA(self):
        import subprocess
        code='import launch11,worker11,torch; assert not torch.cuda.is_initialized()'
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c',code],cwd=HERE11,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
