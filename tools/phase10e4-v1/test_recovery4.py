"""Synthetic CPU scope/precision checks;no real model or GPU updates."""
import ast
import copy
import unittest

from common4 import *
sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
from test_numerics3 import fixture
from stable3 import Capture
from precision4 import combined,box_objective
from objectiveE import objective,base_objective
import torch


class PrecisionTests(unittest.TestCase):
    def setup_data(self):
        head,x,q,native,snapshot=fixture(torch.float16)
        torch.manual_seed(6106)
        box=torch.nn.Sequential(torch.nn.Linear(6,6),torch.nn.GELU(),torch.nn.Linear(6,6),torch.nn.GELU(),torch.nn.Linear(6,4),torch.nn.Sigmoid())
        with torch.autocast('cpu',dtype=torch.float16):b=box(x)[0].float()
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={})
        e=dict(row=row,source_regions=[],legacy_protected=[],distill_regions=[],distill_protected=[])
        teacher=native.detach()[0].clone();tb=b.detach().clone()
        return head,box,snapshot,b,teacher,tb,e

    def test_zero_coefficient_same_loss_and_gradients_on_same_widened_path(self):
        h,box,s,b,t,tb,e=self.setup_data()
        from stable3 import observed_logits
        z=observed_logits(h,s,enabled=True)[0]
        old,_,_=base_objective(z,b,e['row'],[],[],True)
        new,_,_=combined(h,s,b,t,tb,e,0.,enabled=True)
        self.assertTrue(torch.equal(old,new))
        params=[*h.dense0.parameters(),*box.parameters()]
        a=torch.autograd.grad(old,params,retain_graph=True);c=torch.autograd.grad(new,params,retain_graph=True)
        for aa,cc in zip(a,c):torch.testing.assert_close(aa,cc,rtol=0,atol=0)

    def test_combined_backward_eight_tensors_no_frozen_gradients_or_updates(self):
        h,box,s,b,t,tb,e=self.setup_data()
        # Synthetic test grant and labels only;no real evidence creation.
        e['distill_regions']=[dict(candidate_id='SYNTHETIC',image_id='SYNTHETIC',category='chair',box=[0.,0.,1.,1.],source_verified_under_policy=True)]
        t[:,1]-=2
        params=[*h.dense0.parameters(),*box.parameters()]
        before=[p.detach().clone() for p in params]
        loss,parts,_=combined(h,s,b,t,tb,e,1.,enabled=True)
        grads=torch.autograd.grad(loss,params)
        self.assertEqual(len(grads),8)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(torch.equal(p,old) and p.grad is None for p,old in zip(params,before)))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in h.logit_shift.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in h.logit_scale.parameters()))

    def test_box_helper_matches_original_scalar(self):
        h,box,s,b,t,tb,e=self.setup_data()
        _,_,info=base_objective(s.logits[0],b,e['row'],[],[],True)
        loss=box_objective(s.logits[0],b,e['row'])
        self.assertAlmostEqual(float(loss.detach()),info['box_loss'],places=6)
        grads=torch.autograd.grad(loss,[b,*box.parameters(),*h.dense0.parameters()],allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads[:7]))
        self.assertTrue(all(g is None for g in grads[7:]))

    def test_training_precision_requires_explicit_opt_in(self):
        h,box,s,b,t,tb,e=self.setup_data()
        self.assertRaises(PermissionError,combined,h,s,b,t,tb,e,1.)
        self.assertRaises(ValueError,combined,h,s,b,t,tb,e,None,enabled=True)


class BoundaryTests(unittest.TestCase):
    def test_exact_budgets(self):
        state=dict(forwards=25,contractions=25,VJPs=48,optimizer_updates=0)
        budget(state,239,6979321856,134217728)
        for field in ('forwards','contractions','VJPs','optimizer_updates'):
            bad=dict(state);bad[field]+=1
            self.assertRaises(RuntimeError,budget,bad,1,0,0)
        self.assertRaises(RuntimeError,budget,state,240,0,0)

    def test_worker_never_constructs_optimizer_or_steps(self):
        for path in HERE.glob('*.py'):compile(path.read_text(),str(path),'exec')
        tree=ast.parse((HERE/'worker4.py').read_text())
        attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(attrs&{'step','backward','AdamW','SGD','optimizer','GradScaler'})
        text=(HERE/'worker4.py').read_text()
        self.assertNotIn('loss*128',text)

    def test_modules_import_without_torch_or_cuda_execution(self):
        code='import launch4,worker4;import sys;assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
