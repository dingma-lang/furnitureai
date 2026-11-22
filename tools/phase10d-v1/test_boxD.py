"""CPU-only executor/scope regression tests; no detector or photograph inference."""
import copy
import importlib.util
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from commonD import *
from scopeD import BoxScopeD,BOX_KEYS,unregistered_tensors
from box21 import original_heads,restore_delta
sys.path.insert(0,str(ROOT/'tools/phase10c10-v1'))
from test_preflight10 import fixture


def fixtureD():
    m=fixture(original_heads())
    m.num_patches_height=60;m.num_patches_width=60
    m.compute_box_bias=types.MethodType(lambda self,h,w:torch.zeros(h*w,4),m)
    m.box_bias=m.compute_box_bias(60,60)
    return m


class ScopeDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.model=fixtureD();self.scope=BoxScopeD(self.model);self.opt=self.scope.optimizer()

    def test_only_six_original_box_tensors(self):
        self.scope.check(self.opt,check_hash=True)
        self.assertEqual({n for n,p in self.model.named_parameters() if p.requires_grad},BOX_KEYS)
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),1184260)

    def test_derived_plain_tensor_is_not_a_buffer_but_is_protected(self):
        self.assertNotIn('box_bias',dict(self.model.named_buffers()))
        self.assertEqual(set(unregistered_tensors(self.model)),{'box_bias'})
        self.model.box_bias[0,0]=1
        self.assertRaises(ValueError,self.scope.check,self.opt)

    def test_unexpected_plain_tensor_rejected(self):
        m=fixtureD();m.unapproved_tensor=torch.ones(2)
        self.assertRaises(ValueError,BoxScopeD,m)

    def test_initial_bias_mismatch_rejected(self):
        m=fixtureD();m.box_bias[0,0]=1
        self.assertRaises(ValueError,BoxScopeD,m)

    def test_mode_transition_does_not_reenable_or_hide_gradients(self):
        self.model.eval();train_mode(self.scope,self.opt)
        self.assertTrue(self.model.box_head.training);self.assertFalse(self.model.class_head.training)
        self.model.class_head.dense0.weight.requires_grad_(True)
        self.assertRaises(ValueError,train_mode,self.scope,self.opt)

    def test_box_only_cpu_gradient_and_checkpoint(self):
        row=dict(width=480,height=640,annotations=[dict(label='chair',box=[.1,.2,.6,.9],ignore=False),dict(label='table',box=[.6,.4,.9,.7],ignore=True)],explicit_native_negative={})
        x=torch.randn(8,768)
        logits=self.model.class_head.dense0(x)[:,:7]
        boxes=self.model.box_head.dense2(torch.relu(self.model.box_head.dense1(torch.relu(self.model.box_head.dense0(x))))).sigmoid()
        before=logits.detach().clone();value,parts=self.scope.loss(logits,boxes,row)
        value.backward();self.scope.check(self.opt,True);self.opt.step();self.scope.check(self.opt,check_hash=True)
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if n not in BOX_KEYS))
        self.assertEqual(parts['matched'],1);self.assertFalse(parts['unknowns_as_background'])
        self.assertTrue(torch.equal(before,self.model.class_head.dense0(x)[:,:7]))
        payload=self.scope.checkpoint_payload(1,'a'*64)
        self.assertEqual(payload['unregistered_tensor_sha256'],self.scope.derived_hashes)
        restored=fixtureD();restore_delta(restored,payload,'a'*64)
        self.assertTrue(all(torch.equal(dict(restored.named_parameters())[n],payload['weights'][n]) for n in BOX_KEYS))

    def test_frozen_zero_gradient_rejected(self):
        p=self.model.class_head.logit_scale.bias;p.grad=torch.zeros_like(p)
        self.assertRaises(ValueError,self.scope.check,self.opt)

    def test_optimizer_settings_bound(self):
        self.opt.param_groups[0]['weight_decay']=0
        self.assertRaises(ValueError,self.scope.check,self.opt)


class BudgetDTests(unittest.TestCase):
    def test_exact_forward_plan(self):
        self.assertEqual(4+59*5+4+19,LIMITS['forwards'])
        self.assertEqual(15*5,LIMITS['updates'])

    def test_forward_cap_before_execution(self):
        state=dict(forwards=321,updates=75);consume_forward(state)
        self.assertEqual(state['forwards'],322)
        self.assertRaises(RuntimeError,consume_forward,state)
        self.assertEqual(state['forwards'],322)

    def test_time_memory_storage_caps(self):
        state=dict(forwards=322,updates=75)
        budget_check(state,599,6979321856,2147483648)
        for elapsed,memory,output in ((600,0,0),(1,6979321857,0),(1,0,2147483649)):
            self.assertRaises(RuntimeError,budget_check,state,elapsed,memory,output)

    def test_count_overrun_rejected(self):
        for state in (dict(forwards=323,updates=75),dict(forwards=1,updates=76)):
            self.assertRaises(RuntimeError,budget_check,state,0,0,0)

    def test_atomic_exclusive_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'one.json'
            with patch('commonD.RUN',root),patch('commonD.REPORT',root):
                emit(p,dict(ok=True));self.assertRaises(FileExistsError,emit,p,dict(ok=False))
            self.assertEqual(read(p),{'ok':True})
        self.assertRaises(PermissionError,emit,Path('/tmp/not-authorized-D.json'),{})

    def test_no_worker_execution_on_import(self):
        import workerD,launchD
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_grouped_order_same59(self):
        cfg,_,train,dev,_=context()
        from common8 import grouped_epoch
        for epoch in range(5):
            groups=grouped_epoch(train,epoch)
            self.assertEqual([[r['image_id'] for r in g] for g in groups],cfg['orders'][epoch])
            self.assertEqual([len(g) for g in groups],[4]*14+[3])
        for key in ('image_id','sha256','identity_group'):
            self.assertFalse({r[key] for r in train}&{r[key] for r in dev})

    def test_metrics_process_import_is_torch_free(self):
        code='import analyzeD; import sys; assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
