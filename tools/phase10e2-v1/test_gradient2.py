"""CPU regression tests for measurement/replay/resource behavior,no detector inference."""
import unittest,ast,tempfile
import numpy as np
import torch
from common2 import *
from measure2 import replay,vjp,zero_derivative,cosine,summaries
from objectiveE import distill

class ReplayTests(unittest.TestCase):
    def arrays(self):return dict(logits=np.full((5,7),-10,dtype='float32'),square_cxcywh=np.full((5,4),.4,dtype='float32'))
    def test_fp16_ulp_difference_allowed_with_exact_layout(self):
        b=self.arrays();a={k:v.copy() for k,v in b.items()};a['logits'][0,0]+=.0078125
        r=replay(a,b);self.assertTrue(r['logits']['accepted']);self.assertFalse(r['logits']['bit_identical'])
    def test_material_or_box_mismatch_rejected(self):
        b=self.arrays();a={k:v.copy() for k,v in b.items()};a['logits'][0,0]+=1
        self.assertRaises(ValueError,replay,a,b)
        a={k:v.copy() for k,v in b.items()};a['square_cxcywh'][0,0]+=.001;self.assertRaises(ValueError,replay,a,b)
    def test_schema_nonfinite_and_dtype_fail_closed(self):
        a=self.arrays();b=self.arrays();a['logits'][0,0]=np.nan;self.assertRaises(ValueError,replay,a,b)
        a=self.arrays();a['logits']=a['logits'].astype('float16');self.assertRaises(ValueError,replay,a,b)

class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6106);self.p=torch.nn.Linear(3,7);self.box=torch.nn.Linear(3,4);x=torch.randn(4,3)
        self.z=self.p(x);self.b=self.box(x);self.projection=[self.p.weight,self.p.bias];self.bp=[self.box.weight,self.box.bias]
    def test_real_output_and_projection_route_without_box(self):
        mask=torch.zeros_like(self.z,dtype=torch.bool);mask[1,1]=True;t=self.z.detach().clone();t[1,1]-=2
        loss=distill(self.z,t,mask);gz,gp=vjp(loss,self.z,self.b,self.projection,self.bp)
        self.assertTrue((gz[~mask]==0).all());self.assertGreater(summaries(gp)['L2'],0)
        self.assertTrue(all(p.grad is None for p in self.p.parameters()));self.assertTrue(all(p.grad is None for p in self.box.parameters()))
    def test_positive_box_routing_violation_detected(self):
        self.assertRaises(ValueError,vjp,self.z.square().sum()+self.b.square().sum(),self.z,self.b,self.projection,self.bp)
    def test_analytic_zero_proof_not_a_fabricated_vjp(self):
        mask=torch.zeros_like(self.z,dtype=torch.bool);mask[:,1]=True
        self.assertTrue(zero_derivative(self.z,self.z.detach().clone(),mask))
        t=self.z.detach().clone();t[0,1]+=.01;self.assertFalse(zero_derivative(self.z,t,mask))
        self.assertTrue(zero_derivative(self.z,t,torch.zeros_like(mask)))
    def test_gradient_cosine_and_summaries(self):
        self.assertAlmostEqual(cosine([torch.ones(2)],[-torch.ones(2)]),-1)
        self.assertEqual(summaries([torch.tensor([-3.,4.])])['L2'],5)
    def test_budget_counters_and_zero_update_scope(self):
        budget(dict(forwards=28,gradient_calculations=48),179,6979321856,134217728)
        self.assertRaises(RuntimeError,budget,dict(forwards=29,gradient_calculations=48),1,0,0)
        self.assertRaises(RuntimeError,budget,dict(forwards=28,gradient_calculations=49),1,0,0)
    def test_every_module_compiles_and_worker_has_no_optimizer(self):
        for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
        tree=ast.parse((HERE/'worker2.py').read_text())
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&{'step','backward','optimizer','AdamW','SGD'})

if __name__=='__main__':unittest.main()
