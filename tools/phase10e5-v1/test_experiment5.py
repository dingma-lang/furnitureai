"""CPU synthetic checks for the representative gate,loss and monitoring."""
import ast,unittest
from common5 import *
sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
from test_recovery4 import PrecisionTests
from observe5 import gradient_diagnostics,combined_gate
from precision4 import combined,box_objective
import torch

class ExperimentTests(unittest.TestCase):
    def data(self):return PrecisionTests().setup_data()
    def test_complete_combined_gate_keeps_all_parameters_unchanged(self):
        h,box,s,b,t,tb,e=self.data();params=[*h.dense0.parameters(),*box.parameters()]
        old=[p.detach().clone() for p in params];loss,parts,stats=combined(h,s,b,t,tb,e,COEFFICIENT,enabled=True)
        mask=torch.zeros_like(t,dtype=torch.bool)
        value=combined_gate(loss,parts,s.logits[0],t,mask,b,box_objective(s.logits[0],b,e['row']),params,list(h.dense0.parameters()),list(box.parameters()),COEFFICIENT,torch.autograd.grad)
        self.assertEqual(value['status'],'PASS');self.assertEqual(value['loss_scale'],1.)
        self.assertTrue(all(torch.equal(a,p) and p.grad is None for a,p in zip(old,params)))
    def test_gradient_monitors_zero_and_nonzero_without_retuning(self):
        h,box,s,b,t,tb,e=self.data();loss,parts,_=combined(h,s,b,t,tb,e,COEFFICIENT,enabled=True)
        mask=torch.zeros_like(t,dtype=torch.bool)
        value=gradient_diagnostics(parts,s.logits[0],t,mask,list(h.dense0.parameters()),b,list(box.parameters()),COEFFICIENT,torch.autograd.grad)
        self.assertEqual(value['weighted_KD_projection_L2'],0);self.assertFalse(value['KD_dominant'])
        self.assertGreater(value['positive_projection_L2'],0)
    def test_budget_epochs_updates_runtime(self):
        state=dict(updates=75,forwards=373,epochs=5,contractions=354)
        budget(state,599,6979321856,2147483648)
        for field in state:
            bad=dict(state);bad[field]+=1;self.assertRaises(RuntimeError,budget,bad,1,0,0)
        self.assertRaises(RuntimeError,budget,state,600,0,0)
    def test_pinned_loss_scale_coefficient_and_syntax(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
        text=(HERE/'worker5.py').read_text()
        self.assertNotIn('loss*128',text);self.assertNotIn('GradScaler(',text)
        self.assertEqual(COEFFICIENT,7.063459015854303e-6)
        self.assertIn('optimizer=scope.optimizer()',text)
        self.assertLess(text.index("emit(RUN/'gate/complete.json'"),text.index('optimizer=scope.optimizer()'))
    def test_import_launch_worker_without_torch(self):
        r=subprocess.run([sys.executable,'-B','-c','import worker5,launch5;import sys;assert "torch" not in sys.modules'],cwd=HERE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
