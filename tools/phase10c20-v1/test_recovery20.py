"""CPU preflight for bounded recovery;no model forward or optimizer step."""
import ast,unittest,tempfile,copy
from unittest.mock import patch
import torch,numpy as np
from common20 import *
import common20
from restore20 import restore_state,restore_recorded_rng,numpy_state


def function(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)


class Recovery20Tests(unittest.TestCase):
    def test_exact_recovery_budget_order_and_membership(self):
        cfg,_,train,dev,_=inputs20();self.assertEqual(cfg['limits']['forwards'],270);self.assertEqual(cfg['limits']['additional_updates'],47);self.assertEqual(cfg['limits']['seconds'],515);self.assertEqual(len(train),59);self.assertEqual(len(dev),19)
        self.assertEqual([r['update'] for r in cfg['remaining_sequence']],list(range(29,76)));self.assertEqual(sum(len(r['image_ids']) for r in cfg['remaining_sequence']),184)
        self.assertEqual(cfg['remaining_sequence'][0]['image_ids'],cfg['orders'][1][13])
        for key in ('image_id','identity_group','sha256'):self.assertFalse({r[key] for r in train}&{r[key] for r in dev})
    def test_unapproved_limits_fail_closed(self):
        cfg=read(RUN20/'binding.json')
        for bad in (dict(cfg,recovery_authorized=False),dict(cfg,limits=dict(cfg['limits'],forwards=271))):
            with patch.object(common20,'read',return_value=bad):
                with self.assertRaises(PermissionError):inputs20()
    def test_actual_checkpoint_cpu_restore_all_heads_moments_scaler(self):
        from test_preflight10 import fixture
        from expanded import configure_heads
        payload=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True);model=fixture();configure_heads(model);heads=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(heads,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);scaler=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2000);scaler.scale(torch.zeros(()))
        restore_state(model,opt,scaler,payload);self.assertEqual(scaler.state_dict(),payload['scaler']);self.assertEqual(len(opt.state),12);self.assertTrue(all(int(v['step'])==28 for v in opt.state.values()));self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertEqual(sha(CHECKPOINT28),CHECKPOINT28_SHA256)
    def test_restore_rng_does_not_fabricate_numpy_or_call_cuda(self):
        p=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True);before=numpy_state();r=restore_recorded_rng(p,use_cuda=False);self.assertFalse(r['NumPy_restored']);self.assertFalse(r['CUDA_restored']);self.assertTrue(r['PyTorch_CPU_restored']);self.assertEqual(before,numpy_state());self.assertFalse(torch.cuda.is_initialized())
    def test_source_numerical_observations_function_identical(self):
        a=function(ROOT/'tools/phase10c18-v1/worker18.py','observations_for');b=function(HERE20/'worker20.py','observations_for');self.assertEqual(ast.dump(a),ast.dump(b))
    def test_forward_preprocessing_and_precision_unchanged(self):
        a=function(ROOT/'tools/phase10c18-v1/worker18.py','forward');b=function(HERE20/'worker20.py','forward');b.body=b.body[1:];self.assertEqual(ast.dump(a),ast.dump(b))
    def test_optimizer_construction_and_clipping_unchanged(self):
        def calls(path,name):return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name]
        for name in ('AdamW','clip_grad_norm_'):
            self.assertEqual([ast.dump(n) for n in calls(ROOT/'tools/phase10c18-v1/worker18.py',name)],[ast.dump(n) for n in calls(HERE20/'worker20.py',name)])
    def test_no_original_restart_or_legacy_output_paths(self):
        s=(HERE20/'worker20.py').read_text();self.assertIn('restore_state(model,optimizer,scaler,payload)',s);self.assertIn('restore_recorded_rng(payload,use_cuda=True)',s);self.assertNotIn('probe(0)',s);self.assertNotIn('snapshot(0,0)',s);self.assertNotIn('relative_to(RUN18)',s);self.assertIn('relative_to(RUN20)',s);self.assertIn("state['updates']>=75",s);self.assertIn("from guard19 import compose_checked",s);self.assertIn('group_index<payload',s)
    def test_atomic_new_storage_bound_and_historical_path_protection(self):
        with tempfile.TemporaryDirectory(dir=RUN20) as td:
            p=Path(td)/'synthetic.json';emit20(p,dict(synthetic_test=True))
            with self.assertRaises(FileExistsError):emit20(p,{})
            with self.assertRaises(PermissionError):emit20(CHECKPOINT28,{})
            with patch.object(common20,'MAX_OUTPUT20',1):
                with self.assertRaises(RuntimeError):emit20(Path(td)/'forbidden.json',{})
    def test_failure_telemetry_uses_new_destination(self):
        from telemetry20 import RuntimeTelemetry
        from test_guard19 import fixture
        with tempfile.TemporaryDirectory(dir=RUN20/'failures') as td:
            t=RuntimeTelemetry(context=dict(synthetic_test=True,next_update=29));t.folder=Path(td);a=fixture();t.boundary(a)
            try:raise ValueError('synthetic recovery failure')
            except ValueError as exc:t.save_failure(exc)
            d=read(next(Path(td).glob('failure-*.json')));self.assertIsNotNone(d['gradients']['positive_norm']);self.assertEqual(d['context']['next_update'],29)
    def test_import_is_CPU_only(self):
        import worker20,launch20
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
