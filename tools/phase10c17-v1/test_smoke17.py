"""CPU-only scope,sequence,scaled-gradient and cap diagnostic tests."""
import ast,subprocess,unittest
from unittest.mock import patch
import torch
import common17
from common17 import *
from measure17 import scaled_gradients,group_metrics,finish_accumulator
from balanced16 import CLASS_KEYS,Accumulator,Policy


def buffers():
    base={k:torch.ones(2) for k in sorted(HEAD_KEYS)};p={k:torch.ones(2) for k in sorted(CLASS_KEYS)};n={k:-torch.ones(2)*10 for k in sorted(CLASS_KEYS)};return base,p,n


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.folder=RUN17/('CPU-fixture-'+uuid.uuid4().hex[:8]);cls.folder.mkdir()
    def test_exact_full_C8_order_and_positive_counts(self):
        cfg,_,groups,_=inputs17();self.assertEqual([len(g) for g in groups],[4]*14+[3]);self.assertEqual(sum(len(g) for g in groups),59);self.assertEqual(sum(not a['ignore'] for g in groups for r in g for a in r['annotations']),182);self.assertEqual(len({r['identity_group'] for g in groups for r in g}),56)
    def test_wrong_authorization_or_budget_rejected(self):
        cfg=read(RUN17/'authorized-config.json')
        for bad in [dict(cfg,smoke_authorized=False),dict(cfg,training_authorized=True),dict(cfg,limits=dict(cfg['limits'],forwards=60))]:
            with patch.object(common17,'read',return_value=bad):
                with self.assertRaises(PermissionError):inputs17()
    def test_C8_scope_is12_with_all_calibration_trainable(self):
        from test_preflight10 import fixture
        from expanded import configure_heads
        m=fixture();configure_heads(m);self.assertEqual({n for n,p in m.named_parameters() if p.requires_grad},HEAD_KEYS);self.assertEqual(len(CLASS_KEYS),6);self.assertTrue(all(p.requires_grad for n,p in m.named_parameters() if n in CLASS_KEYS))
    def test_GPU_worker_has_no_optimizer_or_adapted_loader(self):
        s=(HERE17/'gpu17.py').read_text();t=ast.parse(s);calls=[n.func.attr for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)];self.assertFalse(set(calls)&{'step','backward','optimizer','zero_grad','load_state_dict'});self.assertNotIn('restore_candidate',s);self.assertNotIn('torch.load',s)
    def test_no_CUDA_context_on_import(self):
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c','import gpu17,launch17,torch;assert not torch.cuda.is_initialized()'],cwd=HERE17,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
    def test_scaled_VJP_normalization_order_and_no_grad_accumulation(self):
        z=torch.ones((2,7),requires_grad=True);b=torch.ones((2,4),requires_grad=True);p=torch.nn.Parameter(torch.ones(2));loss=z.sum()+p.sum();g=scaled_gradients(loss,z,b,[p],3,128.,retain_graph=False);self.assertIsNone(g[1]);self.assertTrue(torch.equal(g[0],torch.full_like(z,128/3)));self.assertTrue(torch.equal(g[2],torch.full_like(p,128/3)));self.assertIsNone(p.grad);self.assertIsNone(z.grad)
    def test_cap_metrics_use_actual_parameter_buffers(self):
        b,p,n=buffers();a=Accumulator(1,128);a.add({k:v*128 for k,v in b.items()},{k:v*128 for k,v in p.items()},{k:v*128 for k,v in n.items()},1);out,cap=finish_accumulator(a);r=group_metrics(b,p,n,n,n,out,cap);self.assertEqual(r['cap_violations'],0);self.assertLessEqual(r['negatives']['C_post_cap']['negative_positive_ratio'],.250001);self.assertTrue(r['box_gradients_unchanged'])
    def test_cap_violation_and_box_mutation_fail_closed(self):
        b,p,n=buffers();a=Accumulator(1,1);a.add(b,p,n,1);out,cap=finish_accumulator(a);bad={k:v.clone() for k,v in out.items()};bad[next(k for k in bad if k.startswith('box_head.'))].add_(1)
        with self.assertRaises(ValueError):group_metrics(b,p,n,n,n,bad,cap)
        with self.assertRaises(ValueError):group_metrics(b,p,n,n,n,{k:(b[k]+n[k] if k in n else b[k]) for k in b},cap)
    def test_C16_cap_buffer_bit_identity_and_AMP_metadata_collision_regression(self):
        b,p,n=buffers();a=Accumulator(1,128);a.add({k:v*128 for k,v in b.items()},{k:v*128 for k,v in p.items()},{k:v*128 for k,v in n.items()},1);old,oldmeta=a.finish(Policy(enabled=True));new,diag=finish_accumulator(a);self.assertTrue(all(torch.equal(old[k],new[k]) for k in old));self.assertEqual(oldmeta['scale'],128);self.assertEqual(diag['amp_scale'],128);self.assertAlmostEqual(diag['scale'],.025);self.assertLess(diag['scale'],1)
    def test_atomic_paths_overwrite_and_output_limit(self):
        p=self.folder/'record.json';emit17(p,dict(synthetic=True))
        with self.assertRaises(FileExistsError):emit17(p,{})
        with self.assertRaises(PermissionError):emit17(DATA16/'must-not-create.json',{})
        with patch.object(common17,'MAX_OUTPUT17',1):
            with self.assertRaises(RuntimeError):emit17(self.folder/'must-not-create.json',{})
    def test_all24_original_cached_source_outputs_preserve_masks(self):
        from common6 import candidates
        from balanced16 import branches
        from loss15 import controlled_losses
        _,train,dev,_,reg=training_context();byid={r['image_id']:r for r in train};ids=[r['image_id'] for r in read(SOURCE/'proposed-smoke-photos.json')['photos']];frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');context=read(DATA16/'catalog.json')
        for i,ident in enumerate(ids,1):
            row=byid[ident];a=load_npz(C8/'B-corrected/probes/epoch-0'/(str(i).zfill(2)+'.npz'));z=torch.from_numpy(a['logits']).requires_grad_();b=torch.from_numpy(a['square_cxcywh']).requires_grad_();e=[e for e in bundle['records'] if e['image_id']==ident];protect=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain'];old,m,s=controlled_losses(z,b,row,e,protect);r=branches(z,b,row,e,protect,context,Policy(enabled=True));self.assertTrue(torch.equal(m['mask'],r['masks']['mask']));self.assertTrue(torch.equal(m['added'],r['masks']['added']));self.assertTrue(torch.equal(old['positive'],r['positive_anchor']));self.assertEqual(s['protected_violations'],0)


if __name__=='__main__':unittest.main()
