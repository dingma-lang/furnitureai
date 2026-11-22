"""CPU-only regression tests; no old phase outputs are written."""
import ast, unittest, tempfile
from unittest.mock import patch
import torch
from common18 import *
import common18
from update18 import compose_checked, assign_scaled
from balanced16 import Accumulator,CLASS_KEYS,Policy
from measure17 import scaled_gradients


def buffers():
    return ({k:torch.ones(2) for k in sorted(HEAD_KEYS)}, {k:torch.ones(2) for k in sorted(CLASS_KEYS)}, {k:torch.full((2,),-10.) for k in sorted(CLASS_KEYS)})


class TrainingTests(unittest.TestCase):
    def test_bound_C8_all75_order_and_parameter_scope(self):
        cfg,_,train,dev,_=inputs18()
        self.assertEqual(len(train),59);self.assertEqual(len(dev),19)
        self.assertEqual(len({r['identity_group'] for r in train}),56)
        self.assertEqual(sum(not a['ignore'] for r in train for a in r['annotations']),182)
        self.assertEqual(cfg['trainable_keys'],sorted(HEAD_KEYS))
        self.assertEqual(cfg['unchanged_C8_settings']['learning_rate'],1e-5)
        for epoch,order in enumerate(cfg['orders']):
            self.assertEqual([len(g) for g in order],[4]*14+[3])
            for i,group in enumerate(order,1):self.assertEqual(group,read(C8/'B-corrected'/('update-'+str(epoch*15+i).zfill(3)+'.json'))['source_order'])
        for key in ('image_id','sha256','identity_group'):self.assertFalse({r[key] for r in train}&{r[key] for r in dev})
    def test_authorization_and_unchanged_objective_fail_closed(self):
        cfg=read(RUN18/'binding.json')
        for bad in (dict(cfg,training_authorized=False),dict(cfg,arms=2),dict(cfg,limits=dict(cfg['limits'],updates=76))):
            with patch.object(common18,'read',return_value=bad):
                with self.assertRaises(PermissionError):inputs18()
    def test_cap_preserves_positive_box_and_composes_actual_parameter_buffers(self):
        b,p,n=buffers();a=Accumulator(1,128)
        a.add({k:v*128 for k,v in b.items()},{k:v*128 for k,v in p.items()},{k:v*128 for k,v in n.items()},1)
        out,m=compose_checked(a);self.assertLess(m['cap']['scale'],1);self.assertEqual(m['cap_violations'],0)
        self.assertLessEqual(m['source_post']['norm'],.250001*m['positive']['norm'])
        for k in p:self.assertTrue(torch.equal(a.positive[k],p[k]*128))
        for k in b:
            if k.startswith('box_head.'):self.assertTrue(torch.equal(out[k],b[k]))
    def test_actual_GradScaler_assignment_clipping_AdamW_and_frozen_preservation(self):
        b,p,n=buffers();a=Accumulator(3,128)
        for _ in range(3):a.add({k:v*128/3 for k,v in b.items()},{k:v*128/3 for k,v in p.items()},{k:v*128/3 for k,v in n.items()},1)
        out,m=compose_checked(a);heads={k:torch.nn.Parameter(torch.ones(2)) for k in b};frozen=torch.nn.Parameter(torch.ones(2),requires_grad=False);before=frozen.clone()
        optimizer=torch.optim.AdamW(list(heads.values()),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        scaler=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2000)
        scaler.scale(torch.zeros(()));assign_scaled(heads,out,scaler.get_scale());scaler.unscale_(optimizer)
        self.assertTrue(all(torch.equal(heads[k].grad,out[k]) for k in heads))
        torch.nn.utils.clip_grad_norm_(list(heads.values()),1.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update()
        self.assertTrue(all(int(s['step'])==1 for s in optimizer.state.values()));self.assertEqual(scaler.get_scale(),128)
        self.assertTrue(torch.equal(frozen,before));self.assertIsNone(frozen.grad)
    def test_invalid_assignment_nonfinite_scope_existing_grad(self):
        b,p,n=buffers();heads={k:torch.nn.Parameter(torch.ones(2)) for k in b}
        with self.assertRaises(ValueError):assign_scaled(heads,{k:v for k,v in b.items() if k!=next(iter(b))},128)
        bad={k:v.clone() for k,v in b.items()};bad[next(iter(bad))][0]=float('nan')
        with self.assertRaises(ValueError):assign_scaled(heads,bad,128)
        heads[next(iter(heads))].grad=torch.ones(2)
        with self.assertRaises(ValueError):assign_scaled(heads,b,128)
    def test_last3_VJP_matches_C8_scale_order(self):
        z=torch.ones((2,7),requires_grad=True);b=torch.ones((2,4),requires_grad=True);p=torch.nn.Parameter(torch.ones(2))
        g=scaled_gradients(z.sum()+p.sum(),z,b,[p],3,128,retain_graph=False)
        self.assertTrue(torch.equal(g[2],torch.full_like(p,128/3)));self.assertIsNone(p.grad)
    def test_atomic_no_overwrite_bad_path_budget(self):
        with tempfile.TemporaryDirectory(dir=RUN18) as td:
            p=Path(td)/'fixture.json';emit18(p,dict(synthetic=True))
            with self.assertRaises(FileExistsError):emit18(p,{})
            with self.assertRaises(PermissionError):emit18(ROOT/'unauthorized-test.json',{})
            with patch.object(common18,'MAX_OUTPUT18',1):
                with self.assertRaises(RuntimeError):emit18(Path(td)/'forbidden.json',{})
    def test_CPU_import_does_not_initialize_CUDA(self):
        import worker18,launch18
        self.assertFalse(torch.cuda.is_initialized())
    def test_launcher_one_attempt_and_no_alternative_arm(self):
        text=(HERE18/'launch18.py').read_text();tree=ast.parse(text)
        self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in ast.walk(tree)),1)
        self.assertIn("'launch-once.json'",text);self.assertIn('600-',text)
        text=(HERE18/'worker18.py').read_text();self.assertNotIn('load_delta(',text);self.assertNotIn('L2SP',text);self.assertIn('range(5)',text);self.assertIn("state['updates']>=75",text)
    def test_all59_cached_masks_positive_native_box_and_ignored_preserved(self):
        from balanced16 import branches
        from common6 import candidates
        cfg,prepared,train,dev,reg=inputs18();rows={r['image_id']:r for r in train};frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');ctx=read(DATA16/'catalog.json')
        done=read(RUN17/'complete.json')
        for rec in done['records']:
            meta=read(RUN17/rec['file']);row=rows[meta['image_id']];data=RUN17/meta['raw_file']
            self.assertEqual(sha(data),meta['raw_sha256'])
            import numpy as np
            with np.load(data,allow_pickle=False) as raw:arrays={k:raw[k] for k in raw.files}
            z=torch.from_numpy(arrays['logits']).requires_grad_();b=torch.from_numpy(arrays['square_cxcywh']).requires_grad_();ev=[e for e in bundle['records'] if e['image_id']==row['image_id']];pro=[p['box'] for p in reg[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain'];r=branches(z,b,row,ev,pro,ctx,Policy(enabled=True))
            for key in ('target','legacy','mask','added'):self.assertTrue(np.array_equal(r['masks'][key].detach().numpy(),arrays[key]))
            self.assertTrue(np.array_equal(r['weights'].numpy(),arrays['weights']));self.assertEqual(r['stats']['protected_violations'],0)
            source=torch.autograd.grad(r['source_raw'],z)[0];self.assertFalse(source[~r['masks']['added']].any());self.assertFalse((r['masks']['added']&r['masks']['target'].bool()).any())


if __name__=='__main__':unittest.main()
