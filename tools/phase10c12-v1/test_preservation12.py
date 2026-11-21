"""CPU-only math/synthetic fixtures; no image/model training or CUDA calls."""
import copy,io,random,subprocess,unittest,math
from unittest.mock import patch
import torch
from common12 import *
from preservation12 import ProjectionPreserver
from checkpoint12 import make_payload,restore_candidate,restore_training_state
from objective12 import objective,existing_objective
from config12 import specification,check_spec,require_training_approval,required_code_pins
from freeze10 import FrozenScope,head_weights
from test_preflight10 import fixture,toy_step
from expanded import xyxy_to_cxcywh


class PreservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.model=fixture();self.reference=head_weights(self.model);self.scope=FrozenScope(self.model,self.reference);self.opt=self.scope.optimizer();self.hashes={n:tensor_sha(self.reference[n]) for n in PROJECTION_KEYS}

    def reg(self,enabled=True,strength=2):
        return ProjectionPreserver(self.model,self.reference,enabled=enabled,strength=strength,expected_hashes=self.hashes)

    def drift(self):
        with torch.no_grad():
            for n in PROJECTION_KEYS:self.scope.params[n].add_(1/1024)

    def test_zero_at_original_with_zero_gradient(self):
        r=self.reg();loss=r.penalty();self.assertEqual(float(loss.detach()),0)
        g=torch.autograd.grad(loss,tuple(self.scope.params[n] for n in PROJECTION_KEYS));self.assertTrue(all(torch.count_nonzero(t)==0 for t in g))

    def test_joint_half_squared_sum_and_toward_original_gradient(self):
        r=self.reg();self.drift();loss=r.penalty();expected=.5*sum((self.scope.params[n].double()-self.reference[n].double()).square().sum() for n in PROJECTION_KEYS)
        self.assertAlmostEqual(float(loss.detach()),float(expected.detach()),places=6)
        gs=torch.autograd.grad(r.strength*loss,tuple(self.scope.params[n] for n in PROJECTION_KEYS))
        for n,g in zip(PROJECTION_KEYS,gs):
            delta=self.scope.params[n].detach()-self.reference[n];self.assertTrue(torch.equal(g,2*delta));self.assertGreater(float((g*delta).sum()),0)

    def test_external_reference_mutation_does_not_change_internal_reference(self):
        r=self.reg();saved=r.state()
        self.reference[PROJECTION_KEYS[0]].add_(1)
        self.assertEqual(r.state(),saved);self.assertEqual(float(r.penalty().detach()),0)

    def test_internal_reference_corruption_detected(self):
        r=self.reg();r._reference[PROJECTION_KEYS[0]].add_(1)
        with self.assertRaisesRegex(ValueError,'immutability'):r.penalty()

    def test_device_reference_corruption_detected(self):
        r=self.reg();r.penalty();r._device_reference[PROJECTION_KEYS[0]].add_(1)
        with self.assertRaisesRegex(ValueError,'Device reference'):r.check()

    def test_configuration_and_hashes_are_read_only(self):
        r=self.reg()
        with self.assertRaises(AttributeError):r.strength=9
        with self.assertRaises(AttributeError):r.enabled=False
        with self.assertRaises(TypeError):r.hashes[PROJECTION_KEYS[0]]='other'

    def test_wrong_original_reference_rejected(self):
        self.reference[PROJECTION_KEYS[0]].add_(1)
        with self.assertRaisesRegex(ValueError,'pinned original'):self.reg()

    def test_invalid_strength_and_nonfinite_parameters_rejected(self):
        for value in (-1,float('nan'),float('inf'),0):
            with self.assertRaises(ValueError):self.reg(strength=value)
        r=self.reg();self.scope.params[PROJECTION_KEYS[0]].data[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):r.penalty()

    def test_only_projection_is_regularized(self):
        r=self.reg();self.drift();r.penalty().backward()
        for n,p in self.model.named_parameters():self.assertEqual(p.grad is not None,n in PROJECTION_KEYS)

    def test_calibration_freeze_persists_across_modes_and_optimizer_updates(self):
        r=self.reg();self.drift();toy_step(self.scope,self.opt);self.scope.eval_mode();r.check();self.scope.train_mode();r.check()
        for n in CALIBRATION_KEYS:self.assertTrue(torch.equal(self.scope.params[n],self.reference[n]));self.assertIsNone(self.scope.params[n].grad)

    def test_optimizer_exact_eight_tensors_excludes_reference_and_calibration(self):
        r=self.reg();values=[p for g in self.opt.param_groups for p in g['params']]
        self.assertEqual({id(p) for p in values},{id(self.scope.params[n]) for n in TRAIN_KEYS});self.assertEqual(len(values),8)
        self.assertFalse({id(p) for p in r._reference.values()}&{id(p) for p in values})

    def test_FP32_penalty_under_CPU_autocast(self):
        r=self.reg();self.drift()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=r.penalty()
        self.assertEqual(loss.dtype,torch.float32);self.assertTrue(torch.isfinite(loss))

    def test_accumulation_four_and_last_three_apply_penalty_once(self):
        r=self.reg();self.drift();params=tuple(self.scope.params[n] for n in PROJECTION_KEYS)
        for size in (3,4):
            self.opt.zero_grad(set_to_none=True)
            for _ in range(size):(r.strength*r.penalty()/size).backward()
            for n,p in zip(PROJECTION_KEYS,params):self.assertTrue(torch.allclose(p.grad,r.analytic_gradients()[n],atol=1e-7,rtol=1e-6))

    def test_gradient_partition_diagnostics_do_not_change_gradients(self):
        r=self.reg();self.drift();data=sum(self.scope.params[n].sum()*.001 for n in PROJECTION_KEYS);(data+r.strength*r.penalty()).backward();before={n:self.scope.params[n].grad.clone() for n in PROJECTION_KEYS};parts=r.gradient_diagnostics()
        for n in PROJECTION_KEYS:
            self.assertTrue(torch.equal(before[n],self.scope.params[n].grad));expected=.001*math.sqrt(self.scope.params[n].numel());self.assertAlmostEqual(parts[n]['existing_objective_gradient_norm_after_subtraction'],expected,places=5)

    def loss_case(self,dtype=torch.float32):
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False),dict(label='sofa',box=[.4,.4,.6,.6],ignore=False)],explicit_native_negative={})
        z=torch.zeros((3,7),dtype=dtype,requires_grad=True);b=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.4,.4,.6,.6],[.1,.1,.2,.2]],dtype=dtype)).requires_grad_();ev=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC')]
        return z,b,row,ev,[],True

    def test_disabled_objective_has_identical_loss_stats_parts_and_gradients(self):
        r=self.reg(enabled=False);self.drift();args=self.loss_case();a,ap,ass=existing_objective(*args);b,bp,bss=objective(*args,r)
        self.assertTrue(torch.equal(a,b));self.assertEqual(ass,bss);self.assertEqual(ap.keys(),bp.keys());self.assertTrue(all(torch.equal(ap[k],bp[k]) for k in ap))
        ga=torch.autograd.grad(a,args[:2],retain_graph=True);gb=torch.autograd.grad(b,args[:2]);self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))

    def test_enabled_preserves_matching_masks_and_box_loss_gradients_FP32_FP16(self):
        r=self.reg();self.drift()
        for dtype in (torch.float32,torch.float16):
            args=self.loss_case(dtype);a,ap,ast=existing_objective(*args);b,bp,bst=objective(*args,r)
            for key in ast:
                if key not in ('total_loss','logging_only_original_loss_returned'):self.assertEqual(ast[key],bst[key])
            self.assertFalse(bst['logging_only_original_loss_returned']);self.assertTrue(bst['existing_supervision_and_box_loss_unchanged'])
            self.assertTrue(all(torch.equal(ap[k],bp[k]) for k in ap));self.assertTrue(torch.equal(b,a+r.strength*r.penalty()))
            ga=torch.autograd.grad(a,args[:2],retain_graph=True);gb=torch.autograd.grad(b,args[:2]);self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))

    def test_box_head_still_receives_existing_gradients(self):
        r=self.reg();toy_step(self.scope,self.opt)
        self.assertTrue(all(self.scope.params[n].grad is not None for n in TRAIN_KEYS if n.startswith('box_head.')));r.check()

    def payload(self,r):
        toy_step(self.scope,self.opt)
        return make_payload(self.scope,self.opt,1,'SYNTHETIC-BINDING',preserver=r,epoch=0,next_group=1,rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=[]),scaler={})

    def test_checkpoint_roundtrip_preserves_objective_optimizer_and_next_update(self):
        r=self.reg();payload=self.payload(r);stream=io.BytesIO();torch.save(payload,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu',weights_only=True)
        model=fixture(self.reference);scope=FrozenScope(model,self.reference);opt=scope.optimizer();rr=ProjectionPreserver(model,self.reference,enabled=True,strength=2,expected_hashes=self.hashes)
        restore_candidate(scope,loaded,opt,'SYNTHETIC-BINDING',preserver=rr);self.assertEqual(rr.state(),r.state())
        def regularized_step(s,o,reg):
            o.zero_grad(set_to_none=True);base=sum(s.params[n].sum()/math.sqrt(s.params[n].numel()) for n in TRAIN_KEYS)
            (base+reg.strength*reg.penalty()).backward();s.gradients(o);o.step();s.check(o);reg.check()
        regularized_step(self.scope,self.opt,r);regularized_step(scope,opt,rr)
        self.assertTrue(all(torch.equal(scope.params[n],self.scope.params[n]) for n in TRAIN_KEYS))

    def test_resume_helper_restores_RNG_scaler_and_cursor_on_CPU(self):
        r=self.reg();payload=self.payload(r)
        class Scale:
            def load_state_dict(self,state):self.state=state
        scale=Scale();torch.rand(5);cursor=restore_training_state(self.scope,payload,self.opt,scale,'SYNTHETIC-BINDING',preserver=r)
        self.assertEqual(cursor,(0,1,1));self.assertEqual(scale.state,{});self.assertTrue(torch.equal(torch.get_rng_state(),payload['rng']['cpu']))

    def test_checkpoint_changed_strength_enable_reference_or_binding_rejected_before_copy(self):
        r=self.reg();payload=self.payload(r);before=head_weights(self.model)
        for key,value in (('lambda_value',3),('enabled',False),('reference_hashes',{})):
            p=copy.deepcopy(payload);p['projection_preservation'][key]=value
            with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt,'SYNTHETIC-BINDING',preserver=r)
        with self.assertRaises(ValueError):restore_candidate(self.scope,payload,self.opt,'OTHER',preserver=r)
        self.assertTrue(all(torch.equal(before[n],head_weights(self.model)[n]) for n in HEAD_KEYS))

    def test_legacy_checkpoint_missing_preservation_or_bad_cursor_rejected(self):
        r=self.reg();p=self.payload(r);bad=copy.deepcopy(p);bad.pop('projection_preservation')
        with self.assertRaises(ValueError):restore_candidate(self.scope,bad,self.opt,'SYNTHETIC-BINDING',preserver=r)
        p['next_group']=2
        with self.assertRaisesRegex(ValueError,'cursor'):restore_candidate(self.scope,p,self.opt,'SYNTHETIC-BINDING',preserver=r)


class ContractTests(unittest.TestCase):
    def test_single_disabled_configuration_and_unchanged_C11_settings(self):
        cfg=specification();old=read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json')
        self.assertFalse(cfg['training_authorized']);self.assertFalse(cfg['projection_preservation']['enabled']);self.assertEqual(cfg['experiment_count'],1)
        for key in ('initialization','optimization_settings','data','evaluation','trainable_keys','limits'):self.assertEqual(cfg[key],old[key])

    def test_lambda_uses_only_optimization_photo_evidence(self):
        proposal=read(REPORT12/'lambda-proposal.json');_,train,dev=verify_preparation();eligible={r['image_id'] for r in train}
        self.assertTrue(all(r['image_id'] in eligible for r in proposal['pairs']));self.assertFalse(proposal['development_metrics_used']);self.assertFalse(proposal['hyperparameter_search'])
        self.assertAlmostEqual(proposal['lambda_value']*proposal['unweighted_endpoint_penalty'],proposal['training_probe_classification_reduction'])

    def test_disabled_launcher_never_spawns_worker_or_creates_destination(self):
        import launch12
        with patch.object(launch12.subprocess,'Popen') as spawn:
            with self.assertRaises(PermissionError):launch12.run(REPORT12/'experiment-disabled.json')
            spawn.assert_not_called()
        self.assertFalse(FUTURE.exists());self.assertFalse(torch.cuda.is_initialized())

    def test_C11_approval_cannot_authorize_C12(self):
        cfg=specification();cfg['training_authorized']=True
        with self.assertRaises(PermissionError):require_training_approval(cfg,read(C11_RUN/'approval-record.json'))

    def test_changed_lambda_targets_or_optimizer_rejected(self):
        for section,key,value in (('projection_preservation','lambda_value',1),('projection_preservation','selected_keys',['box_head.dense0.weight']),('optimization_settings','learning_rate',3e-5)):
            cfg=specification();cfg[section][key]=value
            with self.assertRaises(PermissionError):check_spec(cfg)

    def test_complete_bound_activation_contract_without_execution(self):
        cfg=specification();cfg['training_authorized']=True;cfg['projection_preservation']['enabled']=True;cfg['source_configuration'].update(enabled=True,approved=True);cfg['code_hashes']=required_code_pins()
        # Synthetic unit-test approval exists only in memory, never in the ledger.
        approval=dict(user_training_authorized=True,config_semantic_sha256=digest(cfg),user_reference='SYNTHETIC CPU CONTRACT TEST ONLY',max_optimizer_updates=75,max_epochs=5,max_GPU_seconds=600,max_VRAM_bytes=int(6.5*2**30))
        with patch('config12.verify_preparation',return_value=None):self.assertTrue(require_training_approval(cfg,approval))
        cfg['code_hashes'].pop(next(iter(cfg['code_hashes'])))
        with self.assertRaises(PermissionError):check_spec(cfg)

    def test_worker_source_is_exact_documented_C11_delta(self):
        from build_worker12 import transformed_source
        self.assertEqual((HERE12/'future_worker12.py').read_text(),transformed_source())

    def test_future_worker_paths_preserve_input_contract_and_isolate_outputs(self):
        import future_worker12
        self.assertEqual(future_worker12.OUT,ROOT/'data/open-images/preparation/phase10c10-v1')
        self.assertEqual(future_worker12.FUTURE,FUTURE);self.assertNotEqual(FUTURE,C11_RUN)

    def test_future_metrics_do_not_run_without_authorized_outputs(self):
        from evaluate12 import require_complete
        with self.assertRaises(FileNotFoundError):require_complete()

    def test_standalone_imports_do_not_initialize_CUDA(self):
        code='import launch12,future_worker12,evaluate12,torch;assert not torch.cuda.is_initialized()'
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c',code],cwd=HERE12,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
