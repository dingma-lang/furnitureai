"""CPU-only synthetic module/optimizer tests; no real image inference/training."""
import unittest,copy,io
from unittest.mock import patch
import numpy as np
import torch
from common10 import *
from freeze10 import FrozenScope,head_weights,make_payload,restore_candidate,verify_initial_heads
from config10 import specification,check_spec,require_training_approval
from attribution10 import linear_layernorm_envelope,head_combinations,shapley_components,require_feature_cache

def fixture(reference=None):
    torch.manual_seed(6106) # Same synthetic frozen backbone on every restore fixture.
    m=torch.nn.Module();m.class_head=torch.nn.Module();m.box_head=torch.nn.Module();m.backbone=torch.nn.Linear(2,2);m.objectness_head=torch.nn.Linear(2,1)
    m.class_head.dense0=torch.nn.Linear(768,512);m.class_head.logit_shift=torch.nn.Linear(768,1);m.class_head.logit_scale=torch.nn.Linear(768,1)
    m.box_head.dense0=torch.nn.Linear(768,768);m.box_head.dense1=torch.nn.Linear(768,768);m.box_head.dense2=torch.nn.Linear(768,4)
    if reference is not None:
        with torch.no_grad():
            for n,p in m.named_parameters():
                if n in reference:p.copy_(reference[n])
    return m

def toy_step(scope,opt):
    # Synthetic objective on isolated test tensors, not furniture/model training.
    opt.zero_grad(set_to_none=True);sum(scope.params[n].sum()/math.sqrt(scope.params[n].numel()) for n in TRAIN_KEYS).backward();scope.gradients(opt);opt.step();scope.check(opt)

class ScopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6106);torch.set_num_threads(2);self.model=fixture();self.reference=head_weights(self.model);self.scope=FrozenScope(self.model,self.reference);self.opt=self.scope.optimizer()
    def test_exact_frozen_identities_and_count(self):
        self.assertEqual({n for n,p in self.model.named_parameters() if p.requires_grad},TRAIN_KEYS);self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad),TRAIN_COUNT);self.assertEqual(len(CALIBRATION_KEYS),4)
    def test_optimizer_excludes_all_four_calibration_tensors(self):
        ids={id(p) for g in self.opt.param_groups for p in g['params']};self.assertFalse(ids&{id(self.scope.params[n]) for n in CALIBRATION_KEYS});self.assertEqual(len(ids),8)
    def test_frozen_gradients_remain_none(self):
        toy_step(self.scope,self.opt);self.assertTrue(all(self.scope.params[n].grad is None for n in self.scope.frozen_names))
    def test_persistent_across_training_eval_and_updates(self):
        for _ in range(3):self.scope.train_mode();toy_step(self.scope,self.opt);self.scope.eval_mode();self.scope.train_mode()
        self.scope.check(self.opt,hash_frozen=True);self.assertTrue(all(torch.equal(self.scope.params[n].cpu(),self.reference[n]) for n in CALIBRATION_KEYS))
    def test_full_75_synthetic_updates_preserve_calibration(self):
        for _ in range(75):self.scope.train_mode();toy_step(self.scope,self.opt);self.scope.eval_mode()
        self.scope.check(self.opt,hash_frozen=True);p=make_payload(self.scope,self.opt,75,'SYNTHETIC-TEST-ONLY');self.assertTrue(all(torch.equal(p['weights'][n],self.reference[n]) for n in CALIBRATION_KEYS));self.assertTrue(all(int(s['step'])==75 for s in self.opt.state.values()))
    def test_legacy_reenable_is_detected_not_silently_repaired(self):
        from expanded import configure_heads
        configure_heads(self.model)
        with self.assertRaises(ValueError):self.scope.train_mode()
    def test_stray_zero_frozen_gradient_is_rejected(self):
        self.scope.params['class_head.logit_shift.bias'].grad=torch.zeros_like(self.scope.params['class_head.logit_shift.bias'])
        with self.assertRaises(ValueError):self.scope.check(self.opt)
    def test_optimizer_with_frozen_parameter_rejected(self):
        self.opt.add_param_group({'params':[self.scope.params['class_head.logit_scale.bias']],'param_names':['class_head.logit_scale.bias']})
        with self.assertRaises(ValueError):self.scope.check(self.opt)
    def test_frozen_value_mutation_rejected(self):
        with torch.no_grad():self.scope.params['class_head.logit_shift.bias'].add_(1)
        with self.assertRaises(ValueError):self.scope.check()
    def test_full_backbone_mutation_rejected(self):
        with torch.no_grad():self.model.backbone.weight.add_(1)
        with self.assertRaises(ValueError):self.scope.check(hash_frozen=True)
    def test_changed_optimizer_hyperparameter_rejected(self):
        self.opt.param_groups[0]['lr']=3e-5
        with self.assertRaises(ValueError):self.scope.check(self.opt)
    def test_nonfinite_trainable_gradient_rejected(self):
        for n in TRAIN_KEYS:self.scope.params[n].grad=torch.ones_like(self.scope.params[n])
        self.scope.params['class_head.dense0.bias'].grad[0]=float('nan')
        with self.assertRaises(ValueError):self.scope.gradients(self.opt)
    def test_pristine_checkpoint_initialization(self):
        p=make_payload(self.scope,self.opt,0,'synthetic-binding');self.assertFalse(p['optimizer']['state']);self.assertEqual(set(p['weights']),HEAD_KEYS)
    def test_checkpoint_CPU_serialization_roundtrip(self):
        toy_step(self.scope,self.opt);p=make_payload(self.scope,self.opt,1,'synthetic-binding');b=io.BytesIO();torch.save(p,b);b.seek(0);loaded=torch.load(b,map_location='cpu',weights_only=True)
        m=fixture(self.reference);s=FrozenScope(m,self.reference);o=s.optimizer();restore_candidate(s,loaded,o,'synthetic-binding');self.assertTrue(all(torch.equal(head_weights(m)[n],p['weights'][n]) for n in HEAD_KEYS));s.check(o,hash_frozen=True)
    def test_checkpoint_legacy_inference_head_compatibility(self):
        from expanded import configure_heads,load_delta
        toy_step(self.scope,self.opt);p=make_payload(self.scope,self.opt,1,'fixture');m=fixture();configure_heads(m);load_delta(m,p);self.assertTrue(all(torch.equal(head_weights(m)[n],p['weights'][n]) for n in HEAD_KEYS))
    def test_checkpoint_calibration_mismatch_rejected(self):
        p=make_payload(self.scope,self.opt,0,'fixture');p['weights']['class_head.logit_shift.bias']+=1
        with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt)
    def test_checkpoint_optimizer_bad_lr_rejected_before_copy(self):
        toy_step(self.scope,self.opt);p=make_payload(self.scope,self.opt,1,'fixture');p['optimizer']['param_groups'][0]['lr']=.1;m=fixture(self.reference);s=FrozenScope(m,self.reference);o=s.optimizer();before=head_weights(m)
        with self.assertRaises(ValueError):restore_candidate(s,p,o)
        self.assertTrue(all(torch.equal(head_weights(m)[n],before[n]) for n in HEAD_KEYS))
    def test_checkpoint_nonfinite_moment_rejected(self):
        toy_step(self.scope,self.opt);p=make_payload(self.scope,self.opt,1,'fixture');p['optimizer']['state'][0]['exp_avg'].flatten()[0]=float('nan')
        with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt)
    def test_wrong_binding_rejected(self):
        p=make_payload(self.scope,self.opt,0,'old')
        with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt,'different')
    def test_wrong_training_scope_and_shape_rejected(self):
        p=make_payload(self.scope,self.opt,0,'fixture');p['weights']['class_head.dense0.weight']=torch.zeros(1)
        with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt)
    def test_over_budget_checkpoint_rejected(self):
        p=make_payload(self.scope,self.opt,0,'fixture');p['updates']=76
        with self.assertRaises(ValueError):restore_candidate(self.scope,p,self.opt)

class AttributionTests(unittest.TestCase):
    def test_layernorm_linear_bound_contains_synthetic_vectors(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(100,8));u=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-5);gamma=rng.normal(size=8);beta=rng.normal(size=8);w=rng.normal(size=8);bias=.3;b=linear_layernorm_envelope(w,bias,gamma,beta);values=(u*gamma+beta)@w+bias;self.assertTrue((values<=b['max']+1e-10).all());self.assertTrue((values>=b['min']-1e-10).all())
    def test_layernorm_bound_removes_impossible_mean_direction(self):
        b=linear_layernorm_envelope(np.ones(8),0,np.ones(8),np.zeros(8));self.assertEqual(b['radius'],0)
    def test_missing_activations_never_filled(self):
        with self.assertRaises(FileNotFoundError):require_feature_cache(OUT/'NONEXISTENT-FEATURE-CACHE.npz')
    def test_three_group_decomposition_closes_with_interactions(self):
        c={''.join(map(str,b)):np.ones((2,7))*((1+b[0])+(2+3*b[1]))*(4+2*b[2]) for b in __import__('itertools').product((0,1),repeat=3)};p=shapley_components(c);np.testing.assert_allclose(sum(p.values()),c['111']-c['000'])
    def test_missing_counterfactual_combination_rejected(self):
        with self.assertRaises(ValueError):shapley_components({'000':np.zeros((2,7))})
    def test_installed_class_head_matches_CPU_formula(self):
        from transformers.models.owlv2.configuration_owlv2 import Owlv2Config
        from transformers.models.owlv2.modeling_owlv2 import Owlv2ClassPredictionHead
        torch.manual_seed(8);config=Owlv2Config.from_dict(read(BASE/'config.json'));module=Owlv2ClassPredictionHead(config);state=original_heads();module.load_state_dict({n.removeprefix('class_head.'):t for n,t in state.items() if n.startswith('class_head.')});f=torch.randn(4,768);text=torch.randn(7,512)
        with torch.no_grad():actual,_=module(f.unsqueeze(0),text.unsqueeze(0),torch.ones((1,7),dtype=torch.bool))
        combos=head_combinations(f,text,state,state);np.testing.assert_allclose(actual[0].numpy(),combos['000'],atol=1e-5,rtol=1e-5)
    def test_fixed_calibration_does_not_prevent_projection_inflation(self):
        # Pure2D synthetic cosine example, never presented as real furniture features.
        theta=np.linspace(-.08,.08,7);text=np.stack([np.sin(theta),np.cos(theta)],1);before=text@np.array([1.,0.]);after=text@np.array([0.,1.]);shift=-1.;scale=2.;self.assertTrue(((after+shift)*scale>(before+shift)*scale).all())

class ContractTests(unittest.TestCase):
    def test_configuration_deterministic_and_single_candidate(self):
        a=specification();b=specification();self.assertEqual(digest(a),digest(b));self.assertEqual(a['experiment_count'],1);self.assertFalse(a['training_authorized']);self.assertEqual(a['limits']['max_optimizer_updates'],75)
    def test_only_parameter_freeze_changes_training_semantics(self):
        cfg=specification();old=read(C8/'binding.json')['configs']['B-corrected']
        self.assertTrue(all(v==old[k] for k,v in cfg['optimization_settings'].items()));self.assertEqual(cfg['trainable_keys'],sorted(HEAD_KEYS-CALIBRATION_KEYS));self.assertEqual(cfg['data']['new_annotations'],0)
    def test_source_data_integrity_and_isolation(self):
        prepared,train,dev=verify_inputs();self.assertEqual(len(train),59);self.assertEqual(len(dev),19)
        for k in ('image_id','sha256','identity_group'):self.assertFalse({r[k] for r in train}&{r[k] for r in dev})
    def test_optimizer_and_scope_regression_records_exist(self):
        audit=read(REPORT/'checkpoint-parameter-audit.json')
        for arm in ('A','B'):
            for n in CALIBRATION_KEYS:self.assertEqual(audit['arms'][arm][n]['optimizer_steps'],75);self.assertTrue(audit['arms'][arm][n]['gradient_probes'])
    def test_actual_original_heads_initialize_without_drift(self):
        reference=original_heads();m=fixture(reference);verify_initial_heads(m,reference);s=FrozenScope(m,reference);s.check(hash_frozen=True)
    def test_actual_C8_adapted_heads_not_original_initialization(self):
        reference=original_heads();p=read(C8/'B-corrected/complete.json')['final_checkpoint'];state=torch.load(C8/p['file'],map_location='cpu',weights_only=True);m=fixture(state['weights'])
        with self.assertRaises(ValueError):verify_initial_heads(m,reference)
    def test_disabled_gate_precedes_worker_and_directory_creation(self):
        import launch10
        with patch.object(launch10.subprocess,'Popen') as launch:
            with self.assertRaises(PermissionError):launch10.run(REPORT/'experiment-disabled.json')
            launch.assert_not_called()
        self.assertFalse(FUTURE.exists());self.assertFalse(torch.cuda.is_initialized())
    def test_enabled_flag_without_approval_is_insufficient(self):
        cfg=read(REPORT/'experiment-disabled.json');cfg['training_authorized']=True
        with self.assertRaises(PermissionError):require_training_approval(cfg)
    def test_semantic_configuration_drift_rejected(self):
        cfg=specification();cfg['optimization_settings']['learning_rate']=3e-5
        with self.assertRaises(PermissionError):check_spec(cfg)
    def test_no_future_metrics_without_completed_run(self):
        from future_diagnostics10 import require_complete
        with self.assertRaises(FileNotFoundError):require_complete()
    def test_future_sampling_identical_C8(self):
        from common8 import grouped_epoch
        train,dev=subsets();orders=[[[r['image_id'] for r in g] for g in grouped_epoch(train,e)] for e in range(5)];self.assertEqual(sum(map(len,orders)),75);self.assertEqual(orders,[[[r['image_id'] for r in g] for g in grouped_epoch(train,e)] for e in range(5)])
    def test_loss_masking_unchanged(self):
        from loss8 import objective
        from expanded import xyxy_to_cxcywh
        row=dict(image_id='SYNTHETIC',width=100,height=100,annotations=[dict(label='chair',box=[.7,.7,.9,.9],ignore=False)],explicit_native_negative={});z=torch.zeros((2,7),requires_grad=True);boxes=xyxy_to_cxcywh(torch.tensor([[.7,.7,.9,.9],[.1,.1,.2,.2]])).requires_grad_();ev=[dict(image_id='SYNTHETIC',category='chair',box=[.05,.05,.3,.3],candidate_id='SYNTHETIC-REGION')]
        _,parts,s=objective(z,boxes,row,ev,[],True);g=torch.autograd.grad(parts['added'],z)[0];self.assertEqual(s['positive_boxes'],1);self.assertEqual(s['added_active_Chair_terms'],1);self.assertEqual(int((g!=0).sum()),1);self.assertGreater(g[1,1],0)

if __name__=='__main__':unittest.main()
