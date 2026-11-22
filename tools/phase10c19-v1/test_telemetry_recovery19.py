"""CPU-only failure persistence, recovery state and preservation tests."""
import copy
import tempfile
import unittest
from unittest.mock import patch
import torch
from common19 import *
from recovery19 import validate_payload,remaining_sequence,restore_cpu_fixture
from guard19 import compose_checked
from test_guard19 import fixture
import telemetry19
from telemetry19 import FailureTelemetry,gradient_snapshot,model_fingerprint


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.folder=DATA19/('synthetic-telemetry-'+uuid.uuid4().hex[:8]);self.folder.mkdir()
    def test_atomic_before_boundary_and_failure_with_exact_diagnostics(self):
        model=torch.nn.Linear(2,1);before=model_fingerprint(model)
        t=FailureTelemetry(context=dict(synthetic_test=True,current_photo_id='SYNTHETIC-PHOTO',current_source_group_id='SYNTHETIC-GROUP',photo_ids=['SYNTHETIC-PHOTO'],source_group_ids=['SYNTHETIC-GROUP'],epoch_zero_based=1,accumulation_group_zero_based=13,updates_committed=28,next_update=29),model_provider=lambda:model_fingerprint(model),optimizer_provider=lambda:dict(available_checkpoint='synthetic-reference',committed_updates=28),resource_provider=lambda:dict(CPU_only=True,GPU_memory_bytes=None),folder=self.folder)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with t.failure_scope():
                a=fixture();t.boundary(a);compose_checked(a);raise RuntimeError('injected synthetic failure')
        boundary=list(self.folder.glob('boundary-*.json'));failure=list(self.folder.glob('failure-*.json'));self.assertEqual((len(boundary),len(failure)),(1,1))
        d=read(failure[0]);g=d['gradients'];self.assertIsNotNone(g['positive_norm']);self.assertIsNotNone(g['negative_pre_norm']);self.assertIsNotNone(g['negative_post_norm']);self.assertTrue(g['cap_active']);self.assertIsNotNone(g['numerical_tolerance']);self.assertIsNotNone(g['negative_positive_cosine']);self.assertFalse(g['nonfinite_gradient'])
        self.assertEqual(d['context']['next_update'],29);self.assertEqual(d['model_state_fingerprint']['value'],before);self.assertEqual(d['last_boundary']['sha256'],sha(boundary[0]));self.assertIn('injected synthetic failure',d['stack_trace']);self.assertFalse(d['automatic_retry']);self.assertFalse(d['photograph_pixels_saved']);self.assertEqual(model_fingerprint(model),before)
    def test_failure_before_measurements_marks_unknowns(self):
        t=FailureTelemetry(folder=self.folder)
        with self.assertRaises(ValueError):
            with t.failure_scope():raise ValueError('startup synthetic failure')
        d=read(next(self.folder.glob('failure-*.json')));self.assertIsNone(d['gradients']['positive_norm']);self.assertIsNone(d['context']['current_photo_id']);self.assertFalse(d['model_state_fingerprint']['available']);self.assertFalse(d['optimizer_state_availability']['available'])
    def test_provider_failure_does_not_destroy_original_exception(self):
        def broken():raise RuntimeError('provider unavailable')
        t=FailureTelemetry(model_provider=broken,folder=self.folder)
        with self.assertRaisesRegex(ValueError,'original'):
            with t.failure_scope():raise ValueError('original synthetic failure')
        d=read(next(self.folder.glob('failure-*.json')));self.assertFalse(d['model_state_fingerprint']['available']);self.assertIn('provider unavailable',d['model_state_fingerprint']['reason'])
    def test_write_failure_surfaces_without_swallowing_exception(self):
        t=FailureTelemetry(folder=self.folder)
        with patch.object(telemetry19,'emit19',side_effect=OSError('injected disk failure')):
            with self.assertRaisesRegex(ValueError,'original') as caught:
                with t.failure_scope():raise ValueError('original synthetic exception')
        self.assertTrue(any('could not be persisted' in n for n in caught.exception.__notes__))
    def test_nonfinite_diagnostic_is_serializable_and_flagged(self):
        a=fixture();a.negative[next(iter(a.negative))][0]=float('nan');d=gradient_snapshot(a);self.assertTrue(d['nonfinite_gradient'])
        emit19(self.folder/'nonfinite.json',telemetry19.clean(d));self.assertIn('Nonfinite',read(self.folder/'nonfinite.json')['unavailable_reason'])
    def test_atomic_outputs_never_overwrite_or_escape(self):
        p=self.folder/'protected.json';emit19(p,dict(synthetic=True));h=sha(p)
        with self.assertRaises(FileExistsError):emit19(p,{})
        with self.assertRaises(PermissionError):emit19(CHECKPOINT28,{})
        self.assertEqual(sha(p),h)
    def test_integrated_nonfinite_guard_persists_actual_failure(self):
        a=fixture();a.negative[next(iter(a.negative))][0]=float('inf');t=FailureTelemetry(context=dict(synthetic_test=True,updates_committed=28,next_update=29),folder=self.folder)
        with self.assertRaises(ValueError):telemetry19.compose_with_telemetry(a,t)
        d=read(next(self.folder.glob('failure-*.json')));self.assertTrue(d['gradients']['nonfinite_gradient']);self.assertIsNotNone(d['last_boundary']);self.assertIn('ValueError',d['stack_trace'])
    def test_genuine_violation_diagnostics_are_in_failure_receipt(self):
        from guard19 import verify_direct,CapViolation,CLASS_KEYS,norm64
        p={k:torch.ones(2) for k in CLASS_KEYS};n={k:torch.full((2,),100.) for k in CLASS_KEYS};post={k:torch.ones(2) for k in CLASS_KEYS};coefficient=.25*norm64(p)/norm64(n);t=FailureTelemetry(folder=self.folder)
        with self.assertRaises(CapViolation):
            with t.failure_scope():verify_direct(p,n,post,coefficient,1)
        d=read(next(self.folder.glob('failure-*.json')));self.assertGreater(d['guard_exception_diagnostics']['negative_post_norm'],d['guard_exception_diagnostics']['expected_max_negative_norm'])


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=read(RUN18/'binding.json');cls.s=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True);ref=read(next((RUN18/'checkpoints').glob('update-000-*.receipt.json')));cls.initial=torch.load(RUN18/ref['file'],map_location='cpu',weights_only=True)
    def test_actual_checkpoint28_state_and_rng_presence(self):
        d=validate_payload(self.s,self.cfg,self.initial);self.assertEqual(d['optimizer_step'],28);self.assertEqual(d['optimizer_states'],12);self.assertTrue(d['CPU_rng_restored_locally']);self.assertTrue(d['CUDA_rng_available']);self.assertFalse(d['CUDA_restore_executed']);self.assertFalse(d['scheduler_required'])
    def test_remaining_sequence_exact_deterministic_and_no_committed_replay(self):
        a=remaining_sequence(self.cfg,self.s);b=remaining_sequence(self.cfg,self.s);self.assertEqual(a,b);self.assertEqual(len(a),47);self.assertEqual([v['update'] for v in a],list(range(29,76)));self.assertEqual(sum(len(v['image_ids']) for v in a),184)
        for r in a:self.assertEqual(r['image_ids'],read(C8/'B-corrected'/('update-'+str(r['update']).zfill(3)+'.json'))['source_order'])
    def test_wrong_counter_cursor_or_binding_rejected(self):
        for bad in (dict(self.s,updates=29),dict(self.s,next_group=14),dict(self.s,binding_sha256='wrong')):
            with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
    def test_wrong_optimizer_hyperparameter_or_moment_rejected(self):
        bad=copy.deepcopy(self.s);bad['optimizer']['param_groups'][0]['lr']=3e-5
        with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
        bad=copy.deepcopy(self.s);bad['optimizer']['state'][0]['exp_avg'].flatten()[0]=float('nan')
        with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
    def test_wrong_scaler_rng_or_scope_rejected(self):
        bad=copy.deepcopy(self.s);bad['scaler']['_growth_tracker']=27
        with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
        bad=copy.deepcopy(self.s);bad['rng']['cuda']=[]
        with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
        bad=dict(self.s,weights={k:v for k,v in self.s['weights'].items() if 'logit_shift.bias' not in k})
        with self.assertRaises(ValueError):validate_payload(bad,self.cfg,self.initial)
    def test_CPU_head_fixture_restore_and_model_integrity(self):
        before=sha(CHECKPOINT28);r=restore_cpu_fixture(self.s);self.assertEqual(r['status'],'PASS');self.assertTrue(r['synthetic_frozen_parameters_unchanged']);self.assertEqual(r['optimizer_steps'],0);self.assertEqual(r['forward_calls'],0);self.assertEqual(sha(CHECKPOINT28),before)
    def test_all12_heads_including_calibration_remain_trainable(self):
        from test_preflight10 import fixture as model_fixture
        from expanded import configure_heads
        m=model_fixture();configure_heads(m);o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5)
        self.assertEqual({n for n,p in m.named_parameters() if p.requires_grad},HEAD_KEYS);self.assertEqual({id(p) for g in o.param_groups for p in g['params']},{id(p) for n,p in m.named_parameters() if n in HEAD_KEYS});self.assertTrue(all(p.grad is None for p in m.parameters()))
    def test_proposal_disabled_with_correct_budgets_and_no_executor(self):
        index=REPORT19/'recovery-proposal-status.json'
        p=read(ROOT/read(index)['active_proposal'] if index.exists() else REPORT19/'recovery-disabled.json');self.assertFalse(p['approved']);self.assertFalse(p['enabled']);self.assertFalse(p['GPU_authorized']);self.assertFalse(p['training_authorized']);self.assertFalse(p['runner_implemented']);self.assertEqual(p['updates'],list(range(29,76)));self.assertEqual(p['limits']['additional_GPU_seconds'],515);self.assertEqual(p['limits']['additional_forward_passes'],270);self.assertEqual(p['limits']['cumulative_actual_forward_passes'],452);self.assertFalse((ROOT/p['destination']).exists());self.assertEqual(p['guard_sha256'],sha(HERE19/'guard19.py'));self.assertEqual(p['telemetry_sha256'],sha(HERE19/'telemetry19.py'));self.assertFalse(torch.cuda.is_initialized())
    def test_original_and_historical_hashes_match(self):
        self.assertEqual(sha(CHECKPOINT28),CHECKPOINT28_SHA256);self.assertEqual(sha(BASE/'model.safetensors'),self.cfg['initialization']['SHA256']);self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
