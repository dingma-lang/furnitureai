"""CPU-only recovery boundary,telemetry,hash and all24 cached-loss tests."""
import ast,copy,subprocess,unittest
from unittest.mock import patch
import torch
import common15r,launch15r
from common15r import *
from gpu15 import state_fingerprint
# Only these two old CPU classes;SafetyTests would write into historical outputs.
from test_smoke15 import LossTests,CachedRegressionTests


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.folder=RUNR/('CPU-fixture-'+uuid.uuid4().hex[:8]);cls.folder.mkdir()
    def test_exact20_indices_accepted(self):validate_indices(list(range(5,25)))
    def test_reused_or_reserved_indices_rejected(self):
        for v in [list(range(1,21)),list(range(6,26)),list(range(5,24)),[5]*20,['5']+list(range(6,25))]:
            with self.assertRaises(PermissionError):validate_indices(v)
    def test_four_prior_records_and_seals_immutable(self):
        cfg,records,seals=verify_prior();self.assertEqual(len(records),4);self.assertEqual([r['image_id'] for r in records],cfg['photo_ids'][:4]);self.assertGreater(seals['private_files'],60)
    def test_new_inputs_only20_eligible_optimization_photos(self):
        cfg,_,rows,_=inputs_r();self.assertEqual([r['image_id'] for r in rows],cfg['photo_ids'][4:]);self.assertEqual(len({r['identity_group'] for r in rows}),20);self.assertTrue(all(r['internal_partition']=='optimization' for r in rows));self.assertFalse(cfg['training_authorized'])
    def test_changed_limits_or_authorization_fail_closed(self):
        original=read(RUNR/'authorized-config.json')
        for changed in [dict(original,recovery_authorized=False),dict(original,training_authorized=True),dict(original,limits=dict(original['limits'],new_forwards=21))]:
            with patch.object(common15r,'read',return_value=changed):
                with self.assertRaises(PermissionError):inputs_r()
    def test_atomic_writer_refuses_overwrite(self):
        p=self.folder/'immutable.json';emit_r(p,dict(synthetic=True))
        with self.assertRaises(FileExistsError):emit_r(p,dict(synthetic=False))
        self.assertTrue(read(p)['synthetic'])
    def test_output_path_and_budget_rejected(self):
        with self.assertRaises(PermissionError):emit_r(PRIOR_RUN/'never-write.json',{})
        with patch.object(common15r,'MAX_BYTES',1):
            with self.assertRaises(RuntimeError):emit_r(self.folder/'never-created.json',{})
    def test_model_fingerprint_covers_nonpersistent_buffers(self):
        model=torch.nn.Linear(2,2);model.register_buffer('nonpersistent',torch.tensor([1.]),persistent=False);before=state_fingerprint(model);model.nonpersistent.add_(1);self.assertNotEqual(before,state_fingerprint(model));self.assertNotIn('nonpersistent',model.state_dict());self.assertIn('nonpersistent',before['tensors'])
    def test_telemetry_selects_latest_saved_record(self):
        for i in [1,3,2]:emit_r(self.folder/('telemetry-'+str(i)+'.json'),dict(elapsed_seconds=i,last_verified_model_sha256='SYNTHETIC',forwards=i))
        with patch.object(launch15r,'RUNR',self.folder):self.assertEqual(launch15r.latest_telemetry()['forwards'],3)
    def test_worker_has_no_optimizer_step_or_new_loss_implementation(self):
        source=(HERER/'gpu15r.py').read_text();tree=ast.parse(source);calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)];self.assertFalse(set(calls)&{'step','backward','optimizer','zero_grad'});self.assertIn('from loss15 import controlled_losses,gradients',source);self.assertIn("state_fingerprint(model)",source)
    def test_imports_never_initialize_CUDA_or_run_model(self):
        r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-c','import gpu15r,launch15r,torch;assert not torch.cuda.is_initialized()'],cwd=HERER,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
    def test_old_missing_measurements_not_filled(self):
        old=read(PRIOR_REPORT/'results.json');self.assertIsNone(old['peak_allocated_bytes']);self.assertIsNone(old['peak_observed_device_bytes']);self.assertFalse((PRIOR_RUN/'complete.json').exists())


if __name__=='__main__':unittest.main()
