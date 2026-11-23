"""CPU-only authorization,coordinate,output and no-update regression tests."""
import ast,unittest
from unittest.mock import patch
import numpy as np
from common7 import *

class AuthorizationTests(unittest.TestCase):
    def test_forward_limit_consumed_before_execution(self):
        s=dict(attempted_forwards=58);consume_forward(s);self.assertEqual(s['attempted_forwards'],59)
        with self.assertRaises(RuntimeError):consume_forward(s)
        self.assertEqual(s['attempted_forwards'],59)

    def test_time_memory_and_storage_limits(self):
        s=dict(attempted_forwards=59,actual_forwards=59)
        budget(s,179.9,LIMITS['VRAM_bytes'],LIMITS['output_bytes'])
        for args in [(s,180,0,0),(s,1,LIMITS['VRAM_bytes']+1,0),(s,1,0,LIMITS['output_bytes']+1),(dict(attempted_forwards=60,actual_forwards=60),1,0,0)]:
            with self.assertRaises(RuntimeError):budget(*args)

    def test_actual_extra_forward_rejected(self):
        with self.assertRaises(RuntimeError):budget(dict(attempted_forwards=1,actual_forwards=2),1,0,0)

    def test_development_and_unknown_identity_rejected(self):
        _,train,dev=context()
        for ident in [dev[0]['image_id'],'reserved:unknown']:
            with self.assertRaises(PermissionError):require_train(ident,train)

    def test_binding_exact59_ids_and_no_updates(self):
        cfg,_,train=inputs();self.assertEqual(cfg['image_ids'],list(train));self.assertEqual(len(train),59)
        self.assertEqual(cfg['optimizer_updates'],0);self.assertEqual(cfg['backward_calls'],0)
        self.assertEqual(cfg['checkpoint_sha256'],CHECKPOINT_SHA)

    def test_worker_contains_no_training_or_download_calls(self):
        tree=ast.parse((HERE/'worker7.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse([n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','optimizer','download','urlopen','train')])
        self.assertEqual(len([n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='model']),1)
        self.assertTrue([n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='inference_mode'])

    def test_launcher_rejects_retries_by_exclusive_marker(self):
        code=(HERE/'launch7.py').read_text()
        self.assertIn("RUN/'launch-once.json'",code)
        self.assertNotIn('for attempt',code)
        self.assertIn('SIGKILL',code)

class CoordinateTests(unittest.TestCase):
    def fixture(self):
        return np.zeros((3600,7),np.float32),np.tile(np.array([[.75,.5625,.5,.375]],np.float32),(3600,1))

    def test_all_queries_and_seven_category_scores_retained(self):
        z,b=self.fixture();v=raw_values(z,b,640,480)
        np.testing.assert_array_equal(v['query_ids'],np.arange(3600))
        self.assertEqual(v['scores'].shape,(3600,7));self.assertTrue((v['scores']==.5).all())

    def test_bottom_right_padding_mapping_landscape(self):
        z,b=self.fixture();v=raw_values(z,b,640,480)
        np.testing.assert_allclose(v['original_xyxy_pixels'][0],[320,240,640,480],rtol=0,atol=1e-12)

    def test_portrait_not_center_padded(self):
        z,b=self.fixture();b[:]=[.5625,.75,.375,.5];v=raw_values(z,b,480,640)
        np.testing.assert_allclose(v['original_xyxy_pixels'][0],[240,320,480,640],rtol=0,atol=1e-12)

    def test_outside_frame_not_clipped_into_safe_negatives(self):
        z,b=self.fixture();b[:]=[.9,.9,.4,.4];v=raw_values(z,b,640,480)
        self.assertGreater(v['original_xyxy_normalized'][0,3],1)

    def test_invalid_shape_or_nonfinite_rejected(self):
        z,b=self.fixture()
        with self.assertRaises(ValueError):raw_values(z[:1],b,640,480)
        z[0,0]=np.nan
        with self.assertRaises(ValueError):raw_values(z,b,640,480)

class StorageAndStateTests(unittest.TestCase):
    def test_atomic_no_replacement_and_roundtrip(self):
        p=RUN/'cpu-test-fixtures'/('immutable-'+uuid.uuid4().hex+'.json');emit(p,{'fixture':True})
        self.assertEqual(read(p),{'fixture':True})
        with self.assertRaises(PermissionError):emit(p,{'fixture':False})
        self.assertEqual(read(p),{'fixture':True})

    def test_historical_path_is_rejected_before_open(self):
        with self.assertRaises(PermissionError):atomic(BASE/'forbidden-E7-test',b'')

    def test_size_guard_prevents_new_file(self):
        p=RUN/'cpu-test-fixtures'/('not-written-'+uuid.uuid4().hex)
        with patch('common7.allocated',return_value=LIMITS['output_bytes']):
            with self.assertRaises(RuntimeError):atomic(p,b'x')
        self.assertFalse(p.exists())

    def test_buffers_and_parameter_mutation_change_fingerprint(self):
        import torch
        sys.path.insert(0,str(ROOT/'tools/phase10c19-v1'))
        from telemetry19 import model_fingerprint
        m=torch.nn.Linear(2,2);m.register_buffer('test_buffer',torch.zeros(1))
        before=model_fingerprint(m);m.test_buffer.add_(1)
        self.assertNotEqual(model_fingerprint(m),before)
        before=model_fingerprint(m)
        with torch.no_grad():m.weight.add_(1)
        self.assertNotEqual(model_fingerprint(m),before)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
