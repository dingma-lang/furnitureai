"""CPU integration failures, gates, and source/mask/scope preservation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_numerics3 import fixture
from components3 import measure_components
from recovery3 import execute, validate
from stable3 import Capture, measured_vjp
from telemetry3 import Trace


def entry():
    return dict(row=dict(image_id='SYNTHETIC', historical_partition='train', internal_partition='optimization',
                         identity_group='SYNTHETIC-GROUP', sha256='a'*64, width=100, height=100,
                         annotations=[dict(label='chair', box=[.7,.7,.9,.9], ignore=False)],
                         explicit_native_negative={}),
                source_regions=[], legacy_protected=[], distill_regions=[], distill_protected=[])


class IntegrationTests(unittest.TestCase):
    def arguments(self, folder):
        head, _, _, _, s = fixture(torch.float16)
        e = entry()
        b = torch.tensor([[.8,.8,.2,.2], [.1,.1,.1,.1], [.4,.4,.1,.1], [.5,.5,.1,.1]], requires_grad=True)
        context = dict(image_id='SYNTHETIC', source_group='SYNTHETIC-GROUP',
                       source_sha256='a'*64, internal_partition='optimization')
        return (head, s, b, s.logits[0].clone(), b.detach().clone(), e, [], Path(folder)/'photo'), context

    def test_complete_components_preserve_annotations_and_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            args, context = self.arguments(folder)
            before = copy.deepcopy(args[5])
            calls = []
            result = measure_components(*args, context=context, enabled=True, guard=calls.append)
            self.assertEqual(args[5], before)
            self.assertEqual(len(result['positive']), 2)
            self.assertTrue((args[-1]/'positive.json').is_file())
            self.assertTrue((args[-1]/'distillation.json').is_file())
            self.assertTrue((args[-1]/'complete.json').is_file())
            self.assertNotIn('distillation-VJP-intent', calls)

    def test_positive_persisted_before_distillation_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            args, context = self.arguments(folder)
            mask = torch.zeros_like(args[3], dtype=torch.bool)
            mask[1,1] = True
            teacher = args[3].clone()
            teacher[mask] -= 2
            args = (*args[:3], teacher, *args[4:])
            fake_positive = (torch.ones_like(args[3]), [torch.ones_like(p) for p in args[0].dense0.parameters()])
            parts = dict(positive=torch.tensor(1.), distillation=torch.tensor(4.))
            with patch('components3.selective_mask', return_value=(mask, {'terms':1})), \
                 patch('components3.objective', return_value=(torch.tensor(5.), parts, {'base':{}})), \
                 patch('components3.measured_vjp', side_effect=[fake_positive, FloatingPointError('KD test failure')]):
                self.assertRaises(FloatingPointError, measure_components, *args,
                                  context=context, enabled=True, guard=lambda _:None)
            failure = json.loads((args[-1]/'failure.json').read_text())
            self.assertTrue(failure['positive_component_persisted'])
            self.assertEqual(failure['exception_type'], 'FloatingPointError')
            self.assertFalse((args[-1]/'complete.json').exists())

    def test_hook_exception_is_recorded_without_suppressing(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Trace(folder, enabled=True)
            p = torch.nn.Linear(2, 7)
            z = p(torch.ones(3,2))
            def bad_hook(_):
                raise RuntimeError('Synthetic hook exception')
            handle = p.weight.register_hook(bad_hook)
            with self.assertRaisesRegex(RuntimeError, 'Synthetic hook exception'):
                measured_vjp(z.square().mean(), z, torch.zeros(3,4,requires_grad=True),
                             [p.weight,p.bias], [], trace=trace)
            handle.remove()
            last = json.loads(sorted(Path(folder).glob('event-*.json'))[-1].read_text())
            self.assertEqual(last['exception_type'], 'RuntimeError')
            self.assertIn('Synthetic hook exception', last['stack_trace'])

    def test_source_group_or_hash_change_rejected_before_output(self):
        for key in ('image_id','source_group','source_sha256','internal_partition'):
            with tempfile.TemporaryDirectory() as folder:
                args, c = self.arguments(folder)
                c[key] = 'WRONG'
                self.assertRaises(PermissionError, measure_components, *args, context=c,
                                  enabled=True, guard=lambda _:None)
                self.assertFalse(args[-1].exists())

    def test_development_or_reserved_row_cannot_be_used(self):
        for partition in ('inner_development','reserved','frozen'):
            with tempfile.TemporaryDirectory() as folder:
                args, c = self.arguments(folder)
                args[5]['row']['internal_partition'] = partition
                self.assertRaises(PermissionError, measure_components, *args, context=c,
                                  enabled=True, guard=lambda _:None)
                self.assertFalse(args[-1].exists())

    def test_original_source_annotation_never_mutated_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            args, c = self.arguments(folder)
            before = copy.deepcopy(args[5])
            def guard(_):
                raise RuntimeError('Budget reached')
            self.assertRaises(RuntimeError, measure_components, *args, context=c, enabled=True, guard=guard)
            self.assertEqual(args[5], before)
            self.assertTrue((args[-1]/'failure.json').exists())

    def test_diagnostic_disabled_before_any_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args, c = self.arguments(folder)
            self.assertRaises(PermissionError, measure_components, *args, context=c, guard=lambda _:None)
            self.assertFalse(args[-1].exists())

    def test_no_overwrite_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            args, c = self.arguments(folder)
            args[-1].mkdir()
            self.assertRaises(FileExistsError, measure_components, *args, context=c,
                              enabled=True, guard=lambda _:None)

    def test_bad_snapshot_reconstruction_rejected(self):
        head, x, q, _, _ = fixture()
        # Deliberate incompatible head output; no tolerance relaxation allowed.
        handle = head.register_forward_hook(lambda m,a,out:(out[0]+1.,out[1]))
        try:
            with self.assertRaises(ValueError), Capture(head, enabled=True):
                head(x,q,torch.ones(1,7,dtype=torch.bool))
        finally:
            handle.remove()

    def test_recovery_proposal_and_execution_disabled(self):
        from verify3 import public_files
        self.assertTrue(all(not p.name.startswith('.') for p in public_files()))
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root/'reports/phase10e3-v1/recovery-disabled.json').read_text())
        self.assertEqual(validate(plan)['GPU_calls'], 0)
        self.assertRaises(PermissionError, execute, plan)
        for key in ('approved','enabled','training_authorized'):
            bad=copy.deepcopy(plan)
            bad[key]=True
            self.assertRaises(PermissionError, validate, bad)
        self.assertFalse((root/plan['destination']).exists())

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
