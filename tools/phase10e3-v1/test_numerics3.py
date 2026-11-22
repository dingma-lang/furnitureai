"""Synthetic CPU verification, never a reconstruction of missing E2 activations."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Owlv2Config
from transformers.models.owlv2.modeling_owlv2 import Owlv2ClassPredictionHead

from stable3 import Capture, Snapshot, observed_logits, measured_vjp, widened_vjp
from telemetry3 import Trace, stats, atomic_json, persist_component


def fixture(dtype=None):
    torch.manual_seed(6106)
    cfg = Owlv2Config(text_config={'hidden_size': 5}, vision_config={'hidden_size': 6})
    head = Owlv2ClassPredictionHead(cfg).eval()
    head.logit_shift.requires_grad_(False)
    head.logit_scale.requires_grad_(False)
    x, q = torch.randn(1, 4, 6), torch.randn(1, 7, 5)
    m = torch.ones(1, 7, dtype=torch.bool)
    context = torch.autocast('cpu', enabled=dtype is not None, dtype=dtype or torch.bfloat16)
    with Capture(head, enabled=True) as captured, context:
        z, _ = head(x, q, m)
    return head, x, q, z, captured.snapshot


def anchored(value, observed):
    """Exact observed value with the FP64 mathematical local derivative."""
    return observed.double() + (value - value.detach())


def reference(head, s, g):
    """Independent FP64 autograd at the same rounded intermediate values.

    Anchors describe finite-precision nodes, not a changed MSE objective.
    This is a reference-only CPU construction, not the implementation path.
    """
    w = head.dense0.weight.detach().double().requires_grad_()
    b = head.dense0.bias.detach().double().requires_grad_()
    x = s.x.to(s.raw.dtype).double()
    raw = anchored(torch.nn.functional.linear(x, w, b), s.raw)
    # Norm backward uses its SAVED, rounded result as the denominator. Merely
    # anchoring a fresh FP64 norm's output would retain the unrounded saved
    # norm inside autograd and test a different finite-precision graph.
    square = anchored(raw.square().sum(-1, keepdim=True), s.norm.double().square())
    n = torch.where(s.norm > 0, square.clamp_min(1e-300).sqrt(), square * 0)
    den = anchored(n + 1e-6, s.denominator)
    normalized = raw / den
    # Local einsum derivative sees its autocast input values; no weight-path
    # derivative depends on the image input's rounded contraction value.
    cosine = torch.einsum('bpd,bqd->bpq', normalized, s.query_contract.double())
    logits = cosine * s.scale.double()
    logits = torch.where(s.valid, logits, 0.)
    return torch.autograd.grad((logits * g.double()).sum(), (w, b))


class NumericsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def check_reference(self, dtype=None, magnitude=1.):
        h, _, _, _, s = fixture(dtype)
        g = torch.randn_like(s.logits) * magnitude
        actual = widened_vjp(s, g, h.dense0.weight)
        ref = reference(h, s, g)
        for a, b in zip(actual, ref):
            torch.testing.assert_close(a.double(), b, rtol=2e-5, atol=max(1e-6, magnitude * 2e-6))

    def test_fp32_matches_independent_fp64_reference(self):
        self.check_reference()

    def test_cpu_fp16_observed_nodes_match_fp64_reference(self):
        self.check_reference(torch.float16)

    def test_cpu_bf16_observed_nodes_match_fp64_reference(self):
        self.check_reference(torch.bfloat16)

    def test_large_adjoint_reference(self):
        self.check_reference(torch.float16, 1e5)

    def test_exact_forward_and_masks_unchanged(self):
        h, _, _, native, s = fixture(torch.float16)
        z = observed_logits(h, s, enabled=True)
        self.assertTrue(torch.equal(z, native))
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[:, 1, 1] = True
        before = mask.clone()
        teacher = z.detach().clone()
        teacher[mask] -= .002
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase10e1-v1'))
        from objectiveE import distill
        loss = distill(z[0], teacher[0], mask[0])
        gz, gp = measured_vjp(loss, z, torch.zeros(1, 4, 4, requires_grad=True),
                              [h.dense0.weight, h.dense0.bias], [])
        self.assertTrue(torch.equal(mask, before))
        self.assertTrue((gz[~mask] == 0).all())
        self.assertGreater(sum(float(g.norm()) for g in gp), 0)
        self.assertFalse(teacher.requires_grad)

    def derivative(self, delta=0., covered=True):
        h, _, _, _, s = fixture(torch.float16)
        z = observed_logits(h, s, enabled=True)
        mask = torch.zeros_like(z, dtype=torch.bool)
        if covered:
            mask[0, 1, 1] = True
        teacher = z.detach().clone()
        teacher[mask] -= delta
        loss = (z - teacher)[mask].square().mean() if mask.any() else z.sum() * 0
        grad = torch.autograd.grad(loss, (h.dense0.weight, h.dense0.bias))
        return float(loss.detach()), grad

    def test_empty_coverage_zero_loss_and_gradient(self):
        loss, grad = self.derivative(100, False)
        self.assertEqual(loss, 0)
        self.assertTrue(all(not g.any() for g in grad))

    def test_original_teacher_agreement_zero(self):
        loss, grad = self.derivative()
        self.assertEqual(loss, 0)
        self.assertTrue(all(not g.any() for g in grad))

    def test_small_nonzero_deviation(self):
        loss, grad = self.derivative(.0002)
        self.assertGreater(loss, 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grad))
        self.assertGreater(sum(float(g.norm()) for g in grad), 0)

    def test_large_logit_deviation_remains_finite(self):
        loss, grad = self.derivative(1e6)
        self.assertGreater(loss, 1e11)
        self.assertTrue(all(torch.isfinite(g).all() for g in grad))

    def test_scaled_unscaled_fp32_adjoint_agree(self):
        h, _, _, _, s = fixture(torch.float16)
        z = observed_logits(h, s, enabled=True)
        loss = z[0, 1, 1].square()
        args = (loss, z, torch.zeros(1, 4, 4, requires_grad=True), [h.dense0.weight, h.dense0.bias], [])
        a = measured_vjp(*args, scale=1.)
        b = measured_vjp(*args, scale=128.)
        for x, y in zip([a[0], *a[1]], [b[0], *b[1]]):
            torch.testing.assert_close(x, y, rtol=2e-6, atol=1e-6)

    def test_finite_terminal_nonfinite_half_upstream_synthetic(self):
        # An intentionally different tiny CPU graph: no claim to reproduce E2.
        p = torch.tensor(1., requires_grad=True)
        y = p.half() * torch.tensor(1000., dtype=torch.float16)
        z = y.float()
        trace = Trace(enabled=True)
        trace.graph(z)
        g = torch.autograd.grad(z * 128., (z, p))
        self.assertTrue(torch.isfinite(g[0]))
        self.assertFalse(torch.isfinite(g[1]))
        self.assertIsNotNone(trace.first_nonfinite)
        trace.close()

    def test_widened_projection_fp16_overflow_sensitive(self):
        h, _, _, _, s = fixture(torch.float16)
        g = torch.full_like(s.logits, 1e5)
        self.assertTrue(torch.isfinite(g).all())
        self.assertFalse(torch.isfinite(g.half()).all())
        self.assertTrue(all(torch.isfinite(a).all() for a in widened_vjp(s, g, h.dense0.weight)))

    def test_fp32_native_head_autograd_agrees(self):
        h, _, _, native, s = fixture()
        g = torch.randn_like(native)
        a = torch.autograd.grad((native * g).sum(), [h.dense0.weight, h.dense0.bias])
        b = widened_vjp(s, g, h.dense0.weight)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)

    def test_parameter_routing_and_no_accumulation(self):
        h, x, q, _, s = fixture(torch.float16)
        before = {n: p.detach().clone() for n, p in h.named_parameters()}
        z = observed_logits(h, s, enabled=True)
        box = torch.nn.Linear(6, 4)
        boxes = box(x)
        measured_vjp(z.square().mean(), z, boxes, [h.dense0.weight, h.dense0.bias], list(box.parameters()))
        for n, p in h.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))
            self.assertIsNone(p.grad)
        self.assertFalse(x.requires_grad)
        self.assertFalse(q.requires_grad)
        self.assertTrue(all(p.grad is None for p in box.parameters()))
        self.assertFalse(h.logit_shift.weight.requires_grad)
        self.assertFalse(h.logit_scale.weight.requires_grad)

    def test_box_gradient_rejected(self):
        h, _, _, z, _ = fixture()
        b = torch.zeros(1, 4, 4, requires_grad=True)
        self.assertRaises(ValueError, measured_vjp, z.sum() + b.sum(), z, b,
                          [h.dense0.weight, h.dense0.bias], [])

    def test_disabled_by_default(self):
        h, x, q, _, s = fixture()
        self.assertRaises(PermissionError, observed_logits, h, s)
        with Capture(h) as c:
            h(x, q, torch.ones(1, 7, dtype=torch.bool))
        self.assertIsNone(c.snapshot)
        trace = Trace()
        trace.record('unused', [torch.tensor(float('nan'))])
        self.assertEqual(trace.events, [])

    def test_unfrozen_calibration_rejected(self):
        h, _, _, _, _ = fixture()
        h.logit_shift.weight.requires_grad_(True)
        self.assertRaises(ValueError, Capture(h, enabled=True).__enter__)

    def test_stale_snapshot_or_changed_scope_rejected(self):
        for changed in ('weight', 'scope'):
            h, _, _, _, s = fixture()
            if changed == 'weight':
                with torch.no_grad():
                    h.dense0.weight.add_(.01)
            else:
                h.logit_scale.bias.requires_grad_(True)
            self.assertRaises(ValueError, observed_logits, h, s, enabled=True)

    def test_shared_representation_gradient_rejected(self):
        h, x, q, _, _ = fixture()
        with self.assertRaises(ValueError), Capture(h, enabled=True):
            h(x.requires_grad_(), q, torch.ones(1, 7, dtype=torch.bool))

    def test_invalid_numerical_gradient_stops(self):
        h, _, _, _, s = fixture()
        trace = Trace(enabled=True)
        self.assertRaises(FloatingPointError, widened_vjp, s,
                          torch.full_like(s.logits, float('inf')), h.dense0.weight, trace)
        self.assertEqual(trace.first_nonfinite['location'], 'FP32-adjoint.terminal')

    def test_zero_projection_norm_reference(self):
        h, _, _, _, s = fixture()
        raw = torch.zeros_like(s.raw)
        s = dataclasses.replace(s, raw=raw, norm=torch.zeros_like(s.norm),
                                denominator=torch.full_like(s.denominator, 1e-6))
        g = torch.randn_like(s.logits)
        a = widened_vjp(s, g, h.dense0.weight)
        b = reference(h, s, g)
        for x, y in zip(a, b):
            torch.testing.assert_close(x.double(), y, rtol=3e-5, atol=1.)


class PersistenceTests(unittest.TestCase):
    def test_stats_avoid_half_reduction_overflow(self):
        t = torch.full((1000,), 1000., dtype=torch.float16)
        self.assertTrue(stats(t)['finite'])
        self.assertAlmostEqual(stats(t)['L2'], 31622.776601683792, places=6)

    def test_nonfinite_stats_stay_json_serializable(self):
        r = stats(torch.tensor([float('nan'), float('inf'), 4.], dtype=torch.float16))
        self.assertEqual(r['nonfinite'], 2)
        self.assertIsNone(r['L2'])
        json.dumps(r, allow_nan=False)

    def test_hook_failure_persists_exception_and_first_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Trace(folder, enabled=True, context={'photo_id': 'SYNTHETIC', 'source_group': 'CPU'})
            p = torch.tensor(1., requires_grad=True)
            z = (p.half() * torch.tensor(1000., dtype=torch.float16)).float()
            trace.graph(z)
            try:
                measured_vjp(z * 128, z, torch.zeros(1, requires_grad=True), [p, p], [], trace=trace)
            except FloatingPointError:
                pass
            else:
                self.fail('Nonfinite gradient must not be suppressed')
            trace.close()
            files = sorted(Path(folder).glob('event-*.json'))
            records = [json.loads(p.read_text()) for p in files]
            self.assertTrue(any(r['location'] == 'VJP.scaled-return' for r in records))
            self.assertEqual(records[-1]['exception_type'], 'FloatingPointError')
            self.assertIn('Traceback', records[-1]['stack_trace'])
            self.assertIsNotNone(records[-1]['earliest_observed_nonfinite'])

    def test_positive_component_survives_later_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'positive.json'
            persist_component(p, component='positive', output_gradient=torch.ones(2),
                              projection_gradients=[torch.ones(2), torch.ones(1)], context={'source_group': 'CPU'})
            trace = Trace(folder, enabled=True)
            try:
                raise RuntimeError('Later KD failed')
            except RuntimeError as e:
                trace.failure(e)
            self.assertEqual(json.loads(p.read_text())['component'], 'positive')

    def test_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'x.json'
            atomic_json(p, {'one': 1})
            self.assertRaises(FileExistsError, atomic_json, p, {'two': 2})
            self.assertEqual(json.loads(p.read_text()), {'one': 1})

    def test_storage_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertRaises(RuntimeError, atomic_json, Path(folder) / 'x.json', {'a': 'b'}, 1)


if __name__ == '__main__':
    unittest.main()
