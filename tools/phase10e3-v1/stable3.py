"""Opt-in FP32 adjoint of the *observed* OWLv2 classification-head forward.

No model forward or CUDA initialization on import. No training entry point.
Forward logits and masks stay exact. Only diagnostic adjoints are widened:
unlike a new FP32 head forward, this evaluates derivatives at the original
rounded intermediates. It is not bit-equivalent to legacy FP16 backward.
"""
from dataclasses import dataclass
import math

import torch
from telemetry3 import Trace


@dataclass(frozen=True)
class Snapshot:
    x: torch.Tensor
    raw: torch.Tensor
    norm: torch.Tensor
    denominator: torch.Tensor
    query_contract: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor
    logits: torch.Tensor
    parameter_binding: tuple


class Capture:
    """Targeted, optional forward hooks. Transient features never go to disk.

    Use around one normal detector forward. Require exact reconstruction of
    every final logit and normalized projection before accepting a snapshot.
    Reject unsupported head implementations/precision contexts.
    """
    def __init__(self, head, *, enabled=False, trace=None):
        self.head, self.enabled = head, enabled
        self.trace = trace or Trace()
        self.values = {}
        self.handles = []
        self.snapshot = None

    def __enter__(self):
        if not self.enabled:
            return self
        params = dict(self.head.named_parameters())
        if set(params) != {'dense0.weight', 'dense0.bias', 'logit_shift.weight',
                           'logit_shift.bias', 'logit_scale.weight', 'logit_scale.bias'}:
            raise ValueError('Unexpected OWLv2 classification head scope')
        if any(p.dtype != torch.float32 for p in params.values()):
            raise ValueError('Pinned FP32 master parameters required')
        if any(params[n].requires_grad for n in params if not n.startswith('dense0.')):
            raise ValueError('Calibration must remain frozen')
        if not all(params[n].requires_grad for n in ('dense0.weight', 'dense0.bias')):
            raise ValueError('Both classification projection tensors must remain trainable')
        for name in ('dense0', 'logit_shift', 'elu'):
            def hook(module, args, result, name=name):
                self.values[name] = result.detach()
                self.trace.tensor('head.' + name, result)
            self.handles.append(getattr(self.head, name).register_forward_hook(hook))
        self.handles.append(self.head.register_forward_hook(self._finish))
        return self

    def _finish(self, module, args, result):
        if self.snapshot is not None or len(args) != 3:
            raise ValueError('Exactly one text-query head invocation required')
        x, query, mask = args
        if query is None or mask is None or query.shape[1] != 7 or not bool(mask.all()):
            raise ValueError('Require all seven non-padded production prompts')
        if x.requires_grad or query.requires_grad:
            raise ValueError('Shared representations must be frozen/detached')
        z, normalized = result
        raw = self.values['dense0']
        # These replay the pinned source expressions within the SAME autocast
        # context. The extra contraction is a bounded head-only replay, not a
        # detector forward. Failure never triggers another precision attempt.
        with torch.no_grad():
            n = torch.linalg.norm(raw, dim=-1, keepdim=True)
            i = raw / (n + 1e-6)
            q = query / (torch.linalg.norm(query, dim=-1, keepdim=True) + 1e-6)
            cosine = torch.einsum('...pd,...qd->...pq', i, q)
            scale = self.values['elu'] + 1
            recreated = ((cosine + self.values['logit_shift']) * scale).float()
            if not torch.equal(i, normalized) or not torch.equal(recreated, z):
                raise ValueError('Exact observed-head reconstruction failed')
            # einsum uses the autocast low-precision inputs; capture the same
            # values for its widened adjoint. With autocast off it uses i.dtype.
            dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else i.dtype
            q_contract = q.to(dtype)
            valid = mask[:, None, :].expand_as(z).bool()
        binding = tuple((name, id(p), p._version, p.requires_grad) for name, p in module.named_parameters())
        self.snapshot = Snapshot(x.detach(), raw, n, (n + 1e-6).detach(), q_contract.detach(), scale,
                                 valid.detach(), z.detach(), binding)
        self.trace.record('head.forward-verified', [x, raw, n, i, q, q_contract, cosine, scale, z],
                          autocast_enabled=torch.is_autocast_enabled(x.device.type),
                          autocast_dtype=str(torch.get_autocast_dtype(x.device.type)),
                          bit_identical_reconstruction=True)

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def widened_vjp(snapshot, gradient, weight, trace=None):
    """FP32 chain rule at observed nodes; no loss scale or half adjoint casts.

    d = Linear(x); n=||d||; u=d/(n+1e-6); c=u@q';
    z=(c+frozen_shift)*frozen_scale. Cast derivatives are identity,
    as in autograd; removing their FP16 rounding is explicit and diagnostic.
    """
    trace = trace or Trace()
    s = snapshot
    with torch.autocast(gradient.device.type, enabled=False):
        g = gradient.float()
        if g.shape != s.logits.shape:
            raise ValueError('Gradient/query layout mismatch')
        d, n = s.raw.float(), s.norm.float()
        x = s.x.to(s.raw.dtype).float()  # exact input values of autocast linear
        q, scale = s.query_contract.float(), s.scale.float()

        def check(name, t):
            trace.record('FP32-adjoint.' + name, [t], loss_scale=1., unscaled=True)
            if not torch.isfinite(t).all():
                raise FloatingPointError('Nonfinite FP32 adjoint at ' + name)
            return t

        check('terminal', g)
        gc = check('cosine', torch.where(s.valid, g, 0.) * scale)
        gu = check('normalized-projection', torch.einsum('bpq,bqd->bpd', gc, q))
        den = s.denominator.float()
        gn = check('norm', -(gu * d).sum(-1, keepdim=True) / den.square())
        # torch.linalg.norm has zero derivative at an all-zero vector.
        direction = torch.where(n > 0, d / n.clamp_min(torch.finfo(torch.float32).tiny), 0.)
        gd = check('projection-output', gu / den + gn * direction)
        flat = gd.reshape(-1, gd.shape[-1])
        gw = check('projection-weight', flat.T @ x.reshape(-1, x.shape[-1]))
        gb = check('projection-bias', flat.sum(0))
        # Observe input sensitivity but NEVER return a gradient to the frozen
        # shared representation or to calibration/query/box branches.
        if trace.enabled:
            check('projection-input-unrouted', gd @ weight.detach().float())
        return gw, gb


class _ObservedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, bias, snapshot, trace):
        ctx.snapshot, ctx.trace = snapshot, trace
        ctx.save_for_backward(weight)
        return snapshot.logits.clone()

    @staticmethod
    def backward(ctx, gradient):
        weight, = ctx.saved_tensors
        gw, gb = widened_vjp(ctx.snapshot, gradient, weight, ctx.trace)
        return gw, gb, None, None


def observed_logits(head, snapshot, *, enabled=False, trace=None):
    """Explicit opt-in; never substitute this derivative into production training."""
    if not enabled:
        raise PermissionError('FP32 observed-node measurement is disabled by default')
    if snapshot is None:
        raise ValueError('Verified live class-head snapshot required; raw logits are insufficient')
    binding = tuple((name, id(p), p._version, p.requires_grad) for name, p in head.named_parameters())
    if binding != snapshot.parameter_binding:
        raise ValueError('Head parameters/scope changed since the observed forward')
    return _ObservedHead.apply(head.dense0.weight, head.dense0.bias, snapshot, trace or Trace())


def measured_vjp(loss, z, boxes, projection, box_parameters, *, trace=None, scale=1.):
    """Finite/routing checks before unscaling. scale=128 reproduces legacy policy.

    Default scale1 is appropriate for the opt-in all-FP32 adjoint. No GradScaler,
    optimizer, automatic scale search, nonfinite replacement or fallback.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid instrumentation scale')
    trace = trace or Trace()
    trace.record('VJP.begin', [loss, z], scale=scale, scaler=False, unscaled=False)
    try:
        grads = torch.autograd.grad(loss * scale, [z, boxes, *projection, *box_parameters],
                                    retain_graph=True, allow_unused=True)
        # Persist ALL return slots before raising, unlike E2.
        trace.record('VJP.scaled-return', grads, scale=scale, unscaled=False)
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Nonfinite returned gradient; see typed slot telemetry')
        if grads[0] is None or any(g is None for g in grads[2:4]):
            raise ValueError('Missing classification output/projection gradient')
        if grads[1] is not None or any(g is not None for g in grads[4:]):
            raise ValueError('Classification gradient reached box branch')
        gz = grads[0].detach().float() / scale
        gp = [g.detach().float() / scale for g in grads[2:4]]
        trace.record('VJP.unscaled-return', [gz, *gp], scale=scale, unscaled=True)
        return gz, gp
    except BaseException as exc:
        try:
            trace.failure(exc)
        except BaseException as persistence_error:
            exc.add_note('Failure telemetry could not be persisted: ' + str(persistence_error))
        raise
