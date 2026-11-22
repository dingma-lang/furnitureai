"""Direct-component numerical verification; unchanged C16 composition and cap.

No optimizer, model construction, CUDA initialization or loss implementation.
"""
import math
import torch
from common19 import HEAD_KEYS
from balanced16 import CLASS_KEYS, PositiveAnchorUnavailable
from measure17 import finish_accumulator, info, cos

U32 = 2.0**-24
U64 = 2.0**-53
ETA32 = 2.0**-149


class CapViolation(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def norm64(values):
    # Same sorted FP32-to-FP64 square/sum/sqrt frame as C16.
    return math.sqrt(sum(float(v.detach().double().square().sum()) for _,v in sorted(values.items())))


def tolerance_for(budget, elements, coefficient=1.):
    """A priori bound, not fit to failed GPU data.

    FP32 scalar conversion and multiply each contribute unit roundoff u32.
    FP32 values square exactly in FP64 (48 significand bits <=53).
    Each norm uses at most N+T+8 FP64 additions/root/division operations.
    gamma bounds two norm measurements; absolute underflow is sqrt(N)*eta/2.
    The scalar multiplier must be normal FP32 (or exactly zero); otherwise
    stop for insufficient resolution rather than inflate the relative bound.
    """
    if not math.isfinite(budget) or budget<0 or elements<1 or not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('Invalid numerical bound inputs')
    if coefficient and coefficient<torch.finfo(torch.float32).tiny:
        raise ValueError('Cap coefficient below normal FP32 resolution; stop safely')
    operations=elements+len(CLASS_KEYS)+8
    if operations*U64>=1:raise ValueError('Diagnostic reduction is too large')
    gamma=operations*U64/(1-operations*U64)
    # Products/root/division and both norm reductions, rounded conservatively.
    # log1p/expm1 retain terms below one ulp of1.0; round the bound upward.
    relative=math.nextafter(math.expm1(2*math.log1p(U32)+2*math.log1p(gamma)+4*math.log1p(U64)-math.log1p(-gamma)),math.inf)
    absolute=math.sqrt(elements)*ETA32/2*(1+gamma)
    return dict(relative=relative,absolute=absolute,total=math.nextafter(budget*relative+absolute,math.inf),unit_roundoff_FP32=U32,unit_roundoff_FP64=U64,FP32_min_subnormal=ETA32,reduction_operation_bound=operations,reduction_gamma=gamma,scalar_conversion_and_multiply_roundings=2,derivation='(1+u32)^2*(1+gamma)^2*(1+u64)^4/(1-gamma)-1; plus sqrt(N)*eta32/2; log1p/expm1 evaluated and rounded upward')


def verify_direct(positive, negative, post, coefficient, positive_count):
    if set(positive)!=CLASS_KEYS or set(negative)!=CLASS_KEYS or set(post)!=CLASS_KEYS:
        raise ValueError('Exact six classification-head tensors required')
    if type(positive_count) is not int or positive_count<0:
        raise ValueError('Verified positive count required')
    for k,p in positive.items():
        for v in (p,negative[k],post[k]):
            if v.dtype!=torch.float32 or v.shape!=p.shape or v.device!=p.device or not torch.isfinite(v).all():
                raise ValueError('Finite FP32 same-frame components required')
    pn=norm64(positive);nn=norm64(negative);used=norm64(post)
    if positive_count==0:
        if pn:raise ValueError('Positive gradient without verified targets')
        budget=.01;mode='verified-negative-only:absolute1percent-of-original-clip'
    elif nn==0:
        budget=.25*pn;mode='no-negative-gradient'
    elif pn<=1e-8:
        raise PositiveAnchorUnavailable('Verified-positive anchor absent/near zero; stop update')
    else:
        budget=.25*pn;mode='relative-to-all-verified-furniture-positives'
    expected=min(1.,budget/nn) if nn else 1.
    tol=tolerance_for(budget,sum(v.numel() for v in positive.values()),expected)
    diag=dict(positive_norm=pn,negative_pre_norm=nn,negative_post_norm=used,cap_coefficient=coefficient,expected_coefficient=expected,expected_max_negative_norm=budget,actual_negative_positive_ratio=used/pn if pn else None,cap_active=coefficient<1,numerical_tolerance=tol,mode=mode,negative_positive_cosine=cos(negative,positive) if pn and nn else None,nonfinite_gradient=False,scope=sorted(CLASS_KEYS),measurement='direct post-cap accumulated source parameter vector; no composed-minus-base subtraction')
    # Coefficient/branch corruption is not excused by norm tolerance.
    if not math.isfinite(coefficient) or coefficient!=expected:
        raise CapViolation('Incorrect fixed-policy cap coefficient',diag)
    if used>budget+tol['total']:
        raise CapViolation('Direct source vector exceeds derived cap bound',diag)
    if any(not torch.equal(post[k],negative[k]*coefficient) for k in CLASS_KEYS):
        raise CapViolation('Post vector differs from exact original FP32 multiply',diag)
    diag['cap_verified']=True
    return diag


def compose_checked(acc):
    """Drop-in C18 contract. Returned gradient tensors stay bit-identical."""
    positive_before={k:v.clone() for k,v in acc.positive.items()}
    box_before={k:v.clone() for k,v in acc.base.items() if k.startswith('box_head.')}
    composed,cap=finish_accumulator(acc)  # Immutable C16 policy, math and output.
    p={k:v/acc.scale for k,v in acc.positive.items()}
    n={k:v/acc.scale for k,v in acc.negative.items()}
    post={k:v*cap['scale'] for k,v in n.items()}
    direct=verify_direct(p,n,post,cap['scale'],acc.positive_count)
    if any(not torch.equal(positive_before[k],v) for k,v in acc.positive.items()) or any(not torch.equal(box_before[k],acc.base[k]) for k in box_before):
        raise CapViolation('Composition modified original positive/box buffers',direct)
    for k,v in composed.items():
        base=acc.base[k]/acc.scale
        expected=base+post[k] if k in CLASS_KEYS else base
        if not torch.isfinite(v).all() or not torch.equal(v,expected):
            raise CapViolation('Combined/box gradient corruption',direct)
    # FP64 composition-rounding diagnostic only, never a reconstructed cap.
    rounding={k:composed[k].double()-(acc.base[k]/acc.scale).double()-post[k].double() for k in CLASS_KEYS}
    rounding_norm=math.sqrt(sum(float(v.square().sum()) for v in rounding.values()))
    metric=dict(positive=info(p),source_pre=info(n,p),source_post=info(post,p),cap=cap,positive_buffers_unchanged=True,box_buffers_unchanged=True,cap_violations=0,combined_positive_source_cosine=cos({k:p[k]+post[k] for k in p},p),direct_guard=direct,FP64_addition_rounding_residual_norm=rounding_norm,rounding_diagnostic_not_negative_vector=True,per_tensor={k:dict(positive_norm=float(p[k].double().norm()),negative_pre_norm=float(n[k].double().norm()),negative_post_norm=float(post[k].double().norm())) for k in sorted(CLASS_KEYS)})
    return composed,metric
