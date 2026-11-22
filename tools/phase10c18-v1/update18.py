"""Apply only composed unscaled gradients through unchanged AMP/AdamW clipping."""
import torch
from common18 import HEAD_KEYS
from balanced16 import CLASS_KEYS
from measure17 import finish_accumulator, info, l2, cos


def compose_checked(acc):
    positives = {k:v.clone() for k,v in acc.positive.items()}
    boxes = {k:v.clone() for k,v in acc.base.items() if k.startswith('box_head.')}
    out, cap = finish_accumulator(acc)
    if any(not torch.equal(positives[k],v) for k,v in acc.positive.items()) or any(not torch.equal(boxes[k],acc.base[k]) for k in boxes):
        raise ValueError('Composition modified positive/box buffers')
    p = {k:v/acc.scale for k,v in acc.positive.items()}
    n = {k:v/acc.scale for k,v in acc.negative.items()}
    post = {k:v*cap['scale'] for k,v in n.items()}
    effective = {k:out[k]-acc.base[k]/acc.scale for k in CLASS_KEYS}
    if l2(post.values()) > cap['budget']*(1+1e-6)+1e-12 or l2(effective.values()) > cap['budget']*(1+1e-6)+1e-8:
        raise ValueError('Measured cap violation')
    if any(not torch.equal(out[k],acc.base[k]/acc.scale) for k in boxes):
        raise ValueError('Source altered base box gradients')
    if any(not torch.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite composed gradients')
    metric = dict(positive=info(p),source_pre=info(n,p),source_post=info(post,p),cap=cap,positive_buffers_unchanged=True,box_buffers_unchanged=True,cap_violations=0,combined_positive_source_cosine=cos({k:p[k]+post[k] for k in p},p),per_tensor={k:dict(positive_norm=float(p[k].double().norm()),negative_pre_norm=float(n[k].double().norm()),negative_post_norm=float(post[k].double().norm())) for k in sorted(CLASS_KEYS)})
    return out, metric


def assign_scaled(heads, gradients, scale):
    if set(heads)!=HEAD_KEYS or set(gradients)!=HEAD_KEYS or scale<=0:
        raise ValueError('Exact12-head scope and valid scale required')
    for key,p in heads.items():
        value=gradients[key]
        if value.dtype!=torch.float32 or value.shape!=p.shape or value.device!=p.device or not torch.isfinite(value).all() or p.grad is not None:
            raise ValueError('Invalid gradient assignment')
    for key,p in heads.items():
        p.grad=gradients[key]*scale
    if any(not torch.isfinite(p.grad).all() for p in heads.values()):
        raise ValueError('Nonfinite scaled optimizer gradients')
