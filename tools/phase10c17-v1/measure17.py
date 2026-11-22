"""Pure tensor diagnostics,not an optimizer or model loader."""
import math
import torch
from common17 import HEAD_KEYS
from balanced16 import CLASS_KEYS,compose,Policy


def finish_accumulator(acc):
    """C16 exact buffer math;separate cap scale from shadowed AMP metadata.

    C16.finish overwrites its diagnostic 'scale' with the AMP scale. Invoke
    the same unchanged compose implementation on the same unscaled buffers.
    This changes no algorithm,coefficient,target,normalization or gradient.
    """
    if acc.count!=acc.expected:raise ValueError('Incomplete accumulation group')
    base={k:v/acc.scale for k,v in acc.base.items()};positive={k:v/acc.scale for k,v in acc.positive.items()};negative={k:v/acc.scale for k,v in acc.negative.items()}
    result,diag=compose(base,positive,negative,acc.positive_count,Policy(enabled=True))
    return result,dict(diag,actual_microbatches=acc.count,amp_scale=acc.scale,positive_targets=acc.positive_count,normalization_order='Identical C16/C8 scaled-buffer sum then single unscale;diagnostic cap scale preserved separately')


def l2(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))


def cos(a,b):
    if set(a)!=set(b):raise ValueError('Vector frames differ')
    keys=sorted(a);den=l2(a[k] for k in keys)*l2(b[k] for k in keys)
    return sum(float((a[k].detach().double()*b[k].detach().double()).sum()) for k in keys)/den if den else None


def info(values,positive=None):
    result=dict(norm=l2(values[k] for k in sorted(values)),per_tensor={k:float(v.detach().double().norm()) for k,v in sorted(values.items())})
    result['projection_norm']=l2(v for k,v in values.items() if k.startswith('class_head.dense0.'))
    result['shift_norm']=l2(v for k,v in values.items() if k.startswith('class_head.logit_shift.'))
    result['scale_norm']=l2(v for k,v in values.items() if k.startswith('class_head.logit_scale.'))
    if positive is not None:
        p=l2(positive[k] for k in sorted(positive));result.update(negative_positive_ratio=result['norm']/p if p else None,cosine_positive=cos(values,positive))
    return result


def scaled_gradients(loss,z,b,params,microbatches,scale,retain_graph=True):
    # Preserve C8's loss/group-size THEN scaler order,especially the final3.
    raw=torch.autograd.grad((loss/microbatches)*scale,(z,b,*params),retain_graph=retain_graph,allow_unused=True)
    if any(x is not None and (x.dtype!=torch.float32 or not torch.isfinite(x).all()) for x in raw):raise ValueError('Malformed/nonfinite scaled branch gradient')
    return tuple(None if x is None else x.detach() for x in raw)


def group_metrics(base,positive,focal,BCE,source,composed,cap):
    clsbase={k:base[k] for k in sorted(CLASS_KEYS)};post={k:source[k]*cap['scale'] for k in sorted(CLASS_KEYS)};effective={k:composed[k]-base[k] for k in CLASS_KEYS}
    if l2(post.values())>cap['budget']*(1+1e-6)+1e-12 or l2(effective.values())>cap['budget']*(1+1e-6)+1e-8:raise ValueError('Measured parameter-gradient cap violation')
    if any(not torch.equal(composed[k],base[k]) for k in HEAD_KEYS if k.startswith('box_head.')):raise ValueError('Composition changed box gradients')
    combined={'base':clsbase,'A_focal':{k:clsbase[k]+focal[k] for k in CLASS_KEYS},'B_BCE':{k:clsbase[k]+BCE[k] for k in CLASS_KEYS},'C_pre_cap':{k:clsbase[k]+source[k] for k in CLASS_KEYS},'C_post_cap':{k:composed[k] for k in CLASS_KEYS}}
    tensors={}
    for k in sorted(CLASS_KEYS):
        p=positive[k];n=post[k];pn=float(p.double().norm());nn=float(n.double().norm());den=pn*nn;dot=float((p.double()*n.double()).sum());tensors[k]=dict(positive_norm=pn,post_cap_negative_norm=nn,ratio=nn/pn if pn else None,cosine=dot/den if den else None,opposing_negative_exceeds_positive=bool(nn>pn and dot<0),positive_zero_negative_nonzero=bool(pn==0 and nn>0))
    pure={k:positive[k]+post[k] for k in CLASS_KEYS};p2=l2(positive.values())**2;dot=sum(float((positive[k].double()*pure[k].double()).sum()) for k in CLASS_KEYS)
    if cap['positive_norm']>1e-8 and dot<.75*p2-1e-5*max(1,p2):raise ValueError('Positive-plus-source direction guarantee failed')
    return dict(positive=info(positive),negatives={'A_focal':info(focal,positive),'B_BCE':info(BCE,positive),'C_pre_cap':info(source,positive),'C_post_cap':info(post,positive)},combined={k:dict(info(v),cosine_positive=cos(v,positive),cosine_A=cos(v,combined['A_focal'])) for k,v in combined.items()},per_tensor=tensors,cap=cap,effective_FP32_source_perturbation_norm=l2(effective.values()),positive_plus_post_source_cosine=cos(pure,positive),box_head_norm=l2(v for k,v in base.items() if k.startswith('box_head.')),box_gradients_unchanged=True,positive_buffers_unchanged=True,cap_violations=0,comparison='Additive FP32 branch-buffer composition;not claimed bit-identical to a separate single-backward AMP loss')
