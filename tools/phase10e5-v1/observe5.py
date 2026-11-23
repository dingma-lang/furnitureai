"""Read-only actual gradient measurements;fixed scale1,no weight updates."""
import math
import torch

def norm(gs):return float(sum(g.detach().double().square().sum() for g in gs).sqrt())
def cosine(a,b):
    an=norm(a);bn=norm(b)
    return float(sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b)))/(an*bn) if an and bn else None

def gradient_diagnostics(parts,raw_logits,teacher,mask,projection,boxes,box_parameters,coefficient,grad):
    pg=grad(parts['positive'],[*projection,boxes,*box_parameters],retain_graph=True,allow_unused=True)
    if any(g is None or not torch.isfinite(g).all() for g in pg[:2]) or any(g is not None for g in pg[2:]):raise ValueError('Invalid positive gradient route/value')
    zero=not bool(mask.any()) or bool(torch.equal(raw_logits.detach()[mask],teacher[mask]))
    if zero:kg=[torch.zeros_like(p) for p in projection];method='analytic exact zero,no KD VJP'
    else:
        result=grad(parts['distillation'],[*projection,boxes,*box_parameters],retain_graph=True,allow_unused=True)
        if any(g is None or not torch.isfinite(g).all() for g in result[:2]) or any(g is not None for g in result[2:]):raise ValueError('Invalid KD gradient route/value')
        kg=list(result[:2]);method='actual FP32 observed-node VJP'
    pn=norm(pg[:2]);kn=norm(kg);weighted=coefficient*kn;ratio=weighted/pn if pn else None
    if not all(math.isfinite(v) for v in (pn,kn,weighted)):raise ValueError('Nonfinite gradient norms')
    return dict(positive_projection_L2=pn,unit_KD_projection_L2=kn,weighted_KD_projection_L2=weighted,
                KD_positive_cosine=cosine(pg[:2],kg),weighted_KD_positive_ratio=ratio,
                zero_positive_nonzero_KD=pn==0 and kn>0,above_recovery_half_bound=bool(ratio is not None and ratio>.5),
                KD_dominant=bool((ratio is not None and ratio>1) or (pn==0 and kn>0)),
                method=method,loss_scale=1.,box_routes_absent=True)

def combined_gate(loss,parts,raw_logits,teacher,mask,boxes,box_loss,params,projection,box_parameters,coefficient,grad):
    full=grad(loss,params,retain_graph=True)
    base=grad(parts['base'],params,retain_graph=True)
    kd=grad(parts['distillation'],params,retain_graph=True,allow_unused=True)
    if any(not torch.isfinite(g).all() for g in full+base) or any(g is not None and not torch.isfinite(g).all() for g in kd):raise ValueError('Nonfinite combined/base/KD gate gradient')
    for actual,b,k in zip(full,base,kd):
        expected=b+(coefficient*k if k is not None else 0.)
        if not torch.allclose(actual,expected,atol=1e-6,rtol=2e-5):raise ValueError('Coefficient/combined gradient decomposition failed')
    box=grad(box_loss,[boxes,*box_parameters,*projection],retain_graph=True,allow_unused=True)
    if any(g is None or not torch.isfinite(g).all() for g in box[:7]) or any(g is not None for g in box[7:]):raise ValueError('Invalid native box gate gradient')
    diagnostics=gradient_diagnostics(parts,raw_logits,teacher,mask,projection,boxes,box_parameters,coefficient,grad)
    return dict(status='PASS',combined_gradient_norm=norm(full),base_gradient_norm=norm(base),
                native_box_parameter_norm=norm(box[1:7]),coefficient_application_verified=True,
                KD_coverage=int(mask.sum()),KD_loss=float(parts['distillation'].detach()),
                finite_KD_parameter_gradients=True,diagnostics=diagnostics,loss_scale=1.)
