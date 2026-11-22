"""CPU-testable replay tolerances and actual single-VJP model-gradient measurement."""
import numpy as np
import torch

def replay(actual,expected):
    """Predeclared historical C11 probe=True FP16 tolerances,not post-hoc relaxation."""
    if set(actual)!=set(expected):raise ValueError('Replay fields changed')
    result={}
    for key in ('logits','square_cxcywh'):
        a=actual[key];b=expected[key]
        if a.shape!=b.shape or a.dtype!=np.float32 or b.dtype!=np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Replay shape/dtype/nonfinite mismatch')
        atol,rtol=(.02,.001) if key=='logits' else (1e-5,1e-5)
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
        accepted=bool(np.allclose(a,b,atol=atol,rtol=rtol))
        result[key]=dict(bit_identical=bool(np.array_equal(a,b)),accepted=accepted,max_absolute_difference=float(difference.max()),RMS_difference=float(np.sqrt(np.mean(difference**2))),absolute_difference_quantiles=quantiles(difference),atol=atol,rtol=rtol)
        if not accepted:raise ValueError('Numerical cache mismatch: '+key+' maxAbs='+str(result[key]['max_absolute_difference']))
    return result

def quantiles(values):
    a=np.asarray(values,dtype=np.float64).ravel()
    return {str(q):float(np.quantile(a,q)) if len(a) else None for q in (0,.01,.5,.9,.99,1)}
def summaries(tensors):
    v=torch.cat([t.detach().cpu().double().flatten() for t in tensors]) if tensors else torch.zeros(0,dtype=torch.float64)
    return dict(L2=float(v.norm()),L1=float(v.abs().sum()),maximum_absolute=float(v.abs().max()) if len(v) else 0.,nonzero=int((v!=0).sum()),signed_quantiles=quantiles(v.numpy()),absolute_quantiles=quantiles(v.abs().numpy()))
def cosine(a,b):
    aa=torch.cat([v.flatten().double() for v in a]);bb=torch.cat([v.flatten().double() for v in b]);den=float(aa.norm()*bb.norm())
    return float(aa@bb)/den if den else None

def zero_derivative(student,teacher,mask):
    return not bool(mask.any()) or bool(torch.equal(student.detach()[mask].float(),teacher.detach()[mask].float()))

def vjp(loss,z,boxes,projection,box_parameters):
    """One loss VJP,with independent output/parameter routes checked explicitly."""
    targets=[z,boxes,*projection,*box_parameters]
    grads=torch.autograd.grad(loss*128,targets,retain_graph=True,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite scaled gradient')
    gz=grads[0];gb=grads[1];gp=grads[2:4];gbox=grads[4:]
    if gz is None or any(g is None for g in gp):raise ValueError('Required projection/output gradient missing')
    if gb is not None or any(g is not None for g in gbox):raise ValueError('Classification/distillation loss reached box branch')
    return gz.detach().float()/128,[g.detach().float()/128 for g in gp]
