"""Honest bounds now; actual cached-feature counterfactuals only when available."""
import itertools
import numpy as np
from common10 import *

def linear_layernorm_envelope(weight,bias,gamma,beta):
    """Real-arithmetic bound for f=gamma*u+beta, sum(u)=0, ||u||<=sqrt(d)."""
    w=np.asarray(weight,dtype=np.float64).reshape(-1);g=np.asarray(gamma,dtype=np.float64).reshape(-1);b=np.asarray(beta,dtype=np.float64).reshape(-1)
    if w.shape!=g.shape or g.shape!=b.shape:raise ValueError('LayerNorm dimension mismatch')
    a=w*g;center=float(w@b+float(np.asarray(bias).reshape(-1)[0]));radius=float(np.sqrt(len(w))*np.linalg.norm(a-a.mean()))
    return dict(center=center,radius=radius,min=center-radius,max=center+radius,max_abs=abs(center)+radius)

def head_combinations(features,text,original,trained):
    """CPU arithmetic on explicitly supplied activations; never reconstruct f from logits.

    Tests use synthetic vectors. Real use requires a separately acquired,
    provenance-verified feature cache plus endpoint reconstruction checks.
    """
    import torch
    import torch.nn.functional as F
    if features.device.type!='cpu' or text.device.type!='cpu' or features.ndim!=2 or features.shape[1]!=768 or text.shape!=(7,512):raise ValueError('Explicit CPU pre-head/image and text activations required')
    if not torch.isfinite(features).all() or not torch.isfinite(text).all():raise ValueError('Nonfinite activations')
    def linear(state,name):return F.linear(features,state[name+'.weight'].cpu(),state[name+'.bias'].cpu())
    with torch.no_grad():
        text=text/(text.norm(dim=-1,keepdim=True)+1e-6);cos=[];shifts=[];scales=[]
        for state in (original,trained):
            image=linear(state,'class_head.dense0');image=image/(image.norm(dim=-1,keepdim=True)+1e-6);cos.append(image@text.T);shifts.append(linear(state,'class_head.logit_shift'));scales.append(F.elu(linear(state,'class_head.logit_scale'))+1)
        return {''.join(map(str,bits)):((cos[bits[0]]+shifts[bits[1]])*scales[bits[2]]).double().numpy() for bits in itertools.product((0,1),repeat=3)}

def shapley_components(combinations):
    if set(combinations)!={''.join(map(str,k)) for k in itertools.product((0,1),repeat=3)}:raise ValueError('All eight functional counterfactuals required')
    result={name:np.zeros_like(combinations['000'],dtype=float) for name in ('projection','shift','scale')}
    for order in itertools.permutations(range(3)):
        bits=[0,0,0];previous=combinations['000']
        for index in order:
            bits[index]=1;current=combinations[''.join(map(str,bits))];result[('projection','shift','scale')[index]]+=(current-previous)/6;previous=current
    if not np.allclose(sum(result.values()),combinations['111']-combinations['000'],atol=1e-10,rtol=1e-10):raise ValueError('Counterfactual attribution does not close')
    return result

def require_feature_cache(path):
    if not Path(path).exists():raise FileNotFoundError('No OWLv2 pre-head activations: exact shift/scale/projection attribution is unavailable; no zero or inferred activation fallback')

def run():
    import torch
    from safetensors import safe_open
    verify_inputs();orig=original_heads()
    with safe_open(BASE/'model.safetensors',framework='pt',device='cpu') as f:g=f.get_tensor('layer_norm.weight').double().numpy();beta=f.get_tensor('layer_norm.bias').double().numpy()
    def env(state,module):return linear_layernorm_envelope(state['class_head.'+module+'.weight'].double().numpy(),state['class_head.'+module+'.bias'].double().numpy(),g,beta)
    s0=env(orig,'logit_shift');raw0=env(orig,'logit_scale');scale_upper=max(raw0['max']+1,math.exp(min(raw0['max'],0)))
    bounds={}
    for name,arm in (('A','A-control'),('B','B-corrected')):
        done=read(C8/arm/'complete.json');trained=torch.load(C8/done['final_checkpoint']['file'],map_location='cpu',weights_only=True)['weights'];diff={k:trained[k]-orig[k] for k in CALIBRATION_KEYS};ds=env(diff,'logit_shift');dr=env(diff,'logit_scale')
        db_shift=abs(float(diff['class_head.logit_shift.bias'][0]));db_scale=abs(float(diff['class_head.logit_scale.bias'][0]));r1_upper=scale_upper+db_scale
        bias_only=db_shift*r1_upper+(1+s0['max_abs'])*db_scale
        full_calibration_bound=ds['max_abs']*(scale_upper+dr['max_abs'])+(1+s0['max_abs'])*dr['max_abs']
        bounds[name]=dict(raw_shift_change_envelope=ds,raw_scale_change_envelope=dr,activated_scale_change_abs_bound=dr['max_abs'],
            bias_only_counterfactual_logit_abs_bound=bias_only,calibration_only_counterfactual_logit_abs_bound=full_calibration_bound,
            projection_only_logit_abs_bound_with_pretrained_calibration=2*scale_upper,
            interpretation='Universal envelopes, not observed contributions or additive percentages. Feature-dependent calibration changes are not ruled out; bias-only drift is far too small. Projection remains capable of recreating broad drift.')
    c9=read(ROOT/'reports/phase10c9-v1/prediction-summary.json')['logit_changes']
    save(REPORT/'attribution-limits.json',dict(exact_real_query_attribution_available=False,observed_common_logit_statistics=c9,
        original_shift_envelope=s0,original_raw_scale_envelope=raw0,original_activated_scale_upper_bound=scale_upper,bounds=bounds,
        derivation='Final frozen LayerNorm gives f=gamma*u+beta; sum(u)=0 and||u||2<=sqrt(768) in real arithmetic. Bound each linear form by its centered gamma-weighted norm. ELU+1 is1-Lipschitz and normalized cosine magnitude<=1.',
        numeric_caveat='Analytic real-arithmetic envelopes using saved FP32 parameter values, not measured activations or a formal IEEE rounding-error certificate.',
        unchanged_feature_parameters='C8 original-backbone frozen hashes agree for both arms; vision/text/final LayerNorm parameters were not optimized. Post-projection normalization can still change when dense0 changes.',
        missing_evidence='Actual frozen pre-head feature and text vectors, per-query shift/scale values and normalized projection outputs were not cached. Weight norms/common-mode statistics cannot yield a unique causal component split.',
        future_method='Cache real pre-head f and text from existing planned initial diagnostic forwards, verify endpoints against saved original/A/B logits, then calculate all8 CPU head-group combinations and six-order Shapley contributions. This is functional head attribution conditional on fixed activations, not causal training-dynamics attribution.',
        real_counterfactuals_computed=0,synthetic_tests_are_not_real_activation_evidence=True,real_model_forwards=0))
    print('C10 attribution limitation and analytic bounds:',bounds,flush=True)

if __name__=='__main__':run()
