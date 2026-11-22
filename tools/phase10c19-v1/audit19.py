"""Saved-state attribution and training-only recovery assessment. No inference."""
import math
import torch
from common19 import *
from recovery19 import audit_checkpoint,disabled_proposal


def tensor_change(a,b):
    d=b.double()-a.double();den=float(a.double().norm())
    return dict(different_elements=int((d!=0).sum()),elements=d.numel(),delta_L2=float(d.norm()),delta_max_abs=float(d.abs().max()),relative_L2=float(d.norm())/den if den else None)


def load_ref(base,ref):
    path=base/ref['file']
    if sha(path)!=ref['sha256']:raise ValueError('Matched checkpoint reference changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def compare_matched():
    original=original_heads();old_done=read(C8/'B-corrected/complete.json');old=load_ref(C8,old_done['initial_checkpoint']);new_initial=list((RUN18/'checkpoints').glob('update-000-*.receipt.json'))
    previous=[old,load_ref(RUN18,read(new_initial[0]))];results=[]
    for i in range(1,29):
        old_ref=read(C8/'B-corrected'/('update-'+str(i).zfill(3)+'.json'))['receipt'];new_ref=read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json'))['checkpoint'];current=[load_ref(C8,old_ref),load_ref(RUN18,new_ref)]
        if any(s['updates']!=i for s in current):raise ValueError('Matched step counter differs')
        weights={k:tensor_change(current[0]['weights'][k],current[1]['weights'][k]) for k in HEAD_KEYS};steps={k:tensor_change(current[0]['weights'][k]-previous[0]['weights'][k],current[1]['weights'][k]-previous[1]['weights'][k]) for k in HEAD_KEYS}
        moments={}
        for index,k in enumerate(current[1]['weights']):
            moments[k]={field:tensor_change(current[0]['optimizer']['state'][index][field],current[1]['optimizer']['state'][index][field]) for field in ('exp_avg','exp_avg_sq')}
        adaptation={k:tensor_change(original[k],current[1]['weights'][k]) for k in HEAD_KEYS}
        results.append(dict(update=i,weight_differences=weights,incremental_step_differences=steps,AdamW_moment_differences=moments,C18_displacement_from_original=adaptation,identical_CPU_RNG=torch.equal(current[0]['rng']['cpu'],current[1]['rng']['cpu']),identical_CUDA_RNG=torch.equal(current[0]['rng']['cuda'][0],current[1]['rng']['cuda'][0]),snapshots_sha256=dict(C8=old_ref['sha256'],C18=new_ref['sha256'])))
        previous=current
    return dict(matched_updates=28,records=results,source='Actual immutable checkpoint heads andAdamWmoments;no forward or new optimizer step',causality_limit='Differences demonstrate that the method changes actual steps; not every element is attributable solely to the source loss. C18 uses three isolated VJPs versus C8 single-backward summation,with disclosed upstream AMP roundoff. Later matching and box gradients can diverge.')


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('C19 must be CPU-only')
    checkpoint=audit_checkpoint();matched=compare_matched();emit19(REPORT19/'matched-parameter-comparison.json',matched)
    old=read(REPORT18/'results.json');updates=[read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,29)]
    scores=[s for u in updates for r in u['images'] for s in r['source_scores']]
    ratio=[u['metrics']['source_pre']['negative_positive_ratio'] for u in updates]
    norms=[u['metrics']['source_pre']['norm'] for u in updates]
    nonzero=[u['update'] for u in updates if u['metrics']['source_pre']['norm']>0]
    assessment=dict(status='CONDITIONAL_DIAGNOSTIC_VALUE_NOT_PERFORMANCE_EVIDENCE',diagnostic_completion_conditionally_justified=True,improvement_demonstrated=False,optimization_state_available=True,GPU_binary_reproducibility_certified=False,actual_parameter_updates_differ=True,matched_updates_compared=28,matched_E1_metrics_equal={k:old['comparison']['C18_E1']['headline'][k]==old['comparison']['C8_corrected_B_E1']['headline'][k] for k in ('precision','recall','tp','fp','fn')},Chair_FP_E1=139,caps_activated_committed=sum(u['metrics']['cap']['scale']<1 for u in updates),cap_violations_committed=0,failed_group_cap_activation_unknown=True,negative_supervision_nonzero_update_ids=nonzero,maximum_committed_negative_positive_ratio=max(ratio),maximum_source_Chair_probability=max(scores),source_terms_ge_015=sum(s>=.15 for s in scores),source_gradient_norm_update28=norms[-1],source_positive_ratio_update28=ratio[-1],confidence_inflation_evidence=dict(C18_E1=old['logit_drift']['C18_E1']['mean_all_categories'],C8_E1=old['logit_drift']['C8_corrected_B_E1']['mean_all_categories'],update28_inference_not_available=True,training_only_E1_probe=old['training']['probes'][1]),scientific_rationale=['Completing the originally predeclared fixed75-update mechanistic endpoint would test cap engagement as positive norms decline and source influence grows.','A complete matched-duration comparison would avoid confusing a15-update C18 snapshot with75-update C8. No hyperparameters,thresholds or endpoint changes are proposed.','The marginal runtime and data budget remain bounded,with47updates retained and no need to re-run committed updates.'],counterevidence=['No meaningful15-update precision/recall improvement,near-identical broad inflation,and no trusted negatives above0.15.','28completed updates provide no naturally active cap. Failed group29 cannot be reconstructed from missing buffers.','Additional duration is not demonstrated to improve precision; source coverage remains a small,unverified-hardness subset.'],recommendation='Offer only conditional completion of the existing mechanism experiment,not a new optimization cycle or a production-quality training recommendation. User approval and future executor/RNG validation required; cancellation remains scientifically reasonable.',missing_update28_evaluation='Prevents an endpoint performance conclusion at28. No new inference is needed for this preflight,and no claim is made about checkpoint28 precision/recall.',continuation_classification='Intended optimization-state continuation with unchanged output tensors andsourceobjective; mechanically modified restart/telemetry,not certified bitwise uninterrupted replay.',known_noncheckpointed_reporting_state='Epoch2 per-source accumulated diagnostic vectors before the stop are not in checkpoint28. Reconstruct only by summing retained update16–28 receipts if sufficient vectors existed (they do not); label resumed groups separately. This affects whole-epoch diagnostics,not model/optimizer gradients.')
    emit19(REPORT19/'scientific-assessment.json',assessment);disabled_proposal(checkpoint,assessment)
    emit19(REPORT19/'saved-evaluation-reference.json',dict(source=str((REPORT18/'results.json').relative_to(ROOT)),source_sha256=sha(REPORT18/'results.json'),comparison=old['comparison'],protocol=old['protocol'],new_metrics_computed=False,checkpoint28_scored=False))
    assert not torch.cuda.is_initialized()
    print('C19 checkpoint/order/CPUrestore and28matched-state comparisons complete;GPU calls0',flush=True)


if __name__=='__main__':run()
