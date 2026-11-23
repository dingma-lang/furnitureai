"""Training-only CPU analysis of persisted GPU measurements, no new inference."""
import math
import numpy as np
from common4 import *


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max())) if len(a) else dict(count=0)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only saved-result analysis')
    cfg,*_=inputs()
    if not (RUN/'complete.json').exists():
        failure=read(RUN/'failure.json')
        emit(REPORT/'decision.json',dict(status='STOPPED_RECOVERY_INCOMPLETE',coefficient=None,training_ready=False,failure=failure,new_GPU_attempts=0))
        return
    done=read(RUN/'complete.json');rows=done['records'];groups=read(RUN/'source-groups.json');original=read(RUN/'original-correspondence.json')
    if len(rows)!=15 or len(groups['records'])!=15 or sum(r['terms'] for r in rows)!=267:
        raise ValueError('Complete15-group coverage required for coefficient')
    for r in rows:
        if sha(RUN/r['raw']['file'])!=r['raw']['sha256']:raise ValueError('Raw output integrity failure')
    p=[r['positive_projection']['L2'] for r in rows];k=[r['unit_KD_projection']['L2'] for r in rows]
    sufficient=all(math.isfinite(a) and a>0 for a in p+k) and all(r['parameter_grads_none'] for r in rows)
    sufficient=sufficient and read(RUN/'original-before.json')==read(RUN/'original-after.json') and read(RUN/'adapted-before.json')==read(RUN/'adapted-after.json')
    # One closed-form conservative bound, not a sweep or a development-score fit.
    photo_ceiling=min(a/b for a,b in zip(p,k)) if sufficient else None
    sum_p=groups['positive_sum']['L2'];sum_k=groups['KD_sum']['L2']
    group_sum_ceiling=sum_p/sum_k if sum_k else None
    coefficient=.5*min(photo_ceiling,group_sum_ceiling) if sufficient and group_sum_ceiling else None
    details=[]
    for r in rows:
        ratio=coefficient*r['unit_ratio'] if coefficient is not None else None
        details.append(dict(image_id=r['image_id'],source_group=r['source_group'],terms=r['terms'],
                            positive_L2=r['positive_projection']['L2'],unit_KD_L2=r['unit_KD_projection']['L2'],
                            unit_ratio=r['unit_ratio'],cosine=r['cosine'],weighted_ratio=ratio,
                            positive_descent_fraction=1+ratio*r['cosine'] if ratio is not None else None,
                            KD_cosine_past_projection_displacement=r['KD_cosine_past_projection_displacement']))
    norms=sorted((r['KD']['L2'] for r in groups['records']),reverse=True)
    actual=dict(status='PASS' if sufficient else 'INSUFFICIENT',forwards=done['forwards'],contractions=done['contractions'],
        VJPs=done['VJPs'],seconds=done['seconds'],wall_seconds=read(RUN/'launch-result.json')['wall_seconds'],
        peak_allocated_bytes=done['peak_allocated_bytes'],peak_reserved_bytes=done['peak_reserved_bytes'],
        GPU_execution_new_attempts=1,optimizer_updates=0,photos=15,source_groups=15,terms=267,
        original_KD_loss=original['unit_KD_loss'],original_exact_zero=original['exact_zero_derivative'],
        original_forward_logit_max_error=original['replay']['logits']['max_absolute_difference'],
        all_adapted_forwards_bit_identical=all(all(v['bit_identical'] for v in r['replay'].values()) for r in rows),
        protected_output_overlaps=sum(r['protected_overlap'] for r in rows),
        positive_projection_L2=distribution(p),unit_KD_projection_L2=distribution(k),
        ratios=distribution([r['unit_ratio'] for r in rows]),cosines=distribution([r['cosine'] for r in rows]),
        positive_KD_opposition_photos=sum(r['cosine']<0 for r in rows),
        KD_opposes_past_displacement_photos=sum(r['KD_cosine_past_projection_displacement']>0 for r in rows),
        KD_displacement_cosines=distribution([r['KD_cosine_past_projection_displacement'] for r in rows]),
        source_sum_cosine=groups['cosine_of_source_sums'],
        top1_share_individual_KD_norms=norms[0]/sum(norms),top3_share_individual_KD_norms=sum(norms[:3])/sum(norms),
        source_norm_share_is_not_vector_sum_share=True,weighted_source_sum_ratio=coefficient*sum_k/sum_p if coefficient else None,
        selected_Chair_scores_ge015=sum(r['selected_student_scores_ge015'] for r in rows),
        numerical_earliest_nonfinite=None,
        loss_scale=1.,backward='FP32 observed-node classification adjoint;one native box VJP at scale1',
        box_probe=read(RUN/'box-probe.json'),coefficient=coefficient,
        formula='0.5*min(min_photo(||g_positive||/||g_unit_KD||),||sum_source_mean(g_positive)||/||sum_source_mean(g_unit_KD)||)',
        photograph_bound=photo_ceiling,source_sum_bound=group_sum_ceiling,
        weighted_ratios=distribution([d['weighted_ratio'] for d in details if d['weighted_ratio'] is not None]),
        positive_descent_fractions=distribution([d['positive_descent_fraction'] for d in details if d['positive_descent_fraction'] is not None]),
        limitations=['Local Euclidean first-order bound,not an AdamW-preconditioned update guarantee',
                     '15 source groups only;267terms mostly already-confident negatives;other6classes unanchored',
                     'All measured KD directions conflict with positive gradients;small coefficient is intentional',
                     'Does not prove shared seven-class confidence drift will fall or detection precision/recall improve',
                     'Box backward checked on one adapted photograph;not every future training state',
                     'No development or independent evaluation data used for coefficient selection'])
    # Verify typed telemetry, not just the absence of an exception.
    telemetry=[]
    for path in (RUN/'measurements').rglob('event-*.json'):
        event=read(path);telemetry.append(event)
        if event.get('earliest_observed_nonfinite') or any(t.get('nonfinite',0) for t in event.get('tensors',[])):
            raise ValueError('Persisted nonfinite telemetry despite terminal PASS')
    actual['typed_measurement_telemetry_events']=len(telemetry)
    actual['measurement_dtypes']=sorted({t['dtype'] for e in telemetry for t in e.get('tensors',[]) if t.get('present')})
    emit(REPORT/'gradient-findings.json',actual);emit(REPORT/'per-photo.json',dict(records=details))
    emit(REPORT/'decision.json',dict(status='FINITE_NONZERO_RECOVERY',coefficient=coefficient,
         coefficient_proposed_not_authorized_for_training=True,training_ready=sufficient,training_authorized=False,
         reason='All15 active source groups recovered finite nonzero parameter gradients with unchanged states;one conservative training-only coefficient bounded by positive gradients.',
         remaining_cautions=actual['limitations'],GPU_additional_attempts=0,performance_improvement_claimed=False))
    print('CPU analysis:coefficient',coefficient,';weighted ratio range',actual['weighted_ratios'],';no training',flush=True)


if __name__=='__main__':run()
