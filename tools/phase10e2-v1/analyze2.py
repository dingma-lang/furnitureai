"""CPU-only analysis of saved gradients;no inference,holdout metrics,or coefficient sweep."""
from common2 import *
import numpy as np

def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),minimum=float(a.min()) if len(a) else None,median=float(np.median(a)) if len(a) else None,mean=float(a.mean()) if len(a) else None,maximum=float(a.max()) if len(a) else None)

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU analysis only')
    terminal=read(RUN/('complete.json' if (RUN/'complete.json').exists() else 'failure.json'))
    original=[read(p) for p in sorted((RUN/'original').glob('*.json'))];adapted=[read(p) for p in sorted((RUN/'adapted').glob('*.json'))]
    for r in original+adapted:
        if sha(RUN/r['raw']['file'])!=r['raw']['sha256']:raise ValueError('Saved per-photo output changed')
    replay={}
    for name,rows in (('original',original),('C11',adapted)):
        replay[name]=dict(photos=len(rows),bit_identical_photos=sum(all(v['bit_identical'] for v in r['replay'].values()) for r in rows),maximum_logit_error=max([r['replay']['logits']['max_absolute_difference'] for r in rows],default=None),maximum_square_box_error=max([r['replay']['square_cxcywh']['max_absolute_difference'] for r in rows],default=None),all_within_predeclared_tolerances=all(v['accepted'] for r in rows for v in r['replay'].values()))
    active=[r for r in adapted if r['measurement']['eligible_distillation_terms']>0]
    ratios=[r['measurement']['unit_distill_positive_ratio'] for r in active if r['measurement']['unit_distill_positive_ratio'] is not None]
    cosines=[r['measurement']['projection_cosine'] for r in active if r['measurement']['projection_cosine'] is not None]
    opposition=[r['measurement']['distill_gradient_cosine_observed_projection_drift'] for r in active if r['measurement']['distill_gradient_cosine_observed_projection_drift'] is not None]
    # One analytic safety ceiling,not a coefficient sweep or an automatic recommendation.
    ceilings=[r['measurement']['positive_projection']['L2']/r['measurement']['unit_distill_projection']['L2'] for r in active if r['measurement']['unit_distill_projection']['L2']>0]
    ceiling=min(ceilings) if ceilings else None
    conservative=.5*ceiling if ceiling is not None else None
    scaled=[]
    if conservative is not None:
        for r in active:
            m=r['measurement'];ratio=conservative*m['unit_distill_positive_ratio'];c=m['projection_cosine']
            scaled.append(dict(image_id=r['image_id'],source_group=r['source_group'],unit_ratio=m['unit_distill_positive_ratio'],weighted_ratio=ratio,positive_descent_fraction=1+ratio*c if c is not None else None,projection_cosine=c))
    source=read(RUN/'source-groups.json') if (RUN/'source-groups.json').exists() else None
    source_active=[r for r in source['records'] if r['terms']] if source else []
    norms=sorted([r['distillation']['L2'] for r in source_active],reverse=True)
    group_ratios=[r['distillation']['L2']/r['positive']['L2'] for r in source_active if r['positive']['L2']]
    measurements=dict(status=terminal.get('status','INCOMPLETE'),resources={k:terminal.get(k,terminal.get('state',{}).get(k)) for k in ('forwards','actual_forwards','gradient_calculations','analytic_zero_distill_derivatives','seconds','peak_allocated_bytes','peak_reserved_bytes','allocated_output_bytes')},replay=replay,original=dict(photos=len(original),selected_terms=sum(r['measurement']['eligible_distillation_terms'] for r in original),positive_projection_norms=stats([r['measurement']['positive_projection']['L2'] for r in original]),unit_distillation_projection_norms=stats([r['measurement']['unit_distill_projection']['L2'] for r in original]),maximum_distill_loss=max([r['measurement']['losses']['unit_distillation'] for r in original],default=None)),adapted=dict(photos=len(adapted),photos_with_distillation=len(active),source_groups_with_distillation=len({r['source_group'] for r in active}),selected_terms=sum(r['measurement']['eligible_distillation_terms'] for r in adapted),positive_projection_norms=stats([r['measurement']['positive_projection']['L2'] for r in adapted]),active_distill_projection_norms=stats([r['measurement']['unit_distill_projection']['L2'] for r in active]),unit_distill_positive_ratios=stats(ratios),positive_distill_cosines=stats(cosines),cosine_gradient_vs_observed_projection_drift=stats(opposition),opposes_past_projection_displacement_photos=sum(v>0 for v in opposition),positive_gradient_conflict_photos=sum(v<0 for v in cosines),mean_seven_class_logit_drift=float(np.mean([r['measurement']['diagnostics']['mean_seven_category_logit_drift'] for r in adapted])) if adapted else None,per_class_logit_drift={c:float(np.mean([r['measurement']['diagnostics']['per_class_logit_drift'][c] for r in adapted])) if adapted else None for c in CLASSES},matched_Chair_scores_below_teacher=sum(r['measurement']['mask']['matched_Chair_student_score_below_teacher'] for r in adapted),protected_positive_distill_overlap=sum(r['measurement']['positive_distill_output_overlap'] for r in original+adapted),teacher_conflicts_excluded=sum(r['measurement']['mask']['teacher_conflicts_excluded'] for r in adapted)),source_groups=dict(measured=len(source['records']) if source else 0,active=len(source_active),unit_norm_ratios=stats(group_ratios),top1_share_sum_individual_norms=norms[0]/sum(norms) if norms else None,top3_share_sum_individual_norms=sum(norms[:3])/sum(norms) if norms else None,actual_gradient_sum_cosine=source['cosine_of_sums'] if source else None,individual_norm_share_is_not_fraction_of_vector_sum=True),one_analytic_scale=dict(non_domination_ceiling=ceiling,half_ceiling_diagnostic_scalar=conservative,formula='0.5 * min_measured_photo(||positive_projection_gradient||/||unit_KD_projection_gradient||)',worst_case_rationale='KDnorm<=halfpositiveonmeasuredactivephotos;evenantiparalleldirectionretains>=halfpositivefirstorderdescentinplainEuclideanmetric',weighted_ratios=stats([r['weighted_ratio'] for r in scaled]),positive_descent_fractions=stats([r['positive_descent_fraction'] for r in scaled if r['positive_descent_fraction'] is not None]),selected_for_training=False,limitations=['Not a bound on AdamW-preconditionedupdates','Not a guaranteeacrossfuturetrainingstatesorall59positivegradients','No informationaboutperformanceonunknown/uncoveredrooms orother6classes']),other_six_classes_distilled=False,box_gradient_routing_violations=0,optimizer_steps=0,training_authorized=False,holdout_scoring=False,metrics_improvement_claimed=False)
    emit(REPORT/'measurements.json',measurements);emit(REPORT/'analytic-scale-details.json',dict(records=scaled))
    emit(REPORT/'source-groups.json',dict(records=[{k:r[k] for k in ('source_group','photos','terms','cosine')}|dict(positive_L2=r['positive']['L2'],unit_distill_L2=r['distillation']['L2']) for r in source['records']] if source else []))
    print('E2 CPU gradient analysis complete',json.dumps(measurements['adapted']['unit_distill_positive_ratios']),';analytic scalar',conservative,flush=True)

if __name__=='__main__':run()
