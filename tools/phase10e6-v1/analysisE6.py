"""CPU aggregation of already saved observations;never evaluates a detector."""
from collections import Counter,defaultdict
from commonE6 import *

def run():
    import numpy as np
    _,train,_=context()
    obs=[json.loads(s) for s in (DATA/'high-confidence-observations.jsonl').read_text().splitlines()]
    grouped=defaultdict(list)
    for r in obs:grouped[(r['snapshot_id'],r['target'])].append(r)
    duplicate_rows=[]
    for group in grouped.values():
        for r in group:
            # A spatial relationship,not a claim of object identity or error.
            alternatives=[s for s in group if s['query_index']!=r['query_index'] and s['reported_target_detection']]
            best=max(alternatives,key=lambda s:iou(r['box'],s['box']),default=None)
            duplicate_rows.append(dict(candidate_id=r['candidate_id'],snapshot_id=r['snapshot_id'],query_index=r['query_index'],target=r['target'],
                  overlaps_another_reported_same_class_query=bool(best and iou(r['box'],best['box'])>=.5),
                  best_reported_query=best['query_index'] if best else None,best_IoU=iou(r['box'],best['box']) if best else 0,
                  not_a_confirmed_duplicate_or_false_positive=True))
    emit(DATA/'duplicate-query-relationships.json',dict(records=duplicate_rows))
    per_target={}
    for t in ('chair','sofa','bed'):
        rows=[r for r in obs if r['target']==t]
        per_target[t]=dict(observations=len(rows),ge08=sum(r['student_score']>=.8 for r in rows),
           photographs=len({r['image_id'] for r in rows}),source_groups=len({r['source_group'] for r in rows}),
           source_context_ge05=dict(Counter(r['source_context'] for r in rows)),
           source_context_ge08=dict(Counter(r['source_context'] for r in rows if r['student_score']>=.8)),
           source_groups_ge08=len({r['source_group'] for r in rows if r['student_score']>=.8}),
           confidence_bands={'0.50_to_under_0.80':sum(r['student_score']<.8 for r in rows),'0.80_to_1.00':sum(r['student_score']>=.8 for r in rows)},
           original_same_query_below015=sum(r['original_score_same_query']<.15 for r in rows),
           original_same_query_ge05=sum(r['original_score_same_query']>=.5 for r in rows),
           exact_protected_overlap_observations=sum(r['protected_overlap_count']>0 for r in rows),
           suspected_spatial_redundancy=sum(r['overlaps_another_reported_same_class_query'] for r in duplicate_rows if r['target']==t),
           existing_negative_region_overlap_observations=sum(bool(r['existing_negative_overlap']) for r in rows),
           prior_inactive_negative_overlap_observations=sum(bool(r['prior_inactive_negative_overlap']) for r in rows))
    emit(REPORT/'confidence-coverage.json',dict(per_target=per_target,
         counts_repeat_model_states=True,raw_queries_are_not_independent_mistakes=True,
         duplicate_measurement='Same-image/state/class query IoU>=0.5 with another reported query;not instance identity',
         original_comparison='Same pinned patch/query index on same source image;box heads can differ,so not necessarily the same object',
         source_context='>=70%query area inside native BBOX is context only,not full visible mask containment',
         training_eligible_new=0))
    per_photo=[]
    for ident,row in train.items():
        rows=[r for r in obs if r['image_id']==ident]
        per_photo.append(dict(image_id=ident,source_group=row['identity_group'],sha256=row['sha256'],
             targets={t:dict(ge05=sum(r['target']==t for r in rows),ge08=sum(r['target']==t and r['student_score']>=.8 for r in rows)) for t in ('chair','sofa','bed')}))
    emit(DATA/'per-photo-confidence-coverage.json',dict(records=per_photo))
    # Task1 integrity-only comparison,not mining or new holdout scoring.
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from analyzeD import load_raw
    _,_,dev=context()
    a=load_raw(ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/predictions',dev)
    b=load_raw(ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/evaluation/epoch-5',dev)
    diffs=np.concatenate([(x['logits'].astype(float)-y['logits'].astype(float)).ravel() for x,y in zip(a,b)])
    emit(REPORT/'numerical-difference.json',dict(raw_logit_RMS_difference=float(np.sqrt(np.mean(diffs**2))),raw_logit_mean_difference=float(diffs.mean()),
          raw_logit_max_abs_difference=float(np.abs(diffs).max()),
          raw_argmax_changes=sum(int((x['logits'].argmax(1)!=y['logits'].argmax(1)).sum()) for x,y in zip(a,b)),
          query_count=19*3600,maximum_square_box_difference=max(float(np.abs(x['square_cxcywh'].astype(float)-y['square_cxcywh']).max()) for x,y in zip(a,b)),
          new_inference=False,development_used_for_mining=False))
    template=read(DATA/'independent-review-template.json')
    template['required_fields']+=['creator','source_url','license_version','rights_verified_at']
    emit(DATA/'independent-review-template-v2.json',template)
    print('Existing-query coverage,spatial redundancy and numerical differences recorded;no new labels')

if __name__=='__main__':run()
