"""Training-only mechanism, teacher availability and actual-update evidence."""
import math
from collections import Counter,defaultdict
import numpy as np
from common14 import *
from mining14 import read_derived


def run():
    _,train,dev,photos,registry=training_context();summaries=read(RAW14/'query-summaries.json')['records'];byphoto=defaultdict(dict)
    for r in summaries:byphoto[r['image_id']][r['state']]=r
    drift={};assignment_change={};teacher_coverage={};negative=[]
    for state in STATES:
        if state['key']=='original':continue
        delta=[];same_gt=changed_gt=0;common_negative=0;teacher_positive_on_safe=0
        for ident,versions in byphoto.items():
            orig=versions['original'];current=versions[state['key']];a=load_npz(ROOT/orig['query_file']);b=load_npz(ROOT/current['query_file']);delta.append(b['logits'].astype(float)-a['logits'].astype(float));d0=read_derived(ROOT/orig['derived_file']);d1=read_derived(ROOT/current['derived_file'])
            aa={r['annotation_index']:r['query_index'] for r in orig['matches']};bb={r['annotation_index']:r['query_index'] for r in current['matches']}
            if set(aa)!=set(bb):raise ValueError('Positive annotation membership changed')
            same_gt+=sum(aa[k]==bb[k] for k in aa);changed_gt+=sum(aa[k]!=bb[k] for k in aa)
            # Conservative KD-admission diagnostic: same patch AND safe Chair
            # negative in both original and student; this is not a training mask.
            shared=d0['added_available'][:,1].astype(bool)&d1['added_available'][:,1].astype(bool);common_negative+=int(shared.sum());prob=1/(1+np.exp(-a['logits'][:,1].astype(float)));teacher_positive_on_safe+=int((shared&(prob>=.15)).sum())
        d=np.concatenate(delta);drift[state['key']]=dict(mean_logit_change_by_category={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},fraction_all7_logits_increase=float((d>0).all(1).mean()),fraction_Chair_logits_increase=float((d[:,1]>0).mean()),common_shift_quantiles=quantiles(d.mean(1)),photo_query_count=len(d),precision='Saved FP16-forward logits cast toFP32;not mixed with development FP32 caches')
        assignment_change[state['key']]=dict(positive_objects=same_gt+changed_gt,original_match_query_retained=same_gt,positive_objects_with_different_query_index=changed_gt,query_indices_are_dense_patches_not_persistent_objects=True)
        teacher_coverage[state['key']]=dict(common_safe_Chair_query_indices=common_negative,teacher_Chair_probability_at_least015_on_these=teacher_positive_on_safe,no_all7_absence_claim=True)
    actual={}
    for key,root,applied in [('c8_control',C8/'A-control',False),('c8_corrected',C8/'B-corrected',True),('c11',C11_RUN,True),('c13',RUN13,True)]:
        updates=[read(root/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)];images=[r for u in updates for r in u['images']];scores=[s for r in images for s in r['added_Chair_scores']]
        if len(images)!=295 or {r['image_id'] for r in images}!={r['image_id'] for r in train}:raise ValueError('Actual training records outside exact59photo pool')
        actual[key]=dict(updates=75,photo_presentations=295,available_source_term_occurrences=sum(r['added_active_Chair_terms'] for r in images),source_terms_applied=applied,source_negative_score_quantiles=quantiles(scores),source_negative_terms_at_least015=sum(s>=.15 for s in scores),source_negative_terms_at_least05=sum(s>=.5 for s in scores),source_negative_terms_at_least08=sum(s>=.8 for s in scores),protected_violations=sum(r['protected_violations'] for r in images),
            caveat='Actual per-update aggregates and source-negative scores;full logits/boxes/assignments for these295 presentations were not retained. Control source terms are counterfactual,not applied.')
    c7=read(ROOT/'reports/phase10c7-v1/results.json')
    save(REPORT14/'training-mechanism.json',dict(training_positive_boxes=dict(Counter(a['label'] for r in train for a in r['annotations'] if not a['ignore'])),explicit_native_negative_photos=dict(Counter(c for r in train for c,v in r['explicit_native_negative'].items() if v)),
        actual_training=actual,cached_training_logit_drift=drift,matching_query_changes=assignment_change,C7_reference=dict(actual_new_negative_queries=c7['actual_new_Chair_negative_queries'],photos_with_coverage=c7['photos_with_nonzero_coverage'],regions_with_coverage=c7['regions_with_coverage'],uncovered_regions=c7['zero_coverage_regions']),
        normalization='Every image uses sum(active focal terms)/positiveGTcount. Positives and negatives have the SAME divisor. Focal alpha-positive=.25/negative=.75,gamma2. No count-normalization or all-background loss was added.',
        confirmed=['Only one Hungarian-positive query per activeGT class;other class-query terms remain ignored unless separately evidenced','All four unsupported positive classes have no direct classification target in this training pool','Uniform logit offset directional derivatives are computed conditional on reconstructed fixed assignment/masks;negative derivative means this objective locally rewards a shared boost','C11/C13 training probes show increases in directly unsupervised category scores with frozen backbone/calibration and shared projection adaptation','Current source-region containment is strict whole predicted box,not center overlap;covered negatives are predominantly low-confidence'],
        limitations=['Saved norms do not recover separate signed positive/native-negative/source-negative head gradients','Without training pre-head features,CPU output gradients cannot be converted into exact per-query projection gradients','Hungarian assignments were reconstructed onCPU;matching loss/mask agreement cannot prove original GPU tie breaking','Raw repeated/fragmented high-score queries are not automatically incorrect detections or distinct physical objects'],model_forwards=0,training_updates=0))
    save(REPORT14/'teacher-feasibility.json',dict(teacher='Immutable original OWLv2',available_training_teacher_photos=24,training_photos_missing_teacher_outputs=35,teacher_classes=list(CLASSES),teacher_queries_per_photo=3600,teacher_precision='Original FP16 training-probe forwards cast toFP32;bound source/preprocessing/prompt hashes',teacher_cache='Canonical original state in private cache-index.json;raw arrays remain in historical C11 optimization probes',
        cache_reusable_without_duplicate_forward_every_update=True,conditions=['Same original photo bytes/orientation,processor resize/padding,prompts/precision,and no unrecorded crop/flip/augmentation','Frozen vision/text/backbone and identical60x60 dense patch lattice;verify index/frame correspondence','Index denotes a fixed patch,not stable object identity;predicted boxes/Hungarian matched indices can change','Exclude known positives/seating stools,uncertain/native ignored regions,teacher disagreements and unverified background from any negative behavior-preservation target','Only Chair-negative evidence is established by current source regions;seven saved outputs do not establish seven-category absence'],
        conservative_current_safe_negative_correspondence=teacher_coverage,
        missing_artifacts=['Original teacher outputs for35 optimization photos','Training pre-head/image and text features for exact parameter-gradient attribution','Per-update query boxes/logits,matching indices and masks for all59 photos','Saved objectness outputs','Separate signed per-image positive/native-negative/source-negative projection gradients'],
        recoverable_without_model_inference=['CPU reconstruct selected-probe assignments,masks,output-logit and box gradients','Approximate clipped aggregate parameter gradients from consecutive Adam first moments;not per-image/per-term attribution','Read already saved whole-update parameter-gradient norms and source-negative scores'],
        not_recoverable_from_saved_logits_alone=['Unique actual pre-head embeddings or objectness','Uncached teacher/student outputs for35photos or unsaved intermediate update-time queries'],
        development_features_or_errors_used_for_teacher_or_mining=False,all7_unmasked_distillation_ready=False,safe_Chair_masked_distillation_possible_on_cached24_but_narrow=True,teacher_mistakes_not_ground_truth=True,model_forwards=0))
    print('C14 training-only mechanisms and teacher dependency audit complete',flush=True)


if __name__=='__main__':run()
