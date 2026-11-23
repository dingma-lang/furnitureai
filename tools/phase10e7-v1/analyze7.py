"""CPU-only summaries/evidence joins from immutable E5 optimization caches."""
import math
from collections import Counter,defaultdict
import numpy as np
from common7 import *

THRESHOLDS=(.15,.5,.8)

def evidence_for_query(q,box,score,original_score,row,photo,protected,legacy,reported,poly_cache):
    positives=[a for a in row['annotations'] if a['label']=='chair' and not a['ignore']]
    best=max([iou(box,a['box']) for a in positives],default=0.)
    fragment=any(intersection(box,a['box'])/max(area(box),1e-30)>=.8 and area(box)<.6*area(a['box']) and iou(box,a['box'])<.5 for a in positives)
    blocked=[p['key'] for p in protected if overlap(box,p['box'])]
    valid=0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
    source=[]
    for ann in photo['annotations']:
        f=intersection(box,ann['box'])/max(area(box),1e-30)
        if f>=.7:source.append((f,ann))
    source.sort(key=lambda a:(-a[0],a[1]['annotation_key']))
    ann=source[0][1] if source else None
    geometry=[]
    if valid and not blocked:
        sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
        from geometry6 import polygons,certify
        sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
        from review21 import semantic_protected
        px=[math.floor(box[0]*row['width']),math.floor(box[1]*row['height']),math.ceil(box[2]*row['width']),math.ceil(box[3]*row['height'])]
        if min(px[2]-px[0],px[3]-px[1])>=32:
            for _,a in source:
                if a['native_ignored'] or semantic_protected(a):continue
                key=(row['image_id'],a['annotation_key'])
                if key not in poly_cache:poly_cache[key]=polygons(a['original_annotation'],row['width'],row['height'])
                for pi,p in enumerate(poly_cache[key]):
                    if certify(px,p):geometry.append(dict(annotation_key=a['annotation_key'],native_label=a['native_label'],polygon_index=pi))
    contained=[r['candidate_id'] for r in legacy if r['image_id']==row['image_id'] and r['box'][0]<=box[0]<box[2]<=r['box'][2] and r['box'][1]<=box[1]<box[3]<=r['box'][3]]
    existing_overlap=[r['candidate_id'] for r in legacy if r['image_id']==row['image_id'] and overlap(box,r['box'])]
    if best>=.5:relation='verified_Chair_positive_overlap_not_a_mistake'
    elif fragment:relation='potential_real_Chair_fragment_protected'
    elif blocked:relation='protected_or_incompletely_annotated_furniture_overlap'
    elif not valid:relation='outside_original_frame_not_negative_eligible'
    elif not reported:relation='not_reported_Chair_detection'
    else:relation='unmatched_source_context_or_unknown_not_confirmed_false_positive'
    plausible=reported and valid and not blocked and not fragment and best<.5 and ann is not None
    return dict(query_index=q,box=box,student_Chair_score=score,original_same_query_Chair_score=original_score,
        verified_Chair_IoU=best,potential_fragment=fragment,protected_overlap_keys=blocked,within_original_frame=valid,
        source_context=semantic(ann) if ann else 'uncertain_or_unannotated_real_object',native_label=ann['native_label'] if ann else None,
        native_annotation_key=ann['annotation_key'] if ann else None,geometry_source_support=geometry,
        reported_Chair_detection=reported,interpretation=relation,existing_negative_region_overlap=existing_overlap,
        existing_negative_region_full_containment=contained,trusted_existing_region_high_query_coverage=bool(contained and not blocked and valid),
        evidence_tier='B_automatic_unverified' if plausible else 'C_uncertain_protected_or_rejected',
        training_eligible=False,training_activated=False,human_reviewed=False,negative_label_created=False)

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only analysis required')
    cfg,c,train=inputs();done=read(RUN/'complete.json');raw=read(RUN/'raw-complete.json')
    if done['status']!='PASS' or not done['model_before']==done['model_after'] or len(raw['records'])!=59:raise ValueError('Successful immutable59-image cache required')
    native=read(ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json');photos={p['image_id']:p for p in native['photos']}
    reg={p['image_id']:p for p in read(ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json')['photos']}
    legacy=read(ROOT/'data/open-images/preparation/phase10c6-v1/source-negative-evidence.json')['records']
    refs={r['image_id']:r for r in read(PREP/'teacher-index.json')['records']}
    old=read(ROOT/'data/open-images/preparation/phase10e6-v1/query-index-v3.json')
    old_snapshots={r['snapshot_id'] for r in old['records']}
    old_candidates={r['candidate_id'] for r in read(ROOT/'data/open-images/preparation/phase10e6-v1/candidate-evidence-v1.json')['records']}
    old_hotspots=set()
    for line in (ROOT/'data/open-images/preparation/phase10e6-v1/high-confidence-observations.jsonl').read_text().splitlines():
        o=json.loads(line)
        if o['target']=='chair':old_hotspots.add((photos[o['image_id']]['sha256'],int(o['query_index'])))
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from review21 import semantic_protected
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));import metric_cpu as metric
    totals={str(t):{k:Counter() for k in ('raw_class_terms','raw_argmax_queries','postNMS_argmax')} for t in THRESHOLDS}
    counts=Counter();per_photo=[];observations=[];poly_cache={};new_snapshots=set();current_hotspots=set();current_footprints=set();canon_seen={};known_group_photos=defaultdict(list)
    for n,rec in enumerate(raw['records'],1):
        row=require_train(rec['image_id'],train);photo=photos[row['image_id']];path=RUN/rec['file']
        if rec['source_sha256']!=row['sha256'] or rec['source_group']!=row['identity_group'] or sha(path)!=rec['sha256']:raise ValueError('Cached source/array mismatch')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        logits=values['logits'];square=values['square_cxcywh'];norm=values['original_xyxy_normalized'];scores=1/(1+np.exp(-logits.astype(float)))
        expected=raw_values(logits,square,row['width'],row['height'])
        if set(expected)!=set(values) or any(not np.array_equal(values[k],v) for k,v in expected.items()):raise ValueError('Array/query/coordinate reconstruction mismatch')
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Original photo changed')
        canonical=photo['rights']['source_photo_id'];known_group_photos[row['identity_group']].append(row['image_id'])
        key=(row['sha256'],canonical)
        duplicate_photo=key in canon_seen
        # An exact SHA or same source-photo-ID is sufficient to avoid double
        # counting;known creator groups alone are not photograph identities.
        if row['sha256'] in {v[0] for v in canon_seen} or canonical in {v[1] for v in canon_seen}:duplicate_photo=True
        canon_seen[key]=row['image_id']
        numerical_hash=hashlib.sha256(logits.tobytes()+square.tobytes()).hexdigest()
        sid='Q-'+digest(row['image_id']+':'+numerical_hash)[:20];new_snapshots.add(sid)
        argmax=logits.argmax(1);clipped=np.clip(norm,0,1);proposals=[]
        for q in np.flatnonzero(scores.max(1)>=.15):
            if area(clipped[q])>0:proposals.append(dict(q=int(q),box=clipped[q].tolist(),score=float(scores[q,argmax[q]]),label=CLASSES[int(argmax[q])]))
        kept=metric.nms(proposals,.5,300);reported_chair={p['q'] for p in kept if p['label']=='chair' and p['score']>=.5}
        threshold_counts={}
        for t in THRESHOLDS:
            tkey=str(t);byclass={}
            for ci,cl in enumerate(CLASSES):
                item=dict(raw_class_terms=int((scores[:,ci]>=t).sum()),raw_argmax_queries=int(((argmax==ci)&(scores[:,ci]>=t)).sum()),postNMS_argmax=sum(p['label']==cl and p['score']>=t for p in kept))
                byclass[cl]=item
                if not duplicate_photo:
                    for metric_name,value in item.items():totals[tkey][metric_name][cl]+=value
            threshold_counts[tkey]=byclass
        entry=next(e for e in c['entries'] if e['row']['image_id']==row['image_id'])
        protected=[dict(key='active:'+str(i),box=a['box']) for i,a in enumerate(row['annotations'])]
        protected += [dict(key='registry:'+str(i),box=a['box']) for i,a in enumerate(reg[row['image_id']]['protected'])]
        protected += [dict(key='native:'+a['annotation_key'],box=a['box']) for a in photo['annotations'] if semantic_protected(a) or a['native_ignored']]
        for k in ('legacy_protected','distill_protected'):protected += [dict(key=k+':'+str(i),box=b) for i,b in enumerate(entry[k])]
        teacher=refs[row['image_id']]
        if sha(ROOT/teacher['cache'])!=teacher['cache_sha256']:raise ValueError('Original teacher cache changed')
        with np.load(ROOT/teacher['cache'],allow_pickle=False) as z:ts=1/(1+np.exp(-z['logits'].astype(float)));tb=registered(z['square_cxcywh'],row['width'],row['height'])
        chair=CLASSES.index('chair');photo_obs=[]
        for q in np.flatnonzero(scores[:,chair]>=.5):
            q=int(q);box=norm[q].tolist()
            r=evidence_for_query(q,box,float(scores[q,chair]),float(ts[q,chair]),row,photo,protected,legacy,q in reported_chair,poly_cache)
            cid='HN-'+digest(dict(image_id=row['image_id'],target='chair',box_pixels=[round(v*s,3) for v,s in zip(box,[row['width'],row['height']]*2)]))[:20]
            r.update(image_id=row['image_id'],source_sha256=row['sha256'],source_photo_id=canonical,source_group=row['identity_group'],snapshot_id=sid,
                  candidate_id=cid,target='chair',width=row['width'],height=row['height'],source_path=row['path'],
                  previously_observed_exact_footprint=cid in old_candidates,previously_observed_high_Chair_query=(row['sha256'],q) in old_hotspots,
                  original_same_query_box_IoU=iou(box,tb[q].tolist()),query_comparison_not_guaranteed_same_object=True)
            # Spatial redundancy is a measured overlap,not an inferred label.
            others=[p for p in kept if p['q']!=q and p['label']=='chair' and p['score']>=.5]
            other=max(others,key=lambda p:iou(box,p['box']),default=None)
            r['other_reported_Chair_query_IoU']=iou(box,other['box']) if other else 0.
            r['other_reported_Chair_query']=other['q'] if other else None
            observations.append(r);photo_obs.append(r)
            current_hotspots.add((row['sha256'],q));current_footprints.add(cid)
        counts.update(dict(duplicate_photographs_not_double_counted=int(duplicate_photo)))
        per_photo.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],source_group=row['identity_group'],source_photo_id=canonical,
             snapshot_id=sid,snapshot_seen_in_E6=sid in old_snapshots,duplicate_photo_not_double_counted=duplicate_photo,thresholds=threshold_counts,
             high_Chair_queries=len(photo_obs),Chair_evidence_tiers=dict(Counter(r['evidence_tier'] for r in photo_obs))))
        if n%15==0:print('CPU analyzed',n,'/59',flush=True)
    atomic(RUN/'analysis/high-confidence-Chair.jsonl',b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in observations))
    emit(RUN/'analysis/per-photo-summary.json',dict(records=per_photo))
    grouped_candidates=defaultdict(list)
    for r in observations:grouped_candidates[r['candidate_id']].append(r)
    candidate_B={k for k,v in grouped_candidates.items() if any(r['evidence_tier']=='B_automatic_unverified' for r in v)}
    candidate_C=set(grouped_candidates)-candidate_B
    emit(RUN/'analysis/candidate-manifest.json',dict(schema='phase10e7-inactive-training-query-evidence-v1',binding_sha256=sha(RUN/'binding.json'),
        raw_manifest_sha256=sha(RUN/'raw-complete.json'),candidate_observations='analysis/high-confidence-Chair.jsonl',
        candidate_sha256=sha(RUN/'analysis/high-confidence-Chair.jsonl'),training_eligible_new=[],new_labels_activated=0,
        source_supported_geometry_candidate_ids=sorted({r['candidate_id'] for r in observations if r['geometry_source_support']}),
        B_automatic_unverified=sorted(candidate_B),C_uncertain_protected_or_rejected=sorted(candidate_C),
        prior_valid_positives_and_negatives_unchanged=True,uncertain_background_ignored=True,human_reviews_created=0))
    summary=dict(photos=59,known_source_groups=56,raw_queries=59*3600,class_logit_and_score_terms=59*3600*7,
        thresholds={k:{n:dict(v) for n,v in d.items()} for k,d in totals.items()},NMS=dict(IoU=.5,max_per_photo=300,minimum_score=.15,class_aware=True),
        high_Chair=dict(raw_ge05=len(observations),raw_ge08=sum(r['student_Chair_score']>=.8 for r in observations),
            photographs=len({r['image_id'] for r in observations}),source_groups=len({r['source_group'] for r in observations}),
            distinct_footprints=len(grouped_candidates),footprint_B_unverified=len(candidate_B),footprint_C_uncertain=len(candidate_C),
            source_contexts=dict(Counter(r['source_context'] for r in observations)),interpretations=dict(Counter(r['interpretation'] for r in observations)),
            tiers=dict(Counter(r['evidence_tier'] for r in observations)),
            native_polygon_geometry_candidates=sum(bool(r['geometry_source_support']) for r in observations),
            protected_overlap=sum(bool(r['protected_overlap_keys']) for r in observations),known_positive_IoU_ge05=sum(r['verified_Chair_IoU']>=.5 for r in observations),
            potential_fragments=sum(r['potential_fragment'] for r in observations),potential_spatial_duplicates=sum(r['other_reported_Chair_query_IoU']>=.5 for r in observations),
            existing_negative_overlap=sum(bool(r['existing_negative_region_overlap']) for r in observations),existing_negative_containment=sum(bool(r['existing_negative_region_full_containment']) for r in observations),
            trusted_existing_region_new_high_query_coverage=sum(r['trusted_existing_region_high_query_coverage'] for r in observations),
            original_same_query_below015=sum(r['original_same_query_Chair_score']<.15 for r in observations)),
        deduplication=dict(E6_distinct_snapshots=len(old_snapshots),E7_distinct_snapshots=len(new_snapshots),E7_exact_snapshot_repeats_of_E6=len(new_snapshots&old_snapshots),
            combined_distinct_snapshots=len(new_snapshots|old_snapshots),high_Chair_current_unique_patch_hotspots=len(current_hotspots),
            high_Chair_patch_hotspots_previously_seen=len(current_hotspots&old_hotspots),high_Chair_new_patch_hotspots=len(current_hotspots-old_hotspots),
            combined_unique_Chair_patch_hotspots=len(current_hotspots|old_hotspots),E7_exact_Chair_footprints_seen_in_E6=len(current_footprints&old_candidates),
            E7_exact_Chair_footprints_not_seen_in_E6=len(current_footprints-old_candidates),
            duplicate_photos_not_double_counted=counts['duplicate_photographs_not_double_counted'],unique_photo_hashes=len({r['source_sha256'] for r in per_photo}),unique_source_photo_ids=len({r['source_photo_id'] for r in per_photo}),
            known_multi_photo_creator_groups=sum(len(v)>1 for v in known_group_photos.values()),
            limitations='Exact image/source-photo identity and row-major query trajectory are deduplicated;creator groups are not merged into one photo,nearby boxes do not establish physical identity,and new hotspots are not confirmed new errors'),
        genuinely_new_trusted_negative_labels=0,automatic_training_labels=0,unmatched_predictions_are_not_confirmed_FP=True,
        all_confusion_candidates_unverified=True,new_inference_during_analysis=0,development_or_reserved_mined=False)
    emit(REPORT/'query-summary.json',summary)
    print('E7CPU summary complete;Chair>=0.5',len(observations),';new labels0;no automatic negative activation',flush=True)

if __name__=='__main__':run()
