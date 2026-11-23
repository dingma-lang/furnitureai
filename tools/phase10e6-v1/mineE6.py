"""Saved optimization predictions only. Source geometry is not automatic truth."""
import math
from collections import Counter,defaultdict
import numpy as np
from commonE6 import *

def run():
    c,train,dev=context()
    index_path=DATA/('query-index-v3.json' if (DATA/'query-index-v3.json').exists() else 'query-index-v2.json' if (DATA/'query-index-v2.json').exists() else 'query-index.json')
    index=read(index_path)
    full=read(ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json')
    old=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json')
    photos={p['image_id']:p for p in full['photos']};oldkeys={(p['image_id'],a['annotation_key']) for p in old['photos'] for a in p['annotations']}
    reg={p['image_id']:p for p in read(ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json')['photos'] if p['role']=='optimization'}
    legacy=read(ROOT/'data/open-images/preparation/phase10c6-v1/source-negative-evidence.json')['records']
    previous=read(ROOT/'data/open-images/preparation/phase10c21-v1/hard-negative-manifest-v1.json')
    priorA=[r for r in previous['records'] if r['candidate_id'] in previous['accepted_records']]
    teacher={r['image_id']:r for r in read(PREP/'teacher-index.json')['records']};teacher_arrays={}
    sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
    from geometry6 import polygons,certify
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from review21 import semantic_protected
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    import metric_cpu as metric
    # Revalidate recorded source and licensing provenance,offline only.
    files={}
    for source in full['source_files']:
        path=ROOT/source['source'];h=sha(path)
        if h!=source['sha256']:raise ValueError('Official local annotation source changed')
        files[source['source']]=h
    omitted=[];native_groups=Counter();rights_verified=0
    for photo in full['photos']:
        if sha(ROOT/photo['path'])!=photo['sha256']:raise ValueError('Local photo changed')
        rights=photo['rights']
        for k,hk in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
            if sha(ROOT/rights[k])!=rights[hk]:raise ValueError('Stored rights evidence changed')
        rights_verified+=1
        for a in photo['annotations']:
            if digest(a['original_annotation'])!=a['original_annotation_sha256']:raise ValueError('Native annotation lineage changed')
            if photo['image_id'] in train and (photo['image_id'],a['annotation_key']) not in oldkeys:
                omitted.append(dict(image_id=photo['image_id'],source_group=photo['source_group'],annotation_key=a['annotation_key'],native_label=a['native_label'],dataset=a['dataset'],context=semantic(a),has_mask=a['has_mask'],ignored=a['native_ignored']))
            native_groups[(photo['cohort'],semantic(a))]+=1
    if len(omitted)!=381:raise ValueError('Expected381 omitted optimization records')
    emit(DATA/'omitted-native-records.json',dict(records=omitted))
    emit(REPORT/'native-annotation-audit.json',dict(total_native_records=2272,optimization_native_records=1386,older_optimization_records=1005,omitted_records_revisited=len(omitted),
        omitted_native_labels=dict(Counter(r['native_label'] for r in omitted)),omitted_contexts=dict(Counter(r['context'] for r in omitted)),
        omitted_masks=sum(r['has_mask'] for r in omitted),omitted_ignored=sum(r['ignored'] for r in omitted),
        local_photos=101,active_optimization_photos=59,inactive_photos=42,inactive_not_activated=True,licensing_evidence_rehashed_photos=rights_verified,
        fresh_online_rights_verification=False,source_files=files,records_not_independent_objects=True))
    observations=[];snapshot_stats=[];candidates={};poly_cache={};counts=Counter();families=Counter()
    for si,snapshot in enumerate(index['records'],1):
        row=require_train(snapshot['image_id'],train);photo=photos[row['image_id']]
        if any(photo[k]!=row[k] for k in ('path','sha256','width','height')):raise ValueError('Native/photo frame mismatch')
        protected=[a['box'] for a in row['annotations']]+[r['box'] for r in reg[row['image_id']]['protected']]+[a['box'] for a in photo['annotations'] if semantic_protected(a) or a['native_ignored']]
        # Carry forward every existing E1 protected/ambiguous region too.
        entry=next(e for e in c['entries'] if e['row']['image_id']==row['image_id']);protected+=entry['distill_protected']+entry['legacy_protected']
        record=snapshot['canonical']
        if sha(ROOT/record['path'])!=record['sha256']:raise ValueError('Selected cache changed')
        with np.load(ROOT/record['path'],allow_pickle=False) as z:logits=z['logits'];square=z['square_cxcywh']
        b=registered(square,row['width'],row['height']);scores=1/(1+np.exp(-logits.astype(float)));argmax=logits.argmax(1)
        if row['image_id'] not in teacher_arrays:
            t=teacher[row['image_id']]
            if sha(ROOT/t['cache'])!=t['cache_sha256']:raise ValueError('Teacher cache changed')
            with np.load(ROOT/t['cache'],allow_pickle=False) as z:teacher_arrays[row['image_id']]=z['logits'].copy()
        ts=1/(1+np.exp(-teacher_arrays[row['image_id']].astype(float)))
        clipped=np.clip(b,0,1);proposals=[]
        for q in np.flatnonzero(scores.max(1)>=.5):
            if area(clipped[q])>0:proposals.append(dict(box=clipped[q].tolist(),score=float(scores[q,argmax[q]]),label=CLASSES[int(argmax[q])],q=int(q)))
        kept=metric.nms(proposals,.5,300);kept_ids={p['q'] for p in kept}
        state_names=sorted({s for a in snapshot['aliases'] for s in a['states']})
        stat=dict(snapshot_id=snapshot['snapshot_id'],image_id=row['image_id'],source_group=row['identity_group'],states=state_names,targets={})
        for target in ('chair','sofa','bed'):
            ci=CLASSES.index(target);high=np.flatnonzero(scores[:,ci]>=.5)
            counts[(target,'raw_ge05')]+=len(high);counts[(target,'raw_ge08')]+=int((scores[:,ci]>=.8).sum())
            stat['targets'][target]=dict(raw_ge05=len(high),raw_ge08=int((scores[:,ci]>=.8).sum()),postNMS_argmax_ge05=sum(p['label']==target for p in kept),postNMS_argmax_ge08=sum(p['label']==target and p['score']>=.8 for p in kept))
            for q in high:
                box=b[q].tolist();score=float(scores[q,ci]);positive=[a for a in row['annotations'] if a['label']==target and not a['ignore']]
                best=max([iou(box,a['box']) for a in positive],default=0.)
                fragment=any(intersection(box,a['box'])/max(area(box),1e-30)>=.8 and area(box)<.6*area(a['box']) and iou(box,a['box'])<.5 for a in positive)
                blocked=[p for p in protected if overlap(box,p)]
                valid=0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
                context_anns=[]
                for a in photo['annotations']:
                    fraction=intersection(box,a['box'])/max(area(box),1e-30)
                    if fraction>=.7:context_anns.append((fraction,a))
                context_anns.sort(key=lambda v:(-v[0],v[1]['annotation_key']))
                src=context_anns[0][1] if context_anns else None
                category=semantic(src) if src else 'uncertain_or_unannotated_real_object'
                if best>=.5:interpretation='source_annotated_target_overlap_not_a_confirmed_mistake'
                elif fragment:interpretation='furniture_fragment_candidate_protected_not_negative'
                elif q not in kept_ids or int(argmax[q])!=ci:interpretation='suppressed_or_not_argmax_query_not_a_reported_target_detection'
                else:interpretation='source_context_conflict_or_unknown_not_confirmed_false_positive'
                existing=[e['candidate_id'] for e in legacy if e['image_id']==row['image_id'] and overlap(box,e['box'])]
                previous_regions=[e['candidate_id'] for e in priorA if e['image_id']==row['image_id'] and overlap(box,e['box'])]
                geometric=[]
                if target=='chair' and valid and not blocked:
                    px=[math.floor(box[0]*row['width']),math.floor(box[1]*row['height']),math.ceil(box[2]*row['width']),math.ceil(box[3]*row['height'])]
                    if min(px[2]-px[0],px[3]-px[1])>=32:
                        for _,ann in context_anns:
                            if ann['native_ignored'] or semantic_protected(ann):continue
                            key=ann['annotation_key']
                            if key not in poly_cache:poly_cache[key]=polygons(ann['original_annotation'],row['width'],row['height'])
                            for pi,poly in enumerate(poly_cache[key]):
                                if certify(px,poly):geometric.append(dict(annotation_key=key,native_label=ann['native_label'],polygon_index=pi))
                cid='HN-'+digest(dict(image_id=row['image_id'],target=target,box_pixels=[round(v*s,3) for v,s in zip(box,[row['width'],row['height']]*2)]))[:20]
                item=dict(candidate_id=cid,snapshot_id=snapshot['snapshot_id'],image_id=row['image_id'],source_group=row['identity_group'],query_index=int(q),target=target,student_score=score,
                    original_score_same_query=float(ts[q,ci]),box=box,reported_target_detection=q in kept_ids and int(argmax[q])==ci,
                    source_context=category,source_annotation_key=src['annotation_key'] if src else None,native_label=src['native_label'] if src else None,
                    protected_overlap_count=len(blocked),existing_negative_overlap=existing,prior_inactive_negative_overlap=previous_regions,
                    known_target_IoU=best,fragment_heuristic=fragment,interpretation=interpretation,within_original_frame=valid,
                    source_geometry_support=geometric,human_reviewed=False,assistant_visual_reviewed=False,negative_label_proven=False)
                observations.append(item);families[(target,category)]+=1
                cand=candidates.setdefault(cid,dict(candidate_id=cid,image_id=row['image_id'],source_group=row['identity_group'],source_sha256=row['sha256'],path=row['path'],width=row['width'],height=row['height'],target=target,box=box,
                     peak_score=score,source_context=category,native_label=item['native_label'],source_annotation_key=item['source_annotation_key'],
                     observations=[],source_geometry_support=geometric,protected_overlap_count=len(blocked),already_represented=bool(existing or previous_regions),
                     evidence_tier='B' if geometric else 'C',training_eligible=False,training_activated=False,human_reviewed=False,assistant_visual_reviewed=False))
                cand['peak_score']=max(score,cand['peak_score']);cand['observations'].append(dict(snapshot_id=snapshot['snapshot_id'],query_index=int(q),score=score,original_score=float(ts[q,ci]),reported_detection=item['reported_target_detection']))
                cand['source_geometry_support']=geometric or cand['source_geometry_support']
        snapshot_stats.append(stat)
        if si%50==0:print('CPU mined',si,'/',len(index['records']),'snapshots;',len(observations),'high-score observations',flush=True)
    atomic(DATA/'high-confidence-observations.jsonl',b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in observations))
    emit(DATA/'candidates-initial.json',dict(records=list(candidates.values()),automatic_source_contexts_not_labels=True))
    emit(DATA/'snapshot-coverage.json',dict(records=snapshot_stats))
    summary=dict(unique_snapshots=len(snapshot_stats),raw_query_observations=len(observations),unique_candidate_footprints=len(candidates),
         per_target={t:dict(raw_ge05=counts[(t,'raw_ge05')],raw_ge08=counts[(t,'raw_ge08')],postNMS_ge05=sum(r['targets'][t]['postNMS_argmax_ge05'] for r in snapshot_stats),postNMS_ge08=sum(r['targets'][t]['postNMS_argmax_ge08'] for r in snapshot_stats)) for t in ('chair','sofa','bed')},
         confidence_and_source_context={t:dict(Counter(r['source_context'] for r in observations if r['target']==t)) for t in ('chair','sofa','bed')},
         geometric_source_candidates=sum(bool(r['source_geometry_support']) for r in candidates.values()),
         new_trusted_hard_negatives=0,unmatched_not_automatically_FP=True,all_candidates_unactivated=True,
         snapshot_counts_not_independent_examples=True,development_or_reserved_mined=False,new_inference=0)
    emit(REPORT/'mining-summary-initial.json',summary)
    print('Mining complete:high scores',len(observations),'geometry-qualified footprints',summary['geometric_source_candidates'],';no automatic labels',flush=True)

if __name__=='__main__':run()
