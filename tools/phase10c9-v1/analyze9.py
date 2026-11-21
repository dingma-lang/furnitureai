"""Preserve query identities, native metric statuses and conservative error hypotheses."""
import csv
from collections import Counter
import numpy as np
from common9 import *

def annotated_mechanisms(p,row,native,m):
    chair=[a for a in row['annotations'] if a['label']=='chair' and not a['ignore']]
    other=[a for a in row['annotations'] if a['label']!='chair' and not a['ignore']]
    flags=[];verified=[]
    if p.get('duplicate'):
        flags.append('D');verified.append(dict(category='D',method='Same-category duplicate of an already-matched native Chair at IoU>=0.5'))
    max_chair=max((m.iou(p['box'],g['box']) for g in chair),default=0)
    fragments=[g for g in chair if m.iou(p['box'],g['box'])<.5 and m.inter(p['box'],g['box'])/m.area(p['box'])>=.8 and m.area(p['box'])<.6*m.area(g['box'])]
    if fragments or .1<=max_chair<.5:flags.append('E')
    if any(m.iou(p['box'],g['box'])>=.5 for g in other):flags.append('B')
    nearby=[]
    for a in native:
        ov=m.inter(p['box'],a['box'])/m.area(p['box']);j=m.iou(p['box'],a['box'])
        if ov>=.5 or j>=.25:nearby.append(dict(annotation_key=a['annotation_key'],native_label=a['native_label'],category=a['category'],intersection_over_prediction=ov,iou=j))
        if a['native_label'].replace('_',' ').lower() in ('chair','armchair','deck chair','folding chair','rocking chair','music stool','stool') and j>=.5 and max_chair<.5:
            if 'C' not in flags:flags.append('C')
    # Native loose boxes describe overlap, not proof that a region excludes seating.
    primary='D' if verified else 'F'
    return dict(taxonomy_primary=primary,evidence_tier='source_metric_geometry' if verified else 'unresolved',source_verified_findings=verified,
        automatic_hypotheses=sorted(set(flags)),native_overlap_context=sorted(nearby,key=lambda a:-a['intersection_over_prediction'])[:6],
        nearest_Chair_IoU=max_chair,fragment_heuristic=bool(fragments),near_Chair_heuristic=max_chair>=.1,human_reviewed=False)

def reconstruct(row,arrays,old,m):
    predictions=[]
    for q,(box,s,c) in enumerate(zip(arrays['boxes'].tolist(),arrays['scores'].tolist(),arrays['labels'].tolist())):
        if s>=.001 and box[2]>box[0] and box[3]>box[1]:predictions.append(dict(box=box,score=s,label=CLASSES[c],query_index=q))
    result=m.nms(predictions,.5,300)
    if [{k:p[k] for k in ('box','score','label')} for p in result]!=old:raise ValueError('Saved postprocessing/query registration mismatch')
    return result

def statistical_mode_change(x,y):
    """Seven-category common-mode decomposition; not a recovered internal layer."""
    delta=y.astype(np.float64)-x.astype(np.float64);common=delta.mean(1);residual=delta-common[:,None]
    return dict(mean_logit_delta_by_category={c:float(delta[:,i].mean()) for i,c in enumerate(CLASSES)},
        fraction_queries_all_seven_logits_increase=float((delta>0).all(1).mean()),fraction_queries_Chair_logit_increases=float((delta[:,1]>0).mean()),
        common_mode_shift_quantiles=quantiles(common),common_mode_fraction_squared_delta=float((7*common**2).sum()/(delta**2).sum()) if (delta**2).sum() else 0,
        semantic_residual_RMS=float(np.sqrt((residual**2).mean())),
        caveat='Statistical common category shift only; cannot identify dense0 versus shift/scale causal contributions without saved intermediate embeddings')

def threshold_summary(rows,predictions,m,t):
    result={}
    for c in CLASSES:
        events,n,unknown=m.events(rows,predictions,c,floor=t);tp=sum(e['outcome']=='tp' for e in events);fp=sum(e['outcome']=='fp' for e in events)
        ps=[p for r in rows for p in predictions[r['image_id']] if p['label']==c and p['score']>=t]
        scored=tp+fp
        result[c]=dict(predictions=len(ps),tp=tp,scored_fp=fp,fn=n-tp,unknown_unmatched=unknown,ignored=len(ps)-scored-unknown,
            precision=tp/scored if scored else None,recall=tp/n if n else None,
            scored_FP_area_bins=dict(Counter(geometry(predictions[e['image_id']][e['index']]['box'],1,1)['size_bin'] for e in events if e['outcome']=='fp')))
    return result

def run():
    m=metric_module();train,dev=subsets();rows=[m.scopes(r) for r in dev];native={p['image_id']:p['annotations'] for p in read(OUT/'development-native-context.json')['photos']};photos=[];all_records=[];raw={};summary={};per_photo=[]
    for row in dev:
        photos.append(dict(row,role='internal_development_diagnostic_only',native=native[row['image_id']],models={},coverage_warning='Unmatched predictions count as FP only where unchanged native completeness permits it; this photo never supplies training negatives.'))
    byid={p['image_id']:p for p in photos}
    for model,(arm,epoch) in MODELS.items():
        folder=C8/arm/'evaluation'/('epoch-'+str(epoch));done=read(folder/'raw-complete.json');saved=read(folder/'predictions.json');predictions={};raw[model]=[]
        if done['rows_sha256']!=digest(dev) or len(done['records'])!=19:raise ValueError('Exact development membership mismatch')
        for row,receipt in zip(dev,done['records']):
            path=folder/receipt['file']
            if receipt['image_id']!=row['image_id'] or receipt['source_sha256']!=row['sha256'] or sha(path)!=receipt['sha256']:raise ValueError('Prediction source/receipt mismatch')
            arrays=load_npz(path);raw[model].append(arrays['logits']);predictions[row['image_id']]=reconstruct(row,arrays,saved[row['image_id']],m)
        statuses={};unknown_counts={}
        for c in CLASSES:
            ev,_,_=m.events(rows,predictions,c,floor=.001)
            for e in ev:statuses[e['image_id'],e['index']]=e
        for row in rows:
            out=[]
            for j,p in enumerate(predictions[row['image_id']]):
                if p['score']<.15:continue
                event=statuses.get((row['image_id'],j));status=event['outcome'] if event else 'unknown_unmatched'
                if not event and row['coverage'][p['label']] and any(m.inter(p['box'],a['box'])/m.area(p['box'])>=.5 for a in row['annotations'] if a['ignore'] and a['label']==p['label']):status='ignored_native_region'
                rec=dict(p,prediction_id=f'{model}:{row["image_id"]}:q{p["query_index"]}',model=model,image_id=row['image_id'],post_nms_index=j,status=status,
                    duplicate=bool(event and event.get('duplicate')),source_sha256=row['sha256'],geometry=geometry(p['box'],row['width'],row['height']),
                    pixel_box=[p['box'][i]*[row['width'],row['height'],row['width'],row['height']][i] for i in range(4)],
                    coverage=row['coverage'][p['label']],fp_coverage=row['fp_coverage'][p['label']])
                if p['label']=='chair' and status!='tp':rec.update(annotated_mechanisms(rec,row,native[row['image_id']],m))
                else:rec.update(taxonomy_primary=None,evidence_tier='matched_positive' if status=='tp' else 'not_Chair_analysis',automatic_hypotheses=[],native_overlap_context=[],source_verified_findings=[],human_reviewed=False)
                out.append(rec);all_records.append(rec)
            byid[row['image_id']]['models'][model]=out
            chair=[p for p in out if p['label']=='chair'];per_photo.append(dict(model=model,image_id=row['image_id'],source_group=row['identity_group'],fp_coverage=row['fp_coverage']['chair'],Chair_TP=sum(p['status']=='tp' for p in chair),Chair_scored_FP=sum(p['status']=='fp' for p in chair),Chair_unknown=sum(p['status']=='unknown_unmatched' for p in chair),Chair_ignored=sum(p['status']=='ignored_native_region' for p in chair),Chair_FP_above_05=sum(p['status']=='fp' and p['score']>=.5 for p in chair),Chair_FP_above_08=sum(p['status']=='fp' and p['score']>=.8 for p in chair)))
        summary[model]=dict(thresholds={str(t):threshold_summary(rows,predictions,m,t) for t in THRESHOLDS},all_raw_logits_quantiles={c:quantiles(np.concatenate(raw[model])[:,i]) for i,c in enumerate(CLASSES)})
        # Exact metric headline reproduction; preserves protocol rather than inventing one.
        measured=m.evaluate(rows,predictions);old=read(folder/'metrics-cpu.json')['metrics']
        if any(measured[k]!=old[k] for k in ('tp','fp','fn','precision','recall','ap50','ap75')):raise ValueError('Native metric reproduction failed')
    a=np.concatenate(raw['original']);changes={model:statistical_mode_change(a,np.concatenate(raw[model])) for model in ('control','corrected')}
    sample=[];corrected=[r for r in all_records if r['model']=='corrected' and r['label']=='chair'];fpids=sorted({p['image_id'] for p in corrected if p['status']=='fp'})
    for ident in fpids:
        pool=sorted([p for p in corrected if p['image_id']==ident and p['status']=='fp'],key=lambda p:(-p['score'],p['query_index']));selected=[]
        for lo,hi,n in ((.8,1.01,3),(.5,.8,3),(.15,.5,2)):
            selected.extend([p for p in pool if lo<=p['score']<hi][:n])
        chosen={p['prediction_id'] for p in selected}
        selected.extend([p for p in pool if p['prediction_id'] not in chosen][:max(0,8-len(selected))]);sample.extend(selected)
    unknown=sorted([p for p in corrected if p['status']=='unknown_unmatched'],key=lambda p:(-p['score'],p['prediction_id']));seen=set()
    for p in unknown:
        if p['image_id'] not in seen:sample.append(p);seen.add(p['image_id'])
        if len(seen)==8:break
    samples=[dict(p,sample_id='S'+str(i).zfill(3),sample_selection='deterministic targeted score/photo strata, not prevalence-estimating random sample') for i,p in enumerate(sample,1)]
    # Optional geometric NMS counterfactual from saved boxes only; never production.
    nms_analysis={}
    corrected_ps=read(C8/'B-corrected/evaluation/epoch-5/predictions.json')
    for threshold in (.3,.4):
        ps={r['image_id']:m.nms(corrected_ps[r['image_id']],threshold,300) for r in rows}
        nms_analysis[str(threshold)]=threshold_summary(rows,ps,m,.15)['chair']
    save(OUT/'prediction-records.json',dict(records=all_records,source='C8 saved raw arrays and original unchanged NMS',training_negative_export=False))
    save(OUT/'review-sample.json',dict(records=samples,decisions_prefilled=False,human_reviews_created=0))
    save(OUT/'dashboard-data-base.json',dict(photos=photos,taxonomy=TAXONOMY,summary=summary,per_photo=per_photo,samples=samples))
    save(REPORT/'prediction-summary.json',dict(models=summary,logit_changes=changes,NMS_diagnostic_counterfactual=nms_analysis,thresholds_adopted=False,per_photo=per_photo,
        source_verified_taxonomy_counts=dict(Counter(r['taxonomy_primary'] for r in corrected if r['status']=='fp' and r['evidence_tier']=='source_metric_geometry')),
        unresolved_scored_FP=sum(r['status']=='fp' and r['evidence_tier']=='unresolved' for r in corrected),sample_count=len(samples),models_inferred=0))
    print('C9 SAVED PREDICTIONS ANALYZED',len(all_records),'operating boxes;',len(samples),'targeted samples;no model calls',flush=True)

if __name__=='__main__':run()
