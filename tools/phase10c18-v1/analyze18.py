"""CPU-only unchanged historical metrics and saved-query diagnostics."""
import importlib.util, csv
import numpy as np
from common18 import *


def metric_module():
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('c18_unchanged_cpu_metric',ROOT/'tools/phase10c-v1/metric_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def quantiles(values):
    return {str(q):float(np.quantile(values,q)) if len(values) else None for q in (0,.5,.9,.99,1)}


def envelope(metric,target):
    possible=[p for p in metric['pr_curve'] if p['recall'] is not None and p['recall']>=target]
    return max(possible,key=lambda p:p['precision']) if possible else None


def raw_outputs(folder,dev):
    marker=read(folder/'raw-complete.json')
    if not marker['FP32'] or marker['rows_sha256']!=digest(dev) or len(marker['records'])!=19:raise ValueError('Evaluation protocol/size mismatch')
    arrays=[]
    for row,ref in zip(dev,marker['records']):
        if row['image_id']!=ref['image_id'] or row['sha256']!=ref['source_sha256'] or sha(folder/ref['file'])!=ref['sha256']:raise ValueError('Saved raw identity/hash mismatch')
        arrays.append(load_npz(folder/ref['file']))
    return arrays


def predictions_for(arrays,dev,m):
    predictions={}
    for row,a in zip(dev,arrays):
        ps=[]
        for box,s,c in zip(a['boxes'].tolist(),a['scores'].tolist(),a['labels'].tolist()):
            if s>=.001 and box[2]>box[0] and box[3]>box[1]:ps.append(dict(box=box,score=s,label=CLASSES[c]))
        predictions[row['image_id']]=m.nms(ps,.5,300)
    return predictions


def threshold_diagnostics(rows,pred,m,threshold):
    result={}
    for c in CLASSES:
        events,n,unknown=m.events(rows,pred,c,floor=threshold)
        tp=sum(e['outcome']=='tp' for e in events);fp=sum(e['outcome']=='fp' for e in events)
        result[c]=dict(tp=tp,fp=fp,fn=n-tp,unknown_unmatched=unknown,duplicates=sum(e.get('duplicate',False) for e in events),all_predictions=sum(p['label']==c and p['score']>=threshold for ps in pred.values() for p in ps))
    return result


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU analysis must hide CUDA')
    cfg,prepared,train,dev,_=inputs18();done=read(RUN18/'complete.json')
    if done['status']!='PASS' or done['updates']!=75:raise ValueError('Complete single arm required;no inference recovery here')
    m=metric_module();rows=[m.scopes(r) for r in dev];cpu=RUN18/'cpu-analysis-v1';cpu.mkdir();metrics={};raw={};predictions={};entries=[]
    history=read(C8/'metric-results.json')['entries']
    for key,arm,epoch in [('original','A-control',0),('C8_corrected_B','B-corrected',5)]:
        folder=C8/arm/'evaluation'/('epoch-'+str(epoch));raw[key]=raw_outputs(folder,dev);predictions[key]=predictions_for(raw[key],dev,m)
        if predictions[key]!=read(folder/'predictions.json'):raise ValueError('Historical prediction reconstruction differs')
        old=next(e['metrics'] for e in history if e['arm']==arm and e['epoch']==epoch);measured=m.evaluate(rows,predictions[key])
        if measured!=old:raise ValueError('CPU metric definitions differ from C8')
        metrics[key]=measured
    for epoch in (1,3,5):
        key='C18_epoch'+str(epoch);folder=RUN18/'evaluation'/('epoch-'+str(epoch));a=raw_outputs(folder,dev);pred=predictions_for(a,dev,m)
        # Durable prediction JSON is complete before any metric invocation.
        emit18(cpu/('predictions-E'+str(epoch)+'.json'),pred)
        measured=m.evaluate(rows,pred);entry=dict(epoch=epoch,updates=epoch*15,metrics=measured,predictions_sha256=sha(cpu/('predictions-E'+str(epoch)+'.json')))
        emit18(cpu/('metrics-E'+str(epoch)+'.json'),entry);entries.append(entry)
        metrics[key]=measured
        if epoch==5:raw['C18_source_balanced']=a;predictions['C18_source_balanced']=pred
        print('CPU METRICS',epoch,'P',measured['precision'],'R',measured['recall'],'FP',measured['fp'],flush=True)
    metrics['C18_source_balanced']=metrics['C18_epoch5'];comparison={};z={k:np.concatenate([a['logits'] for a in arrays]).astype(np.float64) for k,arrays in raw.items()}
    for key in raw:
        measured=metrics[key];score=1/(1+np.exp(-z[key]));thresholds={str(t):threshold_diagnostics(rows,predictions[key],m,t) for t in (.15,.5,.8)}
        scored_FP_scores={c:[e['score'] for e in m.events(rows,predictions[key],c,floor=.15)[0] if e['outcome']=='fp'] for c in CLASSES}
        comparison[key]=dict(headline={k:v for k,v in measured.items() if k not in ('categories','pr_curve')},categories=measured['categories'],precision_at_original_recall=envelope(measured,metrics['original']['recall']),precision_at_C8_recall=envelope(measured,metrics['C8_corrected_B']['recall']),thresholds=thresholds,raw_query_confidence={c:quantiles(score[:,j]) for j,c in enumerate(CLASSES)},raw_Chair_at_least={str(t):int((score[:,1]>=t).sum()) for t in (.15,.5,.8)},scored_FP_confidence={c:quantiles(v) for c,v in scored_FP_scores.items()})
    drifts={}
    for key in ('C8_corrected_B','C18_source_balanced'):
        delta=z[key]-z['original'];common=delta.mean(1);residual=delta-common[:,None]
        drifts[key]=dict(per_category_mean_delta={c:float(delta[:,j].mean()) for j,c in enumerate(CLASSES)},mean_all_categories=float(delta.mean()),common_query_shift_quantiles=quantiles(common),common_RMS=float(np.sqrt(np.mean(common**2))),category_residual_RMS=float(np.sqrt(np.mean(residual**2))),caveat='Aligned raw query indices on same frozen backbone/photos; descriptive common-mode drift,not causal parameter attribution')
    updates=[read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)]
    all_images=[r for u in updates for r in u['images']]
    if len(all_images)!=295 or len({r['image_id'] for r in all_images})!=59 or any(r['protected_violations'] for r in all_images):raise ValueError('Training count/protection failure')
    for i,u in enumerate(updates):
        if u['update']!=i+1 or u['source_order']!=cfg['orders'][i//15][i%15] or u['metrics']['cap_violations'] or not u['metrics']['positive_buffers_unchanged'] or not u['metrics']['box_buffers_unchanged']:raise ValueError('Committed group integrity failure')
    training_epochs=[]
    for epoch in range(1,6):
        us=[u for u in updates if u['epoch']==epoch];ims=[r for u in us for r in u['images']];sources=read(RUN18/('sources-epoch-'+str(epoch)+'.json'));norms=sorted([v['norm'] for v in sources.values()],reverse=True);scores=[s for r in ims for s in r['source_scores']];active_sources={r['source_group'] for r in ims if r['source_gradient']['norm']>0}
        training_epochs.append(dict(epoch=epoch,photo_presentations=len(ims),mean_loss={k:float(np.mean([r[k] for r in ims])) for k in ('positive_loss','native_negative_loss','base_loss','box_loss','source_loss','legacy_source_focal_loss')},mean_parameter_gradient={name:float(np.mean([u['metrics'][name]['norm'] for u in us])) for name in ('positive','source_pre','source_post')},mean_projection_gradient={name:float(np.mean([u['metrics'][name]['projection_norm'] for u in us])) for name in ('positive','source_pre','source_post')},caps_activated=sum(u['metrics']['cap']['scale']<1 for u in us),cap_scale=quantiles([u['metrics']['cap']['scale'] for u in us]),negative_positive_ratio_pre=quantiles([u['metrics']['source_pre']['negative_positive_ratio'] for u in us]),negative_positive_ratio_post=quantiles([u['metrics']['source_post']['negative_positive_ratio'] for u in us]),cosine=quantiles([u['metrics']['source_pre']['cosine_positive'] for u in us if u['metrics']['source_pre']['cosine_positive'] is not None]),negative_terms=sum(r['source_terms'] for r in ims),nonzero_gradient_source_groups=len(active_sources),sources_with_covered_queries=len({r['source_group'] for r in ims if r['source_terms']}),top3_sum_source_norm_share=sum(norms[:3])/sum(norms) if sum(norms) else None,source_scores=quantiles(scores),source_terms_ge_015=sum(s>=.15 for s in scores),source_terms_ge_05=sum(s>=.5 for s in scores),regions_covered=len({k for r in ims for k,v in r['region_query_counts'].items() if v}),per_source=sources,parameters=read(RUN18/('parameters-epoch-'+str(epoch)+'.json'))))
    probes=[]
    for epoch in (0,1,3,5):
        records=read(RUN18/'probes'/('epoch-'+str(epoch))/'complete.json')['records'];scores=[s for r in records for s in r['source_scores']]
        probes.append(dict(epoch=epoch,terms=len(scores),source_scores=quantiles(scores),terms_ge_015=sum(s>=.15 for s in scores),terms_ge_05=sum(s>=.5 for s in scores),covered_source_groups=len({r['source_group'] for r in records if r['source_terms']}),nonzero_source_groups=len({r['source_group'] for r in records if r['source_gradient']['norm']>0}),regions_covered=len({k for r in records for k,v in r['region_query_counts'].items() if v}),mean_positive_norm=float(np.mean([r['positive_gradient']['norm'] for r in records])),mean_source_norm=float(np.mean([r['source_gradient']['norm'] for r in records])),mean_category_logits={c:float(np.mean([r['mean_logits_by_category'][j] for r in records])) for j,c in enumerate(CLASSES)}))
    epochs=[dict(epoch=e['epoch'],**{k:e['metrics'][k] for k in ('precision','recall','ap50','ap75','tp','fp','fn')}) for e in entries]
    training=dict(epochs=training_epochs,probes=probes,caps_activated=sum(e['caps_activated'] for e in training_epochs),cap_violations=0,all_positive_and_box_buffers_preserved=True,updates=75,training_presentations=295,gradient_normalization_note='Three isolated scaledVJPs per photo,divide by actual4/3 group size;unscaleonce,sourcecapthenunchangedglobalclip/AdamW. Per-image loss means descriptive,not capped scalar objective.')
    result=dict(comparison=comparison,logit_drift=drifts,training_summary={k:v for k,v in training.items() if k!='epochs'},epoch_evaluation=epochs,resources=done,protocol=dict(photos=19,GT=sum(not a['ignore'] for r in dev for a in r['annotations']),supported_classes=['sofa','chair','bed'],unsupported=['table','desk','cabinet','lamp'],fp_scoring_photos={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},confidence=.15,AP='Unchanged101point macro over supported classes',NMS=.5,cap=300,score_floor=.001,rows_sha256=digest(dev),metric_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'),independent_accuracy=False,frozen_reserved_official_scored=False,unmatched_unknown_not_confirmed_FP=True,taxonomy='Project seating/bar stools protected;native unchanged historical class semantics remain'))
    emit18(REPORT18/'results.json',result);emit18(REPORT18/'training-curves.json',training);emit18(REPORT18/'precision-recall.json',{k:metrics[k]['pr_curve'] for k in raw})
    emit18(cpu/'complete.json',dict(status='PASS',CPU_only=True,new_GPU_forwards=0,epochs=[1,3,5],same_historical_metrics_verified=True))
    csvrows=[]
    for key,r in comparison.items():
        for c,v in r['categories'].items():csvrows.append(dict(model=key,category=c,**v))
    b=io.StringIO();writer=csv.DictWriter(b,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows);atomic18(REPORT18/'category-metrics.csv',b.getvalue().encode())
    print('C18 CPU analysis complete;no extra inference or training.',flush=True)


if __name__=='__main__':run()
