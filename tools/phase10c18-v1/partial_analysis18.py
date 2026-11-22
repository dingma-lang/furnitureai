"""Recover CPU-only diagnostics from the one stopped C18 attempt. Never infer."""
from analyze18 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU analysis must hide CUDA')
    cfg,prepared,train,dev,_=inputs18();failure=read(RUN18/'failure.json');launch=read(RUN18/'launch-result.json');count=failure['state']['updates']
    if count!=28 or (RUN18/'complete.json').exists():raise ValueError('Exact stopped28-update attempt required')
    m=metric_module();rows=[m.scopes(r) for r in dev];cpu=RUN18/'cpu-partial-analysis-v1';cpu.mkdir();comparison={};metrics={};logits={};predictions={}
    history=read(C8/'metric-results.json')['entries']
    jobs=[('original',C8/'A-control/evaluation/epoch-0',0,0),('C8_corrected_B_E1',C8/'B-corrected/evaluation/epoch-1',1,15),('C8_corrected_B_E5',C8/'B-corrected/evaluation/epoch-5',5,75),('C18_E1',RUN18/'evaluation/epoch-1',1,15)]
    for key,folder,epoch,updates in jobs:
        arrays=raw_outputs(folder,dev);pred=predictions_for(arrays,dev,m)
        if key!='C18_E1' and pred!=read(folder/'predictions.json'):raise ValueError('Historical predictions differ')
        emit18(cpu/('predictions-'+key+'.json'),pred)
        measured=m.evaluate(rows,pred)
        if key!='C18_E1':
            old=next(e['metrics'] for e in history if e['arm']==('A-control' if key=='original' else 'B-corrected') and e['epoch']==epoch)
            if measured!=old:raise ValueError('Historical metrics differ')
        metrics[key]=measured;predictions[key]=pred;logits[key]=np.concatenate([a['logits'] for a in arrays]).astype(np.float64)
        emit18(cpu/('metrics-'+key+'.json'),dict(epoch=epoch,updates=updates,metrics=measured))
        print('SAVED CPU METRICS',key,'P',measured['precision'],'R',measured['recall'],'FP',measured['fp'],flush=True)
    for key in metrics:
        measured=metrics[key];scores=1/(1+np.exp(-logits[key]));comparison[key]=dict(updates=0 if key=='original' else 75 if key.endswith('E5') else 15,headline={k:v for k,v in measured.items() if k not in ('categories','pr_curve')},categories=measured['categories'],precision_at_original_recall=envelope(measured,metrics['original']['recall']),precision_at_C8_E1_recall=envelope(measured,metrics['C8_corrected_B_E1']['recall']),precision_at_C8_E5_recall=envelope(measured,metrics['C8_corrected_B_E5']['recall']),thresholds={str(t):threshold_diagnostics(rows,predictions[key],m,t) for t in (.15,.5,.8)},raw_Chair_at_least={str(t):int((scores[:,1]>=t).sum()) for t in (.15,.5,.8)},raw_query_confidence={c:quantiles(scores[:,i]) for i,c in enumerate(CLASSES)},scored_FP_confidence={c:quantiles([e['score'] for e in m.events(rows,predictions[key],c,floor=.15)[0] if e['outcome']=='fp']) for c in CLASSES})
    drift={}
    for key in metrics:
        if key=='original':continue
        d=logits[key]-logits['original'];common=d.mean(1)
        drift[key]=dict(mean_all_categories=float(d.mean()),per_category_mean_delta={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},common_query_shift_quantiles=quantiles(common),common_RMS=float(np.sqrt(np.mean(common**2))),category_residual_RMS=float(np.sqrt(np.mean((d-common[:,None])**2))),causal_parameter_attribution=False)
    updates=[read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,count+1)];allimages=[r for u in updates for r in u['images']];epochs=[]
    for epoch in (1,2):
        us=[u for u in updates if u['epoch']==epoch];ims=[r for u in us for r in u['images']];scores=[s for r in ims for s in r['source_scores']]
        ep=dict(epoch=epoch,complete_epoch=epoch==1,updates=len(us),photo_presentations=len(ims),mean_loss={k:float(np.mean([r[k] for r in ims])) for k in ('positive_loss','native_negative_loss','base_loss','box_loss','source_loss','legacy_source_focal_loss')},mean_parameter_gradient={k:float(np.mean([u['metrics'][k]['norm'] for u in us])) for k in ('positive','source_pre','source_post')},mean_projection_gradient={k:float(np.mean([u['metrics'][k]['projection_norm'] for u in us])) for k in ('positive','source_pre','source_post')},source_negative_positive_pre=quantiles([u['metrics']['source_pre']['negative_positive_ratio'] for u in us]),source_negative_positive_post=quantiles([u['metrics']['source_post']['negative_positive_ratio'] for u in us]),cosines=quantiles([u['metrics']['source_pre']['cosine_positive'] for u in us if u['metrics']['source_pre']['cosine_positive'] is not None]),cap_activations=sum(u['metrics']['cap']['scale']<1 for u in us),source_scores=quantiles(scores),source_terms=len(scores),terms_ge_015=sum(s>=.15 for s in scores),terms_ge_05=sum(s>=.5 for s in scores),query_covered_source_groups=len({r['source_group'] for r in ims if r['source_terms']}),nonzero_gradient_source_groups=len({r['source_group'] for r in ims if r['source_gradient']['norm']>0}),covered_regions=len({k for r in ims for k,n in r['region_query_counts'].items() if n}))
        if epoch==1:
            sources=read(RUN18/'sources-epoch-1.json');norms=sorted([v['norm'] for v in sources.values()],reverse=True);ep.update(source_group_norms=sources,top3_sum_source_norm_share=sum(norms[:3])/sum(norms) if sum(norms) else None,parameters=read(RUN18/'parameters-epoch-1.json'))
        else:ep['source_norm_aggregate_not_saved_before_stop']=True
        epochs.append(ep)
    probes=[]
    for epoch in (0,1):
        rs=read(RUN18/'probes'/('epoch-'+str(epoch))/'complete.json')['records'];scores=[s for r in rs for s in r['source_scores']]
        probes.append(dict(epoch=epoch,terms=len(scores),confidence=quantiles(scores),terms_ge_015=sum(s>=.15 for s in scores),terms_ge_05=sum(s>=.5 for s in scores),covered_source_groups=len({r['source_group'] for r in rs if r['source_terms']}),nonzero_source_groups=len({r['source_group'] for r in rs if r['source_gradient']['norm']>0}),covered_regions=len({k for r in rs for k,n in r['region_query_counts'].items() if n}),mean_positive_gradient=float(np.mean([r['positive_gradient']['norm'] for r in rs])),mean_source_gradient=float(np.mean([r['source_gradient']['norm'] for r in rs])),mean_logits={c:float(np.mean([r['mean_logits_by_category'][i] for r in rs])) for i,c in enumerate(CLASSES)}))
    training=dict(status='STOPPED_SAFETY_GUARD',committed_updates=28,training_photo_presentations_committed=len(allimages),training_photo_forwards=115,uncommitted_photos=4,epochs=epochs,probes=probes,cap_activations_committed=sum(e['cap_activations'] for e in epochs),committed_group_cap_violations=0,failed_group_values_not_retained=True,all_committed_positive_and_box_buffers_preserved=True)
    result=dict(status='STOPPED_SAFETY_GUARD',outcome='D_INCONCLUSIVE_INCOMPLETE_SAFETY_STOP',comparison=comparison,logit_drift=drift,training=training,resource=failure['state'],launcher=launch,latest_checkpoint=updates[-1]['checkpoint'],evaluated_checkpoint=updates[14]['checkpoint'],final_epoch5_comparison_available=False,latest_update28_scored=False,automatic_retry=False,protocol=dict(photos=19,GT=sum(not a['ignore'] for r in dev for a in r['annotations']),supported_classes=['sofa','chair','bed'],unsupported=['table','desk','cabinet','lamp'],fp_scoring_photos={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},confidence=.15,score_floor=.001,NMS=.5,cap=300,metric_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'),rows_sha256=digest(dev),historical_reference_metrics_reproduced_exactly=True,independent_accuracy=False,frozen_reserved_official_scored=False,unknown_unmatched_not_confirmed_FP=True),limitations=['Stopped before update29;only C18 E1/15update predictions exist. No new inference attempted.','The failed group gradient/budget values were not persisted; cannot quantify actual failed inequality.','C18 E1 vs C8 E1 is update-matched; C8 E5 comparison is descriptive and confounded by duration.','FP32 effective-difference guard can false-positive despite an admissible source vector;CPU witness is not a measurement of the failed GPU buffers.','No frozen-backbone live hash was captured after the failure;last full frozen hash verified at epoch1. Original on-disk weights and all completed snapshot metadata remain auditable.'])
    emit18(REPORT18/'results.json',result);emit18(REPORT18/'training-curves.json',training);emit18(REPORT18/'precision-recall.json',{k:v['pr_curve'] for k,v in metrics.items()})
    csvrows=[dict(model=key,category=c,**v) for key,r in comparison.items() for c,v in r['categories'].items()];b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows);atomic18(REPORT18/'category-metrics.csv',b.getvalue().encode())
    emit18(cpu/'complete.json',dict(status='PASS',CPU_only=True,new_GPU_forwards=0,epochs_scored=[1],historical_reference_metrics_reproduced_exactly=True))
    print('Partial CPU diagnostics recovered;D,incomplete;no inference or training resumed.',flush=True)


if __name__=='__main__':run()
