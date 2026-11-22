"""CPU-only scoring of saved recovery predictions; never starts model inference."""
from common20 import *
sys.path.insert(0,str(ROOT/'tools/phase10c18-v1'))
from analyze18 import metric_module,raw_outputs,predictions_for,quantiles,envelope,threshold_diagnostics
import numpy as np


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('Metrics must be CPU-only')
    cfg,prepared,train,dev,_=inputs20()
    success=(RUN20/'complete.json').exists();terminal=read(RUN20/('complete.json' if success else 'failure.json'))
    cpu=RUN20/'CPU-analysis-v1';cpu.mkdir();m=metric_module();rows=[m.scopes(r) for r in dev]
    history=read(C8/'metric-results.json')['entries'];entries={};metrics={};predictions={};raw={}
    jobs=[('original',C8/'A-control/evaluation/epoch-0',0,0),('C8_B_E1',C8/'B-corrected/evaluation/epoch-1',1,15),('C8_B_E3',C8/'B-corrected/evaluation/epoch-3',3,45),('C8_B_E5',C8/'B-corrected/evaluation/epoch-5',5,75),('C18_E1',RUN18/'evaluation/epoch-1',1,15)]
    for epoch in (3,5):
        folder=RUN20/'evaluation'/('epoch-'+str(epoch))
        if (folder/'raw-complete.json').exists():jobs.append(('C20_E'+str(epoch),folder,epoch,epoch*15))
    for key,folder,epoch,updates in jobs:
        arrays=raw_outputs(folder,dev);pred=predictions_for(arrays,dev,m)
        emit20(cpu/('predictions-'+key+'.json'),pred)  # Durable before metrics.
        measured=m.evaluate(rows,pred)
        if key.startswith('C8_') or key=='original':
            old=next(e['metrics'] for e in history if e['arm']==('A-control' if key=='original' else 'B-corrected') and e['epoch']==epoch)
            if measured!=old or pred!=read(folder/'predictions.json'):raise ValueError('Historical prediction/metric replay differs')
        if key=='C18_E1' and measured!=read(RUN18/'cpu-partial-analysis-v1/metrics-C18_E1.json')['metrics']:raise ValueError('C18 E1 replay differs')
        metrics[key]=measured;predictions[key]=pred;raw[key]=arrays;entries[key]=dict(epoch=epoch,updates=updates,metrics=measured)
        emit20(cpu/('metrics-'+key+'.json'),entries[key]);print('METRICS',key,'P',measured['precision'],'R',measured['recall'],'TP/FP/FN',measured['tp'],measured['fp'],measured['fn'],flush=True)
    comparison={};zs={key:np.concatenate([a['logits'] for a in value]).astype(np.float64) for key,value in raw.items()}
    for key,entry in entries.items():
        measured=entry['metrics'];scores=1/(1+np.exp(-zs[key]));pred=predictions[key]
        comparison[key]=dict(epoch=entry['epoch'],updates=entry['updates'],headline={k:v for k,v in measured.items() if k not in ('categories','pr_curve')},categories=measured['categories'],precision_at_original_recall=envelope(measured,metrics['original']['recall']),precision_at_C8_E5_recall=envelope(measured,metrics['C8_B_E5']['recall']),precision_at_C18_E1_recall=envelope(measured,metrics['C18_E1']['recall']),thresholds={str(t):threshold_diagnostics(rows,pred,m,t) for t in (.15,.5,.8)},raw_query_confidence={c:quantiles(scores[:,j]) for j,c in enumerate(CLASSES)},raw_Chair_at_least={str(t):int((scores[:,1]>=t).sum()) for t in (.15,.5,.8)},scored_FP_confidence={c:quantiles([e['score'] for e in m.events(rows,pred,c,floor=.15)[0] if e['outcome']=='fp']) for c in CLASSES})
    drift={}
    for key in zs:
        if key=='original':continue
        d=zs[key]-zs['original'];common=d.mean(1)
        drift[key]=dict(mean_all_categories=float(d.mean()),per_category_mean_delta={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},common_query_shift_quantiles=quantiles(common),common_RMS=float(np.sqrt(np.mean(common**2))),category_residual_RMS=float(np.sqrt(np.mean((d-common[:,None])**2))),scope='Aligned query indices on same19photos and frozen backbone;descriptive,not causal parameter attribution')
    old_updates=[read(RUN18/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,29)];new_updates=[read(p) for p in sorted((RUN20/'updates').glob('update-???.json'))];all_updates=old_updates+new_updates
    if [u['update'] for u in all_updates]!=list(range(1,29+len(new_updates))):raise ValueError('Skipped/repeated optimizer history')
    all_images=[r for u in all_updates for r in u['images']];epochs=[]
    for epoch in sorted({u['epoch'] for u in all_updates}):
        us=[u for u in all_updates if u['epoch']==epoch];ims=[r for u in us for r in u['images']];scores=[s for r in ims for s in r['source_scores']]
        row=dict(epoch=epoch,updates=len(us),photo_presentations=len(ims),mean_losses={k:float(np.mean([r[k] for r in ims])) for k in ('positive_loss','source_loss','native_negative_loss','base_loss','box_loss','legacy_source_focal_loss')},mean_class_gradient={k:float(np.mean([u['metrics'][k]['norm'] for u in us])) for k in ('positive','source_pre','source_post')},mean_projection_gradient={k:float(np.mean([u['metrics'][k]['projection_norm'] for u in us])) for k in ('positive','source_pre','source_post')},source_positive_ratio_before=quantiles([u['metrics']['source_pre']['negative_positive_ratio'] for u in us]),source_positive_ratio_after=quantiles([u['metrics']['source_post']['negative_positive_ratio'] for u in us]),negative_positive_cosine=quantiles([u['metrics']['source_pre']['cosine_positive'] for u in us if u['metrics']['source_pre']['cosine_positive'] is not None]),caps_activated=sum(u['metrics']['cap']['scale']<1 for u in us),cap_scale=quantiles([u['metrics']['cap']['scale'] for u in us]),source_terms=len(scores),source_scores=quantiles(scores),source_terms_ge_015=sum(s>=.15 for s in scores),source_terms_ge_05=sum(s>=.5 for s in scores),source_groups_with_queries=len({r['source_group'] for r in ims if r['source_terms']}),source_groups_with_nonzero_gradients=len({r['source_group'] for r in ims if r['source_gradient']['norm']>0}),covered_regions=len({k for r in ims for k,n in r['region_query_counts'].items() if n}),protected_violations=sum(r['protected_violations'] for r in ims))
        if epoch==1:row['parameters']=read(RUN18/'parameters-epoch-1.json')
        elif (RUN20/('parameters-epoch-'+str(epoch)+'.json')).exists():row['parameters']=read(RUN20/('parameters-epoch-'+str(epoch)+'.json'))
        sources_file=(RUN18 if epoch==1 else RUN20)/('sources-epoch-'+str(epoch)+'.json')
        if sources_file.exists():
            sources=read(sources_file);norms=sorted([v['norm'] for v in sources.values()],reverse=True);row.update(source_vector_segment='Complete epoch' if epoch!=2 else 'Resumed groups14–15 only;old groups1–13 vectors unavailable',top3_sum_source_norm_share=sum(norms[:3])/sum(norms) if sum(norms) else None,per_source_vectors=sources)
        epochs.append(row)
    probes=[]
    for epoch,base in ((0,RUN18),(1,RUN18),(3,RUN20),(5,RUN20)):
        folder=base/'probes'/('epoch-'+str(epoch));marker=folder/'complete.json'
        if not marker.exists():continue
        rs=read(marker)['records'];scores=[s for r in rs for s in r['source_scores']]
        probes.append(dict(epoch=epoch,terms=len(scores),source_scores=quantiles(scores),terms_ge_015=sum(s>=.15 for s in scores),terms_ge_05=sum(s>=.5 for s in scores),covered_source_groups=len({r['source_group'] for r in rs if r['source_terms']}),nonzero_source_groups=len({r['source_group'] for r in rs if r['source_gradient']['norm']>0}),covered_regions=len({k for r in rs for k,n in r['region_query_counts'].items() if n}),mean_positive_gradient=float(np.mean([r['positive_gradient']['norm'] for r in rs])),mean_source_gradient=float(np.mean([r['source_gradient']['norm'] for r in rs])),mean_category_logits={c:float(np.mean([r['mean_logits_by_category'][j] for r in rs])) for j,c in enumerate(CLASSES)}))
    training=dict(updates=len(all_updates),recovery_updates=len(new_updates),photo_presentations=len(all_images),unique_training_photos=len({r['image_id'] for r in all_images}),known_source_groups=len({r['source_group'] for r in all_images}),positive_targets=sum(r['positive_boxes'] for r in all_images),epochs=epochs,probes=probes,cap_activations=sum(e['caps_activated'] for e in epochs),cap_violations=sum(u['metrics']['cap_violations'] for u in all_updates),positive_box_buffers_preserved=all(u['metrics']['positive_buffers_unchanged'] and u['metrics']['box_buffers_unchanged'] for u in all_updates),mixed_pre_stop_epoch2_source_vectors_not_reconstructed=True)
    localization={}
    if 'C20_E5' in predictions:
        for c in ('chair','sofa','bed'):
            aa={(e['image_id'],e['gt_index']):e['iou'] for e in m.events(rows,predictions['C8_B_E5'],c,floor=.15)[0] if e['outcome']=='tp'};bb={(e['image_id'],e['gt_index']):e['iou'] for e in m.events(rows,predictions['C20_E5'],c,floor=.15)[0] if e['outcome']=='tp'};shared=sorted(set(aa)&set(bb))
            localization[c]=dict(common_matched_GT=len(shared),C8_only_matches=len(set(aa)-set(bb)),C20_only_matches=len(set(bb)-set(aa)),mean_C8_IoU=float(np.mean([aa[k] for k in shared])) if shared else None,mean_C20_IoU=float(np.mean([bb[k] for k in shared])) if shared else None,mean_delta_IoU=float(np.mean([bb[k]-aa[k] for k in shared])) if shared else None,IoU75_crossings_up=sum(bb[k]>=.75>aa[k] for k in shared),IoU75_crossings_down=sum(aa[k]>=.75>bb[k] for k in shared))
    result=dict(status='COMPLETE' if success else 'SAFELY_INTERRUPTED',comparison=comparison,logit_drift=drift,training_summary={k:v for k,v in training.items() if k!='epochs'},common_GT_localization=localization,resources=terminal,restoration=read(RUN20/'restoration.json') if (RUN20/'restoration.json').exists() else None,protocol=dict(photos=19,GT=sum(not a['ignore'] for r in dev for a in r['annotations']),supported_classes=['sofa','chair','bed'],unsupported_classes=['table','desk','cabinet','lamp'],fp_scoring_photos={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},confidence=.15,score_floor=.001,NMS_IoU=.5,prediction_cap=300,rows_sha256=digest(dev),metric_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'),independent_accuracy=False,reserved_or_frozen_scored=False,unknown_predictions_not_confirmed_FP=True),checkpoint28_scored=False,new_GPU_forwards_in_analysis=0)
    emit20(REPORT20/'results.json',result);emit20(REPORT20/'training-curves.json',training);emit20(REPORT20/'precision-recall.json',{k:v['metrics']['pr_curve'] for k,v in entries.items()})
    emit20(cpu/'complete.json',dict(status='PASS',new_GPU_forwards=0,saved_prediction_models=list(entries),identical_historical_metrics_verified=True))
    print('C20 separate CPU analysis complete;no new inference or training.',flush=True)


if __name__=='__main__':run()
