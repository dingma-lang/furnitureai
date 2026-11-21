"""CPU analysis of already saved predictions. No model/encoder forward calls."""
import csv
import math
from collections import Counter
import numpy as np
import torch
from common11 import *
from common10 import CLASSES
from analyze9 import threshold_summary,statistical_mode_change


def envelope(points,target):
    eligible=[p for p in points if p['recall']>=target]
    return max(eligible,key=lambda p:p['precision']) if eligible else None


def delta_stats(x,y):
    result=statistical_mode_change(x,y);delta=y.astype(float)-x.astype(float)
    result.update(common_mean=float(delta.mean()),common_RMS=float(np.sqrt(np.mean(delta.mean(1)**2))),
        caveat='Aligned raw queries, same photos/backbone. Common-mode statistics are descriptive; separately endpoint-validated functional attribution uses actual cached features.')
    return result


def load_raw(folder,dev):
    marker=read(folder/'raw-complete.json');items=[]
    if marker['rows_sha256']!=digest(dev) or len(marker['records'])!=len(dev) or not marker['FP32']:raise ValueError('Different comparison data or precision')
    for row,record in zip(dev,marker['records']):
        path=folder/record['file']
        if record['image_id']!=row['image_id'] or record['source_sha256']!=row['sha256'] or sha(path)!=record['sha256']:raise ValueError('Raw comparison identity/hash mismatch')
        a=load_npz(path)
        if a['logits'].shape!=(3600,7) or not np.isfinite(a['logits']).all():raise ValueError('Query layout invalid')
        items.append(a)
    return items


def write_csv(path,records,keys):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(records)


def run():
    done=read(FUTURE/'complete.json');_,train,dev=verify_inputs();m=metric_module();rows=[m.scopes(r) for r in dev]
    cpu=FUTURE/'cpu-diagnostics-v1'
    if not (cpu/'complete.json').exists():raise ValueError('Complete CPU metrics and endpoint-validated attribution required')
    historical=read(C8/'metric-results.json')['entries'];old=lambda arm,e:next(r['metrics'] for r in historical if r['arm']==arm and r['epoch']==e)
    metrics={'original':old('A-control',0),'corrected_B':old('B-corrected',5),'frozen_calibration':read(cpu/'metrics-E5.json')['metrics']}
    folders={'original':C8/'A-control/evaluation/epoch-0','corrected_B':C8/'B-corrected/evaluation/epoch-5','frozen_calibration':FUTURE/'evaluation/epoch-5'}
    predictions={k:read((cpu/'predictions-E5.json') if k=='frozen_calibration' else v/'predictions.json') for k,v in folders.items()}
    raw={k:load_raw(v,dev) for k,v in folders.items()};logits={k:np.concatenate([a['logits'] for a in v]) for k,v in raw.items()}
    comparison={};curves=[]
    for key,measured in metrics.items():
        threshold={str(t):threshold_summary(rows,predictions[key],m,t) for t in (.15,.5,.8)}
        scores=1/(1+np.exp(-logits[key].astype(float)))
        common=sum(int(v['unknown_unmatched']) for v in threshold['0.15'].values())
        comparison[key]=dict(headline={k:v for k,v in measured.items() if k not in ('pr_curve','categories')},categories=measured['categories'],thresholds=threshold,
            precision_at_original_recall=envelope(measured['pr_curve'],metrics['original']['recall']),precision_at_B_recall=envelope(measured['pr_curve'],metrics['corrected_B']['recall']),
            raw_aligned_query_score_quantiles={c:quantiles(scores[:,j]) for j,c in enumerate(CLASSES)},unknown_unmatched_at_015=common,
            queries_per_photo=3600,aligned_query_total=len(scores),same_query_Chair_scores_at_least={str(t):int((scores[:,1]>=t).sum()) for t in (.15,.5,.8)})
        curves.extend(dict(model=key,**p) for p in measured['pr_curve'])
    drift={k:delta_stats(logits['original'],logits[k]) for k in ('corrected_B','frozen_calibration')}
    # Follow original GT-matched Chair query indices across models: no changing sample selection.
    original_events,_,_=m.events(rows,predictions['original'],'chair',floor=.15)
    anchor=[]
    from analyze9 import reconstruct
    registered={r['image_id']:reconstruct(r,a,predictions['original'][r['image_id']],m) for r,a in zip(dev,raw['original'])}
    byid={r['image_id']:i for i,r in enumerate(dev)}
    for e in original_events:
        if e['outcome']!='tp':continue
        q=registered[e['image_id']][e['index']]['query_index'];photo=byid[e['image_id']]
        anchor.append({k:float(1/(1+math.exp(-float(v[photo]['logits'][q,1])))) for k,v in raw.items()})
    original=original_heads();states={}
    old_checkpoint=read(C8/'B-corrected/complete.json')['final_checkpoint'];states['corrected_B']=torch.load(C8/old_checkpoint['file'],map_location='cpu',weights_only=True)['weights']
    states['frozen_calibration']=torch.load(FUTURE/done['final_checkpoint']['file'],map_location='cpu',weights_only=True)['weights']
    projection={}
    for key,state in states.items():
        projection[key]={}
        for suffix in ('weight','bias'):
            name='class_head.dense0.'+suffix;delta=(state[name]-original[name]).double()
            projection[key][name]=dict(original_L2=float(original[name].double().norm()),delta_L2=float(delta.norm()),relative_delta_L2=float(delta.norm()/original[name].double().norm()),delta_max_abs=float(delta.abs().max()))
    attribution=read(cpu/'functional-attribution.json');functional={}
    for model,records in attribution['models'].items():
        if len(records)!=19:raise ValueError('Attribution membership differs')
        functional[model]=dict(photos=19,queries=68400,mean_all_category_contribution={k:float(np.mean([r['mean_contribution'][k] for r in records])) for k in ('projection','shift','scale')},
            mean_Chair_contribution={k:float(np.mean([r['mean_Chair_contribution'][k] for r in records])) for k in ('projection','shift','scale')},max_GPU_endpoint_abs_error=max(r['endpoint_max_error'] for r in records))
    for key,attr in (('corrected_B','B'),('frozen_calibration','candidate')):
        if not math.isclose(sum(functional[attr]['mean_all_category_contribution'].values()),drift[key]['common_mean'],abs_tol=1e-4):raise ValueError('Attribution does not close to observed drift')
    updates=[read(FUTURE/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)]
    all_images=[r for u in updates for r in u['images']];epoch_stats=[]
    if len(all_images)!=295 or len({r['image_id'] for r in all_images})!=59:raise ValueError('Training count mismatch')
    for epoch in range(1,6):
        images=[r for u in updates if u['epoch']==epoch for r in u['images']]
        means={k:float(np.mean([r[k] for r in images])) for k in ('total_loss','classification_loss','box_loss','positive_classification','legacy_negative_classification','added_Chair_negative_loss','Chair_positive_classification','Chair_negative_classification','Chair_source_negative_classification')}
        norms={n:quantiles([u['named_preclip_gradients'][n] for u in updates if u['epoch']==epoch]) for n in sorted(TRAIN_KEYS) if n.startswith('class_head.')}
        epoch_stats.append(dict(epoch=epoch,images=len(images),image_mean=means,preclip_gradient_norms=norms,added_Chair_term_occurrences=sum(r['added_active_Chair_terms'] for r in images),protected_violations=sum(r['protected_violations'] for r in images)))
    probes=[]
    for epoch in (0,1,3,5):
        folder=FUTURE/'probes'/('epoch-'+str(epoch));records=[read(folder/(str(i).zfill(2)+'.json')) for i in range(1,25)]
        region_counts=Counter();covered_photos=set()
        for r in records:
            if r['protected_violations'] or not r['calibration_gradients_none']:raise ValueError('Probe protection/calibration failure')
            region_counts.update(r['region_query_counts'])
            if r['added_active_Chair_terms']:covered_photos.add(r['image_id'])
        bytrain={r['image_id']:r for r in train};scores=[s for r in records for s in r['added_Chair_scores']]
        probes.append(dict(epoch=epoch,added_terms=sum(r['added_active_Chair_terms'] for r in records),photos_with_coverage=len(covered_photos),known_groups_with_coverage=len({bytrain[k]['identity_group'] for k in covered_photos}),regions_with_coverage=sum(n>0 for n in region_counts.values()),regions_without_coverage=41-sum(n>0 for n in region_counts.values()),added_Chair_score_quantiles=quantiles(scores),added_terms_at_least_015=sum(s>=.15 for s in scores),added_terms_at_least_05=sum(s>=.5 for s in scores),protected_violations=0))
    scores=[s for r in all_images for s in r['added_Chair_scores']]
    training=dict(epochs=epoch_stats,updates=75,training_photo_presentations=295,known_products_or_independent_physical_identities_not_established=True,
        training_group_count=len({r['identity_group'] for r in train}),negative_term_occurrences=sum(r['added_active_Chair_terms'] for r in all_images),negative_score_quantiles=quantiles(scores),negative_terms_at_least_015=sum(s>=.15 for s in scores),negative_terms_at_least_05=sum(s>=.5 for s in scores),probes=probes,
        image_mean_caveat='Diagnostic unweighted means across59 per-image normalized losses; optimizer accumulation remains original C8B averaging within groups of4,last3.')
    result=dict(comparison=comparison,common_logit_drift=drift,functional_attribution=functional,projection_changes=projection,
        original_Chair_GT_matched_query_anchor=dict(queries=len(anchor),selection='Original@0.15 TP queries followed at identical photo/query indices; not independently rematched predictions',score_quantiles={k:quantiles([r[k] for r in anchor]) for k in raw}),
        resource=done,training=training,protocol=dict(photos=19,GT=61,supported_classes=['chair','sofa','bed'],unsupported_classes=['table','desk','cabinet','lamp'],FP_scoring_photos={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},confidence=.15,AP='101-point macro over3 supported categories',NMS_IoU=.5,maximum_predictions=300,score_floor=.001,reserved_or_frozen_scored=False,independent_verified_accuracy=False,unknown_unmatched_are_not_confirmed_errors=True,stools_taxonomy_caveat='Project Chair includes seating/bar stools; native Chair protocols retain historical separate Stool semantics. No labels or evaluation rules changed.'))
    save(REPORT11/'results.json',result)
    save(REPORT11/'training-curves.json',training)
    save(REPORT11/'confidence-attribution.json',dict(common_logit_drift=drift,functional_attribution=functional,scope=attribution['scope'],projection_changes=projection,extra_encoder_forwards=0))
    save(REPORT11/'execution-provenance.json',dict(worker=read(FUTURE/'complete.json'),queue=read(FUTURE/'queue-complete.json'),GPU_device=read(FUTURE/'environment.json') if (FUTURE/'environment.json').exists() else None,authorized_config_file='reports/phase10c11-v1/authorized-config-v2.json',authorized_config_sha256=sha(REPORT11/'authorized-config-v2.json'),atomic_raw_prediction_markers=[dict(epoch=e,sha256=sha(FUTURE/'evaluation'/('epoch-'+str(e))/'raw-complete.json')) for e in (0,1,3,5)],CPU_metrics_environment='Same existing .venv Python; CUDA_VISIBLE_DEVICES empty,2 BLAS/OMP threads; no environment or metric changes',initial_zero_GPU_startup_error='Standalone module import failed before destination/CUDA creation;fixed isolated import path and regression-tested,then ran the one authorized arm',new_downloads=0,another_arm_executed=False,phase10D=False))
    write_csv(REPORT11/'precision-recall.csv',curves,['model','threshold','precision','recall','tp','fp'])
    write_csv(REPORT11/'epoch-losses.csv',[dict(epoch=r['epoch'],**r['image_mean'],added_Chair_term_occurrences=r['added_Chair_term_occurrences']) for r in epoch_stats],['epoch']+list(epoch_stats[0]['image_mean'])+['added_Chair_term_occurrences'])
    plot(result,metrics)
    print('C11 comparison,losses,query drift,functional attribution and curves saved',flush=True)


def plot(result,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'original':'#222222','corrected_B':'#c45a24','frozen_calibration':'#147a96'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for key,m in metrics.items():
        ps=m['pr_curve'];axes[0,0].plot([p['recall'] for p in ps],[p['precision'] for p in ps],label=key,color=colors[key]);axes[0,0].scatter([m['recall']],[m['precision']],color=colors[key],s=30)
    axes[0,0].set(xlabel='Recall',ylabel='Precision',title='Exposed19-photo partial-label panel;dots at0.15',xlim=(0,1),ylim=(0,1));axes[0,0].legend(fontsize=8)
    epochs=result['training']['epochs']
    for key in ('classification_loss','box_loss','Chair_positive_classification','Chair_negative_classification'):
        axes[0,1].plot([r['epoch'] for r in epochs],[r['image_mean'][key] for r in epochs],marker='o',label=key)
    axes[0,1].set(xlabel='Epoch',ylabel='Mean per-image normalized loss',title='Frozen-calibration losses');axes[0,1].legend(fontsize=7)
    cats=CLASSES;x=np.arange(7);width=.38
    for offset,key in ((-.5,'corrected_B'),(.5,'frozen_calibration')):
        d=result['common_logit_drift'][key]['mean_logit_delta_by_category'];axes[1,0].bar(x+offset*width,[d[c] for c in cats],width,label=key,color=colors[key])
    axes[1,0].set_xticks(x,cats,rotation=25);axes[1,0].set(ylabel='Mean raw logit change',title='Same68400 queries vs pretrained');axes[1,0].legend(fontsize=8)
    x=np.arange(3)
    for offset,key in ((-.5,'corrected_B'),(.5,'frozen_calibration')):
        d=result['comparison'][key]['thresholds'];axes[1,1].bar(x+offset*width,[d[str(t)]['chair']['scored_fp'] for t in (.15,.5,.8)],width,label=key,color=colors[key])
    axes[1,1].set_xticks(x,['0.15','0.50','0.80']);axes[1,1].set(xlabel='Diagnostic score threshold',ylabel='Chair scored false positives',title='Unmatched unknown-scope predictions excluded');axes[1,1].legend(fontsize=8)
    svg=REPORT11/'diagnostics.svg';fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':run()
