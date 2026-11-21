"""Saved-output CPU analysis only; no forward calls or model selection."""
import math
from collections import Counter,defaultdict
import numpy as np
import torch
from common13 import *
from analyze11 import envelope,delta_stats,load_raw,write_csv
from analyze9 import threshold_summary


def paired_uncertainty(rows,predictions,metrics,m):
    groups=sorted({r['identity_group'] for r in rows});byid={r['image_id']:r for r in rows};counts={}
    for model,ps in predictions.items():
        matrix=np.zeros((len(groups),3),dtype=int)
        for c in CLASSES:
            events,_,_=m.events(rows,ps,c,floor=.15)
            for e in events:matrix[groups.index(byid[e['image_id']]['identity_group']),0 if e['outcome']=='tp' else 1]+=1
        for row in rows:matrix[groups.index(row['identity_group']),2]+=sum(not a['ignore'] and row['coverage'][a['label']] for a in row['annotations'])
        total=matrix.sum(0)
        if tuple(total)!=(metrics[model]['tp'],metrics[model]['fp'],metrics[model]['tp']+metrics[model]['fn']):raise ValueError('Bootstrap counts differ from frozen metric')
        counts[model]=matrix
    draws=np.random.default_rng(6106).integers(0,len(groups),size=(2000,len(groups)));sampled={k:v[draws].sum(1).astype(float) for k,v in counts.items()};result={}
    for baseline in ('original','corrected_B','frozen_calibration'):
        a=sampled[baseline];b=sampled['projection_regularized'];pv=(a[:,0]+a[:,1]>0)&(b[:,0]+b[:,1]>0);rv=(a[:,2]>0)&(b[:,2]>0)
        dp=b[pv,0]/(b[pv,0]+b[pv,1])-a[pv,0]/(a[pv,0]+a[pv,1]);dr=b[rv,0]/b[rv,2]-a[rv,0]/a[rv,2]
        result[baseline]=dict(precision_valid_resamples=int(pv.sum()),recall_valid_resamples=int(rv.sum()),precision_delta_95_percentile_interval=np.quantile(dp,[.025,.975]).tolist() if len(dp) else None,recall_delta_95_percentile_interval=np.quantile(dr,[.025,.975]).tolist() if len(dr) else None)
    return dict(known_source_groups=len(groups),resamples=2000,seed=6106,comparisons=result,caveat='Exploratory paired known-source-group bootstrap of fixed-threshold counts on the exposed partial-label panel;does not repair incomplete annotations or prove population performance/causal significance')


def run():
    done=read(FUTURE/'complete.json');integrity=read(REPORT13/'integrity.json');_,train,dev=verify_inputs13();m=metric_module();rows=[m.scopes(r) for r in dev];cpu=FUTURE/'cpu-evaluation-v1'
    if integrity['status']!='PASS' or not (cpu/'complete.json').exists():raise ValueError('Verified completed execution and separate CPU metrics required')
    entries=read(C8/'metric-results.json')['entries'];metrics={'original':read(C11_RUN/'cpu-diagnostics-v1/metrics-E0.json')['metrics'],'corrected_B':next(e['metrics'] for e in entries if e['arm']=='B-corrected' and e['epoch']==5),'frozen_calibration':read(C11_RUN/'cpu-diagnostics-v1/metrics-E5.json')['metrics'],'projection_regularized':read(cpu/'metrics-E5.json')['metrics']}
    if read(cpu/'metrics-E0.json')['metrics']!=metrics['original']:raise ValueError('Original metric reproduction mismatch')
    folders={'original':C11_RUN/'evaluation/epoch-0','corrected_B':C8/'B-corrected/evaluation/epoch-5','frozen_calibration':C11_RUN/'evaluation/epoch-5','projection_regularized':FUTURE/'evaluation/epoch-5'}
    prediction_files={'original':C11_RUN/'cpu-diagnostics-v1/predictions-E0.json','corrected_B':C8/'B-corrected/evaluation/epoch-5/predictions.json','frozen_calibration':C11_RUN/'cpu-diagnostics-v1/predictions-E5.json','projection_regularized':cpu/'predictions-E5.json'}
    ps={k:read(path) for k,path in prediction_files.items()};raw={k:load_raw(path,dev) for k,path in folders.items()};logits={k:np.concatenate([a['logits'] for a in v]) for k,v in raw.items()};comparison={};curves=[]
    for model,metric in metrics.items():
        if metric['confidence']!=.15 or metric['nms_iou']!=.5:raise ValueError('Historical threshold mismatch')
        score=1/(1+np.exp(-logits[model].astype(float)));thresholds={str(t):threshold_summary(rows,ps[model],m,t) for t in (.15,.5,.8)}
        comparison[model]=dict(headline={k:v for k,v in metric.items() if k not in ('pr_curve','categories')},categories=metric['categories'],thresholds=thresholds,unknown_unmatched_at_015=sum(v['unknown_unmatched'] for v in thresholds['0.15'].values()),precision_at_matched_recall={k:envelope(metric['pr_curve'],metrics[k]['recall']) for k in ('original','corrected_B','frozen_calibration')},raw_aligned_query_score_quantiles={c:quantiles(score[:,j]) for j,c in enumerate(CLASSES)},raw_Chair_query_counts={str(t):int((score[:,1]>=t).sum()) for t in (.15,.5,.8)})
        curves.extend(dict(model=model,**p) for p in metric['pr_curve'])
    drift={k:delta_stats(logits['original'],v) for k,v in logits.items() if k!='original'}
    original=original_heads();states={};parameter={}
    for label,root in (('corrected_B',C8/'B-corrected'),('frozen_calibration',C11_RUN),('projection_regularized',FUTURE)):
        complete=read(root/'complete.json');cp=(C8 if label=='corrected_B' else root)/complete['final_checkpoint']['file']
        if sha(cp)!=complete['final_checkpoint']['sha256']:raise ValueError('Historical/candidate checkpoint changed')
        w=torch.load(cp,map_location='cpu',weights_only=True)['weights'];states[label]=w;parameter[label]={}
        for n in PROJECTION_KEYS:
            d=w[n].double()-original[n].double();parameter[label][n]=dict(L2=float(d.norm()),relative_L2=float(d.norm()/original[n].double().norm()),max_abs=float(d.abs().max()),tensor_sha256=tensor_sha(w[n]))
        parameter[label]['joint_L2']=math.sqrt(sum(parameter[label][n]['L2']**2 for n in PROJECTION_KEYS))
    snapshots={r['update']:r for r in integrity['displacement_by_update']};updates=[read(FUTURE/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)];epoch_records=[];gradient_records=[]
    probe0=[read(FUTURE/'probes/epoch-0'/(str(i).zfill(2)+'.json')) for i in range(1,25)]
    loss_keys=('existing_total_loss','total_loss','classification_loss','positive_classification','legacy_negative_classification','added_Chair_negative_loss','Chair_positive_classification','Chair_negative_classification','box_loss','projection_penalty','weighted_projection_penalty')
    epoch_records.append(dict(epoch=0,updates=0,scope='Fixed24 optimization probes before training',photos=24,loss_means={k:float(np.mean([r[k] for r in probe0])) for k in loss_keys},endpoint=snapshots[0],regularizer_gradient_norm=0.0))
    for u in updates:
        d=u['projection_gradient_parts'];reg=math.sqrt(sum(d[n]['weighted_regularizer_gradient_norm']**2 for n in PROJECTION_KEYS));data=math.sqrt(sum(d[n]['existing_objective_gradient_norm_after_subtraction']**2 for n in PROJECTION_KEYS));total=math.sqrt(sum(d[n]['combined_preclip_gradient_norm']**2 for n in PROJECTION_KEYS))
        if not math.isclose(reg,math.sqrt(2*snapshots[u['updates']-1]['unweighted_LSP'])*done['projection_preservation']['lambda_value'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Regularization gradient disagrees with analytical prior')
        gradient_records.append(dict(update=u['updates'],epoch=u['epoch'],data_gradient_norm=data,regularization_gradient_norm=reg,combined_gradient_norm=total,regularization_to_data_ratio=reg/data if data else None))
    for epoch in range(1,6):
        group=[u for u in updates if u['epoch']==epoch];images=[r for u in group for r in u['images']];grad=[r for r in gradient_records if r['epoch']==epoch]
        epoch_records.append(dict(epoch=epoch,updates=15*epoch,scope='Means across59 optimization-photo presentations during epoch;endpoint is post-update snapshot',photos=len(images),loss_means={k:float(np.mean([r[k] for r in images])) for k in loss_keys},endpoint=snapshots[15*epoch],gradient_norm_quantiles={k:quantiles([r[k] for r in grad if r[k] is not None]) for k in ('data_gradient_norm','regularization_gradient_norm','regularization_to_data_ratio')},added_Chair_term_occurrences=sum(r['added_active_Chair_terms'] for r in images),protected_violations=sum(r['protected_violations'] for r in images)))
    images=[r for u in updates for r in u['images']];neg=[s for r in images for s in r['added_Chair_scores']]
    uncertainty=paired_uncertainty(rows,ps,metrics,m)
    changes=dict(projection_joint_L2_reduction_fraction_vs_C11=1-parameter['projection_regularized']['joint_L2']/parameter['frozen_calibration']['joint_L2'],mean_shared_logit_increase_reduction_fraction_vs_C11=1-drift['projection_regularized']['common_mean']/drift['frozen_calibration']['common_mean'],precision_delta_at015_vs_C11=metrics['projection_regularized']['precision']-metrics['frozen_calibration']['precision'],recall_delta_at015_vs_C11=metrics['projection_regularized']['recall']-metrics['frozen_calibration']['recall'])
    result=dict(comparison=comparison,logit_drift_vs_original=drift,projection_parameter_drift=parameter,changes_vs_C11=changes,resource=done,queue=read(FUTURE/'queue-complete.json'),epochs=epoch_records,gradient_records=gradient_records,paired_uncertainty=uncertainty,
        negative_training_diagnostics=dict(term_occurrences=len(neg),score_quantiles=quantiles(neg),terms_at_least015=sum(s>=.15 for s in neg),terms_at_least05=sum(s>=.5 for s in neg),protected_violations=0),protocol=dict(internal_photos=19,known_source_groups=len({r['identity_group'] for r in dev}),GT=61,supported_classes=['chair','sofa','bed'],unsupported_classes=['table','desk','cabinet','lamp'],FP_complete_photo_counts={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},confidence=.15,NMS=.5,cap=300,AP_floor=.001,AP='101-point macro over3 supported classes',unknown_unmatched_are_not_confirmed_mistakes=True,all_comparison_manifests_match=True,independent_verified_accuracy=False,reserved_or_frozen_scored=False,Chair_stool_native_taxonomy_limit_retained=True))
    save(REPORT13/'results.json',result);save(REPORT13/'training-diagnostics.json',dict(epochs=epoch_records,gradients=gradient_records,negative_training=result['negative_training_diagnostics']))
    save(REPORT13/'confidence-drift.json',dict(logit_drift=drift,parameter_drift=parameter,changes=changes,query_count=68400,photo_count=19,alignment='Exact photo and query indices across original/C8B/C11/C13;score quantiles in results.json',projection_weight_and_bias_not_functionally_separated=True))
    write_csv(REPORT13/'precision-recall.csv',curves,['model','threshold','precision','recall','tp','fp'])
    write_csv(REPORT13/'epoch-losses.csv',[dict(epoch=r['epoch'],updates=r['updates'],**r['loss_means'],endpoint_projection_L2=r['endpoint']['joint_L2'],endpoint_weighted_LSP=r['endpoint']['weighted_LSP']) for r in epoch_records],['epoch','updates']+list(epoch_records[0]['loss_means'])+['endpoint_projection_L2','endpoint_weighted_LSP'])
    plots(result,metrics)
    print('C13 four-model comparison,matched recall,paired uncertainty,loss/gradient/drift diagnostics saved',flush=True)


def plots(result,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'original':'#222222','corrected_B':'#cb7420','frozen_calibration':'#3082b1','projection_regularized':'#438b35'}
    for name,m in metrics.items():
        p=m['pr_curve'];axes[0,0].plot([v['recall'] for v in p],[v['precision'] for v in p],label=name,color=colors[name]);axes[0,0].scatter(m['recall'],m['precision'],color=colors[name],s=25)
    axes[0,0].set(xlabel='Recall',ylabel='Precision',title='Exposed partial-label internal panel;dots at0.15',xlim=(0,1),ylim=(0,1));axes[0,0].legend(fontsize=7)
    epochs=result['epochs'][1:]
    for key in ('classification_loss','box_loss','weighted_projection_penalty'):
        axes[0,1].plot([r['epoch'] for r in epochs],[r['loss_means'][key] for r in epochs],marker='o',label=key)
    axes[0,1].set(xlabel='Epoch',ylabel='Mean per-image loss',title='Actual C13 training losses');axes[0,1].legend(fontsize=8)
    grad=result['gradient_records']
    for key in ('data_gradient_norm','regularization_gradient_norm'):
        axes[1,0].plot([r['update'] for r in grad],[r[key] for r in grad],label=key)
    axes[1,0].set(xlabel='Optimizer update',ylabel='Projection gradient norm before clipping',title='Data component by subtraction;FP32/AMP rounding');axes[1,0].legend(fontsize=8)
    x=np.arange(7);width=.26
    for offset,key in ((-1,'corrected_B'),(0,'frozen_calibration'),(1,'projection_regularized')):
        axes[1,1].bar(x+offset*width,[result['logit_drift_vs_original'][key]['mean_logit_delta_by_category'][c] for c in CLASSES],width,label=key,color=colors[key])
    axes[1,1].set_xticks(x,CLASSES,rotation=25);axes[1,1].set(ylabel='Mean logit increase vs pretrained',title='Same68,400 query indices');axes[1,1].legend(fontsize=7)
    path=REPORT13/'diagnostics.svg';fig.savefig(path,metadata={'Date':None});plt.close(fig);path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':run()
