"""Identical native protocol in isolated CPU/Torch-free process,saved outputs only."""
import csv
import numpy as np
from common5 import *
sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
from analyzeD import load_raw,predictions,envelope,quantiles
import metric_cpu as metric

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:raise PermissionError('Isolated Torch-free CPU metrics required')
    cfg,context,_=inputs();done=read(RUN/'complete.json');dev=context['development'];rows=[metric.scopes(r) for r in dev]
    if done['status']!='PASS' or done['resources']['updates']!=75 or done['coefficient']!=COEFFICIENT:raise ValueError('Complete authorized endpoint required')
    history=read(PREP/'historical-evidence.json')['models']
    folders=dict(original=ROOT/'data/open-images/experiments/phase10c8-ab-v2/A-control/evaluation/epoch-0',
                 C8=ROOT/'data/open-images/experiments/phase10c8-ab-v2/B-corrected/evaluation/epoch-5',
                 C11=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/evaluation/epoch-5',E5=RUN/'predictions')
    raw={};pred={};measured={};comparison={};logits={}
    for name,folder in folders.items():
        a=load_raw(folder,dev);raw[name]=a;p=predictions(a,dev);pred[name]=p
        emit(RUN/'CPU-analysis'/('predictions-'+name+'.json'),p)
        m=metric.evaluate(rows,p);measured[name]=m;headline={k:v for k,v in m.items() if k not in ('pr_curve','categories')}
        if name!='E5' and headline!=history[name]['headline']:raise ValueError('Historical baseline equivalence failed:'+name)
        z=np.concatenate([v['logits'] for v in a]).astype(np.float64);logits[name]=z;scores=1/(1+np.exp(-z))
        unknown={c:metric.events(rows,p,c,floor=.15)[2] for c in CLASSES}
        comparison[name]=dict(headline=headline,categories=m['categories'],
              precision_at_original_recall=envelope(m,history['original']['headline']['recall']),
              precision_at_E5_recall=None,raw_logit_quantiles={c:quantiles(z[:,j]) for j,c in enumerate(CLASSES)},
              raw_score_quantiles={c:quantiles(scores[:,j]) for j,c in enumerate(CLASSES)},
              raw_Chair_scores_ge={str(t):int((scores[:,1]>=t).sum()) for t in (.15,.5,.8)},
              unknown_unmatched_at_015=unknown,scored_FP_score_quantiles={c:quantiles([e['score'] for e in metric.events(rows,p,c,floor=.15)[0] if e['outcome']=='fp']) for c in CLASSES})
        emit(RUN/'CPU-analysis'/('metrics-'+name+'.json'),m)
        print('CPU METRICS',name,'P/R',m['precision'],m['recall'],'TP/FP/FN',m['tp'],m['fp'],m['fn'],flush=True)
    for name in comparison:comparison[name]['precision_at_E5_recall']=envelope(measured[name],measured['E5']['recall'])
    drift={}
    for name in ('C8','C11','E5'):
        d=logits[name]-logits['original']
        drift[name]=dict(mean_all_categories=float(d.mean()),per_class_mean={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},
             per_class_delta_quantiles={c:quantiles(d[:,j]) for j,c in enumerate(CLASSES)},
             raw_query_argmax_disagreements=int((logits[name].argmax(1)!=logits['original'].argmax(1)).sum()),
             baseline_Chair_score_ge015=int(((1/(1+np.exp(-logits['original'][:,1])))>=.15).sum()),
             student_Chair_score_ge015=comparison[name]['raw_Chair_scores_ge']['0.15'])
    localization={}
    for c in ('sofa','chair','bed'):
        aa={(e['image_id'],e['gt_index']):e['iou'] for e in metric.events(rows,pred['original'],c,floor=.15)[0] if e['outcome']=='tp'}
        bb={(e['image_id'],e['gt_index']):e['iou'] for e in metric.events(rows,pred['E5'],c,floor=.15)[0] if e['outcome']=='tp'}
        shared=sorted(set(aa)&set(bb))
        localization[c]=dict(common_matches=len(shared),original_only=len(set(aa)-set(bb)),E5_only=len(set(bb)-set(aa)),
                mean_original_IoU=float(np.mean([aa[k] for k in shared])) if shared else None,
                mean_E5_IoU=float(np.mean([bb[k] for k in shared])) if shared else None,
                mean_delta_IoU=float(np.mean([bb[k]-aa[k] for k in shared])) if shared else None,
                crossings_up75=sum(bb[k]>=.75>aa[k] for k in shared),crossings_down75=sum(aa[k]>=.75>bb[k] for k in shared))
    updates=[read(RUN/'updates'/f'{i:03}.json') for i in range(1,76)];epochs=[]
    for epoch in range(1,6):
        photos=[r for u in updates if u['epoch']==epoch for r in u['images']]
        active=[r for r in photos if r['stats']['mask']['terms']]
        ratio=[r['gradients']['weighted_KD_positive_ratio'] for r in active if r['gradients']['weighted_KD_positive_ratio'] is not None]
        losses=dict(positive=float(np.mean([r['stats']['base']['positive_classification'] for r in photos])),
                    classification=float(np.mean([r['stats']['base']['classification_loss'] for r in photos])),
                    box=float(np.mean([r['stats']['base']['box_loss'] for r in photos])),
                    unit_KD=float(np.mean([r['stats']['distillation'] for r in photos])),
                    weighted_KD=float(np.mean([r['weighted_KD_loss'] for r in photos])))
        epochs.append(dict(epoch=epoch,updates=epoch*15,photo_presentations=len(photos),losses=losses,
                active_KD_photos=len(active),active_KD_source_groups=len({r['source_group'] for r in active}),
                KD_term_occurrences=sum(r['stats']['mask']['terms'] for r in photos),
                weighted_KD_positive_ratio=quantiles(ratio),
                positive_projection_L2=quantiles([r['gradients']['positive_projection_L2'] for r in photos]),
                weighted_KD_projection_L2=quantiles([r['gradients']['weighted_KD_projection_L2'] for r in active]),
                positive_KD_cosine=quantiles([r['gradients']['KD_positive_cosine'] for r in active if r['gradients']['KD_positive_cosine'] is not None]),
                above_recovery_half_bound=sum(r['gradients']['above_recovery_half_bound'] for r in photos),
                KD_dominant=sum(r['gradients']['KD_dominant'] for r in photos),
                training_logit_drift_mean=float(np.mean([r['stats']['logit_diagnostics']['mean_seven_category_logit_drift'] for r in photos])),
                checkpoint=read(RUN/'checkpoints'/f'epoch-{epoch}.json')))
    e5=measured['E5'];orig=measured['original'];baseline=envelope(orig,orig['recall']);matched=envelope(e5,orig['recall'])
    serious_localization=e5['ap75']<orig['ap75']-.02 or any(e5['categories'][c]['ap75']<orig['categories'][c]['ap75']-.05 for c in ('sofa','chair','bed'))
    if e5['precision']>=.60 and e5['recall']>orig['recall'] and not serious_localization:outcome='A — Promising'
    elif e5['precision']<.30 and (drift['E5']['mean_all_categories']>.5 or e5['fp']>2*orig['fp']):outcome='C — Ineffective'
    elif e5['recall']<orig['recall']:outcome='B — Overconstrained'
    else:outcome='D — Inconclusive'
    result=dict(status='PASS',comparison=comparison,logit_drift=drift,localization=localization,training_epochs=epochs,
          decision=dict(outcome=outcome,precision60_recall_gain=e5['precision']>=.60 and e5['recall']>orig['recall'],
                        localization_regression_flag=serious_localization,
                        matched_recall_precision_delta=matched['precision']-baseline['precision'] if matched and baseline else None,
                        independent_accuracy_established=False,coefficient_changed=False,more_training_authorized=False),
          protocol=dict(photos=19,GT=61,confidence=.15,NMS_IoU=.5,cap=300,score_floor=.001,AP='Unchanged101pointmacro',
                        matching_IoUs=[.5,.75],supported_positive_classes=['sofa','chair','bed'],
                        categories_with_native_FP_scope={c:sum(r['fp_coverage'][c] for r in rows) for c in CLASSES},
                        unknown_unmatched_not_confirmed_errors=True,rows_sha256=digest(dev),metric_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'),
                        frozen_or_reserved_scored=False,thresholds_not_tuned=True),
          metrics_process=dict(interpreter=sys.executable,version=sys.version,torch_imported='torch' in sys.modules),
          resources=done['resources'],checkpoint=done['final_checkpoint'],coefficient=COEFFICIENT)
    emit(REPORT/'results.json',result);emit(REPORT/'decision.json',result['decision']);emit(REPORT/'training-curves.json',dict(epochs=epochs))
    emit(REPORT/'precision-recall.json',{n:m['pr_curve'] for n,m in measured.items()})
    stream=io.StringIO();fields=['model','precision','recall','ap50','ap75','tp','fp','fn'];writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for name,r in comparison.items():writer.writerow(dict(model=name,**{k:r['headline'][k] for k in fields[1:]}))
    atomic(REPORT/'comparison.csv',stream.getvalue().encode())
    emit(RUN/'CPU-analysis/complete.json',dict(status='PASS',models=list(comparison),historical_baselines_equivalent=True,new_GPU_forwards=0))
    print('E5 CPU evaluation complete:',outcome,flush=True)

if __name__=='__main__':run()
