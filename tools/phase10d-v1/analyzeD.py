"""Unchanged native metrics in a separate CPU-only, Torch-free process."""
import csv
import zipfile
import numpy as np
from commonD import *
import metric_cpu as metric


def quantiles(values):
    a=np.asarray(values,dtype=np.float64)
    return {str(q):float(np.quantile(a,q)) if a.size else None for q in (0,.5,.9,.99,1)}


def envelope(measured,recall):
    candidates=[p for p in measured['pr_curve'] if p['recall'] is not None and p['recall']>=recall]
    return max(candidates,key=lambda p:p['precision']) if candidates else None


def load_raw(folder,rows):
    marker=read(folder/'raw-complete.json')
    if marker['FP32'] is not True or marker['rows_sha256']!=digest(rows) or len(marker['records'])!=19:
        raise ValueError('Exact19-photo FP32 saved protocol required')
    raw=[]
    for r,ref in zip(rows,marker['records']):
        path=(folder/ref['file']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path)!=ref['sha256'] or ref['image_id']!=r['image_id'] or ref['source_sha256']!=r['sha256']:
            raise ValueError('Raw cache identity/hash mismatch')
        with zipfile.ZipFile(path) as z:
            if len(z.infolist())!=5 or sum(v.file_size for v in z.infolist())>600000:raise ValueError('Unsafe raw archive')
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if set(a)!={'logits','square_cxcywh','scores','labels','boxes'} or a['logits'].shape!=(3600,7) or a['boxes'].shape!=(3600,4) or a['square_cxcywh'].shape!=(3600,4) or a['scores'].shape!=(3600,) or a['labels'].shape!=(3600,):raise ValueError('Raw array schema mismatch')
        for k,v in a.items():
            if v.dtype!=np.dtype('int64' if k=='labels' else 'float32') or not np.isfinite(v).all():raise ValueError('Raw dtype/nonfinite failure')
        if (a['labels']<0).any() or (a['labels']>6).any():raise ValueError('Invalid native query category')
        raw.append(a)
    return raw


def predictions(raw,rows):
    pred={}
    for a,row in zip(raw,rows):
        ps=[]
        for box,score,label in zip(a['boxes'].tolist(),a['scores'].tolist(),a['labels'].tolist()):
            if score>=.001 and box[2]>box[0] and box[3]>box[1]:ps.append(dict(box=box,score=score,label=CLASSES[label]))
        pred[row['image_id']]=metric.nms(ps,.5,300)
    return pred


def success_gates(original,adapted,localization):
    recall=original['recall'];a=envelope(adapted,recall);o=envelope(original,recall)
    milestone=adapted['precision']>=.60 and adapted['recall']>=recall
    positive_localization=adapted['ap75']>original['ap75'] and any(v['mean_delta_IoU'] is not None and v['mean_delta_IoU']>0 for v in localization.values())
    regressed=adapted['precision']<original['precision'] or adapted['recall']<original['recall']
    outcome='C' if positive_localization and regressed else 'A' if positive_localization else 'B'
    gates=dict(precision_not_below_original=adapted['precision']>=original['precision'],recall_not_below_original=adapted['recall']>=original['recall'],AP75_gain_ge002=adapted['ap75']>=original['ap75']+.02,per_category_AP75_drop_le002=all(adapted['categories'][c]['ap75']>=original['categories'][c]['ap75']-.02 for c in ('sofa','chair','bed')),TP_floors=all(adapted['categories'][c]['tp']>=n for c,n in {'sofa':11,'chair':24,'bed':7}.items()),matched_recall_precision_gain_ge003=a is not None and o is not None and a['precision']>=o['precision']+.03)
    return dict(outcome=outcome,development_60_precision_original_recall_milestone=milestone,localization_improved=positive_localization,precision_or_recall_regressed=regressed,C21_stronger_gates=gates,C21_stronger_gates_all_pass=all(gates.values()),independent_acceptance=False)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise PermissionError('Separate CPU-only Torch-free metrics process required')
    cfg=read(RUN/'binding.json');done=read(RUN/'complete.json');dev=read(RUN/'development-rows.json')
    if done['status']!='PASS' or done['updates']!=75 or digest(dev)!=cfg['development_rows_sha256']:
        raise ValueError('Complete single arm and exact diagnostic rows required')
    if sha(ROOT/'tools/phase10c-v1/metric_cpu.py')!=cfg['files']['tools/phase10c-v1/metric_cpu.py']:
        raise ValueError('Historical scoring changed')
    rows=[metric.scopes(r) for r in dev]
    folders={'original':ROOT/'data/open-images/experiments/phase10c8-ab-v2/A-control/evaluation/epoch-0','C20_E5':ROOT/'data/open-images/experiments/phase10c19-recovery-v1/evaluation/epoch-5','Phase10D':RUN/'evaluation'}
    raw={};pred={};measures={};comparison={};logits={}
    previous=read(ROOT/'reports/phase10c20-v1/results.json')
    for name,folder in folders.items():
        a=load_raw(folder,dev);raw[name]=a;p=predictions(a,dev);pred[name]=p
        emit(RUN/'CPU-analysis'/('predictions-'+name+'.json'),p)
        m=metric.evaluate(rows,p);measures[name]=m
        if name!='Phase10D':
            old=previous['comparison'][name]
            if old['headline']!={k:v for k,v in m.items() if k not in ('categories','pr_curve')} or old['categories']!=m['categories']:
                raise ValueError('Saved baseline metric equivalence failed')
        emit(RUN/'CPU-analysis'/('metrics-'+name+'.json'),m)
        z=np.concatenate([v['logits'] for v in a]).astype(np.float64);logits[name]=z;s=1/(1+np.exp(-z))
        comparison[name]=dict(headline={k:v for k,v in m.items() if k not in ('categories','pr_curve')},categories=m['categories'],precision_at_original_recall=envelope(m,previous['comparison']['original']['headline']['recall']),raw_query_confidence={c:quantiles(s[:,j]) for j,c in enumerate(CLASSES)},raw_Chair_at_least={str(t):int((s[:,1]>=t).sum()) for t in (.15,.5,.8)},scored_FP_confidence={c:quantiles([e['score'] for e in metric.events(rows,p,c,floor=.15)[0] if e['outcome']=='fp']) for c in CLASSES})
        print('CPU METRICS',name,'P/R',m['precision'],m['recall'],'TP/FP/FN',m['tp'],m['fp'],m['fn'],flush=True)
    localization={}
    for c in ('sofa','chair','bed'):
        aa={(e['image_id'],e['gt_index']):e['iou'] for e in metric.events(rows,pred['original'],c,floor=.15)[0] if e['outcome']=='tp'}
        bb={(e['image_id'],e['gt_index']):e['iou'] for e in metric.events(rows,pred['Phase10D'],c,floor=.15)[0] if e['outcome']=='tp'}
        shared=sorted(set(aa)&set(bb))
        localization[c]=dict(common_matched_GT=len(shared),original_only_matches=len(set(aa)-set(bb)),adapted_only_matches=len(set(bb)-set(aa)),mean_original_IoU=float(np.mean([aa[k] for k in shared])) if shared else None,mean_adapted_IoU=float(np.mean([bb[k] for k in shared])) if shared else None,mean_delta_IoU=float(np.mean([bb[k]-aa[k] for k in shared])) if shared else None,IoU75_crossings_up=sum(bb[k]>=.75>aa[k] for k in shared),IoU75_crossings_down=sum(aa[k]>=.75>bb[k] for k in shared))
    drift={}
    for name in ('C20_E5','Phase10D'):
        d=logits[name]-logits['original']
        drift[name]=dict(raw_logits_bit_identical=bool(np.array_equal(logits[name],logits['original'])),mean_all_categories=float(d.mean()),maximum_absolute_difference=float(np.abs(d).max()),per_category_mean_delta={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},per_category_difference_quantiles={c:quantiles(d[:,j]) for j,c in enumerate(CLASSES)})
    checks=read(RUN/'logit-checks/complete.json')
    decision=success_gates(measures['original'],measures['Phase10D'],localization)
    if not checks['all_original_logits_bit_identical']:raise ValueError('Actual before/after original classification check failed')
    epochs=[read(RUN/('epoch-'+str(i)+'.json')) for i in range(1,6)]
    result=dict(status='PASS',comparison=comparison,common_GT_localization=localization,logit_drift=drift,fixed_training_probe_FP32=checks,decision=decision,training_epochs=epochs,resources=done,protocol=dict(photos=19,positive_boxes=61,classes=list(CLASSES),supervised_and_scored_positive_classes=['sofa','chair','bed'],confidence=.15,IoU=.5,AP75_IoU=.75,NMS=.5,cap=300,score_floor=.001,AP='Unchanged101pointmacro',rows_sha256=digest(dev),metric_sha256=sha(ROOT/'tools/phase10c-v1/metric_cpu.py'),unknown_unmatched_not_confirmed_errors=True,independent_or_reserved_scored=False,hyperparameters_selected_from_panel=False),metrics_process=dict(interpreter=sys.executable,version=sys.version,Torch_imported=False,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),no_new_inference=True)
    emit(REPORT/'results.json',result);emit(REPORT/'decision.json',decision)
    emit(REPORT/'precision-recall.json',{k:m['pr_curve'] for k,m in measures.items()})
    records=[]
    for name,v in comparison.items():
        for c,m in v['categories'].items():records.append(dict(model=name,category=c,**m))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records);atomic(REPORT/'category-metrics.csv',stream.getvalue().encode())
    emit(RUN/'CPU-analysis/complete.json',dict(status='PASS',baseline_metrics_bit_identical=True,models=list(comparison),new_model_forwards=0,Torch_imported='torch' in sys.modules))
    print('Phase10D isolated CPU analysis complete;no follow-on training',flush=True)


if __name__=='__main__':run()
