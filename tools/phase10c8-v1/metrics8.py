"""Separate CPU metric process using unchanged, equivalence-tested recovery code."""
import os
from common8 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU metrics must hide CUDA')
    import importlib.util,numpy as np
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));spec=importlib.util.spec_from_file_location('ab8_metric_cpu',ROOT/'tools/phase10c-v1/metric_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    prepared=verify_binding(read(RUN/'binding.json'));_,dev=subsets(prepared);scoped=[m.scopes(r) for r in dev];result=[]
    jobs=[(ARMS[0],0)]+[(a,e) for a in ARMS for e in EVAL_EPOCHS]
    for arm,epoch in jobs:
        folder=RUN/arm/'evaluation'/('epoch-'+str(epoch));done=read(folder/'raw-complete.json');predictions={};raw_chair=[]
        if done['rows_sha256']!=digest(dev) or done['FP32'] is not True:raise ValueError('Evaluation manifest/precision mismatch')
        for row,rec in zip(dev,done['records']):
            if rec['image_id']!=row['image_id'] or rec['source_sha256']!=row['sha256'] or sha(folder/rec['file'])!=rec['sha256']:raise ValueError('Raw prediction provenance mismatch')
            a=load_npz(folder/rec['file']);ps=[]
            for box,s,c in zip(a['boxes'].tolist(),a['scores'].tolist(),a['labels'].tolist()):
                if s>=.001 and box[2]>box[0] and box[3]>box[1]:ps.append(dict(box=box,score=s,label=CLASSES[c]))
            predictions[row['image_id']]=m.nms(ps,.5,300);z=a['logits'][:,1].astype(np.float64);raw_chair.extend((1/(1+np.exp(-z))).tolist())
        # All postprocessed predictions are atomically complete BEFORE metrics.
        save(folder/'predictions.json',predictions)
        metrics=m.evaluate(scoped,predictions);ev,_,_=m.events(scoped,predictions,'chair',floor=.15);per_photo=[]
        for row in dev:
            es=[e for e in ev if e['image_id']==row['image_id']];per_photo.append(dict(image_id=row['image_id'],Chair_tp=sum(e['outcome']=='tp' for e in es),Chair_fp=sum(e['outcome']=='fp' for e in es)))
        scores=[p['score'] for ps in predictions.values() for p in ps if p['label']=='chair'];scorestats=dict(raw_Chair_quantiles={str(q):float(np.quantile(raw_chair,q)) for q in (0,.5,.9,.99,1)},post_NMS_Chair_quantiles={str(q):float(np.quantile(scores,q)) if scores else None for q in (0,.5,.9,.99,1)},post_NMS_Chair_above_015=sum(s>=.15 for s in scores))
        entry=dict(arm=arm,epoch=epoch,updates=done['updates'],metrics=metrics,score_distributions=scorestats,per_photo_Chair=per_photo,predictions_sha256=sha(folder/'predictions.json'))
        save(folder/'metrics-cpu.json',entry);result.append(entry);print('METRICS',arm,epoch,'P',metrics['precision'],'R',metrics['recall'],'FP',metrics['fp'],flush=True)
    save(RUN/'metric-results.json',dict(protocol=read(RUN/'binding.json')['evaluation_protocol'],entries=result))
    print('CPU METRICS COMPLETE;predictions preserved;0GPU or optimizer calls')

if __name__=='__main__':run()
