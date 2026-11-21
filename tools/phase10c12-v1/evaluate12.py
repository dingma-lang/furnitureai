"""Future CPU consumer: atomic predictions first, unchanged metrics afterward."""
import numpy as np
from common12 import *


def require_complete():
    if not (FUTURE/'complete.json').exists():raise FileNotFoundError('No authorized completed C12 GPU run; no new evaluation now')
    done=read(FUTURE/'complete.json');cfg=read(FUTURE/'authorized-config.json')
    from config12 import require_training_approval
    require_training_approval(cfg,read(FUTURE/'approval-record.json'))
    if done['updates']!=75 or not done['calibration_unchanged'] or done['projection_preservation']['lambda_value']!=cfg['projection_preservation']['lambda_value']:raise ValueError('Incomplete or incompatible C12 run')
    return done


def run():
    done=require_complete();_,_,dev=verify_preparation();m=metric_module();rows=[m.scopes(r) for r in dev]
    folder=FUTURE/'cpu-evaluation-v1';folder.mkdir(exist_ok=False);results=[]
    from analyze9 import threshold_summary,statistical_mode_change
    original_dir=C11_RUN/'evaluation/epoch-0';baseline_dir=C11_RUN/'evaluation/epoch-5'
    def verified_logits(path):
        marker=read(path/'raw-complete.json');arrays=[]
        if marker['rows_sha256']!=digest(dev) or not marker['FP32'] or len(marker['records'])!=19:raise ValueError('Same19 internal photos/FP32 required')
        for row,r in zip(dev,marker['records']):
            if r['image_id']!=row['image_id'] or r['source_sha256']!=row['sha256'] or sha(path/r['file'])!=r['sha256']:raise ValueError('Query identity/hash mismatch')
            arrays.append(load_npz(path/r['file']))
        return arrays
    original=np.concatenate([a['logits'] for a in verified_logits(original_dir)]);baseline=np.concatenate([a['logits'] for a in verified_logits(baseline_dir)])
    historical=read(C11_RUN/'cpu-diagnostics-v1/metrics-E5.json')['metrics'];original_metrics=read(C11_RUN/'cpu-diagnostics-v1/metrics-E0.json')['metrics']
    for epoch in (0,1,3,5):
        arrays=verified_logits(FUTURE/'evaluation'/('epoch-'+str(epoch)));predictions={}
        for row,a in zip(dev,arrays):
            ps=[]
            for q,(b,s,c) in enumerate(zip(a['boxes'].tolist(),a['scores'].tolist(),a['labels'].tolist())):
                if s>=.001 and b[2]>b[0] and b[3]>b[1]:ps.append(dict(box=b,score=s,label=CLASSES[c],query_index=q))
            predictions[row['image_id']]=m.nms(ps,.5,300)
        save(folder/('predictions-E'+str(epoch)+'.json'),predictions)
        measured=m.evaluate(rows,predictions);logits=np.concatenate([a['logits'] for a in arrays]);scores=1/(1+np.exp(-logits.astype(float)))
        def envelope(target):
            ps=[p for p in measured['pr_curve'] if p['recall']>=target]
            return max(ps,key=lambda p:p['precision']) if ps else None
        result=dict(epoch=epoch,metrics=measured,thresholds={str(t):threshold_summary(rows,predictions,m,t) for t in (.15,.5,.8)},shared_logit_drift_vs_original=statistical_mode_change(original,logits),shared_logit_drift_vs_C11=statistical_mode_change(baseline,logits),raw_query_score_quantiles={c:quantiles(scores[:,j]) for j,c in enumerate(CLASSES)},precision_at_original_recall=envelope(original_metrics['recall']),precision_at_C11_recall=envelope(historical['recall']),independent_verified_accuracy=False)
        save(folder/('metrics-E'+str(epoch)+'.json'),result);results.append(result)
    import torch
    p=FUTURE/done['final_checkpoint']['file']
    if sha(p)!=done['final_checkpoint']['sha256']:raise ValueError('Final checkpoint hash mismatch')
    state=torch.load(p,map_location='cpu',weights_only=True);original_state=original_heads()
    drift={n:dict(L2=float((state['weights'][n].double()-original_state[n].double()).norm()),relative_L2=float((state['weights'][n].double()-original_state[n].double()).norm()/original_state[n].double().norm())) for n in PROJECTION_KEYS}
    if any(not torch.equal(state['weights'][n],original_state[n]) for n in CALIBRATION_KEYS):raise ValueError('Frozen calibration changed')
    save(folder/'complete.json',dict(candidate_epochs=results,primary_endpoint=5,baseline_C11=historical,original=original_metrics,projection_drift=drift,losses_and_gradient_decomposition='Immutable per-update training records',no_threshold_or_hyperparameter_selection=True,reserved_or_frozen_scored=False))


if __name__=='__main__':run()
