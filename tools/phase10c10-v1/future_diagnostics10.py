"""CPU consumer prepared for a future approved run. No image/model forward calls.

Current phase has no FUTURE directory or pre-head activations, so this consumer
cannot produce attribution results now. Never substitutes invented activations.
"""
import numpy as np
from common10 import *
from attribution10 import head_combinations,shapley_components,require_feature_cache

def require_complete():
    if not (FUTURE/'complete.json').exists():raise FileNotFoundError('No approved completed C10 training run; no metrics or attribution can be generated')
    done=read(FUTURE/'complete.json')
    if done['updates']!=75 or done['calibration_unchanged'] is not True:raise ValueError('Incomplete or incompatible future experiment')
    return done

def run():
    done=require_complete();_,_,dev=verify_inputs();m=metric_module();rows=[m.scopes(r) for r in dev];folder=FUTURE/'cpu-diagnostics-v1';folder.mkdir(exist_ok=False);results=[]
    for epoch in (0,1,3,5):
        path=FUTURE/'evaluation'/('epoch-'+str(epoch));raw=read(path/'raw-complete.json');predictions={};logits=[]
        if raw['rows_sha256']!=digest(dev) or raw['FP32'] is not True or len(raw['records'])!=19:raise ValueError('Diagnostic partition/precision mismatch')
        for row,r in zip(dev,raw['records']):
            p=path/r['file']
            if r['image_id']!=row['image_id'] or r['source_sha256']!=row['sha256'] or sha(p)!=r['sha256']:raise ValueError('Raw prediction identity mismatch')
            a=load_npz(p);ps=[];logits.append(a['logits'])
            for q,(b,s,c) in enumerate(zip(a['boxes'].tolist(),a['scores'].tolist(),a['labels'].tolist())):
                if s>=.001 and b[2]>b[0] and b[3]>b[1]:ps.append(dict(box=b,score=s,label=CLASSES[c],query_index=q))
            predictions[row['image_id']]=m.nms(ps,.5,300)
        save(folder/('predictions-E'+str(epoch)+'.json'),predictions) # atomic before metrics
        measured=m.evaluate(rows,predictions);chair={}
        for t in (.15,.5,.8):
            events,n,unknown=m.events(rows,predictions,'chair',floor=t);chair[str(t)]=dict(tp=sum(e['outcome']=='tp' for e in events),scored_fp=sum(e['outcome']=='fp' for e in events),unknown_unmatched=unknown,GT=n)
        ref=next(e for e in read(C8/'metric-results.json')['entries'] if e['arm']=='A-control' and e['epoch']==0)['metrics']['recall'];eligible=[p for p in measured['pr_curve'] if p['recall']>=ref]
        from analyze9 import statistical_mode_change
        original=[];old_dir=C8/'A-control/evaluation/epoch-0'
        for r in read(old_dir/'raw-complete.json')['records']:original.append(load_npz(old_dir/r['file'])['logits'])
        scores=(1/(1+np.exp(-np.concatenate(logits).astype(float)[:,1]))).tolist()
        result=dict(epoch=epoch,metrics=measured,Chair_threshold_counts=chair,Chair_score_quantiles=quantiles(scores),shared_logit_drift=statistical_mode_change(np.concatenate(original),np.concatenate(logits)),precision_at_original_recall=max(eligible,key=lambda p:p['precision']) if eligible else None,independent_acceptance=False)
        save(folder/('metrics-E'+str(epoch)+'.json'),result);results.append(result)
    # CPU functional attribution on ACTUAL cached f/text, validated at every endpoint.
    import torch,zipfile
    require_feature_cache(FUTURE/'prehead-cache/manifest.json');cache=read(FUTURE/'prehead-cache/manifest.json')
    if cache.get('synthetic') is not False or len(cache['records'])!=19 or cache['backbone_sha256']!=done['original_backbone_sha256']:raise ValueError('Actual frozen-feature cache provenance required')
    original=original_heads();heads={};raws={}
    for label,arm in (('A','A-control'),('B','B-corrected')):
        checkpoint=read(C8/arm/'complete.json')['final_checkpoint'];heads[label]=torch.load(C8/checkpoint['file'],map_location='cpu',weights_only=True)['weights'];raws[label]=[load_npz(C8/arm/'evaluation/epoch-5'/r['file'])['logits'] for r in read(C8/arm/'evaluation/epoch-5/raw-complete.json')['records']]
    checkpoint=done['final_checkpoint']
    if sha(FUTURE/checkpoint['file'])!=checkpoint['sha256']:raise ValueError('Future checkpoint changed')
    heads['candidate']=torch.load(FUTURE/checkpoint['file'],map_location='cpu',weights_only=True)['weights'];raws['candidate']=[load_npz(FUTURE/'evaluation/epoch-5'/r['file'])['logits'] for r in read(FUTURE/'evaluation/epoch-5/raw-complete.json')['records']]
    stats={key:[] for key in heads};original_raws=[load_npz(C8/'A-control/evaluation/epoch-0'/r['file'])['logits'] for r in read(C8/'A-control/evaluation/epoch-0/raw-complete.json')['records']]
    for i,(row,r) in enumerate(zip(dev,cache['records'])):
        p=FUTURE/'prehead-cache'/r['file']
        if r['image_id']!=row['image_id'] or r['source_sha256']!=row['sha256'] or sha(p)!=r['sha256'] or p.stat().st_size>12000000:raise ValueError('Feature identity/size mismatch')
        with zipfile.ZipFile(p) as z:
            if len(z.infolist())!=2 or sum(q.file_size for q in z.infolist())>12000000:raise ValueError('Unsafe feature archive')
        with np.load(p,allow_pickle=False) as a:f=torch.from_numpy(a['image_features'].copy());text=torch.from_numpy(a['text_features'].copy())
        for label,state in heads.items():
            combos=head_combinations(f,text,original,state)
            if not np.allclose(combos['000'],original_raws[i],atol=1e-4,rtol=1e-5) or not np.allclose(combos['111'],raws[label][i],atol=1e-4,rtol=1e-5):raise ValueError('CPU head reconstruction differs from saved GPU endpoints; no attribution claim permitted. Predictions/metrics already preserved; no automatic inference retry.')
            parts=shapley_components(combos);stats[label].append(dict(image_id=row['image_id'],mean_contribution={key:float(a.mean()) for key,a in parts.items()},mean_Chair_contribution={key:float(a[:,1].mean()) for key,a in parts.items()},endpoint_max_error=float(np.max(np.abs(combos['111']-raws[label][i])))))
    save(folder/'functional-attribution.json',dict(models=stats,scope='Endpoint-validated functional head counterfactuals conditional on actual cached activations, not causal training dynamics',real_extra_encoder_forwards=0))
    save(folder/'complete.json',dict(metrics=results,parameter_and_gradient_records='GPU worker update/probe records with calibration gradients None and immutable hashes',another_training_run_authorized=False))

if __name__=='__main__':run()
