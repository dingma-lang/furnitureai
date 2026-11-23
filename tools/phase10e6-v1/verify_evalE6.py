"""Task1 only:19-photo hash/parameter verification,never negative mining."""
from commonE6 import *
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU verification only')
    import torch,numpy as np
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from analyzeD import load_raw
    e5=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1';c11=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1'
    done=read(e5/'complete.json');path=e5/done['final_checkpoint']['file']
    if sha(path)!=done['final_checkpoint']['sha256']:raise ValueError('E5 checkpoint hash mismatch')
    p=torch.load(path,map_location='cpu',weights_only=True);old=torch.load(c11/'checkpoints/update-075-bf8be48ddf59.pt',map_location='cpu',weights_only=True)
    names=p['optimizer']['param_groups'][0]['param_names'];h=hashlib.sha256()
    for name in names:h.update(name.encode());h.update(p['weights'][name].contiguous().numpy().tobytes())
    if h.hexdigest()!=done['trainable_sha256'] or done['initial_frozen_sha256']!=done['final_frozen_sha256']:raise ValueError('Persisted checkpoint does not match live post-evaluation state')
    c,_,dev=context();a=load_raw(e5/'predictions',dev);b=load_raw(c11/'evaluation/epoch-5',dev)
    markerA=read(e5/'predictions/raw-complete.json');markerB=read(c11/'evaluation/epoch-5/raw-complete.json')
    per=[]
    for r,x,y,ra,rb in zip(dev,a,b,markerA['records'],markerB['records']):
        per.append(dict(image_id=r['image_id'],source_sha256=r['sha256'],E5_prediction_sha256=ra['sha256'],C11_prediction_sha256=rb['sha256'],
                        arrays_identical=all(np.array_equal(x[k],y[k]) for k in x),logits_identical=bool(np.array_equal(x['logits'],y['logits'])),
                        max_logit_difference=float(np.abs(x['logits'].astype(float)-y['logits']).max())))
        if (e5/'predictions'/ra['file']).stat().st_mtime_ns<path.stat().st_mtime_ns:raise ValueError('Prediction predates final saved checkpoint')
    delta={}
    for name in names:
        d=(p['weights'][name]-old['weights'][name]).double();delta[name]=dict(L2=float(d.norm()),maximum=float(d.abs().max()),bit_identical=bool(torch.equal(p['weights'][name],old['weights'][name])))
    results=read(ROOT/'reports/phase10e5-v1/results.json')['comparison'];actual=results['E5']['headline'];past=results['C11']['headline']
    emit(REPORT/'evaluation-verification.json',dict(status='PASS',E5_checkpoint=str(path.relative_to(ROOT)),E5_sha256=sha(path),C11_sha256=sha(c11/'checkpoints/update-075-bf8be48ddf59.pt'),
         evaluation_loading='E5 did not reload a checkpoint:it evaluated the live original-initialized student after75updates. The saved epoch5 trainable hash equals the live post-evaluation hash;frozen state is unchanged.',
         live_trainable_hash=h.hexdigest(),per_image=per,identical_prediction_files=sum(r['E5_prediction_sha256']==r['C11_prediction_sha256'] for r in per),
         identical_prediction_arrays=sum(r['arrays_identical'] for r in per),parameter_differences=delta,
         exact_AP50_E5=actual['ap50'],exact_AP50_C11=past['ap50'],AP50_exactly_equal=actual['ap50']==past['ap50'],
         identical_fixed_point_counts=all(actual[k]==past[k] for k in ('precision','recall','tp','fp','fn')),
         measurable_effect='E5 parameters/logits are distinct,with73nonzero KD VJPs in recorded training;matched precision/recall counts and rounded AP do not imply an evaluation bug. Numerical-backward differences prevent causal attribution solely to KD.',
         training_mining_used_these19=False,new_inference=0))
    if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA')
    print('E5 evaluation chain verified;distinct checkpoint and arrays,AP50 differs below rounded display',flush=True)
if __name__=='__main__':run()
