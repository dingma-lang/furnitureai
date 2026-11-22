"""Prepared future endpoint scoring:separate CPU/Torch-free process,no inference."""
from commonE import *
import numpy as np
import metric_cpu as metric
sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
from analyzeD import load_raw,predictions,envelope

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:raise PermissionError('CPU-only metric process')
    done=read(FUTURE/'complete.json')
    if done['status']!='PASS' or done['updates']!=75:raise ValueError('Complete approved fixed endpoint required')
    context=read(PREP/'training-context.json');dev=context['development'];rows=[metric.scopes(r) for r in dev]
    if digest(dev)!=context['development_rows_sha256']:raise ValueError('Fixed19-photo diagnostic changed')
    folder=FUTURE/'CPU-analysis'
    if folder.exists():raise FileExistsError('Versioned endpoint metrics already exist')
    folder.mkdir()
    def save(name,data):
        p=folder/name;raw=(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if sum(v.stat().st_blocks*512 for v in FUTURE.rglob('*'))+2*len(raw)>2**31:raise RuntimeError('Future output budget')
        pending=p.with_name('.'+p.name+'.pending')
        with pending.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.link(pending,p)
    raw=load_raw(FUTURE/'predictions',dev);pred=predictions(raw,dev);save('predictions.json',pred)
    measured=metric.evaluate(rows,pred)
    history=read(PREP/'historical-evidence.json');baseline=history['models']['original']['headline']
    original=load_raw(ROOT/'data/open-images/experiments/phase10c8-ab-v2/A-control/evaluation/epoch-0',dev)
    d=np.concatenate([a['logits']-b['logits'] for a,b in zip(raw,original)]).astype(np.float64)
    save('results.json',dict(metrics=measured,precision_at_original_recall=envelope(measured,baseline['recall']),historical_comparisons=history,mean_seven_class_logit_drift=float(d.mean()),per_class_logit_drift={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},independent_acceptance=False,threshold_tuned=False,unknown_unmatched_not_confirmed_FP=True,new_GPU_forwards=0))

if __name__=='__main__':run()
