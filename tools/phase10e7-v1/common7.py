"""Exact E7 authorization, bounded immutable outputs; no GPU work on import."""
import hashlib,io,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'data/open-images/experiments/phase10e6-training-query-cache-v1'
REPORT=ROOT/'reports/phase10e7-v1'
PARENT='8bbd17e338c82173629c3e2570858aebe56ac1e4'
CHECKPOINT=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt'
CHECKPOINT_SHA='416620a947a96579bdd20c85e7adeb8b60f65c8983e73d3b3943a58ceabef4f0'
sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'))
from commonE6 import BASE,BASE_SHA,PREP,CLASSES,PROMPTS,read,sha,digest,context,require_train,registered,area,intersection,overlap,iou,semantic
LIMITS=dict(forwards=59,seconds=180,VRAM_bytes=6979321856,output_bytes=134217728)
LOCK=threading.RLock()

def allocated():
    return sum(p.stat().st_blocks*512 for root in (RUN,REPORT,HERE) if root.exists() for p in [root,*root.rglob('*')])

def atomic(path,data):
    path=Path(path)
    with LOCK:
        if path.exists() or not any(path.resolve().is_relative_to(r) for r in (RUN,REPORT)):
            raise PermissionError('New isolated E7 outputs only;no replacement')
        if allocated()+2*((len(data)+4095)//4096)*4096+65536>LIMITS['output_bytes']-4*2**20:
            raise RuntimeError('128MiB diagnostic output bound;4MiB failure reserve')
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def emit(path,value):atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def arrays(path,values):
    import numpy as np
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite raw query arrays')
    out=io.BytesIO();np.savez_compressed(out,**values);atomic(path,out.getvalue())

def consume_forward(state):
    if state['attempted_forwards']>=59:raise RuntimeError('59-forward limit BEFORE detector call')
    state['attempted_forwards']+=1

def budget(state,seconds,memory,storage):
    if seconds>=180 or memory>LIMITS['VRAM_bytes'] or storage>LIMITS['output_bytes'] or not 0<=state['attempted_forwards']<=59 or not 0<=state['actual_forwards']<=state['attempted_forwards']:
        raise RuntimeError('E7 authorization/resource limit')

def inputs():
    cfg=read(RUN/'binding.json')
    if cfg.get('approved') is not True or cfg.get('limits')!=LIMITS or cfg.get('optimizer_updates')!=0 or cfg.get('backward_calls')!=0 or cfg.get('checkpoint_sha256')!=CHECKPOINT_SHA:
        raise PermissionError('Exact inference-only E7 binding required')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned input changed:'+rel)
    c,train,dev=context()
    if cfg['image_ids']!=list(train) or len(train)!=59 or len({r['identity_group'] for r in train.values()})!=56:
        raise ValueError('Exact59optimization/56known-source-group scope changed')
    for ident in cfg['image_ids']:require_train(ident,train)
    return cfg,c,train

def raw_values(logits,square,width,height):
    import numpy as np
    if logits.shape!=(3600,7) or square.shape!=(3600,4) or logits.dtype!=np.float32 or square.dtype!=np.float32 or not np.isfinite(logits).all() or not np.isfinite(square).all():
        raise ValueError('Pinned3600query/seven-class FP32 output layout required')
    norm=registered(square,width,height)
    return dict(query_ids=np.arange(3600,dtype=np.int32),logits=logits,scores=(1/(1+np.exp(-logits.astype(float)))).astype(np.float32),
                square_cxcywh=square,original_xyxy_normalized=norm,original_xyxy_pixels=norm*np.asarray([width,height]*2))
