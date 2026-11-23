"""One E5 authorization;no GPU initialization or execution on import."""
import hashlib,io,json,os,subprocess,sys,threading,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1'
REPORT=ROOT/'reports/phase10e5-v1'
PARENT='59e9bb0fef90b8fd616b837c4190398536804775'
BRANCH='experiment/phase10e5-selective-distillation-training'
COEFFICIENT=7.063459015854303e-6
LIMITS=dict(epochs=5,updates=75,seconds=600,VRAM_bytes=6979321856,output_bytes=2147483648,forwards=373,head_contractions=354)
sys.path.insert(0,str(ROOT/'tools/phase10e1-v1'))
from commonE import PREP,BASE,BASE_SHA,PROMPTS,CLASSES,read,sha,digest
LOCK=threading.RLock()

def allocated():return sum(p.stat().st_blocks*512 for root in (RUN,REPORT) if root.exists() for p in [root,*root.rglob('*')])

def atomic(path,data):
    path=Path(path)
    with LOCK:
        if not any(path.resolve().is_relative_to(r) for r in (RUN,REPORT)) or path.exists():raise PermissionError('Exclusive E5 outputs only')
        if len(data)>128*2**20 or allocated()+2*((len(data)+4095)//4096)*4096+65536>LIMITS['output_bytes']-16*2**20:raise RuntimeError('E5 output bound')
        path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path);fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)

def emit(path,value):atomic(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())

def arrays(path,values):
    import numpy as np
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite output')
    out=io.BytesIO();np.savez_compressed(out,**values);atomic(path,out.getvalue())

def budget(state,seconds,memory,storage):
    if seconds>=600 or memory>LIMITS['VRAM_bytes'] or storage>LIMITS['output_bytes'] or state.get('updates',0)>75 or state.get('forwards',0)>373 or state.get('contractions',0)>354 or state.get('epochs',0)>5:
        raise RuntimeError('E5 authorization/resource boundary')

def inputs():
    cfg=read(RUN/'binding.json')
    if cfg.get('approved') is not True or cfg.get('limits')!=LIMITS or cfg.get('coefficient')!=COEFFICIENT or cfg.get('configurations')!=1:
        raise PermissionError('Exact single E5 training authorization required')
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
    from future4 import gate
    gate(cfg['prepared_proposal_approval'])
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned E5 input changed: '+rel)
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json')
    if len(context['entries'])!=59 or len(context['development'])!=19 or digest(context['development'])!=context['development_rows_sha256']:raise ValueError('Data membership changed')
    return cfg,context,index
