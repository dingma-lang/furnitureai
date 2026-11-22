"""One Phase10E.2 diagnostic;never an optimizer or training entry point."""
import io,json,os,sys,time,uuid,threading,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'data/open-images/experiments/phase10e-gradient-smoke-v1'
REPORT=ROOT/'reports/phase10e2-v1'
PARENT='953b909654f44533b9772a448f7b404afa2bb49c'
BRANCH='experiment/phase10e2-distillation-gradient-verification'
LIMITS=dict(updates=0,forwards=28,gradient_VJPs=48,seconds=180,VRAM_bytes=6979321856,output_bytes=134217728)
sys.path.insert(0,str(ROOT/'tools/phase10e1-v1'))
from commonE import read,sha,digest,PREP,BASE,BASE_SHA,CLASSES,PROMPTS
LOCK=threading.RLock()

def allocated():return sum(p.stat().st_blocks*512 for folder in (RUN,REPORT) if folder.exists() for p in [folder,*folder.rglob('*')])
def atomic(path,data):
    path=Path(path)
    with LOCK:
        if not any(path.resolve().is_relative_to(folder) for folder in (RUN,REPORT)):raise PermissionError('Outside E2 output roots')
        if path.exists():raise FileExistsError('No overwrite/retry')
        if allocated()+2*((len(data)+4095)//4096)*4096+16384>LIMITS['output_bytes']:raise RuntimeError('128MiB cumulative new output bound')
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
def emit(path,value):atomic(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def arrays(path,values):
    import numpy as np
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite diagnostics')
    raw=io.BytesIO();np.savez_compressed(raw,**values);atomic(path,raw.getvalue())
def inputs():
    cfg=read(RUN/'binding.json')
    if cfg.get('approved') is not True or cfg.get('limits')!=LIMITS or cfg.get('training_authorized') is not False or cfg.get('mode')!='gradient_smoke':raise PermissionError('Exact zero-update E2 authorization required')
    from configE import require_execution
    require_execution(cfg['E1_approval'],'gradient_smoke')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Bound E2 input changed: '+rel)
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json');byid={e['row']['image_id']:e for e in context['entries']}
    refs={r['image_id']:r for r in index['records']}
    if len(refs)!=59 or len(byid)!=59 or cfg['photo_ids']!=read(ROOT/'reports/phase10e1-v1/gradient-smoke-disabled.json')['photo_ids'] or len(cfg['photo_ids'])!=24:raise ValueError('Exact59 caches/24 ordered optimization photos required')
    for ident in cfg['photo_ids']:
        r=byid[ident]['row']
        if r['historical_partition']!='train' or r['internal_partition']!='optimization':raise PermissionError('No development/reserved images')
    return cfg,context,byid,index,refs

def budget(state,seconds,memory,storage):
    if seconds>=180 or memory>LIMITS['VRAM_bytes'] or storage>LIMITS['output_bytes'] or state['forwards']>28 or state['gradient_calculations']>48:raise RuntimeError('E2 resource/counter boundary')
