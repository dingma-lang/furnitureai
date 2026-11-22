"""Exactly one C17 zero-update GPU smoke;immutable bounded diagnostics."""
import sys,os,json,time,uuid,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c16-v1'))
from common16 import *
HERE17=Path(__file__).resolve().parent
RUN17=ROOT/'data/open-images/experiments/phase10c16-balanced-gradient-smoke-v1'
REPORT17=ROOT/'reports/phase10c17-v1'
PARENT17='8f4987348765e4cbd4a3dc6efca9f37598d55bf3'
BRANCH17='experiment/phase10c17-balanced-gradient-smoke'
MAX_OUTPUT17=128*2**20
MAX_VRAM17=int(6.5*2**30)
WRITE_LOCK17=threading.RLock()


def allocation17():
    return sum(p.stat().st_blocks*512 for folder in (RUN17,REPORT17) if folder.exists() for p in folder.rglob('*'))


def atomic17(path,data):
    with WRITE_LOCK17:
        path=Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (RUN17,REPORT17)):raise PermissionError('Outside approved smoke output roots')
        if path.exists():raise FileExistsError('Immutable smoke output exists')
        if allocation17()+2*((len(data)+4095)//4096)*4096+8192>MAX_OUTPUT17:raise RuntimeError('128MiB output allocation limit')
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path) # Retain all pending links;no historical cleanup.


def emit17(path,value):atomic17(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def inputs17():
    cfg=read(RUN17/'authorized-config.json')
    if cfg.get('smoke_authorized') is not True or cfg.get('training_authorized') is not False or cfg['limits']!=dict(forwards=59,gradient_calculations=295,seconds=180,VRAM_bytes=MAX_VRAM17,output_bytes=MAX_OUTPUT17,optimizer_updates=0):raise PermissionError('Specific one-smoke authorization required')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned smoke input/code changed: '+rel)
    prepared,train,dev,_,registry=training_context()
    from common8 import grouped_epoch
    groups=grouped_epoch(train,0)
    if [[r['image_id'] for r in g] for g in groups]!=cfg['group_order'] or [len(g) for g in groups]!=[4]*14+[3]:raise ValueError('Original C8 accumulation sequence changed')
    for group in groups:
        for row in group:require_training_row(row,train,dev)
    return cfg,prepared,groups,registry
