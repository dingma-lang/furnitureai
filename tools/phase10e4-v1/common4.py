"""One approved E4 recovery. Imports no torch and never starts a GPU implicitly."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'data/open-images/experiments/phase10e3-recovery-v1'
REPORT = ROOT/'reports/phase10e4-v1'
PARENT = '5dff95fdc9427106f2aa793b9bcb09ac7bf4dcae'
BRANCH = 'experiment/phase10e4-distillation-gradient-recovery'
LIMITS = dict(forwards=25, contractions=25, VJPs=48, seconds=240,
              VRAM_bytes=6979321856, output_bytes=134217728, optimizer_updates=0)
sys.path.insert(0, str(ROOT/'tools/phase10e1-v1'))
from commonE import read, sha, digest, PREP, BINDING, BASE, BASE_SHA, PROMPTS, CLASSES
LOCK = threading.RLock()


def allocated():
    # Conservative: count retained staging hardlinks twice, including CPU logs.
    return sum(p.stat().st_blocks*512 for root in (RUN,REPORT) if root.exists()
               for p in [root,*root.rglob('*')])


def atomic(path, data):
    path = Path(path)
    with LOCK:
        if not any(path.resolve().is_relative_to(r) for r in (RUN,REPORT)):
            raise PermissionError('Outside approved recovery/report roots')
        if path.exists():
            raise FileExistsError('No replacement or automatic retry')
        if len(data) > 4*2**20 or allocated()+2*((len(data)+4095)//4096)*4096+65536 > LIMITS['output_bytes']-8*2**20:
            raise RuntimeError('Global128MiB/new-file4MiB diagnostic bound')
        path.parent.mkdir(parents=True,exist_ok=True)
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def emit(path,value):
    atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def arrays(path,values):
    import numpy as np
    if any(not np.isfinite(a).all() for a in values.values()):
        raise ValueError('Nonfinite raw model output')
    out=io.BytesIO();np.savez_compressed(out,**values);atomic(path,out.getvalue())


def budget(state,elapsed,memory,storage):
    if (elapsed>=240 or memory>LIMITS['VRAM_bytes'] or storage>LIMITS['output_bytes'] or
        any(state.get(k,0)>LIMITS[k] for k in ('forwards','contractions','VJPs')) or
        state.get('optimizer_updates',0)!=0):
        raise RuntimeError('E4 authorization/resource boundary')


def inputs():
    cfg=read(RUN/'binding.json')
    if cfg.get('approved') is not True or cfg.get('limits')!=LIMITS or cfg.get('training_authorized') is not False:
        raise PermissionError('Exact E4 zero-update approval required')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Bound input changed: '+rel)
    proposal=read(ROOT/'reports/phase10e3-v1/recovery-disabled.json')
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json')
    byid={e['row']['image_id']:e for e in context['entries']};refs={r['image_id']:r for r in index['records']}
    if cfg['permitted_photo_ids']!=proposal['adapted_photo_ids'] or not set(cfg['active_photo_ids'])<=set(cfg['permitted_photo_ids']):
        raise PermissionError('Only exact approved optimization evidence')
    if len(byid)!=59 or len(refs)!=59:raise ValueError('59 source/teacher pairs required')
    for ident in cfg['active_photo_ids']+cfg['original_photo_ids']:
        row=byid[ident]['row']
        if row['historical_partition']!='train' or row['internal_partition']!='optimization':
            raise PermissionError('No development or reserved input')
    return cfg,context,byid,index,refs
