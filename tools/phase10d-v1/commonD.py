"""Phase 10D isolation, immutable output and exact single-arm authorization."""
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'data/open-images/experiments/phase10c21-box-only-v1'
REPORT=ROOT/'reports/phase10d-v1'
BASE=ROOT/'models/owlv2-original-baseline-20261103'
PARENT='d65f6db398fb0e5be45f8aa106d894f11d0846f5'
BRANCH='experiment/phase10d-original-classifier-box-only'
BASE_SHA='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
LIMITS=dict(epochs=5,updates=75,forwards=322,seconds=600,allocated_VRAM_bytes=6979321856,output_bytes=2147483648)
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
from c10 import read,sha
LOCK=threading.RLock()


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def allocated():
    return sum(p.stat().st_blocks*512 for root in (RUN,REPORT) if root.exists() for p in [root,*root.rglob('*')])


def atomic(path,data):
    """Atomic no-replace completion; keep and count pending recovery hard links."""
    path=Path(path)
    with LOCK:
        if not any(path.resolve().is_relative_to(root) for root in (RUN,REPORT)):
            raise PermissionError('Outside Phase10D output roots')
        if path.exists():raise FileExistsError('No historical/output replacement')
        if allocated()+2*((len(data)+4095)//4096)*4096+8192>LIMITS['output_bytes']:
            raise RuntimeError('Phase10D cumulative storage limit')
        pending=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(pending,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def emit(path,value):
    atomic(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def arrays(path,values):
    import numpy as np
    if any(not np.isfinite(a).all() for a in values.values()):raise ValueError('Nonfinite cached predictions')
    stream=io.BytesIO();np.savez_compressed(stream,**values);atomic(path,stream.getvalue())


def context():
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from common21 import context21
    cfg,prepared,train,dev,registry,_,_=context21()
    return cfg,prepared,train,dev,registry


def inputs():
    cfg=read(RUN/'binding.json')
    if cfg.get('training_authorized') is not True or cfg.get('arms')!=1 or cfg.get('limits')!=LIMITS or cfg.get('base_sha256')!=BASE_SHA or cfg.get('learning_rate')!=1e-5 or cfg.get('precision')!='FP32':
        raise PermissionError('Exact one-arm Phase10D approval required')
    for rel,h in cfg['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned Phase10D input changed: '+rel)
    old,prepared,train,dev,reg=context()
    if digest(train)!=cfg['training_rows_sha256'] or digest(dev)!=cfg['development_rows_sha256'] or old['orders']!=cfg['orders']:
        raise ValueError('Training/diagnostic membership or order changed')
    if len(train)!=59 or len(dev)!=19 or len({r['identity_group'] for r in train})!=56 or sum(not a['ignore'] for r in train for a in r['annotations'])!=182:
        raise ValueError('Exact59/56/182 and19-photo scope required')
    for key in ('image_id','sha256','identity_group'):
        if {r[key] for r in train}&{r[key] for r in dev}:raise ValueError('Partition overlap')
    if len(cfg['smoke_ids'])!=4 or len(set(cfg['smoke_ids']))!=4 or not set(cfg['smoke_ids'])<={r['image_id'] for r in train}:
        raise PermissionError('Four optimization-only fixed probes required')
    return cfg,prepared,train,dev,reg


def budget_check(state,elapsed,allocated_VRAM,output):
    if elapsed>=LIMITS['seconds'] or allocated_VRAM>LIMITS['allocated_VRAM_bytes'] or output>LIMITS['output_bytes']:
        raise RuntimeError('Phase10D time/VRAM/storage limit')
    if not 0<=state['forwards']<=322 or not 0<=state['updates']<=75:
        raise RuntimeError('Phase10D forward/update limit')


def consume_forward(state):
    if state['forwards']>=322:raise RuntimeError('Forward cap before execution')
    state['forwards']+=1


def train_mode(scope,opt):
    scope.check(opt)
    scope.model.eval();scope.model.box_head.train()
    scope.check(opt)


def model_state_matches_base(model):
    """All persisted tensors, not just the classifier, must match original."""
    from safetensors import safe_open
    import torch
    with safe_open(str(BASE/'model.safetensors'),framework='pt',device='cpu') as source:
        state=model.state_dict()
        if set(state)!=set(source.keys()):raise ValueError('Actual checkpoint/model state-key mismatch')
        for name,value in state.items():
            if not torch.equal(value.detach().cpu(),source.get_tensor(name)):
                raise ValueError('Non-original initialization: '+name)
    return len(state)
