"""Strict durable optimizer-call accounting for exactly the two approved fits."""
import json
import os
import time
from pathlib import Path
from common import *
from checkpoints import Lease
RUN_ROOT=ROOT/'data/open-images/experiments/phase5c-v1'
FIT_IDS=('heads-lr1e-5','heads-lr3e-5')

def atomic(path,value):
 path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def ledger_init(root=RUN_ROOT):
 root=Path(root);path=root/'update-budget.json'
 with Lease(root/'budget.lock'):
  if not path.exists():atomic(path,dict(max_per_fit=70,max_total=140,fits={k:dict(reserved_calls=0,applied_calls=0,events=[]) for k in FIT_IDS}))
 return path

def check_resume(fit_id,checkpoint_step,root=RUN_ROOT):
 data=read(Path(root)/'update-budget.json');rec=data['fits'][fit_id]
 if rec['reserved_calls']!=checkpoint_step or rec['applied_calls']!=checkpoint_step:
  raise RuntimeError('Optimizer/checkpoint boundary is uncertain: preserve state; do not replay or reclaim budget automatically')
 return data

def reserve(fit_id,step,root=RUN_ROOT):
 root=Path(root)
 with Lease(root/'budget.lock'):
  data=read(root/'update-budget.json')
  if fit_id not in FIT_IDS:raise ValueError('Unapproved third experiment')
  rec=data['fits'][fit_id]
  if rec['reserved_calls']!=step or rec['applied_calls']!=step:raise RuntimeError('Update boundary not committed')
  if rec['reserved_calls']>=70 or sum(r['reserved_calls'] for r in data['fits'].values())>=140:raise RuntimeError('Approved optimizer budget exhausted')
  rec['reserved_calls']+=1;rec['events'].append(dict(ordinal=rec['reserved_calls'],status='reserved_before_optimizer_call',unix_ns=time.time_ns()))
  atomic(root/'update-budget.json',data)

def applied(fit_id,step,root=RUN_ROOT):
 root=Path(root)
 with Lease(root/'budget.lock'):
  data=read(root/'update-budget.json');rec=data['fits'][fit_id]
  if rec['reserved_calls']!=step or rec['applied_calls']!=step-1:raise RuntimeError('Unexpected optimizer accounting')
  rec['applied_calls']=step;rec['events'][-1].update(status='optimizer_applied',completed_ns=time.time_ns());atomic(root/'update-budget.json',data)

def frozen_hash(model):
 import hashlib
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if not p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def assert_scope(model,optimizer):
 named={n:p for n,p in model.named_parameters() if p.requires_grad}
 if not named or any(not n.startswith(('class_head.','box_head.')) for n in named):raise ValueError('Non-head parameter trainable')
 if sum(p.numel() for p in named.values())!=1579526:raise ValueError('Prepared trainable parameter count differs')
 if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in named.values()}:raise ValueError('Optimizer scope violation')
 if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter received gradient')
