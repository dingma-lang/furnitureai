"""Immutable adaptation snapshots, commit receipts and exact resume state."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import time
import fcntl
from common import sha

class Lease:
 def __init__(self,path):self.path=Path(path);self.handle=None
 def __enter__(self):
  self.handle=self.path.open('a+')
  try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:self.handle.close();raise
  return self
 def __exit__(self,*args):self.handle.close()

def commit_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path)
 finally:tmp.unlink()

def snapshot(out,state,binding):
 import torch
 folder=Path(out)/'checkpoints';folder.mkdir(exist_ok=True)
 allocated=sum(p.stat().st_blocks*512 for p in [Path(out),*Path(out).rglob('*')])
 estimate=sum(t.numel()*16 for t in state['weights'].values())+2097152
 if allocated+estimate>2147483648:raise RuntimeError('2 GiB per-fit allocated-storage bound')
 tag=f"step-{state['step']:05d}-epoch-{state['epoch']:03d}-"+uuid.uuid4().hex[:12]
 path=folder/(tag+'.pt');tmp=folder/('.'+tag+'.tmp')
 with tmp.open('xb') as f:torch.save(dict(binding=binding,**state),f);f.flush();os.fsync(f.fileno())
 os.link(tmp,path);tmp.unlink()
 sequence=max([json.loads(p.read_text()).get('commit_sequence',0) for p in folder.glob('*.receipt.json')],default=0)+1
 receipt=dict(commit_sequence=sequence,file=path.name,sha256=sha(path),step=state['step'],epoch=state['epoch'],next_group=state['next_group'],commit_ns=time.time_ns())
 commit_json(folder/(tag+'.receipt.json'),receipt)
 return str(path),receipt['sha256']

def recover(out,binding):
 import torch
 candidates=[]
 for path in sorted((Path(out)/'checkpoints').glob('*.receipt.json')):
  r=json.loads(path.read_text());f=path.parent/r['file']
  if not f.resolve().is_relative_to(path.parent.resolve()) or sha(f)!=r['sha256']:raise ValueError('Committed checkpoint corruption')
  state=torch.load(f,map_location='cpu',weights_only=True)
  if state['binding']!=binding:raise ValueError('Checkpoint/dataset/config/source mismatch')
  if (state['step'],state['epoch'],state['next_group'])!=(r['step'],r['epoch'],r['next_group']):raise ValueError('Checkpoint receipt state mismatch')
  candidates.append(((state['step'],state['epoch'],state['next_group'],r['commit_sequence']),state,str(f),r['sha256']))
 if not candidates:return None
 return max(candidates,key=lambda x:x[0])
