"""Immutable snapshots, exclusive queue lease, conservative durable time/update accounting."""
import contextlib
import fcntl
import math
import time
from p10 import *
from recovery import atomic

class StopRun(RuntimeError):pass

@contextlib.contextmanager
def lease(path):
    with Path(path).open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

class Ledger:
    def __init__(self,root,binding,clock=time.monotonic):
        self.root=Path(root);self.binding=binding;self.clock=clock;self.path=self.root/'ledger.json'
    def start(self,fit):
        self.root.mkdir(parents=True,exist_ok=True)
        if not self.path.exists():atomic(self.path,dict(binding=self.binding,charged_seconds=0,fits={n:dict(reserved=0,applied=0,charged_seconds=0,receipt=None) for n in ('heads-original-lr1e-5','heads-original-lr3e-5')}))
        self.data=read(self.path)
        if self.data['binding']!=self.binding:raise ValueError('Changed queue binding')
        self.fit=fit;self.started=self.clock();self.session_charged=0;self.tick()
    def tick(self):
        # Precharge30-second slices. Crashes never refund unused slices.
        needed=(math.floor((self.clock()-self.started)/30)+1)*30
        delta=needed-self.session_charged;record=self.data['fits'][self.fit]
        if self.data['charged_seconds']+delta>3600 or record['charged_seconds']+delta>1800:raise StopRun('Cumulative wall-time budget exhausted')
        self.data['charged_seconds']+=delta;record['charged_seconds']+=delta;self.session_charged=needed
        atomic(self.path,self.data)
    def resume(self):
        r=self.data['fits'][self.fit]
        if r['reserved']!=r['applied']:raise RuntimeError('Uncertain optimizer call; preserve and reconcile, never replay')
        if r['receipt']:
            p=(self.root/r['receipt']['path']).resolve()
            if not p.is_relative_to(self.root.resolve()) or sha(p)!=r['receipt']['sha256']:raise ValueError('Committed checkpoint corrupt')
            return p
        if r['applied']:raise ValueError('Missing checkpoint receipt')
    def reserve(self,cursor):
        self.tick();r=self.data['fits'][self.fit]
        if (r['reserved'],r['applied'])!=(cursor,cursor):raise RuntimeError('Uncommitted update boundary')
        if cursor>=150 or sum(x['reserved'] for x in self.data['fits'].values())>=300:raise StopRun('Optimizer cap reached')
        r['reserved']+=1;atomic(self.path,self.data)
    def checkpoint(self,payload):
        import torch
        folder=self.root/self.fit/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        used=sum(p.stat().st_blocks*512 for p in self.root.rglob('*') if p.is_file())
        if used+64*2**20>8*2**30:raise StopRun('8GiB proposed queue storage cap')
        p=folder/(f'update-{payload["cursor"]:04d}-'+uuid.uuid4().hex+'.pt');tmp=p.with_suffix('.pending')
        with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,p)
        finally:tmp.unlink()
        return dict(path=str(p.relative_to(self.root)),sha256=sha(p),cursor=payload['cursor'])
    def committed(self,receipt):
        r=self.data['fits'][self.fit];cursor=receipt['cursor'];p=(self.root/receipt['path']).resolve()
        if not p.is_relative_to(self.root.resolve()) or sha(p)!=receipt['sha256']:raise ValueError('Unsafe/unverified snapshot')
        if r['reserved']!=cursor or r['applied'] not in (cursor-1,cursor):raise RuntimeError('Checkpoint/call mismatch')
        r.update(applied=cursor,receipt=receipt);atomic(self.path,self.data)
