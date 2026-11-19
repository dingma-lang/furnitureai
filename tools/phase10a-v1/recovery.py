"""CPU-only durable call-budget scaffold. Never invokes an optimizer or loads a model."""
import fcntl
import json
import math
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from common import sha, read
from protocol import FIT_IDS

def atomic(path, obj):
    path=Path(path); tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex+'.pending')
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

class Budget:
    def __init__(self, root, binding):
        self.root=Path(root);self.binding=binding;self.path=self.root/'update-budget.json'
    @contextmanager
    def lock(self):
        with (self.root/'budget.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            yield
    def create(self):
        self.root.mkdir(parents=True,exist_ok=True)
        with self.lock():
            if self.path.exists():raise FileExistsError('Preserve and validate existing budget')
            atomic(self.path,dict(binding=self.binding,queue_elapsed_seconds=0,
                fits={name:dict(reserved=0,applied=0,receipt=None) for name in FIT_IDS}))
    def checked(self):
        d=read(self.path)
        if d['binding']!=self.binding or set(d['fits'])!=set(FIT_IDS):raise ValueError('Resume binding/config mismatch')
        return d
    def reserve(self, fit_id, cursor, queue_elapsed_seconds):
        with self.lock():
            d=self.checked();r=d['fits'][fit_id]
            if r['reserved']!=cursor or r['applied']!=cursor:raise RuntimeError('Uncertain optimizer boundary; no replay')
            if r['reserved']>=150 or sum(x['reserved'] for x in d['fits'].values())>=300:raise RuntimeError('Update ceiling')
            if not math.isfinite(queue_elapsed_seconds) or queue_elapsed_seconds<d['queue_elapsed_seconds'] or queue_elapsed_seconds>=3600:
                raise RuntimeError('Cumulative queue time bound')
            d['queue_elapsed_seconds']=queue_elapsed_seconds;r['reserved']+=1
            atomic(self.path,d)
    def committed(self, fit_id, ordinal, checkpoint, queue_elapsed_seconds):
        checkpoint=Path(checkpoint).resolve()
        if not checkpoint.is_relative_to(self.root.resolve()) or not checkpoint.is_file():raise ValueError('Untrusted checkpoint path')
        with self.lock():
            d=self.checked();r=d['fits'][fit_id]
            if r['reserved']!=ordinal or r['applied']!=ordinal-1:raise RuntimeError('Unexpected commit boundary')
            if not math.isfinite(queue_elapsed_seconds) or queue_elapsed_seconds<d['queue_elapsed_seconds']:raise RuntimeError('Time rollback')
            r['receipt']=dict(path=str(checkpoint.relative_to(self.root.resolve())),sha256=sha(checkpoint),cursor=ordinal)
            r['applied']=ordinal;d['queue_elapsed_seconds']=queue_elapsed_seconds;atomic(self.path,d)
    def resume(self, fit_id, checkpoint_cursor):
        with self.lock():
            d=self.checked();r=d['fits'][fit_id]
            if r['reserved']!=checkpoint_cursor or r['applied']!=checkpoint_cursor:raise RuntimeError('Uncertain call boundary; preserve everything')
            if checkpoint_cursor:
                rec=r['receipt'];p=(self.root/rec['path']).resolve()
                if not p.is_relative_to(self.root.resolve()) or rec['cursor']!=checkpoint_cursor or sha(p)!=rec['sha256']:
                    raise ValueError('Committed checkpoint integrity failure')
            return d
