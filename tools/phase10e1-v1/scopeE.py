"""C11-comparable eight-tensor scope,with buffers and derived tensor protection."""
import sys
import torch
from commonE import ROOT
sys.path.insert(0,str(ROOT/'tools/phase10c10-v1'))
from freeze10 import FrozenScope,make_payload,restore_candidate,head_weights
from common10 import TRAIN_KEYS,TRAIN_COUNT,CALIBRATION_KEYS,HEAD_KEYS,original_heads
sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
from scopeD import unregistered_tensors
from telemetry19 import tensor_hash

def frozen_state(model):
    import hashlib
    h=hashlib.sha256()
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers()),('derived',unregistered_tensors(model).items())):
        for name,value in sorted(items):
            if kind!='parameter' or name not in TRAIN_KEYS:h.update(kind.encode());h.update(name.encode());h.update(tensor_hash(value).encode())
    return h.hexdigest()

class StudentScope(FrozenScope):
    def __init__(self,model):
        super().__init__(model,original_heads())
        self.immutable_state=frozen_state(model)
    def check(self,optimizer=None,hash_frozen=False):
        super().check(optimizer,hash_frozen)
        if hasattr(self,'immutable_state') and hash_frozen and frozen_state(self.model)!=self.immutable_state:raise ValueError('Frozen representations/calibration/buffers/grid changed')
        allowed=('class_head','box_head')
        if any(module.training for n,module in self.model.named_modules() if not any(n==p or n.startswith(p+'.') for p in allowed)):raise ValueError('Frozen modules must remain eval')
        if self.model.class_head.logit_shift.training or self.model.class_head.logit_scale.training:raise ValueError('Frozen calibration must remain eval')
        return True
    def payload(self,optimizer,updates,binding,**state):
        self.check(optimizer,hash_frozen=True)
        return make_payload(self,optimizer,updates,binding,immutable_state=self.immutable_state,**state)
    def restore(self,payload,optimizer=None,binding=None):
        if payload.get('immutable_state')!=self.immutable_state:raise ValueError('Checkpoint frozen state mismatch')
        restore_candidate(self,payload,optimizer,binding);self.check(optimizer,hash_frozen=True)
