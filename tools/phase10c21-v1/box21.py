"""Isolated pretrained-preserving box-only scope. No training runner or inference."""
import torch
from common21 import ROOT,sha,BASE,HEAD_KEYS,original_heads
sys_path=str(ROOT/'tools/phase10c-v1')
import sys
sys.path.insert(0,sys_path)
import box_only
from telemetry19 import tensor_hash

BOX_KEYS=frozenset(box_only.BOX_KEYS)


def frozen_fingerprint(model):
    import hashlib
    h=hashlib.sha256()
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,t in sorted(items):
            if name not in BOX_KEYS:h.update(kind.encode());h.update(name.encode());h.update(tensor_hash(t).encode())
    return h.hexdigest()


class PretrainedBoxScope:
    """Persistent freeze enforcement;one CPU-tested oldbox loss,never C20 loss."""
    def __init__(self,model):
        self.model=model;box_only.configure(model);reference=original_heads();params=dict(model.named_parameters())
        if any(not torch.equal(params[n].detach().cpu(),reference[n]) for n in HEAD_KEYS-BOX_KEYS):raise ValueError('Original pretrained classification tensors required;rejectC20/adaptedclassifier')
        self.expected_frozen=frozen_fingerprint(model)

    def optimizer(self):
        return box_only.optimizer(self.model,learning_rate=1e-5)

    def check(self,optimizer=None,require_gradients=False,check_hash=False):
        box_only.scope(self.model,optimizer,require_gradients)
        if optimizer is not None:
            if not isinstance(optimizer,torch.optim.AdamW):
                raise ValueError('Prepared optimizer is AdamW')
            for group in optimizer.param_groups:
                if any(group[key]!=expected for key,expected in
                       [('lr',1e-5),('betas',(.9,.999)),('eps',1e-8),('weight_decay',.01)]):
                    raise ValueError('Prepared optimizer settings changed')
        if check_hash and frozen_fingerprint(self.model)!=self.expected_frozen:raise ValueError('Pretrained classifier/backbone/objectness/buffers changed')

    def loss(self,logits,boxes,row):
        self.check();return box_only.loss(logits,boxes,row)

    def checkpoint_payload(self,updates,binding_sha256):
        self.check(check_hash=True)
        if type(updates)is not int or not 0<=updates<=75:raise ValueError('Fixed75-update proposal bound')
        if not valid_hash(binding_sha256):raise ValueError('Exact configuration SHA-256 required')
        return dict(schema='phase10c21-box-only-delta-v1',base_sha256=sha(BASE/'model.safetensors'),binding_sha256=binding_sha256,updates=updates,frozen_sha256=self.expected_frozen,weights={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if n in BOX_KEYS},classifier_adapted=False)


def valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def restore_delta(model,payload,expected_binding_sha256):
    scope=PretrainedBoxScope(model)
    if not valid_hash(expected_binding_sha256) or payload.get('binding_sha256')!=expected_binding_sha256:
        raise ValueError('Wrong or absent experiment binding')
    if type(payload.get('updates')) is not int or not 0<=payload['updates']<=75:
        raise ValueError('Invalid update count')
    if payload.get('schema')!='phase10c21-box-only-delta-v1' or payload.get('base_sha256')!=sha(BASE/'model.safetensors') or set(payload.get('weights',{}))!=BOX_KEYS or payload.get('classifier_adapted') is not False or payload.get('frozen_sha256')!=scope.expected_frozen:raise ValueError('Wrong original base,frozen state or six-box schema')
    params=dict(model.named_parameters())
    for n,v in payload['weights'].items():
        if not isinstance(v,torch.Tensor) or v.shape!=params[n].shape or v.dtype!=params[n].dtype or not torch.isfinite(v).all():raise ValueError('Invalid box delta')
    with torch.no_grad():
        for n,v in payload['weights'].items():params[n].copy_(v.to(params[n].device))
    scope.check(check_hash=True);return scope
