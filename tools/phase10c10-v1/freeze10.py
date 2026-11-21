"""Persistent eight-tensor scope; reject legacy re-enabling and optimizer drift."""
import torch
from common10 import *

def head_weights(model):
    params=dict(model.named_parameters())
    if not HEAD_KEYS<=params.keys():raise ValueError('Incompatible OWLv2 head paths')
    return {n:params[n].detach().cpu().clone() for n in params if n in HEAD_KEYS}

def verify_initial_heads(model,reference):
    current=head_weights(model)
    if set(reference)!=HEAD_KEYS or any(not torch.equal(current[n],reference[n]) for n in HEAD_KEYS):raise ValueError('Initialization must be the original OWLv2, not an adapted head')

def configure_once(model):
    model.requires_grad_(False);model.eval()
    params=dict(model.named_parameters())
    if not HEAD_KEYS<=params.keys():raise ValueError('Missing head components')
    for n in TRAIN_KEYS:params[n].requires_grad_(True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=TRAIN_COUNT:raise ValueError('Unexpected trainable parameter count')
    return [n for n,p in model.named_parameters() if p.requires_grad]

class FrozenScope:
    def __init__(self,model,reference):
        verify_initial_heads(model,reference);self.model=model;self.names=configure_once(model)
        self.params=dict(model.named_parameters());self.frozen_names=set(self.params)-TRAIN_KEYS
        self.calibration={n:reference[n].detach().cpu().clone() for n in CALIBRATION_KEYS}
        self.frozen_sha256=named_sha(model,self.frozen_names)
        self.backbone_sha256=named_sha(model,set(self.params)-HEAD_KEYS)
        self.check();self.train_mode()
    def check(self,optimizer=None,hash_frozen=False):
        actual={n for n,p in self.model.named_parameters() if p.requires_grad}
        if actual!=TRAIN_KEYS:raise ValueError('Trainability drift: never call legacy configure_heads after installing C10 freeze')
        if any(self.params[n].grad is not None for n in self.frozen_names):raise ValueError('Frozen gradient must be None, not a zero tensor')
        if any(not torch.equal(self.params[n].detach().cpu(),t) for n,t in self.calibration.items()):raise ValueError('Original calibration values changed')
        if optimizer is not None:
            values=[p for g in optimizer.param_groups for p in g['params']]
            if len(values)!=len(TRAIN_KEYS) or len({id(p) for p in values})!=len(values) or {id(p) for p in values}!={id(self.params[n]) for n in TRAIN_KEYS}:raise ValueError('Optimizer must contain exactly the eight permitted tensors')
            for group in optimizer.param_groups:
                if group['lr']!=1e-5 or group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8:raise ValueError('Optimizer settings differ from C8')
            if any(p not in set(values) for p in optimizer.state):raise ValueError('Optimizer state for an excluded parameter')
        if hash_frozen and named_sha(self.model,self.frozen_names)!=self.frozen_sha256:raise ValueError('Frozen parameter bytes changed')
        return True
    def train_mode(self):
        self.check();self.model.eval();self.model.class_head.train();self.model.box_head.train();self.model.class_head.logit_shift.eval();self.model.class_head.logit_scale.eval();self.check()
    def eval_mode(self):self.check();self.model.eval();self.check()
    def optimizer(self):
        opt=torch.optim.AdamW([{'params':[self.params[n] for n in self.names],'param_names':self.names}],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);self.check(opt);return opt
    def gradients(self,optimizer):
        self.check(optimizer)
        for n in TRAIN_KEYS:
            p=self.params[n]
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite permitted gradient')
        return {n:float(self.params[n].grad.detach().double().norm()) if self.params[n].grad is not None else None for n in sorted(HEAD_KEYS)}

def make_payload(scope,optimizer,updates,binding_sha256,**state):
    scope.check(optimizer,hash_frozen=True)
    if type(updates) is not int or not 0<=updates<=75 or (updates==0 and optimizer.state) or (updates>0 and (len(optimizer.state)!=8 or any(int(s['step'])!=updates for s in optimizer.state.values()))):raise ValueError('Optimizer cursor/state mismatch')
    return dict(schema='phase10c10-full-heads-eight-trainable-v1',weights=head_weights(scope.model),trainable_keys=sorted(TRAIN_KEYS),frozen_calibration_keys=sorted(CALIBRATION_KEYS),
        calibration_hashes={n:tensor_sha(t) for n,t in scope.calibration.items()},frozen_sha256=scope.frozen_sha256,original_backbone_sha256=scope.backbone_sha256,
        optimizer=optimizer.state_dict(),updates=updates,binding_sha256=binding_sha256,**state)

def restore_candidate(scope,payload,optimizer=None,expected_binding=None):
    """Full12-head payload stays compatible with the legacy inference loader.

    Training restore additionally requires original calibration and eight-key
    optimizer ordering; a C8 adapted head is not a valid C10 initialization.
    """
    scope.check()
    if payload.get('schema')!='phase10c10-full-heads-eight-trainable-v1' or payload.get('trainable_keys')!=sorted(TRAIN_KEYS) or set(payload.get('weights',{}))!=HEAD_KEYS:raise ValueError('Incompatible classifier/head checkpoint')
    if type(payload.get('updates')) is not int or not 0<=payload['updates']<=75:raise ValueError('Checkpoint exceeds approved update scope')
    if expected_binding is not None and payload.get('binding_sha256')!=expected_binding:raise ValueError('Checkpoint/config binding mismatch')
    if payload.get('frozen_sha256')!=scope.frozen_sha256 or payload.get('original_backbone_sha256')!=scope.backbone_sha256:raise ValueError('Frozen initialization identity mismatch')
    for n,t in payload['weights'].items():
        if t.shape!=scope.params[n].shape or not torch.isfinite(t).all():raise ValueError('Checkpoint tensor shape/values invalid')
        if n in CALIBRATION_KEYS and not torch.equal(t.cpu(),scope.calibration[n]):raise ValueError('Checkpoint attempts to change frozen calibration')
    if optimizer is not None:
        groups=payload['optimizer']['param_groups']
        if len(groups)!=1 or groups[0].get('param_names')!=scope.names or groups[0]['params']!=list(range(8)):raise ValueError('Optimizer name/order mismatch')
        g=groups[0]
        if g['lr']!=1e-5 or g['weight_decay']!=.01 or tuple(g['betas'])!=(.9,.999) or g['eps']!=1e-8:raise ValueError('Checkpoint optimizer settings changed')
        if any(int(s['step'])!=payload['updates'] for s in payload['optimizer']['state'].values()):raise ValueError('Optimizer update cursor mismatch')
        states=payload['optimizer']['state']
        if (payload['updates']==0 and states) or (payload['updates']>0 and set(states)!=set(range(8))):raise ValueError('Incomplete optimizer state')
        for i,s in states.items():
            for key in ('exp_avg','exp_avg_sq'):
                if s[key].shape!=scope.params[scope.names[i]].shape or not torch.isfinite(s[key]).all():raise ValueError('Optimizer moment shape/finiteness mismatch')
    with torch.no_grad():
        for n in TRAIN_KEYS:scope.params[n].copy_(payload['weights'][n])
    if optimizer is not None:optimizer.load_state_dict(payload['optimizer'])
    scope.check(optimizer,hash_frozen=True)
