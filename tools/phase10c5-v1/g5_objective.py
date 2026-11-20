"""CPU-testable, opt-in research components. Not connected to any training runner.

Independent sigmoid labels, never a generic Other softmax. Unknown entries have
zero weight. A positive Person label supplies no negative Chair target.
"""
import torch
from torch import nn
from torch.nn import functional as F

def partial_targets(shape,positive,explicit_negative,device='cpu'):
    target=torch.zeros(shape,device=device);weight=torch.zeros_like(target)
    pos=set(positive);neg=set(explicit_negative)
    if pos&neg:raise ValueError('Contradictory positive/negative supervision')
    for collection,value in ((pos,1.),(neg,0.)):
        for index in collection:
            if len(index)!=len(shape) or any(type(i) is not int or not 0<=i<n for i,n in zip(index,shape)):raise ValueError('Out-of-range supervision')
            target[index]=value;weight[index]=1.
    return target,weight

def masked_focal(logits,target,weight,alpha=.25,gamma=2.):
    if logits.shape!=target.shape or logits.shape!=weight.shape:raise ValueError('Shape mismatch')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all() or not torch.isfinite(weight).all():raise ValueError('Nonfinite loss input')
    if not ((target==0)|(target==1)).all() or not ((weight==0)|(weight==1)).all():raise ValueError('Binary targets/masks required')
    prob=logits.sigmoid();pt=prob*target+(1-prob)*(1-target);at=alpha*target+(1-alpha)*(1-target)
    # The real legacy matcher normalizes by positive GT. Do not silently replace
    # that definition: this isolated auxiliary objective is a separate ablation.
    return (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(gamma)*at*weight).sum()/weight.sum().clamp_min(1)

class FurnitureResidual(nn.Module):
    """Optional future architecture ablation, not the primary loss A/B.

    Frozen original features/logits are detached. Only seven explicitly named
    query slots receive a zero-initialized residual. Every other prompt stays
    bit-identical. Requires pinned prompt-slot mapping at call site.
    """
    def __init__(self,dimension=768,hidden=128):
        super().__init__();self.adapter=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Linear(hidden,7))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,features,original_logits,furniture_slots):
        if len(furniture_slots)!=7 or len(set(furniture_slots))!=7 or any(type(i) is not int or not 0<=i<original_logits.shape[-1] for i in furniture_slots):raise ValueError('Exactly seven distinct furniture query slots required')
        if features.shape[:-1]!=original_logits.shape[:-1]:raise ValueError('Feature/query alignment changed')
        out=original_logits.detach().clone();out[...,list(furniture_slots)]+=self.adapter(features.detach());return out
