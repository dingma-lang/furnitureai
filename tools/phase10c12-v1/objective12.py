"""C11/C8 supervision is untouched; optionally add a parameter-only penalty."""
from common12 import *
from observe11 import objective as existing_objective


def objective(logits,boxes,row,selected,protected,corrected,preserver):
    total,parts,stats=existing_objective(logits,boxes,row,selected,protected,corrected)
    preserver.check()
    if not preserver.enabled:return total,parts,stats
    penalty=preserver.penalty();weighted=preserver.strength*penalty;combined=total+weighted
    import torch
    if not torch.isfinite(combined):raise ValueError('Nonfinite regularized objective')
    return combined,dict(parts,projection_preservation=weighted),dict(stats,existing_total_loss=stats['total_loss'],total_loss=float(combined.detach()),projection_penalty=float(penalty.detach()),weighted_projection_penalty=float(weighted.detach()),projection_regularization_enabled=True,logging_only_original_loss_returned=False,existing_supervision_and_box_loss_unchanged=True)
