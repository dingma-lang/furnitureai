"""Explicit proposed training precision; no execution, optimizer or CUDA setup.

Same observed FP16 forward, E3 FP32 classification adjoint, native box backward,
unscaled FP32 combined objective. This opt-in path needs separate training approval.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase10e3-v1'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase10e1-v1'))
from stable3 import observed_logits
from objectiveE import objective


def combined(head,snapshot,boxes,teacher_logits,teacher_boxes,entry,coefficient,*,enabled=False):
    if not enabled:raise PermissionError('Proposed training precision disabled')
    z=observed_logits(head,snapshot,enabled=True)[0]
    return objective(z,boxes.float(),entry['row'],teacher_logits,teacher_boxes,entry,coefficient)


def box_objective(logits,boxes,row):
    """Exact existing matching and5L1+2GIoU, without subtracting classifiers."""
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from objectiveE import to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou,CLASSES
    anns=[a for a in row['annotations'] if not a['ignore']]
    gt=to_square(boxes.new_tensor([a['box'] for a in anns]),row['width'],row['height'])
    cx=xyxy_to_cxcywh(gt);labels=torch.tensor([CLASSES.index(a['label']) for a in anns],device=boxes.device)
    with torch.no_grad():
        cost=-logits.detach().sigmoid()[:,labels]+5*torch.cdist(boxes.detach(),cx,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes.detach()),gt))
        ii,jj=linear_sum_assignment(cost.cpu().numpy())
        ii=torch.as_tensor(ii,device=boxes.device);jj=torch.as_tensor(jj,device=boxes.device)
    return (5*F.l1_loss(boxes[ii],cx[jj],reduction='sum')/len(anns)+
            2*(1-giou(cxcywh_to_xyxy(boxes[ii]),gt[jj]).diagonal()).mean())
