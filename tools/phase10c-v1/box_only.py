"""CPU-testable box-only adaptation, deliberately separate from Phase10B.

No model construction, image loading, optimizer step, checkpoint replacement or
GPU execution occurs in this module. A future authorized worker must use these
scope checks before and after backward and before every optimizer call.
"""
import sys
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from c10 import ROOT, CLASSES

sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from expanded import to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou

BOX_KEYS={f'box_head.{layer}.{suffix}' for layer in ('dense0','dense1','dense2') for suffix in ('weight','bias')}
BOX_PARAMETERS=1_184_260

def configure(model):
    model.requires_grad_(False);model.eval()
    model.box_head.requires_grad_(True);model.box_head.train()
    scope(model)
    return [p for p in model.parameters() if p.requires_grad]

def scope(model,optimizer=None,require_gradients=False):
    train={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(train)!=BOX_KEYS or sum(p.numel() for p in train.values())!=BOX_PARAMETERS:raise ValueError('Exactly six box-head tensors required')
    if any(module.training for name,module in model.named_modules() if name!='box_head' and not name.startswith('box_head.')):raise ValueError('Frozen modules must remain in eval mode')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter acquired gradient')
    if optimizer is not None:
        ps=[p for g in optimizer.param_groups for p in g['params']]
        if len(ps)!=6 or len({id(p) for p in ps})!=6 or {id(p) for p in ps}!={id(p) for p in train.values()}:raise ValueError('Optimizer scope violation')
    if require_gradients:
        for p in train.values():
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite box gradient')

def loss(logits,boxes,row):
    """Same positive Hungarian assignment and 5*L1+2*GIoU; no class loss.

    Classifier logits are detached for matching. Only retained native Sofa,
    Chair and Bed instances are positive targets. Unmatched/ignored objects,
    NEX categories and missing annotations never become negative targets.
    """
    logits=logits.detach().float();boxes=boxes.float()
    if logits.ndim!=2 or logits.shape[1]!=7 or boxes.shape!=(len(logits),4):raise ValueError('Prediction dimensions')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():raise ValueError('Nonfinite detector output')
    if min(row['width'],row['height'])<=0:raise ValueError('Invalid image frame')
    anns=[a for a in row['annotations'] if not a['ignore']]
    if not anns or len(anns)>len(logits):raise ValueError('Require sufficient positive supervision')
    for a in anns:
        b=a['box']
        if a['label'] not in ('sofa','chair','bed') or len(b)!=4 or not all(0<=x<=1 for x in b) or not(b[0]<b[2] and b[1]<b[3]):raise ValueError('Unresolved label/box')
        if row['explicit_native_negative'].get(a['label'],False):raise ValueError('Contradictory positive/negative')
    gt=to_square(boxes.new_tensor([a['box'] for a in anns]),row['width'],row['height']);gc=xyxy_to_cxcywh(gt)
    labels=torch.tensor([CLASSES.index(a['label']) for a in anns],device=boxes.device)
    with torch.no_grad():
        cost=-logits.sigmoid()[:,labels]+5*torch.cdist(boxes,gc,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gt))
        ii,jj=linear_sum_assignment(cost.cpu().numpy());ii=torch.as_tensor(ii,device=boxes.device);jj=torch.as_tensor(jj,device=boxes.device)
    l1=F.l1_loss(boxes[ii],gc[jj],reduction='sum')/len(anns)
    overlap=(1-giou(cxcywh_to_xyxy(boxes[ii]),gt[jj]).diagonal()).mean();total=5*l1+2*overlap
    if not torch.isfinite(total):raise ValueError('Nonfinite box loss')
    return total,dict(l1=float(l1.detach()),giou=float(overlap.detach()),weighted_l1=float(5*l1.detach()),weighted_giou=float(2*overlap.detach()),
        classification=0.,matched=len(anns),unknowns_as_background=False,classifier_frozen=True)

def optimizer(model,learning_rate=1e-5):
    if learning_rate!=1e-5:raise ValueError('Prepared one-configuration LR bound')
    opt=torch.optim.AdamW(configure(model),lr=learning_rate,weight_decay=.01)
    scope(model,opt);return opt
