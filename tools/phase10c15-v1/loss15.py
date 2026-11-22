"""Same graph/masks/coefficients; BCE replaces only explicit added negatives."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from common15 import CLASSES
from loss8 import objective
from expanded import supervision,to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou
from masks6 import target_mask


def controlled_losses(z,b,row,evidence,protected):
    if z.dtype!=torch.float32 or b.dtype!=torch.float32:raise ValueError('Executed loss uses FP32 logits/boxes')
    total,parts,stats=objective(z,b,row,evidence,protected,True)
    anns=[a for a in row['annotations'] if not a['ignore']];n=len(anns)
    with torch.no_grad():
        gt=to_square(b.new_tensor([a['box'] for a in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);labs=torch.tensor([CLASSES.index(a['label']) for a in anns],device=b.device)
        cost=-z.sigmoid()[:,labs]+5*torch.cdist(b,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(b),gt));ii,jj=linear_sum_assignment(cost.cpu().numpy());ii=torch.as_tensor(ii,device=b.device);jj=torch.as_tensor(jj,device=b.device);labels=labs[jj]
        target,legacy=supervision(z,b,row,ii,labels);nt,mask,added=target_mask(z,b,row,ii,labels,evidence,protected)
        if not torch.equal(target,nt) or (added&target.bool()).any() or (added&legacy).any() or added[:,[0,2,3,4,5,6]].any():raise ValueError('Positive/legacy/Chair-negative supervision mismatch')
    p=z.sigmoid();pt=p*target+(1-p)*(1-target);focal=F.binary_cross_entropy_with_logits(z,target,reduction='none')*(1-pt).square()*(.25*target+.75*(1-target))
    if not torch.equal((focal*mask).sum()/n,parts['classification']):raise ValueError('Reconstructed A objective is not bit-identical')
    source_terms=.75*F.binary_cross_entropy_with_logits(z,torch.zeros_like(z),reduction='none')
    source_bce=(source_terms*added).sum()/n
    class_b=(torch.where(added,source_terms,focal)*mask).sum()/n
    # Keep the executed loss's FP32 operation order,not just algebraic equality.
    l1=F.l1_loss(b[ii],cx[jj],reduction='sum')/n
    box=5*l1+2*(1-giou(cxcywh_to_xyxy(b[ii]),gt[jj]).diagonal()).mean()
    if not torch.equal(total,parts['classification']+box):raise ValueError('Box objective differs from executed loss')
    losses=dict(positive=parts['positive'],native_negative=parts['legacy_negative'],source_focal=parts['added'],source_BCE=source_bce,class_A=parts['classification'],class_B=class_b,box=box)
    if any(not torch.isfinite(x) for x in losses.values()):raise ValueError('Nonfinite diagnostic loss')
    return losses,dict(target=target,legacy=legacy,mask=mask,added=added,matched=ii,labels=labels,matched_annotation=jj),stats


def gradients(loss,z,b,params,retain_graph=True,scale=128.):
    """autograd.grad never accumulates .grad; scale matches historical AMP probe."""
    raw=torch.autograd.grad(loss*scale,(z,b,*params),retain_graph=retain_graph,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in raw):raise ValueError('Nonfinite scaled gradient;no retry')
    return tuple(None if g is None else g.detach().float()/scale for g in raw)
