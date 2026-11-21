"""Legacy loss exactly replayed; only the validated additional Chair mask varies."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from common8 import *
from masks6 import target_mask
from expanded import supervision,to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou

def objective(logits,boxes,row,selected,protected,corrected):
    logits,boxes=logits.float(),boxes.float()
    if logits.ndim!=2 or logits.shape[-1]!=7 or boxes.shape!=(len(logits),4) or not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():raise ValueError('Malformed/nonfinite predictions')
    anns=[a for a in row['annotations'] if not a['ignore']]
    if not anns or any(a['label'] not in ('sofa','chair','bed') for a in anns):raise ValueError('Only unchanged supported native positives')
    gt=to_square(boxes.new_tensor([a['box'] for a in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);labs=torch.tensor([CLASSES.index(a['label']) for a in anns],device=boxes.device)
    with torch.no_grad():
        cost=-logits.sigmoid()[:,labs]+5*torch.cdist(boxes,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gt));ii,jj=linear_sum_assignment(cost.cpu().numpy());ii=torch.as_tensor(ii,device=boxes.device);jj=torch.as_tensor(jj,device=boxes.device);labels=labs[jj]
        target,legacy=supervision(logits,boxes,row,ii,labels);nt,mask,added=target_mask(logits,boxes,row,ii,labels,selected,protected)
        if not torch.equal(target,nt) or (legacy&~mask).any() or added[:,[0,2,3,4,5,6]].any() or added[ii,labels].any():raise ValueError('Positive/legacy/non-Chair mask violation')
    p=logits.sigmoid();pt=p*target+(1-p)*(1-target);terms=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(2)*(.25*target+.75*(1-target));n=len(anns)
    positive=(terms*legacy*target).sum()/n;negative=(terms*legacy*(1-target)).sum()/n;added_loss=(terms*added).sum()/n
    old_cls=(terms*legacy).sum()/n;new_cls=(terms*mask).sum()/n;cls=new_cls if corrected else old_cls
    l1=F.l1_loss(boxes[ii],cx[jj],reduction='sum')/n;g=(1-giou(cxcywh_to_xyxy(boxes[ii]),gt[jj]).diagonal()).mean();box_loss=5*l1+2*g;total=cls+box_loss
    if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
    with torch.no_grad():
        z=logits[added[:,1],1].double();prob=z.sigmoid();added64=(.75*prob.square()*F.softplus(z)).sum()/n
        xy=cxcywh_to_xyxy(boxes.detach());bad=0
        for b in protected:
            q=to_square(boxes.new_tensor(b),row['width'],row['height']);hit=((torch.minimum(xy[:,2:],q[2:])-torch.maximum(xy[:,:2],q[:2])).clamp(min=0).prod(-1)>0)&added[:,1];bad+=int(hit.sum())
        if bad:raise ValueError('Protected source positive received Chair-negative target')
        stats=dict(total_loss=float(total.detach()),classification_loss=float(cls.detach()),counterfactual_legacy_classification=float(old_cls.detach()),counterfactual_corrected_classification=float(new_cls.detach()),
            positive_classification=float(positive.detach()),legacy_negative_classification=float(negative.detach()),added_Chair_negative_loss=float(added_loss.detach()),added_Chair_negative_loss_float64=float(added64),
            added_active_Chair_terms=int(added.sum()),legacy_active_terms=int(legacy.sum()),used_active_terms=int((mask if corrected else legacy).sum()),
            box_loss=float(box_loss.detach()),L1=float(l1.detach()),GIoU=float(g.detach()),positive_boxes=n,protected_violations=bad,
            added_Chair_probability_quantiles=[float(prob.quantile(q)) if len(prob) else None for q in (0,.5,.9,1)],
            added_Chair_logit_quantiles=[float(z.quantile(q)) if len(z) else None for q in (0,.5,.9,1)],
            added_Chair_above_015=int((prob>=.15).sum()),added_Chair_scores=prob.cpu().tolist())
        region_counts={}
        for e in selected:
            q=to_square(boxes.new_tensor(e['box']),row['width'],row['height']);inside=(xy[:,:2]>=q[:2]).all(-1)&(xy[:,2:]<=q[2:]).all(-1)&added[:,1];region_counts[e['candidate_id']]=int(inside.sum())
        stats['region_query_counts']=region_counts
    return total,dict(classification=cls,positive=positive,legacy_negative=negative,added=added_loss),stats

def gradient_norm(grads,scale=1.):
    values=[(g.detach().double()/scale).square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.
