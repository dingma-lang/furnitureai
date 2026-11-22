"""One method: evidence-selective Chair-logit MSE over an unchanged C11 base loss."""
import torch
from scipy.optimize import linear_sum_assignment
from commonE import *
sys.path.insert(0,str(ROOT/'tools/phase10c8-v1'))
from loss8 import objective as base_objective
from expanded import to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou

def valid_outputs(logits,boxes):
    if logits.ndim!=2 or logits.shape[1]!=7 or boxes.shape!=(len(logits),4) or logits.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():raise ValueError('Invalid logit/box schema or nonfinite output')

def positive_assignment(logits,boxes,row):
    anns=[a for a in row['annotations'] if not a['ignore']]
    if not anns or any(a['label'] not in ('sofa','chair','bed') for a in anns):raise ValueError('Only existing182 supported positives')
    with torch.no_grad():
        gt=to_square(boxes.new_tensor([a['box'] for a in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);labels=torch.tensor([CLASSES.index(a['label']) for a in anns],device=boxes.device)
        cost=-logits.detach().sigmoid()[:,labels]+5*torch.cdist(boxes.detach(),cx,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes.detach()),gt))
        ii,jj=linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(ii,device=boxes.device),labels[torch.as_tensor(jj,device=boxes.device)]

def selective_mask(student_logits,student_boxes,teacher_logits,teacher_boxes,row,regions,protected):
    valid_outputs(student_logits,student_boxes);valid_outputs(teacher_logits,teacher_boxes)
    if student_logits.shape!=teacher_logits.shape or student_logits.device!=teacher_logits.device:raise ValueError('Unaligned teacher query layout/device')
    if teacher_logits.requires_grad or teacher_boxes.requires_grad:raise ValueError('Frozen teacher targets required')
    ii,labels=positive_assignment(student_logits,student_boxes,row)
    with torch.no_grad():
        frame=to_square(student_boxes.new_tensor([0.,0.,1.,1.]),row['width'],row['height'])
        sb=cxcywh_to_xyxy(student_boxes.detach());tb=cxcywh_to_xyxy(teacher_boxes.detach())
        safe=torch.ones(len(sb),device=sb.device,dtype=torch.bool)
        for xy in (sb,tb):
            safe&=(xy[:,:2]>=frame[:2]).all(1)&(xy[:,2:]<=frame[2:]).all(1)&(xy[:,2:]>xy[:,:2]).all(1)
            for box in [a['box'] for a in row['annotations']]+list(protected):
                p=to_square(sb.new_tensor(box),row['width'],row['height'])
                safe&=~((torch.minimum(xy[:,2:],p[2:])-torch.maximum(xy[:,:2],p[:2])).clamp(min=0).prod(1)>0)
        safe[ii]=False # Any matched positive query,not only Chair,has priority.
        conflict=teacher_logits[:,1].sigmoid()>=.15
        permitted=torch.zeros_like(safe);owner={};region_counts={}
        for e in sorted(regions,key=lambda e:e['candidate_id']):
            if e['image_id']!=row['image_id'] or e['category']!='chair' or e.get('source_verified_under_policy') is not True:raise ValueError('No new/unverified or non-Chair distillation grants')
            p=to_square(sb.new_tensor(e['box']),row['width'],row['height'])
            contained=safe.clone()
            for xy in (sb,tb):contained&=(xy[:,:2]>=p[:2]).all(1)&(xy[:,2:]<=p[2:]).all(1)
            region_counts[e['candidate_id']]=int((contained&~conflict).sum())
            for q in torch.nonzero(contained&~conflict,as_tuple=False).flatten().tolist():owner.setdefault(str(q),e['candidate_id'])
            permitted|=contained
        mask=torch.zeros_like(student_logits,dtype=torch.bool);mask[:,1]=permitted&~conflict
        if mask[ii].any() or mask[:,[0,2,3,4,5,6]].any():raise ValueError('Positive/non-Chair distillation mask violation')
        chairs=ii[labels==1]
        return mask,dict(terms=int(mask.sum()),teacher_conflicts_excluded=int((permitted&conflict).sum()),matched_positive_queries_excluded=len(ii),teacher_argmax_disagrees_with_positive=int((teacher_logits[ii].argmax(1)!=labels).sum()),teacher_positive_score_below_015=int((teacher_logits[ii,labels].sigmoid()<.15).sum()),matched_Chair_queries=len(chairs),matched_Chair_student_score_below_teacher=int((student_logits[chairs,1].sigmoid()<teacher_logits[chairs,1].sigmoid()).sum()),query_region_owner=owner,region_query_counts=region_counts,other_six_classes_unknown=True)

def distill(student_logits,teacher_logits,mask):
    if mask.dtype!=torch.bool or mask.shape!=student_logits.shape or teacher_logits.shape!=student_logits.shape or teacher_logits.requires_grad:raise ValueError('Invalid frozen selective target mask')
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():raise ValueError('Nonfinite distillation output')
    if mask[:,[0,2,3,4,5,6]].any():raise ValueError('Unrestricted seven-category distillation forbidden')
    delta=student_logits.float()-teacher_logits.detach().float()
    return delta[mask].square().mean() if mask.any() else student_logits.sum()*0

def objective(logits,boxes,row,teacher_logits,teacher_boxes,entry,coefficient):
    if type(coefficient) not in (int,float) or coefficient<0 or not __import__('math').isfinite(coefficient):raise ValueError('Explicit finite fixed coefficient required;None is blocked')
    logits=logits.float();boxes=boxes.float()
    base,parts,base_stats=base_objective(logits,boxes,row,entry['source_regions'],entry['legacy_protected'],True)
    mask,mask_stats=selective_mask(logits,boxes,teacher_logits,teacher_boxes,row,entry['distill_regions'],entry['distill_protected'])
    kd=distill(logits,teacher_logits,mask);total=base+coefficient*kd
    if not torch.isfinite(total):raise ValueError('Nonfinite combined objective')
    return total,dict(base=base,positive=parts['positive'],classification=parts['classification'],distillation=kd),dict(base=base_stats,distillation=float(kd.detach()),coefficient=float(coefficient),mask=mask_stats,logit_diagnostics=logit_diagnostics(logits.detach(),teacher_logits,mask))

def logit_diagnostics(student,teacher,mask):
    with torch.no_grad():
        d=student.double()-teacher.double();ps=student.double().sigmoid();pt=teacher.double().sigmoid()
        return dict(mean_seven_category_logit_drift=float(d.mean()),per_class_logit_drift={c:float(d[:,j].mean()) for j,c in enumerate(CLASSES)},per_class_score_drift={c:float((ps[:,j]-pt[:,j]).mean()) for j,c in enumerate(CLASSES)},query_argmax_agreement=float((student.argmax(1)==teacher.argmax(1)).double().mean()),selected_terms=int(mask.sum()),selected_teacher_negative_scores=ps[mask].cpu().tolist(),selected_Chair_above_015=int((ps[mask]>=.15).sum()),unknown_unmatched_not_confirmed_false_positives=True)
