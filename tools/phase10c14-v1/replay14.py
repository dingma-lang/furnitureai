"""Replay the executed objective on saved logits/boxes, never a model forward.

Assignments are reconstructed, not claimed to have been saved. GPU/CPU matching
agreement is assessed through stored losses, mask counts and region coverage.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common14 import *
from expanded import supervision,to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou,from_square
from masks6 import target_mask
from loss8 import objective as executed_objective


def replay(arrays,row,selected,protected,corrected,saved=None):
    z=torch.from_numpy(arrays['logits'].copy()).requires_grad_();b=torch.from_numpy(arrays['square_cxcywh'].copy()).requires_grad_()
    total,parts,stats=executed_objective(z,b,row,selected,protected,corrected)
    with torch.no_grad():
        indices=[i for i,a in enumerate(row['annotations']) if not a['ignore']];anns=[row['annotations'][i] for i in indices];n=len(anns)
        gt=to_square(b.new_tensor([a['box'] for a in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);labs=torch.tensor([CLASSES.index(a['label']) for a in anns])
        cost=-z.sigmoid()[:,labs]+5*torch.cdist(b,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(b),gt));ii,jj=linear_sum_assignment(cost.numpy());ii=torch.as_tensor(ii);jj=torch.as_tensor(jj);labels=labs[jj]
        target,legacy=supervision(z,b,row,ii,labels);nt,valid,added=target_mask(z,b,row,ii,labels,selected,protected);used=valid if corrected else legacy
        assignment=np.full(len(z),-1,dtype=np.int16);assignment[ii.numpy()]=np.asarray(indices,dtype=np.int16)[jj.numpy()]
        boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).numpy();matched_iou=(giou(cxcywh_to_xyxy(b[ii]),gt[jj]).diagonal()).numpy()
    gpos=torch.autograd.grad(parts['positive'],z,retain_graph=True)[0];gnative=torch.autograd.grad(parts['legacy_negative'],z,retain_graph=True)[0];gextra=torch.autograd.grad(parts['added'],z,retain_graph=True)[0]
    gclass=torch.autograd.grad(parts['classification'],z,retain_graph=True)[0];gbox=torch.autograd.grad(total,b)[0]
    actual_extra=gextra if corrected else torch.zeros_like(gextra)
    if not torch.allclose(gclass,gpos+gnative+actual_extra,atol=1e-7,rtol=1e-6) or torch.count_nonzero(gclass[~used]) or not torch.isfinite(gbox).all():raise ValueError('Gradient partition/ignored-mask replay failure')
    warnings=[]
    if saved is not None:
        for key in ('classification_loss','positive_classification','legacy_negative_classification','added_Chair_negative_loss','box_loss','L1','GIoU'):
            if not np.isclose(stats[key],saved[key],atol=1e-5,rtol=1e-4):warnings.append('saved_'+key+'_differs')
        for key in ('positive_boxes','legacy_active_terms','used_active_terms','added_active_Chair_terms','region_query_counts'):
            if stats[key]!=saved[key]:warnings.append('saved_'+key+'_differs')
    score=z.detach().sigmoid();positive=used&target.bool();negative=used&~target.bool();cf_bce=.75*score[:,1]*added[:,1]/n
    def mass(g):return [float(x) for x in g.detach().abs().sum(0)]
    def norm(g):return float(g.detach().double().norm())
    gradients=dict(logit_positive_L1_by_category=mass(gpos),logit_native_negative_L1_by_category=mass(gnative),logit_added_negative_L1_by_category=mass(actual_extra),
        logit_positive_L2=norm(gpos),logit_native_negative_L2=norm(gnative),logit_added_negative_L2=norm(actual_extra),logit_classification_L2=norm(gclass),box_output_L2=norm(gbox),uniform_seven_logit_offset_derivative=float(gclass.detach().sum()),uniform_category_offset_derivative=[float(x) for x in gclass.detach().sum(0)],
        counterfactual_source_BCE_logit_gradient_L1=float(cf_bce.sum()),counterfactual_source_BCE_logit_gradient_L2=norm(cf_bce),counterfactual_source_focal_logit_gradient_L1=float(gextra.detach().abs().sum()),counterfactual_source_BCE_loss=float((.75*torch.nn.functional.softplus(z.detach()[:,1])*added[:,1]).sum()/n),
        note='Output gradients reconstructed on cached arrays,not encoder inference or signed head-parameter gradients;BCE is diagnostic only and was never applied')
    result=dict(target=target.detach().numpy().astype(np.uint8),legacy=legacy.numpy().astype(np.uint8),used=used.numpy().astype(np.uint8),added_available=added.numpy().astype(np.uint8),assignment=assignment,
        grad_positive=gpos.detach().numpy(),grad_native=gnative.detach().numpy(),grad_added=actual_extra.detach().numpy(),grad_classification=gclass.detach().numpy(),grad_box=gbox.detach().numpy())
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite reconstructed result')
    summary=dict(source='CPU reconstruction from actual saved FP16-forward arrays',saved_loss_and_mask_checks_passed=not warnings,reconstruction_warnings=warnings,stats=stats,gradient=gradients,
        positive_terms_by_category=positive.sum(0).tolist(),negative_terms_by_category=negative.sum(0).tolist(),ignored_terms_by_category=(~used).sum(0).tolist(),
        matches=[dict(query_index=int(i),annotation_index=int(assignment[i]),label=CLASSES[int(label)],positive_score=float(score[i,label]),matched_GIoU=float(g)) for i,label,g in zip(ii.tolist(),labels.tolist(),matched_iou)],
        conditional_assignment_caveat='Original Hungarian indices were not saved;CPU-reconstructed indices agree with recorded masks/losses within declared tolerance,not proof of bit-identical GPU tie breaking')
    return result,summary,boxes
