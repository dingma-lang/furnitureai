"""Detached Chair-only instrumentation; returns the ORIGINAL C8 loss/parts.

No altered targets, weights, gradients or optimization decisions. The same
deterministic matching is repeated only to partition existing loss for logging.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from common11 import *
from loss8 import objective as original_objective
from expanded import to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou,supervision
from masks6 import target_mask

def objective(logits,boxes,row,selected,protected,corrected):
    total,parts,stats=original_objective(logits,boxes,row,selected,protected,corrected)
    with torch.no_grad():
        z=logits.detach().float();b=boxes.detach().float();anns=[a for a in row['annotations'] if not a['ignore']];n=len(anns)
        gt=to_square(b.new_tensor([a['box'] for a in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);labs=torch.tensor([list(('sofa','chair','table','bed','desk','cabinet','lamp')).index(a['label']) for a in anns],device=b.device)
        cost=-z.sigmoid()[:,labs]+5*torch.cdist(b,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(b),gt));ii,jj=linear_sum_assignment(cost.cpu().numpy());ii=torch.as_tensor(ii,device=b.device);jj=torch.as_tensor(jj,device=b.device);labels=labs[jj]
        target,legacy=supervision(z,b,row,ii,labels);_,mask,added=target_mask(z,b,row,ii,labels,selected,protected);used=mask if corrected else legacy
        p=z.sigmoid();pt=p*target+(1-p)*(1-target);terms=F.binary_cross_entropy_with_logits(z,target,reduction='none')*(1-pt).pow(2)*(.25*target+.75*(1-target))
        cls=(terms*used).sum()/n
        if not torch.isclose(cls,total.new_tensor(stats['classification_loss']),atol=1e-7,rtol=1e-6) or int(added.sum())!=stats['added_active_Chair_terms']:raise ValueError('Detached diagnostic matching/mask differs; stop before update')
        stats.update(Chair_positive_classification=float((terms[:,1]*used[:,1]*target[:,1]).sum()/n),Chair_negative_classification=float((terms[:,1]*used[:,1]*(1-target[:,1])).sum()/n),
            Chair_legacy_negative_classification=float((terms[:,1]*legacy[:,1]*(1-target[:,1])).sum()/n),Chair_source_negative_classification=float((terms[:,1]*added[:,1]).sum()/n) if corrected else 0.,
            Chair_positive_target_terms=int((used[:,1]&target[:,1].bool()).sum()),Chair_negative_target_terms=int((used[:,1]&~target[:,1].bool()).sum()),
            logging_only_original_loss_returned=True)
    return total,parts,stats

def compare_array_sets(actual,expected,probe=False):
    if set(actual)!=set(expected):raise ValueError('Initial array fields differ')
    result={}
    for key in actual:
        a=actual[key];b=expected[key]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Initial shape/dtype mismatch')
        max_error=float(np.max(np.abs(a.astype(float)-b.astype(float))));exact=bool(np.array_equal(a,b))
        if key=='labels':ok=exact
        elif key=='logits':ok=bool(np.allclose(a,b,atol=.02 if probe else 1e-4,rtol=1e-3 if probe else 1e-5))
        else:ok=bool(np.allclose(a,b,atol=1e-5 if probe else 1e-6,rtol=1e-5))
        if not ok:raise ValueError('Initial original-output mismatch: '+key+' maxAbs='+str(max_error))
        result[key]=dict(bit_identical=exact,max_abs_error=max_error)
    return result

def verify_initial_outputs11():
    records=[]
    # Compare actual already-counted forwards, not a second inference run.
    for index in range(1,25):
        relative=Path('probes/epoch-0')/(str(index).zfill(2)+'.npz');p=FUTURE/relative;old=C8/'B-corrected'/relative
        a=read(p.with_suffix('.json'));b=read(old.with_suffix('.json'))
        if a['image_id']!=b['image_id'] or sha(old)!=b['query_file_sha256']:raise ValueError('Initial probe/source identity mismatch')
        records.append(dict(role='optimization_probe',image_id=a['image_id'],arrays=compare_array_sets(load_npz(p),load_npz(old),True)))
    old_dir=C8/'A-control/evaluation/epoch-0';new_dir=FUTURE/'evaluation/epoch-0';new=read(new_dir/'raw-complete.json');old=read(old_dir/'raw-complete.json')
    if new['rows_sha256']!=old['rows_sha256'] or len(new['records'])!=19 or len(old['records'])!=19:raise ValueError('Initial reference partition differs')
    for a,b in zip(new['records'],old['records']):
        if a['image_id']!=b['image_id'] or a['source_sha256']!=b['source_sha256'] or sha(old_dir/b['file'])!=b['sha256']:raise ValueError('Original FP32 reference identity mismatch')
        records.append(dict(role='internal_diagnostic',image_id=a['image_id'],arrays=compare_array_sets(load_npz(new_dir/a['file']),load_npz(old_dir/b['file']))))
    save(FUTURE/'initial-comparability.json',dict(status='PASS',optimizer_updates=0,probe_photos=24,FP32_reference_photos=19,records=records,
        all_arrays_bit_identical=all(q['bit_identical'] for r in records for q in r['arrays'].values()),extra_model_forwards=0))
