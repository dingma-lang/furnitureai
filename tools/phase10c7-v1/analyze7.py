"""Isolated CPU actual-query matching/loss analysis; never run a model."""
import os
from common7 import *

def classification_loss(logits,target,mask,denominator):
    import torch.nn.functional as F
    p=logits.sigmoid();pt=p*target+(1-p)*(1-target)
    terms=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1-pt).pow(2)*(.25*target+.75*(1-target))*mask
    return terms.sum()/max(denominator,1)

def assigned(logits,boxes,row):
    import torch
    from scipy.optimize import linear_sum_assignment
    from expanded import to_square,xyxy_to_cxcywh,cxcywh_to_xyxy,giou
    anns=[a for a in row['annotations'] if not a['ignore']]
    if not anns:return None,None,0
    gt=to_square(torch.tensor([a['box'] for a in anns],dtype=torch.float32),row['width'],row['height']);cx=xyxy_to_cxcywh(gt);native=torch.tensor([CLASSES.index(a['label']) for a in anns])
    cost=-logits.sigmoid()[:,native]+5*torch.cdist(boxes,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gt));ii,jj=linear_sum_assignment(cost.numpy())
    return torch.as_tensor(ii),native[torch.as_tensor(jj)],len(anns)

def responsible_regions(pixel_boxes,regions):
    import numpy as np
    matches=[]
    for b in pixel_boxes:
        matches.append([e['candidate_id'] for e in regions if e['box_pixels'][0]<=float(b[0])<float(b[2])<=e['box_pixels'][2] and e['box_pixels'][1]<=float(b[1])<float(b[3])<=e['box_pixels'][3]])
    return matches

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU analysis must hide CUDA')
    import numpy as np,torch
    torch.set_num_threads(2)
    from masks6 import validate_bundle,target_mask
    from expanded import supervision,detection_loss,to_square,xyxy_to_cxcywh,giou,cxcywh_to_xyxy
    binding=inputs();rows=validate_binding(binding);done=read(RUN/'raw-complete.json')
    if done['forward_calls']!=24 or done['binding_sha256']!=sha(RUN/'binding.json'):raise ValueError('Raw completion binding changed')
    frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,manifest()['rows'],frozen,observations)
    reg={r['image_id']:r for r in registry()['photos'] if r['role']=='optimization'};region_counts={e['candidate_id']:0 for e in bundle['records']};photo_results=[];total=0;overlaps=0;coordinate_errors=0
    analysis=RUN/'cpu-analysis-v1';analysis.mkdir(exist_ok=False)
    for row,receipt in zip(rows,done['records']):
        raw=RUN/receipt['raw_file']
        if receipt['image_id']!=row['image_id'] or receipt['source_sha256']!=row['sha256'] or sha(raw)!=receipt['raw_sha256']:raise ValueError('Raw-query receipt/source mismatch')
        a=load_arrays(raw);pixels,normalized=original_boxes(a['square_cxcywh'],row['width'],row['height'])
        if not np.array_equal(pixels,a['original_xyxy_pixels']) or not np.array_equal(normalized,a['original_xyxy_normalized']):raise ValueError('Raw inverse preprocessing mismatch')
        logits=torch.from_numpy(a['logits']).clone();boxes=torch.from_numpy(a['square_cxcywh']).clone();ii,labels,denom=assigned(logits,boxes,row)
        old_target,old_mask=supervision(logits,boxes,row,ii,labels);regions=[e for e in bundle['records'] if e['image_id']==row['image_id']]
        protected=[p['box'] for p in reg[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        target,mask,added=target_mask(logits,boxes,row,ii,labels,regions,protected)
        if not torch.equal(old_target,target) or (old_mask&~mask).any() or added[:,[0,2,3,4,5,6]].any():raise ValueError('Positive/legacy/non-Chair supervision changed')
        if ii is not None and (not torch.equal(mask[ii,labels],old_mask[ii,labels]) or added[ii,labels].any()):raise ValueError('Matched-positive supervision changed')
        indices=added[:,1].nonzero().flatten().tolist();why=responsible_regions(a['original_xyxy_pixels'][indices],regions);query_records=[];per_region={e['candidate_id']:[] for e in regions};bad=0;boundary=0
        for qi,ids in zip(indices,why):
            if not ids:boundary+=1
            for ident in ids:region_counts[ident]+=1;per_region[ident].append(qi)
            b=a['original_xyxy_normalized'][qi].tolist();hit=any(overlap(b,p) for p in protected)
            if hit or (old_target[qi,1]>0):bad+=1
            query_records.append(dict(query_index=qi,original_xyxy_pixels=a['original_xyxy_pixels'][qi].tolist(),original_xyxy_normalized=b,
                square_cxcywh=a['square_cxcywh'][qi].tolist(),Chair_logit=float(logits[qi,1]),Chair_score=float(logits[qi,1].sigmoid()),responsible_source_regions=ids,
                matched_positive_class=CLASSES[int(labels[(ii==qi).nonzero().flatten()[0]])] if ii is not None and bool((ii==qi).any()) else None))
        old_cls=classification_loss(logits,old_target,old_mask,denom);new_cls=classification_loss(logits,target,mask,denom);delta_terms=classification_loss(logits,target,added,denom)
        if not torch.allclose(new_cls-old_cls,delta_terms,atol=1e-5,rtol=1e-5):raise ValueError('Classification normalization/term addition mismatch')
        legacy_total,_=detection_loss(logits,boxes,row)
        if denom:
            anns=[g for g in row['annotations'] if not g['ignore']];gt=to_square(torch.tensor([g['box'] for g in anns]),row['width'],row['height']);cx=xyxy_to_cxcywh(gt)
            # Recover exact Hungarian GT indices for independent legacy replay.
            from scipy.optimize import linear_sum_assignment
            native=torch.tensor([CLASSES.index(g['label']) for g in anns]);cost=-logits.sigmoid()[:,native]+5*torch.cdist(boxes,cx,p=1)+2*(1-giou(cxcywh_to_xyxy(boxes),gt));ri,gj=linear_sum_assignment(cost.numpy());l1=(boxes[ri]-cx[gj]).abs().sum()/denom;box_giou=(1-giou(cxcywh_to_xyxy(boxes[ri]),gt[gj]).diagonal()).mean()
            if not torch.allclose(legacy_total,old_cls+5*l1+2*box_giou,atol=1e-5,rtol=1e-5):raise ValueError('Legacy classification or matching replay differs')
        result=dict(image_id=row['image_id'],identity_group=row['identity_group'],source_sha256=row['sha256'],queries=len(boxes),additional_Chair_negative_queries=len(indices),
            legacy_classification_loss=float(old_cls),corrected_classification_loss=float(new_cls),classification_loss_delta=float(new_cls-old_cls),normalization_positive_GT_count=max(denom,1),
            region_query_indices=per_region,zero_coverage_regions=[k for k,v in per_region.items() if not v],positive_or_protected_overlap_targets=bad,coordinate_membership_discrepancies=boundary,
            positive_targets_preserved=True,legacy_mask_preserved=True,query_records=query_records,matched_query_indices=ii.tolist() if ii is not None else [],matched_class_labels=labels.tolist() if labels is not None else [],
            added_Chair_mask_query_indices=indices,legacy_valid_class_terms=int(old_mask.sum()),corrected_valid_class_terms=int(mask.sum()))
        save(analysis/(str(receipt['index']).zfill(2)+'.json'),result);photo_results.append(result);total+=len(indices);overlaps+=bad;coordinate_errors+=boundary
    if torch.cuda.is_initialized():raise RuntimeError('CPU analysis initialized CUDA')
    photos_covered=[r for r in photo_results if r['additional_Chair_negative_queries']>0];zero=[k for k,v in region_counts.items() if v==0]
    summary=dict(status='PASS' if not overlaps and not coordinate_errors else 'REVIEW_REQUIRED',decision='A' if total>0 and not overlaps and not coordinate_errors else 'B',
        original_checkpoint_sha256=binding['original_checkpoint_sha256'],photographs=24,source_groups=24,actual_queries_total=sum(r['queries'] for r in photo_results),actual_new_Chair_negative_queries=total,
        photos_with_nonzero_coverage=len(photos_covered),groups_with_nonzero_coverage=len({r['identity_group'] for r in photos_covered}),negative_regions=41,regions_with_coverage=41-len(zero),zero_coverage_regions=zero,
        positive_or_protected_overlap_targets=overlaps,coordinate_membership_discrepancies=coordinate_errors,all_positives_and_legacy_masks_preserved=True,
        mean_legacy_classification_loss=sum(r['legacy_classification_loss'] for r in photo_results)/24,mean_corrected_classification_loss=sum(r['corrected_classification_loss'] for r in photo_results)/24,
        normalization='Unchanged per-image positive-GT count, focal alpha.25 gamma2; classification1 L1=5 GIoU2 unchanged',
        forward_calls=done['forward_calls'],GPU_runtime_seconds=done['seconds'],peak_allocated_bytes=done['peak_allocated_bytes'],peak_reserved_bytes=done['peak_reserved_bytes'],peak_device_used_bytes=done['peak_device_used_bytes'],
        model_state_unchanged=done['initial_model_state_sha256']==done['final_model_state_sha256'],backward_calls=0,optimizer_steps=0,training_authorized=False,
        loss_difference_is_not_performance_improvement=True,validation_or_evaluation_scoring=False,precision=binding['precision'],
        limitations=['Original model only, eval mode,FP16 autocast; different initialization/training may change coverage','Loss uses Hungarian matches, no gradient or optimizer execution','Source-mask and automatic visual evidence remain fallible; no human labels fabricated','Known protected/uncertain regions checked; incompleteness cannot be excluded globally','No independent real-room performance or model selection performed'])
    save(analysis/'region-coverage.json',region_counts);save(analysis/'complete.json',summary)
    save(REPORT/'results.json',summary)
    public=[{k:r[k] for k in ('additional_Chair_negative_queries','legacy_classification_loss','corrected_classification_loss','positive_or_protected_overlap_targets','coordinate_membership_discrepancies')}|dict(index=i+1,source_region_count=len(r['region_query_indices']),zero_coverage_region_count=len(r['zero_coverage_regions'])) for i,r in enumerate(photo_results)]
    save(REPORT/'per-image-summary.json',public);print('ACTUAL ADDED QUERIES',total,'COVERED PHOTOS',len(photos_covered),'COVERED REGIONS',41-len(zero),'ZERO REGIONS',len(zero),'UNSAFE',overlaps,'COORDINATE ISSUES',coordinate_errors,flush=True)

if __name__=='__main__':run()
