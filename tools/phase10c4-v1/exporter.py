"""Versioned annotation import/export and CPU-only supervision-mask preflight."""
from collections import defaultdict
from datetime import datetime,timezone
import uuid
from core import *

def evidence_records(result,all_regions,reg):
    photos={p['key']:p for p in reg['photos']};byphoto=defaultdict(list)
    for r in result['accepted']:byphoto[r['photo_key']].append(r)
    out=[]
    for key,rects in byphoto.items():
        p=photos[key]
        # One record per rectangle preserves the individual human confirmations;
        # fold only identical photo scope into a combined immutable evidence row.
        reviewers=sorted({r['reviewer'] for r in rects});dates=sorted({r['review_date'] for r in rects})
        record={k:p[k] for k in ('image_id','sha256','identity_group','native_annotation_digest')}
        record.update(historical_partition='train',internal_partition='optimization',category='chair',kind='negative_region',
            native_category={'OpenImages':'/m/01mzpv','COCO':62,'LVIS':232}[next(a['native_dataset'] for a in p['annotations'])],
            regions=[r['box_normalized'] for r in rects],
            potential_unannotated_chair_regions=[normalized(r['box_pixels'],p['width'],p['height']) for r in all_regions if r['photo_key']==key and r['status']!='rejected' and (r['kind']=='potential_positive' or r['status']=='uncertain')],
            taxonomy_sha256=digest(TAXONOMY),review_covers_seating_stools=True,human_reviewed=True,confirmed=True,
            trust_basis='human_category_scope_review',reviewer='; '.join(reviewers),review_date=dates[-1],
            evidence=' | '.join(r['region_id']+': '+r['evidence'] for r in rects),
            individual_confirmation_ids=[r['region_id'] for r in rects],individual_confirmations=rects)
        sys.path.insert(0,str(ROOT/'tools/phase10c3-v1'))
        from preflight import validate_region_record
        row=next(r for r in manifest()['rows'] if r['image_id']==p['image_id'])
        validate_region_record(record,row);out.append(record)
    return out

def mask_comparison(evidence):
    """CPU coordinate fixtures, never real-photo feature/model inference."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'tools/phase10c2-v1'))
    from trusted_loss import negative_mask
    from expanded import supervision,to_square,xyxy_to_cxcywh
    rows=[r for r in manifest()['rows'] if optimizer_member(r)];total=0;records=[]
    for row in rows:
        ev=[e for e in evidence if e['image_id']==row['image_id']]
        # A dense grid plus one strictly internal query/accepted region exercises
        # possible mask effects without claiming real model query coverage.
        ys,xs=torch.meshgrid((torch.arange(30)+.5)/30,(torch.arange(30)+.5)/30,indexing='ij')
        boxes=torch.stack([xs.flatten(),ys.flatten(),torch.full((900,),1/60),torch.full((900,),1/60)],-1)
        anns=[a for a in row['annotations'] if not a['ignore']]
        pos=xyxy_to_cxcywh(to_square(torch.tensor([a['box'] for a in anns]),row['width'],row['height']))
        pieces=[pos,boxes]
        for e in ev:
            for b in e['regions']:
                x1,y1,x2,y2=b;cx=(x1+x2)/2;cy=(y1+y2)/2
                b2=[cx-(x2-x1)/4,cy-(y2-y1)/4,cx+(x2-x1)/4,cy+(y2-y1)/4]
                pieces.append(xyxy_to_cxcywh(to_square(torch.tensor([b2]),row['width'],row['height'])))
        b=torch.cat(pieces);logits=torch.zeros(len(b),7);ii=torch.arange(len(anns));labels=torch.tensor([('sofa','chair','table','bed','desk','cabinet','lamp').index(a['label']) for a in anns])
        ta,ma=supervision(logits,b,row,ii,labels);tb,mb,added=negative_mask(logits,b,row,ii,labels,ev)
        if not torch.equal(ta,tb) or not torch.equal(ma[ii,labels],mb[ii,labels]):raise ValueError('Positive labels/masks changed')
        changed=int((ma!=mb).sum());total+=changed
        records.append(dict(image_id=row['image_id'],changed_mask_terms=changed,added_Chair_negative_terms=int(added.sum()),positive_targets_preserved=True))
    return dict(scope='CPU synthetic grid and internal coordinate probes only; not actual model predictions',changed_mask_terms=total,
        meaningful_fixture_contrast=total>0,records=records,GPU_forwards=0,optimizer_updates=0,actual_model_query_contrast_requires_future_authorized_smoke=True)

def save_export(payload,reg,destination=None):
    result=validate(payload,reg)
    if result['blocked']:raise ValueError('Resolve/reject blocked confirmed rectangles before export: '+str(result['blocked']))
    for key in {r['photo_key'] for r in payload.get('regions',[])}:
        photo_path(next(p for p in reg['photos'] if p['key']==key))
    ev=evidence_records(result,payload.get('regions',[]),reg)
    root=RAW/'exports'
    destination=destination or root/('review-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10])
    destination=Path(destination)
    if destination.exists() or not destination.resolve().is_relative_to(root.resolve()):raise FileExistsError('New isolated export version required')
    destination.mkdir(parents=True)
    save(destination/'human-regions.json',payload);save(destination/'validation.json',result);save(destination/'trusted-evidence-proposed.json',dict(records=ev,training_authorized=False))
    comparison=mask_comparison(ev);save(destination/'mask-comparison.json',comparison)
    source=manifest()
    save(destination/'supervision-manifest.json',dict(schema='phase10c4-additive-negative-supervision-v1',
        source_manifest_sha256=reg['source_manifest_sha256'],registry_sha256=digest(reg),taxonomy_sha256=digest(TAXONOMY),
        existing_optimization_rows=[dict(image_id=r['image_id'],sha256=r['sha256'],identity_group=r['identity_group'],native_annotation_digest=digest(r['annotations'])) for r in source['rows'] if optimizer_member(r)],
        accepted_negative_evidence=ev,rest_of_images_ignored_for_new_negatives=True,source_annotations_modified=False,
        annotation_gate_met=result['minimum_region_group_gate'],CPU_fixture_mask_difference=comparison['changed_mask_terms'],
        training_authorized=False,independent_evaluation_and_GPU_approval_still_required=True))
    with (destination/'SHA256SUMS').open('x') as f:
        for p in sorted(destination.iterdir()):
            if p.is_file() and p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
    return dict(export_id=destination.name,download='human-regions.json',validation={k:v for k,v in result.items() if k!='accepted'},
        mask_changed_terms=comparison['changed_mask_terms'],training_enabled=False)
