"""Propose source-supported hard negatives from saved optimization queries only.

No new image/model inference. Geometry is necessary, never sufficient for visual
approval. New candidates remain inactive and require separate training approval.
"""
from collections import Counter
import numpy as np
from common9 import *
sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
from geometry6 import polygons,certify
from common6 import registry,candidates

def intersects_any(box,protected):
    return any(min(box[2],p[2])>max(box[0],p[0]) and min(box[3],p[3])>max(box[1],p[1]) for p in protected)

def eligible_rectangle(row,query_box,native,protected,old_regions,train,dev):
    assert_mining_scope(row,train,dev)
    b=np.asarray(query_box,dtype=float);w,h=row['width'],row['height']
    if not np.isfinite(b).all() or b.shape!=(4,) or not(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):return None,'out_of_frame'
    pix=[math.floor(b[0]*w),math.floor(b[1]*h),math.ceil(b[2]*w),math.ceil(b[3]*h)]
    if min(pix[2]-pix[0],pix[3]-pix[1])<32:return None,'below_32px_review_size'
    enclosure=[pix[0]/w,pix[1]/h,pix[2]/w,pix[3]/h]
    if intersects_any(enclosure,protected):return None,'protected_or_uncertain_overlap'
    if intersects_any(enclosure,old_regions):return None,'existing_negative_region_overlap'
    if not native.get('metadata_candidate_eligible') or native['mapped_auxiliary_category']=='protected_furniture' or native.get('source_native_ignored'):return None,'unsupported_or_seating_native_label'
    ab=native['box']
    if not(ab[0]<=enclosure[0] and ab[1]<=enclosure[1] and ab[2]>=enclosure[2] and ab[3]>=enclosure[3]):return None,'not_within_native_box'
    for index,poly in enumerate(polygons(native['original_annotation'],w,h)):
        if certify(pix,poly,margin=2):return dict(box=enclosure,box_pixels=pix,polygon=[list(p) for p in poly],polygon_index=index),'native_polygon_certified_pending_visual'
    return None,'no_full_query_source_polygon_certificate'

def run():
    train,dev=subsets();tr={r['image_id']:r for r in train};inventory=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json');native={p['image_id']:p for p in inventory['photos'] if p['eligible_for_primary_AB']}
    reg={p['image_id']:p for p in registry()['photos'] if p['role']=='optimization'};old=read(SOURCE/'source-negative-evidence.json')['records'];frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
    source_photos=read(SOURCE/'proposed-smoke-photos.json')['photos'];proposals=[];uncertain=[];coverage=[];funnel=Counter();seen=[];hard_counts=Counter();per_native=Counter()
    for index,photo in enumerate(source_photos,1):
        row=tr[photo['image_id']];assert_mining_scope(row,train,dev);source_path(row)
        path=C8/'B-corrected/probes/epoch-5'/(str(index).zfill(2)+'.npz');receipt=read(path.with_suffix('.json'))
        if receipt['image_id']!=row['image_id'] or sha(path)!=receipt['query_file_sha256']:raise ValueError('Saved training-query identity/hash mismatch')
        arrays=load_npz(path);prob=1/(1+np.exp(-arrays['logits'].astype(float)));boxes=registered_box(arrays['square_cxcywh'],row['width'],row['height'],clip=False);top=arrays['logits'].argmax(1)==1
        protected=[a['box'] for a in row['annotations']]+[p['box'] for p in reg[row['image_id']]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==row['image_id'] and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        old_boxes=[e['box'] for e in old if e['image_id']==row['image_id']];hard=[int(q) for q in np.flatnonzero(top&(prob[:,1]>=.5))];funnel['raw_top_Chair_queries_ge05']+=len(hard)
        for t in THRESHOLDS:hard_counts[str(t)]+=int((top&(prob[:,1]>=t)).sum())
        c=dict(image_id=row['image_id'],role='optimization',source_group=row['identity_group'],path=row['path'],sha256=row['sha256'],width=row['width'],height=row['height'],protected=protected,old_regions=old_boxes,
            scored_queries=3600,top_Chair_above_015=int((top&(prob[:,1]>=.15)).sum()),top_Chair_above_05=len(hard),top_Chair_above_08=int((top&(prob[:,1]>=.8)).sum()),hard_queries=[])
        for q in sorted(hard,key=lambda q:(-prob[q,1],q)):
            b=boxes[q].tolist();witness=dict(query_index=q,score=float(prob[q,1]),box=b,source_query_file=str(path.relative_to(ROOT)),source_query_sha256=sha(path))
            # Never infer negatives from generic unmatched high-score queries.
            if not(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):funnel['out_of_frame']+=1;continue
            if intersects_any(b,protected):funnel['protected_or_uncertain_overlap']+=1;continue
            c['hard_queries'].append(witness);funnel['not_overlapping_any_protected_box']+=1
            ann=[]
            for a in native[row['image_id']]['annotations']:
                ab=a['box']
                if ab[0]<=b[0] and ab[1]<=b[1] and ab[2]>=b[2] and ab[3]>=b[3] and a['metadata_candidate_eligible']:ann.append(a)
            found=False
            for a in sorted(ann,key=lambda a:a['annotation_key']):
                region,reason=eligible_rectangle(row,b,a,protected,old_boxes,train,dev)
                if region is None:continue
                if any(s['image_id']==row['image_id'] and intersects_any(region['box'],[s['region']['box']]) for s in seen):continue
                record=dict(candidate_id='HN9-'+digest([row['image_id'],a['annotation_key'],region['box']])[:12],image_id=row['image_id'],path=row['path'],source_photo_sha256=row['sha256'],identity_group=row['identity_group'],historical_partition='train',internal_partition='optimization',
                    width=row['width'],height=row['height'],native=a,region=region,witness=witness,rights=native[row['image_id']]['licence'],
                    evidence_tier='source_polygon_certified_pending_visual',human_reviewed=False,human_confirmed=False,training_enabled=False,training_authorized=False,
                    explanation='Entire integer-enclosed high-score query is inside existing native non-Chair polygon with2px clearance, outside positive/ignored/uncertain and prior negative regions; fresh visual verification still required')
                proposals.append(record);seen.append(record);per_native[a['native_label']]+=1;funnel['new_spatial_certificates_pending_visual']+=1;found=True;break
            if not found:
                uncertain.append(dict(image_id=row['image_id'],identity_group=row['identity_group'],witness=witness,native_containing_boxes=[dict(annotation_key=a['annotation_key'],native_label=a['native_label'],has_instance_mask=a['has_instance_mask']) for a in ann],
                    evidence_tier='unverified_hard_query',training_eligible=False,reason='No new full source-polygon certificate after conservative geometry, prior-region and protection checks'))
        coverage.append(c)
    save(OUT/'hard-negative-proposals.json',dict(candidates=proposals,unverified_queries=uncertain,training_enabled=False,visual_confirmation_required=True,official_or_internal_dev_mined=False))
    save(OUT/'training-coverage.json',dict(photos=coverage,training_eligible_photos=59,saved_query_photos=24,unscored_training_photo_ids=sorted(set(tr)-{p['image_id'] for p in source_photos}),unknown_regions_are_not_negatives=True))
    save(REPORT/'hard-negative-summary.json',dict(metadata_training_photos=59,metadata_native_annotations=sum(len(p['annotations']) for p in native.values()),saved_query_photos=24,training_photos_without_saved_model_queries=35,
        thresholds_top_Chair_query_counts=dict(hard_counts),funnel=dict(funnel),new_spatial_candidates_pending_visual=len(proposals),native_labels=dict(per_native),unverified_unprotected_queries=len(uncertain),genuinely_new_eligible_regions_before_fresh_visual_verification=0,
        existing_source_regions=41,model_forwards=0,training_activated=False,development_or_evaluation_mining=False))
    print('C9 MINING spatial candidates',len(proposals),'unverified',len(uncertain),'hard raw queries',funnel['raw_top_Chair_queries_ge05'],flush=True)

if __name__=='__main__':run()
