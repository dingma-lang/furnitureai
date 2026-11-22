"""Record actual assistant inspection of the bounded18-query plates,not human review."""
from common21 import *
from g5_common import overlap
from geometry6 import certify

AMBIGUOUS_FURNITURE={'beanbag','wardrobe','mattress','headboard','chandelier','lampshade','chest of drawers','countertop','crossbar','cushion','tablecloth','hinge','knob'}


def semantic_protected(annotation):
    label=annotation['native_label'].lower().replace('_',' ')
    return annotation['protect_furniture'] or annotation['part_or_semantics_unresolved'] or label in AMBIGUOUS_FURNITURE


def observation_for(record):
    """Findings explicitly authored after viewing query-local1 and2 in this session."""
    label=record['native_label'];ident=record['image_id'];box=record['box_pixels']
    if label=='piano' and ident=='coco2017:000000002270':
        return 'clear_nonchair','The exact red rectangle is on the carved upright piano music-rest/upper wooden panel above the keys. The surrounding image establishes the instrument; no visible Chair or seating stool is inside this panel region. This verifies visible pixels only,not hidden contents or whole-photo completeness.'
    if label=='person' and ident=='coco2017:000000294698':
        detail='long hair and shoulder/clothing at the left image edge' if box[0]<10 else 'lower cheek/neck and long hair' if box[1]<250 else 'hair and the neck/upper-clothing boundary'
        return 'clear_nonchair','The exact red rectangle shows '+detail+'. The local context visibly belongs to the foreground person; no Chair,seating stool or furniture fragment appears within the rectangle. Native person-mask support and protected overlap checks remain mandatory.'
    if label in ('tv','television_set') and ident=='coco2017:000000074638':
        return 'clear_nonchair','The rectangle lies inside the television display and shows a face. No depicted Chair or seating stool is visible inside this exact display region. The inference is limited to these source-supported display pixels,not the television stand or room.'
    if label=='television_set' and ident=='coco2017:000000195507':
        return 'mixed_or_uncertain','The rectangle is a dark reflective television panel. The native TV label alone cannot resolve reflected or obscured content at this scale; keep this region unresolved instead of granting Chair-negative evidence.'
    if label=='handbag' and ident=='coco2017:000000165742':
        return 'clear_nonchair','The rectangle shows the black handbag surface and small metallic hardware. Source instance masks and the nearby bag context support the visible non-Chair object; no visible Chair/seating stool lies inside the exact rectangle. The small32pxminimum makes this a narrow region,not a broad background grant.'
    return 'not_inspected','No actual visual finding authored for this candidate; keep unverified.'


def run():
    records=read(DATA21/'hard-negative-candidates.json')['records'];plates=read(DATA21/'contact-sheets/index.json');source={p['image_id']:p for p in read(DATA21/'full-local-native-inventory.json')['photos']};_,prepared,train,dev,reg,_,_=context21();byid={r['image_id']:r for r in train};plate_by={cid:p for p in plates['pages'] for cid in p['candidate_ids']};observations=[];tiered=[]
    for r in records:
        require_training_row(byid[r['image_id']],train,dev);plate=plate_by[r['candidate_id']]
        if sha(ROOT/plate['file'])!=plate['sha256']:raise ValueError('Actual inspected plate changed')
        finding,evidence=observation_for(r);ann=next(a for a in source[r['image_id']]['annotations'] if a['annotation_key']==r['annotation_key']);poly=[tuple(x) for x in r['native_polygon']]
        extra_protect=[a['box'] for a in source[r['image_id']]['annotations'] if semantic_protected(a)]
        blocked=any(overlap(r['box'],b) for b in extra_protect)
        geometric=certify(r['box_pixels'],poly) and digest(ann['original_annotation'])==r['annotation_sha256'] and not blocked
        observation=dict(candidate_id=r['candidate_id'],candidate_sha256=digest(r),source_sha256=r['source_sha256'],annotation_key=r['annotation_key'],plate=plate['file'],plate_sha256=plate['sha256'],visual_outcome=finding,evidence=evidence,assistant_actual_visual_inspection=finding!='not_inspected',human_reviewed=False,human_confirmed=False,whole_photo_Chair_absence_confirmed=False)
        observations.append(observation);eligible=finding=='clear_nonchair' and geometric
        tier='A' if eligible else 'C' if finding=='mixed_or_uncertain' or blocked else 'B'
        tiered.append(dict(r,evidence_tier=tier,training_eligible=eligible,training_activated=False,assistant_visual_reviewed=finding!='not_inspected',observation_sha256=digest(observation),semantic_overlap_gate=blocked,whole_photo_completeness=False,query_local_source_evidence=True))
    # Do not count cloned/overlapping regions or source annotation aliases as expansion.
    accepted=[];duplicates=[]
    for r in sorted([r for r in tiered if r['evidence_tier']=='A'],key=lambda r:(-max(q['Chair_score'] for q in r['query_observations']),r['candidate_id'])):
        overlap_with=next((x for x in accepted if x['image_id']==r['image_id'] and overlap(x['box'],r['box'])),None)
        if overlap_with:
            r.update(evidence_tier='C',training_eligible=False,duplicate_or_overlap_of=overlap_with['candidate_id'],uncertain_reason='Supported alternate view of same/overlapping visible region;not independently weighted')
            duplicates.append(dict(candidate_id=r['candidate_id'],kept=overlap_with['candidate_id'],native_annotation_alias=r['annotation_key']))
        else:accepted.append(r)
    summary=dict(new_trusted_nonoverlapping_regions=len(accepted),new_trusted_photos=len({r['image_id'] for r in accepted}),new_trusted_source_groups=len({r['source_group'] for r in accepted}),tiers=dict(Counter(r['evidence_tier'] for r in tiered)),by_native_category={label:dict(regions=sum(r['native_label']==label for r in accepted),photos=len({r['image_id'] for r in accepted if r['native_label']==label}),source_groups=len({r['source_group'] for r in accepted if r['native_label']==label})) for label in sorted({r['native_label'] for r in tiered})},target_negative_class='chair_only',other6classes_unknown=True,human_confirmations_created=0,actual_assistant_inspected_candidates=sum(o['assistant_actual_visual_inspection'] for o in observations),duplicate_or_overlapping_supported_regions=len(duplicates),new_regions_at_Chair_score_ge05=sum(max(q['Chair_score'] for q in r['query_observations'])>=.5 for r in accepted),new_training_activated=False,source_polygons_not_whole_room_completeness=True,insufficient_for_broad_hard_negative_coverage=True)
    emit21(DATA21/'query-local-visual-observations.json',dict(records=observations,author='assistant_actual_visual_inspection',not_human_labels=True));emit21(DATA21/'hard-negative-manifest-v1.json',dict(records=tiered,accepted_records=[r['candidate_id'] for r in accepted],duplicates=duplicates,source_annotations_modified=False,training_authorized=False,primary_box_only_experiment_consumes_these_negatives=False));emit21(REPORT21/'hard-negative-summary.json',summary)
    print('C21source-supported newregions',len(accepted),'groups',summary['new_trusted_source_groups'],';inactive;no human labels fabricated',flush=True)


if __name__=='__main__':run()
