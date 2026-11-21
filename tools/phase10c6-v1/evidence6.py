"""Separate automatic source-mask integrity gate; no human flags are fabricated."""
from collections import Counter,defaultdict
from common6 import *
from geometry6 import certify,polygons

def validate_source(r,observation,all_records,all_observations):
    if observation.get('visual_inspected') is not True or observation.get('human_reviewed') is not False or observation.get('human_confirmed') is not False:raise ValueError('Automatic visual provenance required; never human confirmation')
    if observation.get('candidate_sha256')!=digest(r) or observation.get('sha256')!=r['sha256'] or observation.get('annotation_key')!=r['annotation']['annotation_key']:raise ValueError('Visual/source binding changed')
    if observation['visual_outcome']!='clear_nonchair':raise ValueError('Visual evidence not sufficiently clear')
    p=r.get('proposal');a=r['annotation']
    if a['dataset'] not in ('COCO','LVIS') or a['mapped_auxiliary_category']=='protected_furniture' or not a['metadata_candidate_eligible'] or not p:raise ValueError('No eligible native non-Chair polygon')
    b=p['box_pixels'];w,h=r['width'],r['height'];polylist=polygons(a['original_annotation'],w,h)
    if any(type(v) is not int for v in b) or not 0<=b[0]<b[2]<=w or not 0<=b[1]<b[3]<=h or min(b[2]-b[0],b[3]-b[1])<32:raise ValueError('Region unsafe or below spatial review resolution')
    normalized=[b[0]/w,b[1]/h,b[2]/w,b[3]/h]
    if normalized!=p['box'] or p['polygon_index']>=len(polylist) or [list(v) for v in polylist[p['polygon_index']]]!=p['polygon'] or not certify(b,polylist[p['polygon_index']]):raise ValueError('Exact native polygon containment/margin failed')
    ab=a['box']
    if not(ab[0]<=normalized[0]<normalized[2]<=ab[2] and ab[1]<=normalized[1]<normalized[3]<=ab[3]):raise ValueError('Polygon proposal outside source annotation box')
    if any(overlap(normalized,p['box']) for p in r['protected']):raise ValueError('Protected positive/ignored overlap')
    for other in all_records:
        if other['candidate_id']==r['candidate_id'] or other['image_id']!=r['image_id']:continue
        obs=all_observations[other['candidate_id']]
        if obs['visual_outcome']=='mixed_or_uncertain' and overlap(normalized,other['annotation']['box']):raise ValueError('Overlap with visually uncertain source region '+other['candidate_id'])
    return True

def run():
    records=candidates();obs={r['candidate_id']:read(RAW/'observations'/(r['candidate_id']+'.json')) for r in records};rows=manifest()['rows'];train=[r for r in rows if optimizer_member(r)];other=[r for r in rows if not optimizer_member(r)]
    for field in ('image_id','sha256','identity_group'):
        if {r[field] for r in train}&{r[field] for r in other}:raise ValueError('Training/validation leakage')
    training={r['image_id']:r for r in train};inventory=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json');native={a['annotation_key']:(p,a) for p in inventory['photos'] for a in p['annotations']}
    for path,expected in inventory['metadata_sources'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Native source metadata changed')
    results=[];accepted=[]
    for r in records:
        o=obs[r['candidate_id']];row=training.get(r['image_id']);p,a=native[r['annotation']['annotation_key']]
        if row is None or r['sha256']!=row['sha256'] or r['identity_group']!=row['identity_group'] or digest(a)!=digest(r['annotation']):raise ValueError('Candidate/partition/native lineage changed')
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Source photograph changed')
        for k,hk in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
            if sha(ROOT/r['rights'][k])!=r['rights'][hk]:raise ValueError('Rights evidence changed')
        plate=ROOT/o['plate']
        if not plate.resolve().is_relative_to(RAW) or sha(plate)!=o['plate_sha256'] or o['candidate_sha256']!=digest(r):raise ValueError('Inspected visual artifact changed')
        reason=o['visual_observation'];level='C';gate=None
        if o['visual_outcome'] in ('clear_nonchair','automatic_only'):
            level='B'
            if o['visual_outcome']=='clear_nonchair' and r['proposal']:
                try:validate_source(r,o,records,obs);level='A'
                except ValueError as e:level='C';gate=str(e)
        if level=='A':
            evidence=dict(candidate_id=r['candidate_id'],image_id=r['image_id'],source_photo_sha256=r['sha256'],identity_group=r['identity_group'],historical_partition=row['historical_partition'],internal_partition=row['internal_partition'],
                category='chair',kind='source_polygon_visible_negative',box=r['proposal']['box'],box_pixels=r['proposal']['box_pixels'],
                source_native_dataset=a['dataset'],source_native_category=a['native_category_id'],source_annotation_key=a['annotation_key'],source_annotation_sha256=digest(a['original_annotation']),
                native_polygon=r['proposal']['polygon'],native_polygon_index=r['proposal']['polygon_index'],policy_sha256=digest(ADMISSION),
                automatic_visual_observation_sha256=digest(o),human_reviewed=False,human_confirmed=False,source_verified_under_policy=True,
                approval_scope='Only observed visible pixels of named non-Chair object, not hidden contents or whole-photo completeness',
                no_other_furniture_negative_claim=True,training_enabled=False)
            accepted.append(evidence)
        results.append(dict(r,visual_inspected=True,automatic_observation=o,evidence_level=level,eligibility_gate_failure=gate,
            explanation=reason,confidence='source-and-spatial-supported' if level=='A' else 'automatic-unverified' if level=='B' else 'insufficient-or-conflicting',
            model_probability=None,training_eligible_under_new_policy=level=='A',training_activated=False))
    # Avoid overlapping duplicate targets; keep deterministic stronger/larger area
    # only, while retaining all source lineage and excluded results.
    kept=[]
    for e in sorted(accepted,key=lambda e:(-((e['box_pixels'][2]-e['box_pixels'][0])*(e['box_pixels'][3]-e['box_pixels'][1])),e['candidate_id'])):
        if any(e['image_id']==q['image_id'] and overlap(e['box'],q['box']) for q in kept):
            result=next(r for r in results if r['candidate_id']==e['candidate_id']);result.update(evidence_level='C',training_eligible_under_new_policy=False,eligibility_gate_failure='Overlapping accepted source target; preserve without duplicate weighting')
        else:kept.append(e)
    accepted=sorted(kept,key=lambda e:e['candidate_id']);summary=dict(schema='phase10c6-automatic-review-summary-v1',
        photographs_inspected=len({r['image_id'] for r in records}),objects_inspected=len(records),initial_candidates=123,additional_objects_inspected=len(records)-123,
        source_verified_regions=len(accepted),automatically_screened_candidates=sum(r['evidence_level']=='B' for r in results),uncertain_or_rejected=sum(r['evidence_level']=='C' for r in results),
        source_verified_photos=len({e['image_id'] for e in accepted}),source_verified_groups=len({e['identity_group'] for e in accepted}),
        by_category=[dict(category=category,total=sum(r['category']==category for r in results),source_verified=sum(r['category']==category and r['evidence_level']=='A' for r in results),automatic=sum(r['category']==category and r['evidence_level']=='B' for r in results),uncertain=sum(r['category']==category and r['evidence_level']=='C' for r in results),known_groups=len({r['identity_group'] for r in results if r['category']==category})) for category in sorted({r['category'] for r in results})],
        candidate_groups=len({r['identity_group'] for r in records}),human_confirmations_created=0,policy=ADMISSION,policy_sha256=digest(ADMISSION),
        original_furniture_photos=59,original_positive_boxes=182,original_source_groups=56,training_activated=False,model_forwards=0,GPU_forwards=0,downloads=0,
        source_verified_is_not_human_verified_accuracy=True,source_annotations_can_be_imperfect=True,
        source_verified_semantics='Native visible non-Chair polygon, exact isolated spatial region, actual assistant visual inspection and zero protected/uncertain overlap',
        exclusions='All saved Phase10C.2 Chair false-positive review predictions are internal-validation diagnostics; none become training negatives',
        additional_metadata_geometry_audit=read(RAW/'qualified-expansion.json')['remaining_native_records_geometry_scanned'])
    save(RAW/'automatic-evidence.json',dict(results=results,policy=ADMISSION))
    save(RAW/'source-negative-evidence.json',dict(policy=ADMISSION,policy_sha256=digest(ADMISSION),records=accepted,training_authorized=False))
    save(RAW/'training-evidence-manifest.json',dict(schema='phase10c6-additive-source-evidence-v1',source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        training_rows=train,source_negative_evidence=accepted,source_evidence_sha256=sha(RAW/'source-negative-evidence.json'),
        unverified_automatic_candidates=[r['candidate_id'] for r in results if r['evidence_level']=='B'],ignored_candidates=[r['candidate_id'] for r in results if r['evidence_level']=='C'],unknown_background_ignored=True,production_categories=list(CLASSES),training_authorized=False))
    save(REPORT/'summary.json',summary);print('SOURCE-VERIFIED',len(accepted),'AUTO',summary['automatically_screened_candidates'],'UNCERTAIN',summary['uncertain_or_rejected'],'INSPECTED',len(records),flush=True)

if __name__=='__main__':run()
