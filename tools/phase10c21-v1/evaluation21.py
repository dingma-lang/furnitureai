"""Independent seven-class readiness and honest empty review workflow."""
import math,re,datetime
from common21 import *


def registry21():
    ready=read(ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json');ids=set();hashes=set();groups=set();source_ids=set()
    for r in ready['source_rows']:
        ids.add(r['image_id']);hashes.add(r['sha256'])
        groups.update(str(v).lower() for v in (r.get('known_source_component'),r.get('original_source_group')) if v);source_ids.add(r['source_photo_id'])
    for r in read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']:
        ids.add(r['image_id']);groups.update(str(v).lower() for v in (r.get('identity_group'),r.get('source_identity')) if v)
        if r.get('source_identity'):source_ids.add(r['source_identity'])
    rooms=read(ROOT/'reports/phase4-room-comparison-v1/input-lock.json')['rooms']
    for r in rooms:ids.add(r['image_id']);hashes.add(r['sha256'])
    for p in read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json')['photos']:
        groups.add(p['identity_group'].lower());groups.add(p['licence']['creator_group'].lower())
    # Metadata-only exclusion of frozen references and earlier catalogue images.
    lock=read(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json')
    for r in lock['rows']+lock['frozen_references']:
        ids.add(r['image_id']);hashes.add(r['sha256']);groups.add(r['identity_group'].lower())
    for r in read(ROOT/'reports/phase10a-v1/backup-inventory.json')['records']:
        if Path(r['path']).suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
            hashes.add(r['sha256'])
    for split in ('train','validation','test'):
        for r in read(ROOT/f'data/abo/experiments/phase2a-v1/features-v1/{split}/metadata.json')['rows']:
            ids.add(r['image_id']);hashes.add(r['sha256'])
            for key in ('abo_product_id','group_id','family_group_id','human_audited_family_group_id'):
                if r.get(key):groups.add(str(r[key]).lower())
    return dict(image_ids=sorted(ids),sha256=sorted(hashes),source_groups=sorted(groups),source_photo_ids=sorted(source_ids),independent_sources_not_selected=True)


def validate_record(record,registry,verify_files=True):
    if not all(isinstance(record.get(k),str) and record[k].strip() for k in
               ('image_id','source_photo_id','source_group','capture_session_group')):
        raise ValueError('Explicit photograph/source/capture identities required')
    if not re.fullmatch('[0-9a-f]{64}',str(record.get('sha256',''))) or not re.fullmatch('[0-9a-f]{64}',str(record.get('rights_evidence_sha256',''))):
        raise ValueError('Exact image and rights-evidence SHA-256 required')
    if record.get('role')!='independent_acceptance' or record.get('image_id') in registry['image_ids'] or record.get('sha256') in registry['sha256'] or record.get('source_photo_id') in registry['source_photo_ids'] or str(record.get('source_group','')).lower() in registry['source_groups']:raise PermissionError('Historical training/development/frozen/source identity cannot enter new independent acceptance')
    if record['capture_session_group'].lower() in registry['source_groups']:
        raise PermissionError('Historical capture/source group')
    if not record.get('source_group') or not record.get('capture_session_group') or record.get('source_group_independence_verified') is not True or not record.get('source_group_evidence'):raise ValueError('Verified room/source/capture grouping required')
    if record.get('rights_verified') is not True or not record.get('rights_evidence_sha256') or not record.get('licence_url'):raise ValueError('Specific photographic rights evidence required')
    review=record.get('review',{})
    if review.get('human_reviewed') is not True or review.get('blind_to_model_predictions') is not True or not review.get('reviewer') or not review.get('date') or not review.get('evidence'):raise ValueError('Actual independent human completeness evidence required')
    try:datetime.date.fromisoformat(review['date'])
    except (TypeError,ValueError):raise ValueError('Review date must be YYYY-MM-DD')
    if record.get('category_complete')!={c:True for c in CLASSES} or record.get('Chair_includes_seating_and_bar_stools') is not True or record.get('whole_object_taxonomy_confirmed') is not True:raise ValueError('All seven completeness and object-taxonomy checks required')
    for a in record.get('annotations',[]):
        b=a.get('box',[])
        if a.get('label') not in CLASSES or len(b)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in b) or not(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1) or type(a.get('ignore')) is not bool:raise ValueError('Invalid class/box/ignore annotation')
    if verify_files:
        for key,hk in (('path','sha256'),('rights_evidence_path','rights_evidence_sha256')):
            p=(ROOT/record[key]).resolve()
            if not p.is_relative_to(ROOT/'data') or sha(p)!=record[hk]:raise ValueError('Independent local file/evidence identity mismatch')
    return dict(eligible=True,classes_complete=list(CLASSES),synthetic_test_only=not verify_files)


def validate_manifest(records,registry,verify_files=True):
    """No automatic filling; one source/capture group per proposed acceptance room."""
    seen={k:set() for k in ('image_id','sha256','source_photo_id','source_group','capture_session_group')}
    for record in records:
        validate_record(record,registry,verify_files=verify_files)
        for key,values in seen.items():
            value=record[key].lower()
            if value in values:raise ValueError('Duplicate independent identity/group: '+key)
            values.add(value)
    return dict(eligible_photos=len(records),synthetic_test_only=not verify_files)


def run():
    ready=read(ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json');registry=registry21();complete=[r for r in ready['source_rows'] if r.get('full_seven_class_annotation_completeness_independently_verified') is True]
    if complete:raise ValueError('Unexpected completeness evidence requires explicit per-image audit;do not infer eligibility')
    protocol=read(ROOT/'reports/phase10c9-v1/independent-evaluation-protocol.json')
    emit21(DATA21/'historical-source-registry.json',registry)
    emit21(DATA21/'independent-review-template.json',dict(schema='phase10c21-independent-seven-class-review-v1',records=[],accepted_records=0,human_reviews_generated=0,required_fields=['image_id','sha256','path','source_photo_id','source_group','capture_session_group','source_group_independence_verified','source_group_evidence','rights_verified','rights_evidence_path','rights_evidence_sha256','licence_url','role','category_complete','Chair_includes_seating_and_bar_stools','whole_object_taxonomy_confirmed','annotations','review'],review_fields=['human_reviewed','blind_to_model_predictions','reviewer','date','evidence']))
    emit21(REPORT21/'independent-evaluation-readiness.json',dict(actual_independent_complete_eligible_photos=0,actual_complete_category_instance_counts={c:0 for c in CLASSES},source_rows_checked=len(ready['source_rows']),existing_phase4_rooms=3,existing_Phase4_room_annotations='Assistantprovisionalandmodel-exposed;notcompleteindependentlabels',existing_frozen_or_reserved_auto_promoted=0,missing=['Source-disjointindividuallyauthorizednewroommanifest','Independentlyreviewedcompleteall7boxeswithChair+stool,wholeCabinet/Lamp,andTable/Deskpolicy','Rights/source/captureidentityandnear-duplicateverification','Aseparatecalibration/developmentsetifthresholdselectionisneeded','Frozenacceptanceprotocolbeforemodelscoring'],workflow=['Freeze sampling/source IDs without looking at detector outputs','Verify individual rights/creator/room-session groups;quarantineunresolvedrelations','Seed native boxes where available,but have independent annotators confirm all7category completeness andmissedobjects','Explicitignore/occlusion/truncation/crowd/unitsemantics;no unlabeledobjectbecomesnegative','Second-reviewcategory/completeness and>=20percentdoubleboxannotations;adjudicateblindtomodeloutputs','Freezeimage/annotation/protocolhashes;oneoriginalversuspreselectedexperimentalcomparison','ReportPR/AP50/AP75,TP/FP/FN,duplicates,fragmentation,occlusion/size andsource-groupbootstrapuncertainty'],proposed_acceptance_photos=protocol['proposed_photos'],proposed_distinct_source_groups=protocol['proposed_distinct_source_groups'],proposed_min_instances_per_category=protocol['minimum_fully_annotated_instances_per_category'],bootstrap_resamples=2000,threshold=.15,joint_targets=dict(precision=.8,recall=.9),future_qualifying_collection_requires_separate_exact_manifest_approval=True,no_photos_acquired=True,no_scoring=True))
    print('Independent complete all7 readiness:0photos;honest emptyreviewtemplate created',flush=True)


if __name__=='__main__':run()
