"""Actual assistant observations plus unchanged source-evidence eligibility gates."""
from collections import Counter,defaultdict
import numpy as np
from common8q import *

# Authored only after viewing each newly generated,source-bound E8plate.
# These are assistant assessments,not invented human reviews or ground truth.
FINDINGS={
'E7V01':('possible_furniture_fragment','The query spans patterned green upholstery-like material,a rigid frame/laptop edge and part of a glass. The full context could include an unannotated chair back;only17.9%is supported by the best single visible mask. Protect this possible seating fragment.',True,False),
'E7V02':('other_object_handbag','The enlarged crop visibly shows a black handbag surface/curved rim with shiny metal hardware. COCOhandbag mask supports87.8%of the query,with exposed background at the rim. The entire rectangle is not source-certified.',False,False),
'E7V03':('other_object_handbag','Another partial view of the same black handbag and metal hardware is visible. It shares the exact COCOhandbag instance withV02;do not count it as a new physical confuser.',False,False),
'E7V04':('other_object_appliance','The context places this tiny query on the refrigerator door/magnet area. The visible source mask supports98.2%,but the18x24outward-rounded rectangle fails the existing32pixel admission size and boundary clearance.',False,False),
'E7V05':('other_object_sink_and_clutter','The crop shows sink/drainer/kitchen clutter rather than an identifiable whole chair. The sink mask supports58.6%and surrounding pixels are not certified;do not infer Chair absence from the sink bbox.',False,False),
'E7V06':('possible_furniture_fragment','The brown curved object under the foreground seated person could be an armrest or a mixed body/furniture boundary. Only16.6%belongs to the visible person mask;the person bbox is misleading. Protect possible furniture.',True,False),
'E7V07':('person_clothing_fragment','The enlarged narrow crop lies on dark clothing beside the foreground person. Native person-mask coverage is96.3%,but the19pixeloutward-rounded width is too narrow and lacks full source-boundary support.',False,False),
'E7V08':('mixed_person_furniture_boundary','The query crosses the foreground person clothing edge and adjacent brown room/furniture structure. Native person-mask coverage72.9%does not certify the remaining pixels;protect possible seating/furniture content.',True,False),
'E7V09':('other_object_handbag','Bag hardware and a foreground glass edge are visible. The strongest visible mask is the same COCOhandbag instance asV02,not the original broad person-bbox label. It is another partial query on that object.',False,False),
'E7V10':('possible_furniture_fragment','The crop is a narrow shoulder/wooden room-divider or booth-edge strip behind people. Person-mask coverage is49.8%;the wood/furniture boundary must not be turned into a Chair-negative region.',True,False),
'E7V11':('other_object_handbag','The black curved handbag rim is visible within the same COCOhandbag instance asV02. The smaller crop is partial and below the existing32pixel size gate;it is not independent evidence.',False,False),
'E7V12':('other_object_handbag','The crop again contains the same handbag and shiny fittings with a glass edge. Native person-bbox context was not crop identity. This is a repeated partial query on the V02source object.',False,False),
'E7V13':('other_object_toilet_fragment','The context and enlarged crop clearly show a bathroom toilet fragment and adjacent floor/wall. Best mask coverage is63.3%;the full predicted rectangle extends outside the visible toilet mask and remains unverified as a negative.',False,False),
'E7V14':('appliance_annotation_low_resolution','The query is inside the native microwave visible polygon,but its16x11pixelrounded crop is too small for reliable independent object identification and does not satisfy the two-pixel interior/32pixel admission gates.',False,False),
'E7V15':('ambiguous_dark_object_boundary','A dark vertical object/frame next to the green patterned surface and a foreground glass is visible. The best mask covers only21.8%and belongs to the glass;the remaining object cannot be identified reliably.',False,False),
'E7V16':('ambiguous_overexposed_person_background','This backlit blue/white patch overlaps a background person/head silhouette and bright window glare. Overexposure obscures precise contents;60.4%person-mask overlap is not absence evidence. Unsuitable for negative supervision.',False,True),
'E7V17':('ambiguous_reflection_decoration','The tiny crop contains mantel decoration/reflection and a small frame. Depicted or reflected seating cannot be excluded;only18.0%visible mirror-mask coverage and the small width prevent admission.',False,False),
'E7V18':('ambiguous_background','The crop is a green/dark background patch near the window. Native person-mask support is zero despite person-bbox context. A plain-looking patch does not establish a trustworthy Chair-negative label.',False,False),
'E7V19':('mixed_body_plant_background','The narrow crop crosses a pale shirt/body or object edge and plant/background detail. The best visible mask is potted plant at36.3%,rather than the original person-bbox label. It is too small and mixed for admission.',False,False),
'E7V20':('possible_furniture_fragment','Dark vertical/horizontal frame-like elements beneath the flowers could be a furniture or decorative structure. Potted-plant visible support is only19.7%;protect possible unannotated seating instead of using plant-bbox context as absence evidence.',True,False),
'E7V21':('person_clothing_fragment','The crop visibly contains the standing person jeans/hip and a narrow background strip. Native person-mask coverage is69.4%;the unmasked strip prevents full query-local absence certification.',False,False),
}

def decide(record,visual):
    kind,evidence,protect,unusable=visual
    hits=record['native_mask_hits'];gate=hits[0]['geometric_gate'] if hits else None
    if protect:return 'protected_unsuitable','Possible real furniture/seating fragment;preserve as potentially positive'
    if unusable:return 'protected_unsuitable','Insufficient visual quality to certify object/absence'
    if not gate or not gate['bounds_ok'] or not gate['minimum_side_ok']:
        return 'protected_unsuitable','Existing source-region size/bounds gate fails;not an accepted negative'
    supported=[a for a in hits if a['geometric_gate']['geometric_eligible'] and not a['native_ignored'] and not a['protected_furniture']]
    if kind.startswith('other_object_') or kind=='person_clothing_fragment':
        if supported and record['student_Chair_score']>=.5:return 'trusted_negative','Clear visible non-Chair object plus complete original source-polygon admission'
    return 'unverified_candidate','Incomplete visible mask or unresolved pixels;no Chair-absence label'

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU qualification only')
    audit=read(DATA/'candidate-source-audit.json');records=audit['records'];_,train,_=context()
    if {r['sample_id'] for r in records}!=set(FINDINGS):raise ValueError('Actual inspected21sample set mismatch')
    decisions=[];groups=defaultdict(list)
    for r in records:
        require_train(r['image_id'],train)
        if sha(ROOT/r['evidence_plate'])!=r['evidence_plate_sha256']:raise ValueError('Actually viewed plate changed')
        visual=FINDINGS[r['sample_id']];status,reason=decide(r,visual)
        result=dict(r,decision=status,decision_reason=reason,object_assessment=visual[0],visual_evidence=visual[1],
              assistant_actual_visual_inspection=True,human_reviewed=False,human_confirmed=False,whole_photo_Chair_completeness_verified=False,
              source_mask_support_is_not_a_class_label=True,training_activated=False,training_authorized=False,
              training_eligible_under_policy=status=='trusted_negative')
        # Group only the exact same native instance with strong visible-mask
        # overlap and actually observed non-Chair object identity. Same creator,
        # bbox context or generic stock scene is never an object merge key.
        best=r['native_mask_hits'][0] if r['native_mask_hits'] else None
        if visual[0]=='other_object_handbag' and best and best['native_label']=='handbag' and best['maximum_single_polygon_query_fraction']>=.75:
            key=(r['source_photo_id'],best['annotation_key']);groups[key].append(result)
        decisions.append(result)
    duplicate_groups=[]
    for (photo_id,annotation_key),members in groups.items():
        if len(members)<2:continue
        members.sort(key=lambda r:(-r['student_Chair_score'],r['candidate_id']));keep=members[0]
        for r in members:
            r.update(source_instance_representative=keep['candidate_id'],repeated_partial_source_instance=True,
                duplicate_relation='Same exact native handbag instance and actual visible bag context;partial queries,not necessarily IoU>0.5duplicate boxes')
            if r is not keep:r.update(decision='protected_unsuitable',training_eligible_under_policy=False,
                decision_reason='Repeated partial query on the same source-supported handbag;not independently counted evidence')
        duplicate_groups.append(dict(source_photo_id=photo_id,annotation_key=annotation_key,representative=keep['candidate_id'],candidate_ids=[r['candidate_id'] for r in members],extra_observations=len(members)-1))
    # A new grant must cover the actual E5query and remain safe for the original
    # teacher query too,as required by the existing selective E5 objective.
    trusted=[];coverage_rows=[]
    for r in decisions:
        newly_eligible=r['decision']=='trusted_negative'
        source=next((a for a in r['native_mask_hits'] if a['geometric_gate']['geometric_eligible'] and not a['native_ignored'] and not a['protected_furniture']),None)
        if newly_eligible:
            if source is None:raise ValueError('No source grant may be invented')
            r['trusted_region_pixels']=source['geometric_gate']['rectangle_pixels'];r['trusted_region_source']=source['annotation_key'];trusted.append(r)
        coverage_rows.append(dict(candidate_id=r['candidate_id'],image_id=r['image_id'],query_index=r['query_index'],
            E5_Chair_score=r['student_Chair_score'],original_Chair_score=r['original_same_query_Chair_score'],
            full_query_source_polygon_supported=bool(source),newly_qualified_negative=newly_eligible,
            accepted_new_high_confidence_Chair_query=newly_eligible and r['student_Chair_score']>=.5,
            accepted_new_ge08_Chair_query=newly_eligible and r['student_Chair_score']>=.8,
            original_teacher_same_query_not_same_object_claim=True))
    # No coverage is inferred from source bbox overlap or the enlarged display.
    # Report the actual zero as zero,without synthetic center-grid surrogates.
    if trusted:
        raise ValueError('Unexpected newly source-qualified region requires explicit full-cache coverage verification before final publication')
    old=read(ROOT/'data/open-images/preparation/phase10c6-v1/source-negative-evidence.json')['records']
    prior=read(ROOT/'data/open-images/preparation/phase10c21-v1/hard-negative-manifest-v1.json')
    old_count=len(old);prior_count=len(prior['accepted_records'])
    evidence_path=DATA/'automatic-decisions-v1.json';emit(evidence_path,dict(records=decisions,actual_assistant_inspected=21,human_reviews_created=0,policy=POLICY,policy_sha256=digest(POLICY)))
    emit(DATA/'actual-query-coverage.json',dict(records=coverage_rows,actual_saved_E5_queries_checked=21,actual_saved_original_queries_checked=21,
         new_regions=0,additional_valid_high_confidence_Chair_negative_queries=0,additional_ge08_queries=0,
         optimization_photos_with_new_coverage=0,source_groups_with_new_coverage=0,positive_or_protected_queries_incorrectly_targeted=0,
         no_synthetic_queries_counted=True,new_GPU_forwards=0))
    emit(DATA/'hard-negative-dataset-v1.json',dict(schema='phase10e8-source-qualified-hard-negative-dataset-v1',
         exact_candidate_ids=[r['candidate_id'] for r in records],training_authorized=False,policy=POLICY,policy_sha256=digest(POLICY),
         source_audit=str((DATA/'candidate-source-audit.json').relative_to(ROOT)),source_audit_sha256=sha(DATA/'candidate-source-audit.json'),
         decisions=str(evidence_path.relative_to(ROOT)),decisions_sha256=sha(evidence_path),
         trusted_negative_ids=[r['candidate_id'] for r in decisions if r['decision']=='trusted_negative'],
         unverified_candidate_ids=[r['candidate_id'] for r in decisions if r['decision']=='unverified_candidate'],
         protected_unsuitable_ids=[r['candidate_id'] for r in decisions if r['decision']=='protected_unsuitable'],
         existing_41_source_regions_preserved=old_count==41,prior_C21_regions_still_inactive=prior_count,
         base_training_context=str((PREP/'training-context.json').relative_to(ROOT)),base_training_context_sha256=sha(PREP/'training-context.json'),
         source_annotations_or_partitions_modified=False,new_labels_activated=0,uncertain_regions_ignored=True,
         accepted_independent_evaluation_photos=0,model_checkpoints_modified=False))
    summary=dict(exact_candidates=21,photographs=10,source_groups=10,decisions=dict(Counter(r['decision'] for r in decisions)),
         object_assessments=dict(Counter(r['object_assessment'] for r in decisions)),
         new_trusted_negatives=0,new_source_groups_with_trusted_negatives=0,new_high_confidence_Chair_query_coverage=0,
         existing_regions_preserved=old_count,prior_C21_regions_remain_inactive=prior_count,
         repeated_source_instance_groups=duplicate_groups,extra_partial_queries_not_independent=sum(g['extra_observations'] for g in duplicate_groups),
         native_full_query_geometry_supported=sum(r['full_query_source_polygon_supported'] for r in coverage_rows),
         assistant_visual_reviews=21,human_reviews_created=0,training_activated=False,downloads=0,GPU_forwards=0,
         independent_acceptance_photos=0)
    emit(REPORT/'qualification-summary.json',summary)
    print('E8qualified',summary['decisions'],';new valid high-confidence coverage0;all historical evidence unchanged')

if __name__=='__main__':run()
