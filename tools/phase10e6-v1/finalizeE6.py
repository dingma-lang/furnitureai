"""Versioned A/B/C candidate manifest,without annotation activation or invented review."""
from collections import Counter
from commonE6 import *

FINDINGS={
'V01':('mixed_real_seating','The crop contains a seat/back edge,table and background beneath a person;the native person bbox does not establish Chair absence.'),
'V02':('uncertain_seating_fragment','A dark narrow structure beneath the foreground person may include seating;the crop is too ambiguous to certify a negative.'),
'V03':('plant_person_mixed','Plant leaves and adjacent person/background are visible;protected furniture overlaps prevent a trustworthy negative rectangle.'),
'V04':('plant_decoration','Lower Christmas-tree branches,decorations and base fabric dominate;known protected overlaps remain unresolved.'),
'V05':('screen_or_depiction_uncertain','A narrow patterned/display-like strip is visible;native laptop context does not certify every pixel or exclude depicted/reflective content.'),
'V06':('uncertain_furniture_or_appliance','A narrow boundary near kitchen appliances and seating is not visually sufficient to resolve the object.'),
'V07':('furniture_fragment','The small crop contains an upholstered/furniture boundary in a room with known seating;it must remain protected.'),
'V08':('apparent_real_seating','A dark framed bar-stool/chair with a light seat is visible near the balcony table;this is a plausible real seating detection despite the native bed-bbox context.'),
'V09':('mixed_fragment_clothing','The lower dinner-table crop includes fabric/body or seating boundaries;the tablecloth bbox is not absence evidence.'),
'V10':('real_chair_fragment','The crop contains a dining-chair frame and light cushion/seat;it must not be used as a Chair negative.'),
'V11':('real_chair_mixed','A slatted chair is clearly visible together with a table/laptop;the low whole-object IoU does not make it non-Chair.'),
'V12':('unknown_toy_or_object','A small colorful household/toy region is visible,but its seating semantics cannot be established at this scale.'),
'V13':('real_chair','The crop clearly includes a white molded chair and legs;native seating support and visual evidence forbid a negative target.'),
'V14':('apparent_real_seating','Several slatted chair-like frames are visible against the wall;they cannot be labelled non-Chair.'),
'V15':('mixed_bag_person_seating','The crop is around a seated person and a dark bag/furniture boundary;native backpack context alone is insufficient.'),
'V16':('too_small_uncertain','The very thin crop does not provide enough visual information to identify a non-Chair region reliably.'),
'V17':('person_in_real_seating','A person sits in upholstered seating,which occupies much of the crop;person overlap cannot establish Sofa/Chair absence.'),
'V18':('mixed_booth_and_people','People,table and upholstered booth/bench seating are visible together;unannotated seating may explain the Sofa prediction.'),
'V19':('plant_and_household_clutter','The crop shows plant leaves,pots/books and background clutter;the source polygon does not certify the whole query as a Sofa-negative region.'),
'V20':('armchair_sofa_taxonomy_conflict','The crop shows a single upholstered armchair;Sofa prediction and native sofa context need taxonomy review,not automatic negative supervision.'),
'V21':('furniture_fragment','The crop is a small furniture/floor boundary in a rotated room image;whole-object category absence is not established.'),
'V22':('bed_headboard_fragment','The narrow region is around the bed/headboard behind a person;it is not evidence of Bed absence.'),
'V23':('real_bed_fragment','The crop contains the bed headboard,pillows and bedding;it is a partial real Bed detection.'),
'V24':('bed_and_person_mixed','The upper-bed/person/headboard crop contains genuine bed context and cannot be a Bed negative.')}

def run():
    c,train,dev=context();initial=read(DATA/'candidates-initial.json')['records'];obs=[json.loads(v) for v in (DATA/'high-confidence-observations.jsonl').read_text().splitlines()]
    samples=read(DATA/'inspection-samples.json');pages={s:p for p in samples['pages'] for s in p['sample_ids']};visual=[]
    for r in samples['records']:
        page=pages[r['sample_id']]
        if sha(ROOT/page['path'])!=page['sha256']:raise ValueError('Viewed plate changed')
        finding,evidence=FINDINGS[r['sample_id']]
        visual.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],image_id=r['image_id'],source_sha256=train[r['image_id']]['sha256'],
            plate=page['path'],plate_sha256=page['sha256'],assistant_actual_visual_inspection=True,human_reviewed=False,human_confirmed=False,
            finding=finding,evidence=evidence,negative_label_approved=False))
    emit(DATA/'visual-observations.json',dict(records=visual,actual_assistant_reviewed_samples=24,human_reviews_created=0,new_labels=0))
    # Alias-only indexing expansion must not silently change the mined cohort.
    v2=read(DATA/'query-index-v2.json');v3=read(DATA/'query-index-v3.json')
    if {r['snapshot_id'] for r in v2['records']}!={r['snapshot_id'] for r in v3['records']}:raise ValueError('Additional snapshots require a versioned re-mining run')
    emit(REPORT/'final-cache-accounting.json',dict(source_bound_raw_files=v3['raw_source_bound_files'],unique_snapshots=v3['unique_image_array_snapshots'],
         added_initial_cache_aliases=96,new_numerical_snapshots_from_alias_expansion=0,
         remaining_unindexed_files=len(v3['unindexed_non_evaluation_NPZ']),remaining_files_are19_internal_validation_recovery_caches=True,
         no_validation_cache_body_used_for_mining=True,duplicate_file_aliases=v3['raw_source_bound_files']-v3['unique_image_array_snapshots']))
    byid={r['candidate_id']:[] for r in initial}
    for o in obs:byid[o['candidate_id']].append(o)
    A=[];B=[];C=[];reasons=Counter()
    for candidate in initial:
        observations=byid[candidate['candidate_id']]
        qualified=[o for o in observations if o['source_geometry_support'] and o['protected_overlap_count']==0 and o['within_original_frame']]
        if qualified:raise ValueError('A geometric candidate needs actual source/pixel/visual validation;do not auto approve')
        plausible=[o for o in observations if o['reported_target_detection'] and o['within_original_frame'] and o['known_target_IoU']<.5 and not o['fragment_heuristic'] and o['protected_overlap_count']==0 and o['source_annotation_key'] is not None]
        if plausible:
            tier='B';reason='Source-bbox context without certified visible-region absence;not a training negative';B.append(candidate['candidate_id'])
        else:
            tier='C'
            if any(o['known_target_IoU']>=.5 for o in observations):reason='Known target overlap;not a confirmed mistake'
            elif any(o['fragment_heuristic'] for o in observations):reason='Possible real furniture fragment;protected'
            elif any(o['protected_overlap_count'] for o in observations):reason='Known furniture/ignored/ambiguous overlap blocks negative eligibility'
            elif not any(o['within_original_frame'] for o in observations):reason='Outside original image frame'
            else:reason='No sufficient source-supported category/visible-region evidence'
            C.append(candidate['candidate_id'])
        candidate.update(evidence_tier=tier,reason=reason,training_eligible=False,training_activated=False)
        reasons[reason]+=1
    # Record exact containment separately from mere overlap with old grants.
    old=read(ROOT/'data/open-images/preparation/phase10c6-v1/source-negative-evidence.json')['records'];existing=Counter()
    for o in obs:
        if o['target']!='chair':continue
        box=o['box']
        contained=any(e['image_id']==o['image_id'] and e['box'][0]<=box[0]<box[2]<=e['box'][2] and e['box'][1]<=box[1]<box[3]<=e['box'][3] for e in old)
        if contained:
            existing['contained_high_queries']+=1
            if o['protected_overlap_count']==0 and o['within_original_frame']:existing['unblocked_contained_high_queries_upper_bound']+=1
    emit(DATA/'candidate-evidence-v1.json',dict(records=initial,all_unactivated=True,visual_observations_sha256=sha(DATA/'visual-observations.json')))
    manifest=dict(schema='phase10e6-additive-hard-negative-candidates-v1',training_authorized=False,base_training_context=str((PREP/'training-context.json').relative_to(ROOT)),
        base_training_context_sha256=sha(PREP/'training-context.json'),base_optimization_images=59,base_known_source_groups=56,base_positive_boxes=182,
        production_categories=list(CLASSES),Chair_includes_seating_and_bar_stools=True,footstools_ottomans_stepstools_not_silently_Chair=True,
        sets=dict(A_trusted_new_training_eligible_hard_negatives=A,B_automatic_unverified=B,C_uncertain_or_rejected=C),
        candidates=str((DATA/'candidate-evidence-v1.json').relative_to(ROOT)),candidates_sha256=sha(DATA/'candidate-evidence-v1.json'),
        query_observations=str((DATA/'high-confidence-observations.jsonl').relative_to(ROOT)),query_observations_sha256=sha(DATA/'high-confidence-observations.jsonl'),
        existing_41_regions_retained_unchanged=True,prior_C21_five_regions_remain_inactive=True,omitted_381_reviewed_not_automatically_activated=True,
        new_eligible_regions=0,new_eligible_photographs=0,new_eligible_source_groups=0,new_eligible_Chair_queries_ge05=0,new_eligible_Chair_queries_ge08=0,
        no_unmatched_prediction_is_an_absence_label=True,unknown_regions_ignored=True,partition_assignments_unchanged=True)
    emit(DATA/'training-candidate-manifest-v1.json',manifest)
    emit(REPORT/'candidate-dataset-summary.json',dict(A=len(A),B=len(B),C=len(C),reason_counts=dict(reasons),
        new_eligible_hard_negatives=0,new_eligible_photos=0,new_eligible_source_groups=0,
        B_by_target=dict(Counter(r['target'] for r in initial if r['evidence_tier']=='B')),
        B_by_context=dict(Counter(r['source_context'] for r in initial if r['evidence_tier']=='B')),
        B_source_groups=len({r['source_group'] for r in initial if r['evidence_tier']=='B'}),B_photographs=len({r['image_id'] for r in initial if r['evidence_tier']=='B'}),
        existing_region_ge05_coverage=dict(existing),actual_assistant_views=24,human_reviews_created=0,
        safe_full_polygon_query_candidates=0,training_should_proceed=False))
    print('Dataset finalized:A',len(A),'B',len(B),'C',len(C),';no new eligible negatives',flush=True)

if __name__=='__main__':run()
