"""No-download inference/data/evaluation proposals,plus lossless candidate lineage."""
from collections import Counter
from commonE6 import *

def run():
    c,train,dev=context();full_path=ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json';full=read(full_path)
    lineage=[];inactive=[];furniture=[];devgroups={r['identity_group'].lower() for r in dev}
    for p in full['photos']:
        anns=[{k:a[k] for k in ('annotation_key','dataset','native_category_id','native_label','box','has_mask','native_ignored','protect_furniture','part_or_semantics_unresolved','source','source_sha256','original_annotation_sha256')} for a in p['annotations']]
        lineage.append(dict(image_id=p['image_id'],sha256=p['sha256'],source_group=p['source_group'],path=p['path'],rights=p['rights'],annotations=anns))
        for a in anns:
            if a['protect_furniture'] or a['part_or_semantics_unresolved']:
                furniture.append(dict(image_id=p['image_id'],cohort=p['cohort'],source_group=p['source_group'],**a,training_eligible=False,activation='Requires native whole-object/taxonomy/partition audit;not a new positive or negative label'))
        if p['image_id'] not in train:
            inactive.append(dict(image_id=p['image_id'],source_group=p['source_group'],sha256=p['sha256'],cohort=p['cohort'],
                 native_annotations=len(anns),native_context_counts=dict(Counter(semantic(a) for a in p['annotations'])),
                 native_furniture_labels=dict(Counter(a['native_label'] for a in anns if a['protect_furniture'])),
                 known_internal_development_group_overlap=p['source_group'].lower() in devgroups,
                 training_partition_assigned=False,model_query_coverage_unverified=True,licence_evidence_preserved=True))
    emit(DATA/'source-lineage-v1.json',dict(full_native_inventory=str(full_path.relative_to(ROOT)),full_native_inventory_sha256=sha(full_path),photos=lineage,rights_reverified_online=False))
    emit(DATA/'inactive-local-candidates-v1.json',dict(records=inactive,automatic_activation=False))
    emit(DATA/'native-furniture-candidates-v1.json',dict(records=furniture,native_granularity_preserved=True,labels_not_activated=True))
    emit(DATA/'dataset-index-v1.json',dict(schema='phase10e6-composed-training-candidate-dataset-v1',training_authorized=False,
         manifest=str((DATA/'training-candidate-manifest-v1.json').relative_to(ROOT)),manifest_sha256=sha(DATA/'training-candidate-manifest-v1.json'),
         lineage=str((DATA/'source-lineage-v1.json').relative_to(ROOT)),lineage_sha256=sha(DATA/'source-lineage-v1.json'),
         inactive_pool=str((DATA/'inactive-local-candidates-v1.json').relative_to(ROOT)),inactive_pool_sha256=sha(DATA/'inactive-local-candidates-v1.json'),
         native_furniture_candidates=str((DATA/'native-furniture-candidates-v1.json').relative_to(ROOT)),native_furniture_sha256=sha(DATA/'native-furniture-candidates-v1.json'),
         protected_source_context_sha256=sha(PREP/'training-context.json'),new_labels_activated=0))
    emit(REPORT/'local-expansion-audit.json',dict(inactive_local_training_only_photos=len(inactive),inactive_native_annotation_records=sum(r['native_annotations'] for r in inactive),
         inactive_groups=len({r['source_group'] for r in inactive}),inactive_group_overlap_with_current_dev=sum(r['known_internal_development_group_overlap'] for r in inactive),
         all_native_furniture_or_part_records=len(furniture),native_labels=dict(Counter(r['native_label'] for r in furniture)),
         no_added_positive_or_negative_labels=True,ABO='Existing catalogue images/weak categories are not fully annotated real-room detection hard negatives;no catalogue partitions or images were changed.'))
    # One precise missing-data request. No original-model replay is necessary
    # because the59 source-bound FP16 teacher outputs already exist.
    checkpoint=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt'
    emit(REPORT/'inference-request-disabled.json',dict(approved=False,enabled=False,mode='optimization_only_inference',
         checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),original_base=str(BASE.relative_to(ROOT)),original_sha256=BASE_SHA,
         exact_photo_ids=list(train),photos=59,source_manifest=str((PREP/'training-context.json').relative_to(ROOT)),source_manifest_sha256=sha(PREP/'training-context.json'),
         maximum_detector_forwards=59,maximum_seconds=180,maximum_allocated_VRAM_bytes=6979321856,maximum_new_output_bytes=134217728,
         optimizer_updates=0,backward_calls=0,new_teacher_forwards=0,downloads=0,automatic_retry=False,
         destination='data/open-images/experiments/phase10e6-training-query-cache-v1',precision='Existing pinned FP16 forward/FP32 arrays;exact original preprocessing/prompts/query grid;no augmentations',
         outputs=['Full per-image3600x7logits and3600x4unclipped square boxes saved atomically before analysis','Exact image/model/input hashes,dimensions,source groups,query-layout and forward counters','Separate CPU candidate scoring/geometry;no automatic negative labels'],
         exclusions='No internal validation,reserved,acceptance or42inactive images;all59are current optimization IDs',
         purpose='Fill exact E5 endpoint training-query gap;near-equality on19development photographs does not establish59training query coverage.',
         success_does_not_automatically_authorize_training=True))
    # Conditional acquisition route reuses an exact already-local source-ID
    # proposal. All rights remain unverified;no provider requests are resumed.
    original=ROOT/'data/open-images/preparation/phase10c21-v1/additional-rights-candidates-v1.json'
    value=read(original);records=value['records'] if 'records' in value else value.get('candidates',[])
    if not records:
        # Preserve the actual existing proposal schema rather than invent IDs.
        for item in value.values():
            if isinstance(item,list) and item and isinstance(item[0],dict) and 'image_id' in item[0]:records=item;break
    if len(records)!=64:raise ValueError('Expected exact64 already-proposed rights candidates')
    emit(DATA/'conditional-rights-candidates-v1.json',dict(records=records,source_manifest=str(original.relative_to(ROOT)),source_sha256=sha(original),all_rights_unverified=True,no_substitution=True))
    emit(REPORT/'conditional-acquisition-proposal.json',dict(approved=False,priority='Only after exhausting the local42inactive and E5query-cache gaps',
         exact_candidate_manifest=str((DATA/'conditional-rights-candidates-v1.json').relative_to(ROOT)),candidate_manifest_sha256=sha(DATA/'conditional-rights-candidates-v1.json'),source_ids=64,
         rights_verified_new_photos=0,photo_acquisition_ready=False,
         stage_A=dict(scope='Photo-specific rights/creator/source-group evidence audit only',max_transfer_bytes=32000000,max_storage_bytes=134217728,max_requests_per_ID=1,min_interval_seconds=3,
                      paused_296_requests_not_resumed=True,stop_on_429_and_respect_Retry_After=True,photo_bodies=False,separate_approval_required=True),
         stage_B=dict(scope='Only exact rights/identity-verified IDs surviving StageA;later explicit photo approval',maximum_photographs=64,estimated_transfer_bytes=[9600000,25600000],
                      proposed_transfer_cap=100000000,per_photo_maximum=3145728,attempts_per_photo=2,storage_cap=536870912,no_automatic_replacements=True),
         requested_types=['Real indoor rooms with chairs plus visible people/body parts,plants,screen/appliance and small household confusers','Whole-object Sofa/Chair/Table/Bed/Desk/Cabinet/Lamp annotations with project stool semantics','Visible-instance masks for nonfurniture confusers and precise whole-object boxes for furniture;unknown background stays ignored'],
         individual_rights='CC0/CCBY photo-specific evidence,creator credit,license version,source URL,evidence checksum and timestamp;annotation license separate',
         isolation='Training-only roles;creator/source/capture groups quarantined if related to any exposed,frozen or acceptance source;all repeated views grouped',
         annotation_requirements='Do not infer negatives from missing COCO/LVIS labels;source-visible regions plus actual targeted semantic checks;Table/Desk and Cabinetry/front/whole-Lamp ambiguities remain unresolved',
         inferred_eligible_hard_negatives=0,new_requests_or_downloads=0))
    registry=read(ROOT/'data/open-images/preparation/phase10c21-v1/historical-source-registry-v2.json')
    emit(DATA/'independent-exclusion-registry.json',registry)
    template=dict(schema='phase10e6-independent-seven-category-intake-v1',records=[],eligible_photographs=0,human_reviews_created=0,
         required_fields=['image_id','sha256','path','source_photo_id','source_group','capture_session_group','source_group_independence_verified','source_group_evidence','rights_verified','rights_evidence_path','rights_evidence_sha256','license_url','role','category_complete','verified_visible_instance_counts','Chair_includes_seating_and_bar_stools','whole_object_taxonomy_confirmed','unresolved_objects','annotations','review'],
         per_category_unreviewed={k:None for k in CLASSES},review_defaults=dict(human_reviewed=False,blind_to_model_predictions=None,reviewer=None,date=None,evidence=None))
    emit(DATA/'independent-review-template.json',template)
    emit(REPORT/'independent-evaluation-plan.json',dict(verified_eligible_photographs=0,verified_complete_instances={k:0 for k in CLASSES},
         first_collection_batch_target=20,existing_long_term_target_photographs=140,target_minimum_instances_per_class=50,
         approval_required_for_actual_collection=True,existing_exposed_or_frozen_auto_promoted=0,threshold=.15,
         workflow=['Recruit individually authorized new room/source groups;choose sources without inspecting model outputs','Freeze intended independent roles before acquisition;use separate source groups for any calibration/training photos',
                   'Record creator/license/consent,source and capture-session evidence,hashes,dimensions and duplicate-group checks',
                   'A blind annotator reviews ALL7categories,including explicit zero/absent categories,and whole-object bbox extents;annotate seating/bar stools as Chair',
                   'Resolve Table/Desk,whole Cabinet versus fronts,and standalone Lamp versus fixtures;unresolved photos stay ineligible',
                   'Second reviewer checks completeness and ≥20%of boxes;adjudicate without seeing detector predictions',
                   'Validate via acceptanceE6.py,freeze image/annotation/protocol hashes,then request a separately authorized single comparison',
                   'Report all7category PR/AP50/AP75,matched recall,duplicates/fragmentation and source-group bootstrap uncertainty;never train or tune on acceptance'],
         actual_intake_records=0,no_scoring=True,acceptance_data_never_negative_mining=True))
    print('Prepared dataset provenance,59-forward disabled request,conditional64-ID rights route,and honest0-image acceptance workflow',flush=True)

if __name__=='__main__':run()
