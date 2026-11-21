"""Metadata-only readiness; never opens independent photograph bodies."""
from collections import Counter
from common12 import *


def run():
    protocol_path=ROOT/'reports/phase10c9-v1/independent-evaluation-protocol.json';protocol=read(protocol_path)
    readiness_path=ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json';readiness=read(readiness_path)
    complete=[r for r in readiness['source_rows'] if r['full_seven_class_annotation_completeness_independently_verified']]
    index_path=ROOT/'data/open-images/image-pilots/phase5-v1/partition-indexes/frozen_evaluation.json';index=read(index_path)
    screening_path=ROOT/'data/open-images/image-pilots/phase5-v1/screening-final-v1/screening.json';screening=read(screening_path)
    frozen=[r for r in screening['images'] if r['proposed_split']=='frozen_evaluation']
    fields=('image_id','sha256','identity_group');_,train,dev=verify_preparation()
    # Metadata intersection only; membership is never changed or promoted.
    overlaps={k:len({r[k] for r in frozen if r.get(k)}&{r[k] for r in train+dev if r.get(k)}) for k in fields}
    save(REPORT12/'independent-readiness.json',dict(status='NOT_READY',protocol=str(protocol_path.relative_to(ROOT)),protocol_sha256=sha(protocol_path),protocol_status=protocol['status'],proposed_unseen_photos=protocol['proposed_photos'],minimum_instances_per_category=protocol['minimum_fully_annotated_instances_per_category'],combined_readiness_rows=len(readiness['source_rows']),combined_readiness_fully_independently_verified_seven_class_photos=len(complete),
        historical_frozen_metadata_members=len(index['members']),historical_frozen_statuses=dict(Counter(r['eligibility_status'] for r in frozen)),historical_frozen_human_reviewed=sum(r.get('human_reviewed',False) for r in frozen),historical_frozen_scene_verified=sum(r.get('scene_verified') is True for r in frozen),frozen_intersection_with_current_training_and_development=overlaps,frozen_missing_identity_metadata={k:sum(not r.get(k) for r in frozen) for k in fields},
        all_seven_complete_source_disjoint_acceptance_pool_established=False,qualified_independent_photos_established=0,qualification_zero_means_no_pool_proven_not_no_photos_exist=True,photo_bodies_opened=0,model_scores_computed=0,existing_holdout_membership_changed=False,
        missing=['Exact separately authorized unseen source manifest and commercially compatible photo-specific rights','140 proposed unseen room/source groups; minimum50 fully annotated instances/category,actual support audited before scoring','Complete seven-class whole-object taxonomy including seating/bar stools; Cabinet/Desk/Table/Lamp unit policies; ignored uncertain objects','Independent annotation QA and adjudication without model prediction-driven selection','Room/session/creator/source/hash/perceptual isolation from all training and exposed development material','Predeclared common operating point and one frozen selected checkpoint; no acceptance-set tuning','Source-group confidence intervals and power limitations; no80%precision/90%recall claim from current panel'],
        audited_metadata=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in (protocol_path,readiness_path,index_path,screening_path)]))
    print('Independent evaluation NOT_READY:0 proven fully seven-class/source-disjoint acceptance photos; no images accessed',flush=True)


if __name__=='__main__':run()
