"""Future intake metadata validation; never acquires photos or activates labels."""
import re
SCOPES=('storage','annotation_and_modification','model_training_or_evaluation','commercial_model_distribution')
CLASSES=('Sofa','Chair','Table','Bed','Desk','Cabinet','Lamp')
def validate(record,existing=None):
    errors=[];existing=existing or []
    for key in ('photo_ids','creator_id','source_group_id','room_group_ids','permission_evidence','permission_date','explicit_confirmation'):
        if not record.get(key):errors.append('Missing '+key)
    if record.get('explicit_confirmation') is not True:errors.append('Explicit creator confirmation required')
    for key in SCOPES:
        if record.get('scopes',{}).get(key) is not True:errors.append('Consent absent: '+key)
    if record.get('privacy_and_third_party_rights_resolved') is not True:errors.append('Other rights unresolved')
    purpose=record.get('purpose')
    if purpose not in ('training','independent_evaluation'):errors.append('Purpose not fixed')
    for old in existing:
        if set(record.get('photo_ids',[]))&set(old.get('photo_ids',[])):errors.append('Duplicate photograph ID')
        overlap=record.get('creator_id')==old.get('creator_id') or record.get('source_group_id')==old.get('source_group_id') or bool(set(record.get('room_group_ids',[]))&set(old.get('room_group_ids',[])))
        if overlap and purpose!=old.get('purpose'):errors.append('Creator/source/room group crosses purposes')
    return dict(status='unresolved' if errors else 'metadata_complete_pending_evidence_and_identity_verification',errors=sorted(set(errors)),
                human_reviewed=False,photograph_download_authorized=False,annotations_active=False)
def annotation_complete(record):
    # Missing classes are unknown, never inferred negative.
    decisions=record.get('class_completeness',{})
    return all(decisions.get(c) in ('complete_positive_and_negative_review','reviewed_not_applicable_with_evidence') for c in CLASSES) and bool(record.get('independent_reviewer_evidence'))
