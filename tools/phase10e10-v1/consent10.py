"""Future creator-grant validation only; no signatures, consent or reviews generated."""
REQUIRED_SCOPES=('store_photographs','annotate_and_modify','commercial_ML_training','distribute_trained_model')
def validate_grant(record):
    errors=[]
    for field in ('photograph_ids','creator_or_rightsholder','source_group_id','grant_timestamp','affirmative_signature_or_evidence'):
        if not record.get(field):errors.append('Missing '+field)
    if record.get('rights_holder_authority_confirmed') is not True:errors.append('Authority not affirmed')
    scopes=record.get('scopes',{})
    for key in REQUIRED_SCOPES:
        if scopes.get(key) is not True:errors.append('Scope not granted: '+key)
    if record.get('other_people_or_third_party_rights_resolved') is not True:errors.append('Other rights unresolved')
    if record.get('evaluation_or_training_purpose') not in ('training','independent_evaluation'):errors.append('Purpose unresolved')
    return dict(status='unresolved' if errors else 'grant_ready_for_independent_evidence_verification',errors=errors,
                automatically_approved=False,photo_download_authorized=False)
