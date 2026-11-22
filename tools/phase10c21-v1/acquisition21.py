"""Exact metadata-only rights-audit shortlist;no rights/photo requests."""
from common21 import *
from evaluation21 import registry21


def run():
    registry=registry21();known=set(registry['image_ids']);source_ids=set(registry['source_photo_ids']);attempted=set()
    for p in [ROOT/'data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json',ROOT/'data/lvis/rights-audits/phase7-v1/license-and-attribution-ledger.json',ROOT/'data/lvis/rights-audits/phase8a-v1/license-attribution-ledger.json']:
        d=read(p)
        for r in d.get('records',[]):
            ident=r.get('image_id') or ('coco2017:'+str(r.get('coco_image_id',r.get('lvis_image_id'))).zfill(12))
            attempted.add(ident)
    contexts={'person','potted plant','vase','tv','dining table','bed','couch'};candidates=[];exclusions=Counter()
    path=ROOT/'data/coco/metadata/phase6b-v1/candidate-inventory.jsonl'
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r['image_id'] in known or r['source_identity'] in source_ids:exclusions['existing_source_or_dataset']+=1;continue
            if r['image_id'] in attempted:exclusions['previously_attempted_or_paused_rights_ID']+=1;continue
            if r.get('metadata_license_id')!=4 or not r.get('metadata_eligible_for_rights_audit'):exclusions['not_metadata_CC_BY_or_source_eligible']+=1;continue
            overlap=contexts&set(r.get('indoor_metadata_context',[]))
            if not overlap:exclusions['no_relevant_indoor_furniture_metadata']+=1;continue
            score=5*('potted plant' in overlap)+4*('person' in overlap)+2*('vase' in overlap)+len(r.get('usable_category_ids',[]))
            candidates.append((score,r))
    candidates.sort(key=lambda p:(-p[0],p[1]['image_id']));selected=[];groups=set()
    for score,r in candidates:
        if r['source_identity'] in groups:continue
        groups.add(r['source_identity']);selected.append(dict(image_id=r['image_id'],source_photo_id=r['source_identity'],official_image_url=r['proposed_secure_image_url'],original_source_page_resolver=r['proposed_source_page_resolver'],metadata_licence_id=4,metadata_licence_name=r['metadata_license_name'],metadata_licence_url=r['metadata_license_url'],native_furniture_counts=r['native_category_box_counts'],indoor_metadata_context=r['indoor_metadata_context'],rights_status='individually_unverified',creator=None,creator_group_verified=False,photo_download_authorized=False,proposed_role='additional_training_only_not_independent_evaluation',expected_image_bytes_estimate=[150000,400000],source_supported_hard_negative_count_unverified=True,annotation_source='OfficialCOCO2017instances_trainmetadata',annotation_licence='https://creativecommons.org/licenses/by/4.0/'))
        if len(selected)==64:break
    emit21(DATA21/'additional-rights-candidates-v1.json',dict(records=selected,selection='Deterministic metadata-only indoors/furniture/context ranking;notdevmodelerrors',inventory_sha256=sha(path),previously_attempted_IDs_excluded=len(attempted),no_requests_performed=True))
    emit21(REPORT21/'additional-data-proposal.json',dict(approved=False,scope='StageA rights-evidence-only audit for exact64newCOCOsourceIDs;notphotoacquisitionapproval',candidate_manifest=str((DATA21/'additional-rights-candidates-v1.json').relative_to(ROOT)),candidate_manifest_sha256=sha(DATA21/'additional-rights-candidates-v1.json'),count=len(selected),metadata_licence_as_recorded='COCOlicense4Attribution(CCBY2.0);notproof of individual photo rights',individually_verified_current_rights_count=0,previous_rate_limited_and_attempted_IDs_not_resumed=True,max_rights_response_body_bytes=32000000,max_rights_allocated_bytes=128*2**20,max_evidence_attempts_per_ID=1,min_request_interval_seconds=3,stop_on_429_respect_RetryAfter=True,source='OfficialCOCOmetadata-resolvedoriginalFlickrpages;certificateverifiedHTTPSrequired',rights_required=['Exactphotoidentity','Creator/source URL','Photo-specificcommercial-compatiblelicense/version','Attribution andevidencehash/timestamp','Creator/source-groupisolation','CurrentobjectHEADsizeandrightsavailability'],photo_bodies_authorized=False,conditional_later_photo_scope=dict(max_images=64,projected_transfer_range_bytes=[len(selected)*150000,len(selected)*400000],proposed_cumulative_body_cap=100000000,per_image_max=3145728,max_new_storage_bytes=512*2**20,exact_HEAD_budget_and_rights_approval_required_first=True,licensing_status='NOT_PHOTO_ACQUISITION_READY'),preferred_local_remedy=dict(already_local_inactive_photos=42,action='Audit/qualifytheirsource-supportedobjectsandmappingbeforeanynewdownloads;no automatictrainingpartitionactivation',new_downloads_needed=0),missing_training_evidence=['Moreprotected-overlap-freevisiblepeople/bodyparts,plants,architectureandfurniture-partChairconfusers','TrustworthyChair+seating-stoolnegativecoverageatrealscore>=0.5;currentnew5regionsallbelow0.5','Exactsourceinstanceorcompletecategoryevidence;do notuseunmatchedpredictionsaslabels'],exclusions=dict(exclusions),no_source_substitution=True))
    print('Exact rights-only proposal',len(selected),'new IDs;no verified-rights or acquisition claims',flush=True)


if __name__=='__main__':run()
