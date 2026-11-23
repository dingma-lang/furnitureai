"""Offline per-ID rights dispositions,identity audit and approval-ready plan state."""
import collections,csv,gzip,re,urllib.parse,urllib.robotparser
from common9 import *
from evidence9 import account

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only offline finalization')
    local=read(OUT/'local-candidate-audit.json');rows=local['records'];access=read(OUT/'policy-access-preflight.json');net=read(OUT/'network-ledger.json')
    request=access['robots']['request'];p=OUT/request['file'];data=p.read_bytes()
    if sha(p)!=request['sha256']:raise ValueError('Policy document changed')
    headers={k.lower():v for k,v in request['headers'].items()}
    if headers.get('content-encoding')=='gzip':data=gzip.decompress(data)
    parser=urllib.robotparser.RobotFileParser();parser.parse(data.decode('utf-8').splitlines())
    denied=[r['image_id'] for r in rows if not parser.can_fetch(UA,r['original_source_page_resolver'])]
    if len(denied)!=64 or net['candidate_evidence_attempts'] or len(net['requests'])!=1:raise ValueError('This closeout requires the actually observed64-IDpolicy denial;never invent rights outcomes')
    # Persist the policy circuit for future same-audit reads. No additional
    # requests are made,and a closed audit cannot reopen the network transport.
    if not (OUT/'circuits/flickr.com.json').exists():
        emit(OUT/'circuits/flickr.com.json',dict(reason='Robots policy denies this audit client',checked_utc=now(),
              policy_url=request['url'],policy_sha256=request['sha256'],Retry_After=None,not_before=None,
              retry_in_this_phase=False,new_permission_required=True,no_user_agent_rotation=True))
    known=read(OUT/'historical-exclusion-registry.json');wanted={r['source_photo_id']:r for r in rows}
    metadata_matches=collections.defaultdict(list)
    # Recover historical canonical source/profile claims from existing OI
    # metadata only. These claims never become current photo-rights verification.
    for split in ('validation','test'):
        path=ROOT/f'data/open-images/metadata/phase5-v1/{split}-images-with-rotation.csv'
        with path.open(newline='') as f:
            for oi in csv.DictReader(f):
                u=urllib.parse.urlsplit(oi.get('OriginalLandingURL',''));m=re.fullmatch(r'/photos/[^/]+/(\d+)/?',u.path)
                if u.hostname not in ('flickr.com','www.flickr.com') or not m:continue
                sid='flickr-photo:'+m[1]
                if sid not in wanted:continue
                metadata_matches[sid].append(dict(dataset='Open Images',metadata_split=split,image_id=oi['ImageID'],
                     canonical_source_claim=oi.get('OriginalLandingURL'),author_profile_claim=oi.get('AuthorProfileURL'),
                     author_claim=oi.get('Author'),license_claim=oi.get('License'),original_MD5_claim=oi.get('OriginalMD5'),
                     current_ownership_or_license_verified=False))
    records=[];identities=[]
    for row in rows:
        matches=metadata_matches[row['source_photo_id']];aliases=sorted({x for m in matches for x in (account(m.get('author_profile_claim','')),account(m.get('canonical_source_claim',''))) if x})
        group_overlap=sorted(set(aliases)&set(known['source_groups']));source_overlap=[m['image_id'] for m in matches if m['image_id'] in set(known['image_ids'])]
        flags=[]
        if group_overlap:flags.append('metadata_creator_alias_overlaps_historical_sources_unverified')
        if source_overlap:flags.append('source_photo_metadata_matches_historical_project_image')
        if row['local_filename_matches']:flags.append('local_filename_collision_unresolved')
        canonical=next((m['canonical_source_claim'] for m in matches if m.get('canonical_source_claim')),None)
        obj_counts=row['native_category_counts'];support={c:dict(boxes=row['native_furniture_counts'].get(c,0),segmentation_annotations=sum(a['original_annotation']['id'] in row['native_annotation_ids'] and a['native_label']=={'sofa':'couch','table':'dining table'}.get(c,c) and a['segmentation_kind']!='missing' for a in row['native_annotations'])) for c in CLASSES}
        rec=dict(image_id=row['image_id'],original_dataset='COCO2017',native_image_id=row['native_image']['id'],source_photo_id=row['source_photo_id'],original_host='Flickr',
            original_hosting_asset_url=row['native_image'].get('flickr_url'),source_resolver_url=row['original_source_page_resolver'],canonical_source_URL=canonical,
            canonical_source_provenance='historical existing OI metadata only' if canonical else 'unresolved;no source page retrieved',
            official_image_URL=row['official_image_url'],width=row['width'],height=row['height'],
            rights_status=STATUS_B,rights_reason='Source-page evidence not requested:current provider robots policy denies this audit client. Metadata CC BYclaim alone is insufficient.',
            audited_utc=now(),rights_verified_utc=None,individual_source_attempted=False,individual_evidence_requests=0,
            photographer_or_rights_holder_verified=None,metadata_creator_aliases=aliases,metadata_source_matches=matches,
            photograph_license_verified=None,metadata_photo_licence_id=row['metadata_licence_id'],metadata_photo_licence_URL=row['metadata_licence_url'],
            annotation_licence_URL=row['annotation_licence'],annotation_creator='COCO Consortium',native_annotation_IDs=row['native_annotation_ids'],native_categories=obj_counts,
            furniture_annotation_support=support,annotation_source=local['source'],annotation_source_sha256=local['source_sha256'],
            hosting_terms_verified=False,hosting_terms_reason='Automated source access was denied before terms or individual source pages were requested',
            proposed_training_permission_verified=False,attribution_conditions=None,
            attribution_status='Creator,title,current licence and applicable notices must be obtained from acceptable photograph-specific evidence before acquisition',
            licence_evidence_URL=None,licence_evidence_file=None,licence_evidence_sha256=None,
            access_evidence=dict(URL=request['url'],file=str((OUT/request['file']).relative_to(ROOT)),sha256=request['sha256'],checked_utc=request['checked_utc'],
                 HTTP_status=request['status'],TLS_certificate_chain_verified=True,audit_user_agent=UA,decision='Robots check denies this client;not photo copyright evidence'),
            source_group_identity='unverified',capture_group_identity='unverified',known_local_availability=row['known_local_availability'],local_filename_matches=row['local_filename_matches'],
            identity_flags=flags,metadata_creator_group_overlaps=group_overlap,exact_known_source_overlaps=source_overlap,
            actual_photo_sha256=None,visual_quality_verified=False,perceptual_duplicates_checked=False,Chair_absence_inferred=False,
            native_labels_are_positives_only=True,new_training_labels=0,final_partition=None,
            potential_role='training_candidate_only_unassigned',eligible_for_photograph_acquisition=False,
            estimated_image_bytes=row['expected_image_bytes_estimate'],HEAD_verified_image_bytes=None)
        emit(OUT/'records'/(str(row['native_image']['id']).zfill(12)+'.json'),rec);records.append(rec)
        identities.append(dict(image_id=rec['image_id'],source_photo_id=rec['source_photo_id'],canonical_source_URL=canonical,metadata_creator_aliases=aliases,
             known_project_source_overlap=source_overlap,known_creator_group_overlap=group_overlap,local_filename_matches=row['local_filename_matches'],
             flags=flags,group_verified=False,actual_image_duplicates_screened=False,partition_assigned=False))
    emit(OUT/'license-attribution-ledger.json',dict(records=records,primary_status_counts={STATUS_A:0,STATUS_B:64,STATUS_C:0},
          local_metadata_audited=64,individual_photo_rights_sources_attempted=0,policy_requests=1,
          individual_rights_verified=0,unsuitable_not_inferred_from_access_block=True,hosting_access_policy_not_copyright_license=True))
    emit(OUT/'identity-audit.json',dict(records=identities,unique_candidate_image_IDs=64,unique_source_photo_IDs=64,
          exact_prior_photo_ID_overlaps=0,prior_attempted_or_paused_ID_overlaps=0,previous844excluded=True,
          metadata_photo_match_candidates=sum(bool(r['metadata_source_matches']) for r in records),metadata_creator_overlap_candidates=sum(bool(r['metadata_creator_group_overlaps']) for r in records),
          local_filename_matches=sum(bool(r['local_filename_matches']) for r in records),known_intercandidate_creator_alias_groups='Any recovered historical aliases remain unverified and must be grouped before future selection',
          photograph_SHA_or_perceptual_screening_performed=False,physical_scene_or_object_independence_verified=False,partitions_assigned=0))
    emit(OUT/'proposed-training-acquisition.json',dict(records=[],rights_supported_count=0,ready_for_download_approval=False,
          reason='No photograph-specific rights evidence obtainable through this permitted audit path',training_partitions_not_assigned=True,
          required_before_future_approval=['Provider-permitted metadata/API access or acceptable creator-supplied photo-specific rights evidence','Exact ID/source/creator/license/conditions and grouping verification','HEAD sizes for only rights-supported objects','Frozen exact survivor manifest,byte/storage/retry caps and separate explicit photo acquisition approval'],
          previous64IDimage_transfer_estimate=[9600000,25600000],estimate_is_not_a_download_approval=True))
    emit(OUT/'proposed-independent-evaluation-acquisition.json',dict(records=[],eligible_independent_seven_category_photos=0,
          historical_training_development_exposed_reserved_photos_reused=0,scope='Separate future collection of new authorized real rooms;first batch target20,not an acquired or approved count',
          require=['Source/creator/capture disjointness from all training candidates,includingthese64','Independent acceptable photo rights and attribution','Blind full7category furniture/stool and whole-object annotation with explicit completeness/absence decisions','Second review and frozen image/annotation/protocol hashes before any scoring'],
          automatic_partition_assignment=False,acquisition_ready=False,separate_exact_collection_manifest_and_approval_required=True))
    emit(OUT/'later-annotation-verification-plan.json',dict(approved_for_execution=False,requires_separate_photo_acquisition_and_inference_approval=True,
          workflow=['Reverify exact source evidence and rights before acquisition;TLS/size/hash/decoding checks after approval','Display native boxes,masks,IDs and whole-photo/crop context without inventing missing annotations',
                    'Identify Chair plus ordinary seating/bar stools;protect unverified furniture,footstools/ottomans/step-stool ambiguity','Preserve COCO/LVIS/OI native completeness and ignored/crowd/occlusion information',
                    'Propose visible-instance interior regions automatically;source polygons plus actual visual evidence,full containment and protected overlap checks remain necessary',
                    'Keep assistant/model proposals separate from real human confirmation;no mandatory manual rectangle drawing',
                    'Request a separately bounded model-query cache run if needed,then verify actual high-score negative coverage before training','Keep independent evaluation out of negative mining and model selection'],
          labels_created_now=0,GPU_forwards_now=0))
    distribution={c:dict(photographs=sum(r['furniture_annotation_support'][c]['boxes']>0 for r in records),boxes=sum(r['furniture_annotation_support'][c]['boxes'] for r in records),
               mask_annotations=sum(r['furniture_annotation_support'][c]['segmentation_annotations'] for r in records),rights_supported_photographs=0) for c in CLASSES}
    summary=dict(status='COMPLETE_LOCAL_AUDIT_REMOTE_RIGHTS_BLOCKED',metadata_IDs_audited=64,individual_rights_IDs_attempted=0,rights_supported=0,unresolved=64,unsuitable=0,
          policy_document_requests=1,photo_object_HEADs=0,photograph_bodies=0,source_terms_or_licence_page_downloads_after_robots_denial=0,
          category_distribution=distribution,all_native_annotations=786,selected_furniture_boxes=268,
          source_access_reason='Flickrrobots default User-agent:* Disallow:/ applies to this honestly identified audit client;no alternate identity,API or source was used',
          measured_response_body_bytes=net['actual_response_body_bytes'],measured_TLS_sent_bytes=net['measured_TLS_sent_bytes'],measured_TLS_received_bytes=net['measured_TLS_received_bytes'],
          measured_TLS_total_bytes=net['measured_TLS_sent_bytes']+net['measured_TLS_received_bytes'],conservative_transfer_charge=net['conservative_transfer_charge'],transfer_cap=CAP,
          policy_URL=request['url'],policy_document_sha256=request['sha256'],new_training_labels=0,partition_assignments=0,GPU_forwards=0,
          training_acquisition_list_count=0,independent_evaluation_list_count=0,
          next_approval='No photograph-acquisition approval is ready. A future rights-only continuation first requires provider-permitted access or acceptable photograph-specific creator evidence;then freeze exact rights-supported IDs and resource budget for separate photo approval.')
    emit(REPORT/'audit-summary.json',summary)
    # Small per-ID metadata is expressly authorized for Git;raw provider responses
    # and full native annotation bodies stay under the ignored local data root.
    emit(REPORT/'rights-by-ID.json',dict(records=[{k:r[k] for k in ('image_id','source_photo_id','source_resolver_url','canonical_source_URL','rights_status','rights_reason','audited_utc','rights_verified_utc','photographer_or_rights_holder_verified','metadata_photo_licence_id','metadata_photo_licence_URL','photograph_license_verified','annotation_licence_URL','width','height','metadata_creator_aliases','source_group_identity','known_local_availability','identity_flags','licence_evidence_URL','access_evidence')} for r in records],
         individual_licence_evidence_unavailable=True,no_human_or_licence_decisions_invented=True))
    emit(OUT/'audit-complete.json',dict(status='CLOSED_PROVIDER_POLICY_STOP',summary_sha256=sha(REPORT/'audit-summary.json'),closed_utc=now(),additional_network_requests_authorized=False,requests=1,photograph_bodies=0))
    print('Rights audit closed:64locally audited,0individual-source attempts,0A/64B/0C;no photo acquisition list',flush=True)
if __name__=='__main__':run()
