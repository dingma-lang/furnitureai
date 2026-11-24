"""Offline manifest/gate construction; this never turns metadata into training labels."""
import collections,re
from common10 import *
from discover10 import decoded_receipt
from net10 import Client
from decisions10 import category_hints
SELECTED=['efqgz854','bpxzpgg7','x6v3kwb7','y4n8bmat','vugqdsr7','mhnzfe6b','mqxh737p']

def run():
    data=read(OUT/'discovered-candidates-v2.json');rows=data['records'];lookup={r['source_image_id']:r for r in rows}
    if len(rows)!=78 or data['conservative_candidate_record_count']>200:raise ValueError('Actual bounded inventory changed; review before closeout')
    old=ROOT/'data/coco/rights-audits/phase10e8-v1/historical-exclusion-registry.json';registry=read(old)
    prior=set(registry['image_ids'])|set(registry['source_photo_ids'])|set(registry['source_groups'])|set(registry['previous_attempted_or_paused_ids'])
    matches=[]
    for r in rows:
        vals={r['candidate_id'],r['source_image_id'],r['source_work_id'],r.get('original_image_id'),r.get('canonical_source_url'),r.get('image_metadata_URL')}-{None}
        if vals&prior:matches.append(dict(id=r['candidate_id'],matches=sorted(vals&prior)))
    # Also check exact credits/declared contributor labels against previous rights ledgers.
    ledgers=sorted(p for p in (ROOT/'data').rglob('*.json') if 'rights-discovery' not in str(p) and ('license-and-attribution-ledger' in p.name or 'license-attribution-ledger' in p.name or 'license-and-provenance-ledger' in p.name))
    names=set();urls=set();sources={}
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if isinstance(v,str):
                    if 'creator' in k.lower() or 'photographer' in k.lower():names.add(v.casefold().strip())
                    if v.startswith(('https://','http://')):urls.add(v.rstrip('/').casefold())
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    for p in ledgers:
        if p.stat().st_size>16*2**20:raise ValueError('Unexpected oversized historical rights ledger')
        collect(read(p));sources[str(p.relative_to(ROOT))]=sha(p)
    name_matches=[]
    for r in rows:
        hit=[c['name'] for c in r['declared_contributors'] if c.get('name','').casefold().strip() in names]
        if hit:name_matches.append(dict(id=r['candidate_id'],matching_names=hit))
        if any(str(r.get(k,'')).rstrip('/').casefold() in urls for k in ('canonical_source_url','image_metadata_URL')):matches.append(dict(id=r['candidate_id'],matches=['historical URL']))
    exact=collections.defaultdict(list);similar=collections.defaultdict(list)
    for r in rows:
        exact[r['image_metadata_URL']].append(r['candidate_id'])
        title=re.sub(r'\W+',' ',r['metadata_title'].casefold()).strip();similar[(title,r['source_group_id'])].append(r['candidate_id'])
    duplicate_urls=[v for v in exact.values() if len(v)>1];similar_titles=[v for v in similar.values() if len(v)>1]
    identity=dict(historical_registry=str(old.relative_to(ROOT)),historical_registry_sha256=sha(old),previous_registry_counts={k:len(registry[k]) for k in ('image_ids','source_photo_ids','source_groups','previous_attempted_or_paused_ids')},
       historical_licence_ledgers=sources,known_ID_URL_matches=matches,known_contributor_name_matches=name_matches,
       exact_image_metadata_URL_duplicate_groups=duplicate_urls,same_title_contributor_possible_reprints=similar_titles,
       declared_group_buckets=len({r['source_group_id'] for r in rows}),named_contributor_buckets=len({r['source_group_id'] for r in rows if r['declared_contributors']}),
       unknown_creator_photographs=sum(not r['declared_contributors'] for r in rows),verified_independent_creator_or_capture_groups=0,
       pixel_hash_and_perceptual_checks_performed=False,source_reuploads_unresolved=True,group_ids_not_proof_of_independent_physical_rooms=True)
    emit(REPORT/'identity-audit.json',identity)
    if matches or name_matches or duplicate_urls:raise ValueError('Known overlap needs resolution before acquisition proposal')
    workraw=decoded_receipt(read(OUT/'requests/022.json'))['results'];workrows=[]
    for w in workraw:
        locations=[l for it in w.get('items',[]) for l in it.get('locations',[]) if l.get('type')=='DigitalLocation']
        workrows.append(dict(candidate_id='wellcome-work:'+w['id'],title=w['title'],source_url='https://wellcomecollection.org/works/'+w['id'],
          rights_status='B_unresolved',reason='Work-level discovery only; no exact photograph/canvas, attribution or sufficient reuse grant verified',
          physical_description=w.get('physicalDescription'),digital_locations=locations,source_specific_restriction_text_preserved=True,
          evidence=dict(url=read(OUT/'requests/022.json')['url'],sha256=read(OUT/'requests/022.json')['sha256']),final_partition=None))
    emit(REPORT/'work-level-unresolved.json',dict(records=workrows,rights_supported=0,unresolved=len(workrows),unsuitable=0,not_added_to_image_manifest=True))
    training=[]
    for id in SELECTED:
        r=lookup[id];file=OUT/'IIIF-info'/(r['original_image_id']+'.json');info=read(file)
        if r['rights_status']!='A_rights_supported' or r['image_metadata_URL']!=info['evidence']['url']:raise ValueError('Per-image rights/identity mismatch')
        if not info['info']['@id'].startswith('https://iiif.wellcomecollection.org/image/'):raise ValueError('Unsafe image service')
        proposed=info['info']['@id']+'/full/!1600,1600/0/default.jpg'
        training.append(dict(r,exact_download_URL=proposed,width=info['info']['width'],height=info['info']['height'],IIIF_info_sha256=sha(file),
            proposal_purpose='training-side visual and annotation-feasibility inspection only',final_partition=None,
            maximum_bytes_per_object=512000,expected_bytes=None,object_size_not_known_until_future_acquisition=True,
            all_training_and_hard_negative_eligibility_checks_pending=True,privacy_and_sensitive_context_review_required=True))
    common=dict(version='phase10e10-v1',photograph_acquisition_authorized=False,no_training_labels_created=True)
    emit(REPORT/'potential-training-acquisition.json',dict(common,records=training,count=len(training),scope='Seven source-supported room-scene photographs for inspection, not approved training data',
       source_group_count=1,verified_independent_groups=0,all_unknown_creators_grouped_together=True,additional_rights_supported_images_not_proposed=sum(r['rights_status']=='A_rights_supported' for r in rows)-len(training)))
    emit(REPORT/'potential-independent-evaluation-acquisition.json',dict(common,records=[],count=0,reason='No source-disjoint, modern, seven-category, fully annotated room pool established from metadata',
       future_protocol=dict(target_initial_rooms=20,minimum_independent_contributor_groups=5,counts_are_planning_targets_not_verified=True,
         exclude_all_current_training_and_exposed_sources=True,exclude_current_seven_training_side_candidates=True,
         source_creator_capture_and_room_groups_disjoint=True,all_seven_category_positive_and_negative_completeness_review=True,
         Chair_taxonomy='Ordinary chairs, seating stools and bar stools; ottomans/footstools/step stools separate',
         two_pass_annotation_and_blind_review=True,freeze_manifest_hashes_before_any_model_scoring=True,no_hyperparameter_selection=True)))
    emit(REPORT/'unresolved-rights-candidates.json',dict(common,records=workrows+[r for r in rows if r['rights_status']=='B_unresolved'],count=len(workrows)+sum(r['rights_status']=='B_unresolved' for r in rows),preserved_Flickr_records_path='reports/phase10e9-v1/rights-by-ID.json',preserved_Flickr_status_counts=dict(A=0,B=64,C=0),no_historical_rights_updates=True))
    emit(REPORT/'unsuitable-restricted-candidates.json',dict(common,records=[],count=0,note='Unverified rights not reclassified as prohibited. Restricted work-level terms remain unresolved where their exact scope/creator is unknown; no such images proposed.'))
    emit(REPORT/'rights-supported-image-inventory.json',dict(common,records=[r for r in rows if r['rights_status']=='A_rights_supported'],count=sum(r['rights_status']=='A_rights_supported' for r in rows),all_claims_are_photo_specific_catalogue_declarations_not_absolute_rights_warranties=True))
    coverage=collections.Counter(c for r in rows for c in r['category_hints']);counts=collections.Counter(r['rights_status'] for r in rows)
    summary=dict(status='METADATA_DISCOVERY_COMPLETE_NO_MODEL_EXECUTION',parent='9835343ee1beeb2c96e7f739143c02d5d35bd6bb',new_unique_image_metadata_records=len(rows),initial_work_metadata_records=len(workrows),
       conservative_new_record_count=len(rows)+len(workrows),new_record_limit=200,image_rights_supported=counts['A_rights_supported'],image_rights_unresolved=counts['B_unresolved'],image_rights_unsuitable=counts['C_unsuitable'],
       total_new_metadata_status_counts=dict(A=counts['A_rights_supported'],B=len(workrows)+counts['B_unresolved'],C=counts['C_unsuitable']),
       prior64_Flickr_counts=dict(A=0,B=64,C=0),provisional_source_group_buckets=identity['declared_group_buckets'],verified_independent_groups=0,
       categories_with_metadata_hints={k:coverage[k] for k in ('Sofa','Chair','Table','Bed','Desk','Cabinet','Lamp')},
       Chair_hints_are_restraint_chair_and_sedan_chair_not_accepted_target_labels=True,
       Bed_hints_from_bedroom_context_not_verified_objects=True,Cabinet_hints_have_whole_object_vs_front_ambiguity=True,
       available_native_detection_boxes=0,available_native_instance_masks=0,visually_verified_room_photos=0,verified_hard_negatives=0,
       training_side_inspection_proposal=len(training),independent_evaluation_proposal=0,meaningful_training_data_improvement_established=False,
       source_metadata_has_historical_medical_or_unrelated_content=True,
       final_partition_assignments=0,photo_downloads=0,thumbnail_or_preview_downloads=0,GPU_calls=0,new_training_labels=0,
       network=Client().summary(),allocated_bytes_before_final_reports=allocated(),storage_cap=DISK)
    emit(REPORT/'discovery-summary.json',summary)
    emit(REPORT/'next-photo-approval-request.json',dict(common,approval_requested='Seven-photograph training-side inspection pilot only, not training or evaluation',
       manifest='reports/phase10e10-v1/potential-training-acquisition.json',manifest_SHA256=sha(REPORT/'potential-training-acquisition.json'),exact_ids=SELECTED,
       destination='data/licensed-rooms/image-pilots/phase10e11-wellcome-v1/',maximum_photographs=7,max_dimension=1600,
       maximum_bytes_per_image=512000,maximum_attempts_per_image=2,maximum_cumulative_transfer=8000000,maximum_new_allocated_storage=64*2**20,
       expected_transfer=None,expected_size_reason='IIIF metadata exposes dimensions but not encoded JPEG byte lengths; no image HEAD or body requested',
       maximum_single_successful_copy_payload=7*512000,
       preserve_creator_unknown_status=True,attribution='Wellcome Collection, linked source work and CC BY4.0; identify resized derivatives and preserve any declared creator credit',
       no_independent_evaluation_membership=True,no_GPU_or_model_inference_authorized=True,no_training_or_certified_negatives=True,
       stops=['Changed licence or access restrictions','Conflicting source or identity evidence','Image integrity failure','Resource limit','Existing destination'],
       quarantine_instead_of_replace=True))
    emit(REPORT/'later-verification-workflow.json',dict(
       acquisition_gate='Separate exact-scope approval and current provider/licence recheck; HTTPS, metered caps, no new IDs or redirects',
       image_checks=['SHA256/decode/dimensions/orientation','Exact/perceptual duplicate and reprint detection','Creator/room/capture identity against all historical pools','Room relevance, privacy/sensitive content, quality and domain gap'],
       annotation_checks=['No native boxes/masks available in discovered pool; preserve this absence','Propose all seven categories with future separately authorized annotation tooling','Protect ordinary chairs and seating/bar stools; preserve other stool/ottoman classes','Leave uncertain categories, fragments, mixed boxes, occluded furniture and unreviewed areas ignored','Completeness requires affirmative evidence; missing annotations are never negatives'],
       model_query_coverage='Future separately approved inference only; match actual query boxes to evidence-backed regions with protected-positive checks',
       evidence_types=['source-supported','automatic-unverified','human-confirmed-only-after-real-review'],manual_rectangles_mandatory=False,
       independent_evaluation='Separate source/creator/capture groups, complete seven-class independent annotations, blind review, frozen membership and scoring once; no tuning',
       training_gate='Do not propose GPU training until a materially larger and more diverse reliably supervised real-room pool passes integrity/completeness checks',
       preferred_next_development='Recruit genuinely independent modern room photographs with explicit photographer reuse/annotation/ML consent; collect room/capture/source identity, privacy and commercial-use conditions before transfer'))
    emit(OUT/'discovery-closed.json',dict(closed_UTC=now(),network=summary['network'],photo_bodies=0,no_further_requests=True))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
