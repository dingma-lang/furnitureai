"""Offline policy/identity reconciliation and small approved metadata exports."""
import collections,csv
from common9 import *
def run():
    net=read(OUT/'network-ledger.json');circuits={p.stem:read(p) for p in (OUT/'circuits').glob('*.json')}
    emit(OUT/'network-final-ledger.json',dict(net,circuits=circuits,audit_closed=True,source_terms_or_photo_page_requests_after_denial=0,
         no_additional_requests_during_closeout=True,robots_not_copyright_evidence=True))
    records=read(OUT/'license-attribution-ledger.json')['records'];prior=set()
    for rel in manifest()['prior_rights_ledger_hashes']:
        for r in read(ROOT/rel)['records']:
            for key in ('source_identity','source_photo_id'):
                if r.get(key):prior.add(r[key])
    overlap=sorted({r['source_photo_id'] for r in records}&prior)
    if overlap:raise ValueError('Unexpected prior source-photo alias;no acquisition permitted')
    annotations=read(OUT/'local-candidate-audit.json')['records'];contexts=collections.Counter();image_contexts=collections.Counter()
    for row in annotations:contexts.update(row['native_category_counts']);image_contexts.update(row['native_category_counts'].keys())
    emit(REPORT/'identity-and-content-summary.json',dict(prior_unique_rights_source_photos=len(prior),candidate_source_photo_overlap_with_prior_rights=overlap,
         candidate_unique_image_IDs=64,candidate_unique_source_photo_IDs=64,creators_verified=0,capture_groups_verified=0,
         known_local_filename_matches=0,body_hash_or_perceptual_comparison_performed=False,
         native_context_annotation_counts=dict(contexts),native_context_image_counts=dict(image_contexts),
         all_native_boxes_valid_within_recorded_dimensions=all(a['box_valid'] for r in annotations for a in r['native_annotations']),
         native_polygon_segmentation_records=sum(a['segmentation_kind']=='polygon' for r in annotations for a in r['native_annotations']),
         visual_indoor_content_or_hard_negative_suitability_verified=False,missing_annotations_are_not_confirmed_negatives=True))
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(['image_id','source_photo_id','rights_status','source_resolver_url','metadata_photo_licence','verified_photo_licence','verified_creator','individual_source_attempted','final_partition'])
    for r in records:writer.writerow([r['image_id'],r['source_photo_id'],r['rights_status'],r['source_resolver_url'],r['metadata_photo_licence_URL'],'','',False,''])
    atomic(REPORT/'rights-decisions.csv',stream.getvalue().encode())
    emit(REPORT/'acquisition-plans.json',dict(training=read(OUT/'proposed-training-acquisition.json'),independent_evaluation=read(OUT/'proposed-independent-evaluation-acquisition.json'),
         later_verification=read(OUT/'later-annotation-verification-plan.json'),rights_audit_closed=True,photograph_downloads_authorized=False))
    req=net['requests'][0]
    emit(REPORT/'source-access-evidence.json',dict(policy_URL=req['url'],checked_utc=req['checked_utc'],HTTP_status=req['status'],audit_user_agent=UA,
        TLS_certificate_chain_verified=req['TLS_certificate_chain_verified'],TLS_version=req['TLS_version'],peer_certificate_sha256=req['peer_certificate_SHA256'],
        response_body_sha256=req['sha256'],response_file=str((OUT/req['file']).relative_to(ROOT)),
        applicable_rule='User-agent: *; Disallow: /',denied_frozen_resolver_paths=64,
        use_of_an_alternate_user_agent_or_endpoint=False,photo_specific_rights_evidence_retrieved=0,
        source_terms_fetched=False,policy_condition_not_a_copyright_finding=True,
        source_audit_response_body_bytes=net['actual_response_body_bytes'],source_audit_TLS_bytes=net['measured_TLS_sent_bytes']+net['measured_TLS_received_bytes'],
        conservative_source_transfer_charge=net['conservative_transfer_charge'],source_transfer_cap=CAP,
        transfer_measurement_scope=net['measurement'],anonymous_response_cookie_values_not_published=True))
    print('Closed policy/rights audit:64B;source alias checks0;no acquisition-ready photos',flush=True)
if __name__=='__main__':run()
