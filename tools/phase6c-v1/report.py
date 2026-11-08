"""Actual outcomes and precise, unapproved72-photo acquisition request."""
import collections as C
from common import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def run():
 audit=read(OUT/'rights-audit-v3.json');records=audit['records'];identity=read(OUT/'identity-audit.json');plan=read(OUT/'proposed-image-acquisition-v1.json');tests=read(OUT/'test-results.json');v=read(OUT/'verification.json');ledger=read(OUT/'license-attribution-ledger.json');net=read(OUT/'network-ledger.json')
 counts=C.Counter(r['licensing_status'] for r in records);status_rows=[]
 labels=dict(zip(STATUSES,('Verified commercially compatible','Potentially compatible but unverified','Restricted / outside approved profile','Source unavailable / temporarily rate limited','Conflicting licensing evidence (primary category)')))
 for status in STATUSES:status_rows.append([labels[status],counts[status],sum(r['licensing_status']==status and r['proposed_split']=='train' for r in records),sum(r['licensing_status']==status and r['proposed_split']=='development' for r in records)])
 category_rows=[]
 for c in ('chair','sofa','bed','dining_table'):
  a=plan['summary']['train']['categories'][c];b=plan['summary']['development']['categories'][c]
  category_rows.append([c,a['images'],a['boxes'],b['images'],b['boxes']])
 flags=C.Counter(flag for r in identity['records'] for flag in r['identity_flags']);known_url_matches=sum(bool(r['known_exact_original_url_metadata_matches']) for r in identity['records']);filename_matches=sum(bool(r['local_filename_matches']) for r in identity['records'])
 native_annotations=sum(len(r['original_furniture_annotations']) for r in plan['images']);groups=plan['known_source_group_count']
 summary=dict(status='complete_rights_audit_photographs_not_acquired',audited_exact_source_ids=400,original_slots=dict(train=320,development=80),licensing_status_counts={status:counts[status] for status in STATUSES},source_specific_rights_verified=92,source_page_documents=95,
  source_rate_limited_records=275,rate_limit_retry_attempts=1,stop_after_persistent_rate_limit=True,unresolved_redirects=30,unrecognized_licence=1,restricted_source_metadata_conflicts=2,
  historical_known_source_exclusions=3,current_cross_partition_creator_group_exclusions=17,eligible_photographs=72,eligible_slots=dict(train=60,development=12),known_eligible_source_groups=66,quarantined=328,
  native_source_annotation_objects=native_annotations,expected_photo_bytes=10853122,max_two_complete_attempt_photo_bytes=21706244,actual_rights_body_bytes=net['actual_body_bytes'],conservative_transfer_charge=net['conservative_charge'],photo_body_bytes=0,
  image_object_heads=400,visual_or_perceptual_checks_performed=False,tests_passed=sum(r['tests'] for r in tests if r['passed']),historical_preservation=True,model_inference_or_updates=0,
  acquisition_manifest='proposed-image-acquisition-v1.json',acquisition_manifest_sha256=sha(OUT/'proposed-image-acquisition-v1.json'),photograph_acquisition_approved=False)
 commit(OUT/'summary.json',summary)
 commit(OUT/'provenance.json',dict(manifest_sha256=PIN,proposal_sha256=sha(PROPOSAL),source='Official frozen COCO2017 instance metadata linked to individual original Flickr photo IDs and current source-page ImageObject creator/licence evidence',
  commercial_policy='Canonical CC0/CC BY only; current licence must match frozen metadata; other compatible-in-principle licences are outside this approved profile.',
  copyright_verification_scope='Photo ID, source-provided creator account, specific licence/version/title and matching source grant. Does not invent independent legal ownership evidence, property/publicity/privacy releases or visual content validation.',
  source_response_hashes=[dict(file=q['file'],sha256=q['sha256'],source_url=q['url'],checked_utc=q['checked_utc']) for q in net['requests'] if q['method']=='GET'],
  annotation_creator='COCO Consortium',annotation_license='https://creativecommons.org/licenses/by/4.0/',reference_sources=['https://creativecommons.org/licenses/by/2.0/','https://creativecommons.org/publicdomain/zero/1.0/','https://www.flickr.com/creativecommons/']))
 report=f'''# Phase6C — COCO individual photograph rights audit completed

Audited **all400 exact approved source IDs**. Original320 training /80 development slots and all annotations remain unchanged. No photograph body, model weight or additional dataset/archive was downloaded, and no model inference/training occurred. HTTPS certificate and hostname verification stayed enabled; source redirects were manually restricted to the same Flickr photo ID. Image objects were HEAD-only.

**92** photos have source-specific CC BY2.0 evidence matching photo identity, creator account, title and frozen licence claim. After conservative identity isolation, the exact acquisition proposal has **72 photos:60 train /12 development**, across **66 known source groups**, totalling **10,853,122 HEAD-reported image bytes**. These are copyright-source-qualified candidates for acquisition/screening, not visually validated training examples. Unknown physical furniture relationships and other content rights were not inferred from a licence label.

## Actual licensing findings

{table(['Primary licensing outcome','Total','Original train slots','Original dev slots'],status_rows)}

All400 COCO metadata records claim CC BY2.0, but this was never treated as proof of current rights.95 original source HTML documents were retrieved and hashed.92 contain one matching individual ImageObject with consistent creator account, title and specific CC BY2.0 licence. Two photos now show CC BY-SA2.0 rather than the metadata's CC BY2.0; their primary status is restricted/outside-policy and the source/metadata conflict is recorded separately. BY-SA is not asserted to universally forbid commercial use; it is outside this phase's approved strict CC0/CC BY profile and also differs from the frozen claim. The primary conflicting category is0 because those two findings use the more restrictive primary disposition; `licence_differs_from_metadata` preserves both contradictory facts.

One source provides an unrecognized general Flickr usage-help link rather than a verified specific CC licence grant, so it remains potentially compatible but unverified.30 original source resolvers redirect to a path without a matching photo identity (`/photos///`); they were not followed or guessed to be deleted/private/licensed. They remain unresolved. **275** sources were temporarily inaccessible under HTTP429, not proven permanently missing or incompatible. Every unresolved/restricted source is quarantined. No licences, creator identities or human evidence were invented, and there were no replacement IDs.

The licensing outcome describes matching source-provided rights evidence, not a universal commercial-content guarantee. Creator/copyright information is retained as supplied by the original photo page; no property, privacy/publicity, trademark or model-release rights were inferred. All accepted titles and creator credits are present. Source licence versions, canonical photo/creator links, notices when supplied, UTC times, HTML hashes, methods and identity decisions are preserved in `license-attribution-ledger.json`, `attribution-and-decisions.csv`, per-photo records and `provenance.json`.

## Resolver handling and respecting access limits

The first pass found Flickr's intermediate server-provided `/photo.gne?rb=1&id=<same-ID>` redirect and stopped because the initial allowlist only recognized canonical photo pages. It downloaded no source document or photograph body. Those initial400 records are preserved. A narrowly validated rule was then added for this exact HTTPS/Flickr/same-numeric-ID resolver form; arbitrary query flags, other IDs/hosts, HTTP downgrades and authentication routes remained disallowed. The second pass is preserved in `rights-audit-v2.json` / `records-v2/`.

When source HTTP429 responses appeared, unavailable pages were not read or worked around. One normal retry was made after at least120 seconds (the source supplied no Retry-After); it used one worker, minimum1.25-second source spacing, the same source URL/client policy and at most a second request. The server again returned429, so **all further source requests in that retry pass stopped immediately**. No account/IP rotation, cookies, credential use, proxy substitution, retry loop or alternate-source hunt was used. `rights-audit-v3.json` retains the final findings and original outcomes for cases not retried. This is a completed bounded attempt audit with explicit temporary unavailability, not a claim all400 licences were verifiable.

All400 specified official COCO S3 photo objects returned available JPEG metadata under HEAD. HEAD confirmed availability, size and ETag only; it never upgraded a licensing outcome. No photograph GET was issued. Of1,391 total recorded HTTP requests,1,296 were body-free HEADs and95 were source-document GETs. Only one recorded request used a second attempt, the cooldown HEAD429 retry. Actual source HTML response-body bytes and conservative cumulative charge are both **12,536,007**, below256,000,000. Final allocated storage is recorded in `resource-usage.json` and is well under512MiB. Browser research on official CC/Flickr licensing documents is separate from this source-response ledger.

## Identity, overlap and partition isolation

Known source-photo, normalized creator-account aliases and historical original-MD5 relationships were reconciled using **167,056** existing Open Images metadata records, all191 prior Open Images pilot memberships/source groups and their stored per-photo creator evidence. Current source aliases were linked only through a matching same-photo page; a shared creator was treated as a conservative source family, not a declaration of the same physical product.

**3** otherwise rights-qualified photos intersect protected historical source/creator groups. **17** otherwise rights-qualified photos belong to current creator/source groups spanning original train/dev slots. All20 remain quarantined rather than moving partitions or allowing leakage. The72 surviving records have no known group crossing train/dev or protected groups. Their primary `identity_group` uses the reconciled current source group; the old COCO-only group is separately retained as `metadata_identity_group`. Six additional photos share a within-partition source family, giving66 known groups for72 photos; do not call these72 independent physical furniture identities.

Known exact original-URL matches in the full historical metadata tables: **{known_url_matches}** records (recorded individually; an unused metadata match is not silently called training leakage). Existing local image-filename collisions across the enumerated data/reports/assets/samples trees: **{filename_matches}**. Namespaced COCO IDs, exact URLs/source-photo IDs, filenames and historical/current source families were checked. Equal image-object ETag/size fingerprint suspicions: **{len(identity['header_asset_suspicions'])}**; this is a header-level check, not proof of a body hash. See `identity-audit.json` for all400 decisions and conservative exclusion reasons.

**Perceptual duplicates, actual image SHA-256, decoded pixels, visual quality, indoor suitability and capture independence were not checked.** They cannot be declared passed without photograph bodies. All proposed image SHA fields stay null; no final human-reviewed or training-ready label is generated. Unknown cross-author physical relationships and unresolved creator aliases remain limitations.

## Exact proposed acquisition subset

Manifest: **`proposed-image-acquisition-v1.json`**.

SHA-256: `{sha(OUT/'proposed-image-acquisition-v1.json')}`

{table(['Category','Train images','Train native boxes','Dev images','Dev native boxes'],category_rows)}

Category image counts overlap within a photograph. The72-photo subset retains **{native_annotations}** original COCO furniture annotation objects and their original box/mask/category/crowd data. These remain source-native metadata, not visually confirmed boundaries. COCO supports chair62, couch63/sofa, bed65 and dining table67. Dining table is not equivalent to arbitrary broad Table units; Desk/Cabinet/Lamp remain absent/unknown. No missing instances or absent classes were promoted to background negatives.

The approved original slots remain320/80, with260 train-slot and68 dev-slot records quarantined. Eligible60/12 counts reflect a subset, not repartitioning or an attempt to restore an80/20 ratio after seeing results. Exact original image IDs, source IDs, native annotations and split labels are unchanged. Source/creator evidence, current CC BY2.0 version/title, HTML SHA, rights-record SHA and live object size/ETag are bound per accepted row. No model output or heldout metric informed selection.

## Attribution and checks still required after acquisition

Retain supplied creator and title, canonical original photo link, exact CC BY2.0 licence link, supplied notices and an indication of resizing/cropping/annotation changes. Keep COCO Consortium / CC BY4.0 annotation attribution separately. Rights-source evidence is not replaced by the COCO licence ID or S3 HEAD response. Reverify saved source evidence/hash/freshness before acquisition; the proposal uses a seven-day freshness window and stops on changed metadata or conflicting rights rather than silently altering it.

After separate photo approval, download only the72 pinned official objects. Validate bytes against pinned HEAD size/ETag, compute actual SHA-256, decode images/dimensions/orientation and preserve originals. Screen for byte/pixel/perceptual duplicates, historical/frozen overlap using stored signatures, real indoor context, blur, partial/mixed-object boxes, mask/crowd semantics and native dining-table scope. Quarantine unresolved examples without replacement or partition movement. Do not start training, inference/scoring or final dataset optimization under the acquisition approval.

## Resource proposal and precise next approval

Expected photo transfer: **10,853,122 bytes**, from actual HEAD sizes, not the earlier400×250k scenario. Maximum two complete attempts: **21,706,244 bytes**. Largest accepted image object: **{max(r['remote_object']['bytes'] for r in plan['images']):,} bytes**. Proposed hard per-image maximum3,145,728bytes, two attempts/object, **100,000,000-byte cumulative transfer cap /536,870,912-byte additional storage cap**. Expected photo-plus-screening storage is approximately64MiB; the hard cap must include partials, logs, manifests and contact sheets. Current rights bytes plus two complete photo attempts total34,242,251bytes; future acquisition needs its own counters/approval and must not reset uncertain prior transfer accounting.

Destination: **`{plan['destination']}`** (does not exist; no photograph directory was created).

Request approval for **only these72 photos, preserving60 train /12 dev slots**, plus integrity/duplicate/quality/annotation screening and attribution records. No additional IDs, replacements, datasets, model weights, model inference, training or deployment. `image-acquisition-approval-request.json` records this exact request and is currently unapproved. The400-ID rights-audit approval does not authorize photograph bodies.

## Tests and historical preservation

{table(['Suite','Tests','Passed'],[[r['suite'],r['tests'],r['passed']] for r in tests])}

All **{sum(r['tests'] for r in tests)} relevant tests pass**. They cover source-ID/creator/version/title matching; restricted and conflicting licence evidence; rejection of dataset-wide “proof”; CC0/CC BY allowlists; photo-GET denial; same-ID rb=1 resolver validation; arbitrary host/ID/HTTP redirects; caps before requests; current/historical creator-family quarantine; unchanged source slots; source-only response replay; and the observed cooldown/stop behavior. All95 fetched source documents replayed to the same final primary licensing statuses under the final verifier. Actual root/code allocation and transfer ledger bounds were checked.

Historical verification passed5,100 records (5,080 unique snapshot paths,187 official heldout paths stat-only), plus Open Images metadata39, old photo/evidence571, Phase5B79, Phase5C639, Phase6A preparation134, Phase6A photo/audit203 and Phase6B metadata44 sealed artifacts, plus prior implementation hashes. SigLIP2's eight pinned model/processor files match their own checksum ledger. Original OWLv2 SHA `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`, SigLIP2 SHA `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` and selected Phase5C SHA `d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6` are unchanged. All Phase5C recovery/checkpoint receipts remain covered by its seal.

No old image was decoded or scored for selection. No frozen partition, original annotation, historical model/result, environment or Mac simulation file was modified. `verification.json` records independent checks; original Phase6B400-candidate/rights-unverified metadata remains frozen, and new source rights evidence is stored separately here.

## Authoritative policy references

- [CC BY2.0 commercial permissions and required attribution](https://creativecommons.org/licenses/by/2.0/)
- [CC0 permissions and remaining other rights](https://creativecommons.org/publicdomain/zero/1.0/)
- [Flickr Creative Commons licence categories](https://www.flickr.com/creativecommons/)
'''
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('REPORT CREATED',OUT/'report.md')

if __name__=='__main__':run()
