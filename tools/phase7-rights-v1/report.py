"""Final evidence/attribution/coverage report. Does not retrieve or label photographs."""
import collections as C
import csv
from audit_core import *

def run():
 summary=read(OUT/'summary.json');audit=read(OUT/'rights-audit.json');identity=read(OUT/'identity-audit.json');net=read(OUT/'network-ledger.json');proposal=read(OUT/'proposed-image-acquisition-v1.json');registration=read(OUT/'existing-ten-photo-registration-assessment.json');m=manifest();byid={r['coco_image_id']:r for r in m['images']};decision={r['coco_image_id']:r for r in identity['records']}
 evidence=[]
 for r in audit['records']:
  row=byid[r['coco_image_id']]
  evidence.append(dict(coco_image_id=r['coco_image_id'],lvis_image_id=r['lvis_image_id'],source_photo_id=r['source_photo_id'],provisional_partition=r['provisional_partition'],
   annotation_license=row['annotation_license'],annotation_archive_sha256=row['annotation_archive_sha256'],annotation_credit='LVIS / Gupta, Dollar, Girshick / CC BY4.0',
   metadata_license_claim=row['metadata_license_url'],photograph_rights_status=r['licensing_status'],creator=r.get('creator'),creator_url=r.get('creator_url'),
   source_url=r.get('source_url'),license_url=r.get('license_url'),title=r.get('title'),attribution=r.get('attribution'),
   evidence_url=r.get('evidence_url'),evidence_checked_utc=r.get('evidence_checked_utc'),evidence_file=r.get('evidence_file'),evidence_sha256=r.get('evidence_sha256'),
   attempted=r['attempted'],evidence_attempts=r['evidence_attempts'],reason=r.get('reason'),identity_decision=decision[r['coco_image_id']],
   annotation_scope=row['source_image_scope'],source_annotation_ids=[a['id'] for a in row['original_lvis_furniture_annotations']],
   no_photograph_body=True,human_reviewed=False))
 commit(OUT/'license-and-attribution-ledger.json',dict(records=evidence,manifest_sha256=PIN,
  annotation_and_photo_licences_separate=True,licence_evidence_is_source_claim_not_chain_of_title_or_property_release=True,
  new_photograph_bodies=0,visual_quality_or_perceptual_duplicates_verified=False))
 fields=['coco_image_id','source_photo_id','provisional_partition','attempted','evidence_attempts','photograph_rights_status','creator','creator_url','license_url','evidence_url','evidence_checked_utc','evidence_sha256']
 with (OUT/'rights-decisions.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in evidence:w.writerow({k:r[k] for k in fields})
 status_rows='\n'.join(f'| {s} | {summary["licensing_status_counts"].get(s,0)} |' for s in STATUSES)
 category_rows='\n'.join(f"| {name} ({v['native_id']}) | {v['images']} / {v['instances']} | {summary['annotation_value']['eligible_for_acquisition_proposal'][name]['images']} / {summary['annotation_value']['eligible_for_acquisition_proposal'][name]['instances']} |" for name,v in summary['annotation_value']['rights_verified_before_identity_exclusions'].items())
 registration_rows='\n'.join(f"| {r['coco_image_id']:012d} | {r['existing_partition']} | {len(r['annotation_diagnostics'])} | {', '.join(sorted({a['interpreted_category'] for a in r['annotation_diagnostics']}))} |" for r in registration['records'])
 mixed=[r for r in identity['repeated_source_or_creator_groups'] if len(r['partitions'])>1]
 historical=[r for r in identity['records'] if 'historical_source_or_creator_group' in r['identity_flags']]
 limited=[r for r in net['requests'] if r.get('status')==429]
 reasons=C.Counter(r.get('reason','') for r in audit['records'] if r['licensing_status']!=STATUSES[0])
 reason_lines='\n'.join(f'- {count}: {reason}' for reason,count in reasons.most_common())
 next_request=read(OUT/'next-approval-request.json')
 if proposal['image_count']:
  next_text=f"Request separate approval to acquire and screen exactly **{proposal['image_count']} photographs** in proposed-image-acquisition-v1.json, preserving allocation **{proposal['proposed_allocation']}**. Expected object-body sum from approved HEADs: **{proposal['expected_photo_body_bytes']:,} bytes**; proposed cumulative cap **{proposal['proposed_cumulative_body_cap']:,} bytes**, maximum **3,145,728 bytes/photo**, at most **two attempts/photo**, maximum allocated storage **{proposal['proposed_new_allocated_storage_cap']:,} bytes**, destination **data/lvis/image-pilots/phase7-v1/**. Actual source IDs, creators, licences, evidence hashes/timestamps and native annotations are in that exact manifest. Acquisition is not authorized yet. It must revalidate saved evidence/hash/age, source availability and object size/version, preserve all original memberships, and quarantine any new contradictions. No replacement images or training are included."
 else:
  next_text=f"**No photograph acquisition is currently eligible.** A separately approved future rights-only stage may consider the exact **{summary['unattempted_candidates']} never-attempted candidates** in unattempted-checkpoint-v1.json after provider access is restored and any Retry-After window has passed. Proposed future limits are 128,000,000 response-body bytes, 512MiB allocated storage and one attempt per still-unattempted candidate, with no photograph bodies. This is not current resume authorization and does not clear any circuit. Do not retry already-attempted candidates or change source IDs."
 body=f'''# Phase7 LVIS individual photograph rights audit

Prepared {now()}. Workspace `/home/tristary/furniture-ai-ml`, existing `.venv/bin/python`, Windows model-training project only. Approval source and SHA are preserved in authorization.json. No photograph bodies, model weights, training, inference, rendering, production changes or Mac application access.

## Scope and validation

The exact approved manifest SHA-256 is `{PIN}`. All **400 photo IDs**, provisional **320 training / 80 development** memberships and **905 seven-class boxes/masks** match the frozen LVIS proposal. All **1,080 native annotations** including separately preserved subtypes/parts were compared to their original source ZIP dictionaries before network requests: IDs, masks, boxes and geometry are unchanged. The five frozen-related source photo IDs and the previous400 COCO audit IDs remain excluded. Known historical COCO/Open Images source/creator components were checked against local metadata without reading frozen pixels.

Photograph licensing and LVIS annotation CC BY4.0 licensing remain independent. A COCO licence-ID4 claim never automatically passes a candidate. Verification means a public source's photo-specific licence/creator grant was independently checked against the exact source photo ID, not that a legal chain of title, privacy/property/model releases or visual suitability has been established.

## Actual coverage and findings

**{summary['attempted_candidates']} candidates attempted; {summary['unattempted_candidates']} left unattempted.** Candidate attempts include bounded HEAD resolution of the same photo and at most one source-document GET. No candidate consumed more than one evidence attempt. There were **{summary['source_GET_attempts']} source GET attempts**, of which **{summary['source_successful_document_GETs']} completed with HTTP200 evidence**. HTTP HEAD responses did not establish copyright.

| Licensing classification | Candidates |
|---|---:|
{status_rows}

Rights-verified before group/availability exclusions: **{summary['verified_licensing_candidates']}**. Eligible for a separate photo-acquisition proposal: **{summary['eligible_proposed_photos']}**. Rights-verified records excluded for known identity/group conflicts: **{summary['identity_quarantined_verified_candidates']}**; rights-verified records without an acceptable available object HEAD: **{summary['source_unavailable_or_unverified_image_objects']}**. These exclusion reasons can overlap and must not be added as disjoint totals. Licensing findings are retained even when the photograph is identity-quarantined.

Machine-readable records: rights-audit.json, records/*.json, license-and-attribution-ledger.json and rights-decisions.csv. Every retrieved source document retains its exact URL, retrieval timestamp, compressed-byte SHA256, headers and response file. Source ID/creator/name/title/licence/version/attribution comes from same-photo JSON-LD on certificate-verified official Flickr pages. Metadata-versus-current-source licence discrepancies remain conflicting; CC BY-SA or other licences outside the approved CC0/CC BY policy are not falsely asserted to universally forbid commerce. Unrecognized or incomplete grants do not pass.

Nonverified reasons:

{reason_lines}

## Rate limits and durable checkpoints

One worker used paced source requests with a minimum three-second interval and the existing stable client identity. Redirects were bounded and allowed only to HTTPS official Flickr canonical/resolver pages retaining the exact approved photo ID. No proxies, account rotation, identity evasion, TLS bypass, API substitution or unapproved photo sources were used. Source pages were limited to 1MiB encoded / 4MiB decoded; photograph object URLs were HEAD-only.

The previous COCO audit's last429 was retained in preflight.json. Its fallback120-second cooldown had expired before this newly approved400-ID audit; none of those old IDs was retried. Current429 responses: **{len(limited)}**. Durable provider circuits: **{audit['circuits']}**. Any circuit remains open for this phase even after its Retry-After deadline. No later request to that provider is permitted in this phase. Remaining unattempted candidates are explicitly recorded, not reported as individual retrieval failures or licence denials. unattempted-checkpoint-v1.json is a future approval candidate list only.

Actual downloaded response bodies: **{summary['actual_body_bytes']:,} bytes**. Conservative response-body charge including any retained interrupted/error reservations: **{summary['conservative_charge']:,} bytes**, below the128,000,000-byte approved cap. HEAD response bodies were not downloaded. network-ledger.json records every request, one-attempt reservation and provider suspension. Final allocated storage (including this isolated implementation and audit records) is in resource-usage.json and stays below512MiB with a16MiB reserve. No installation or environment changes occurred.

## Identity and partition controls

Source identity and creator aliases were linked for conservative isolation; they are not silent merges of unrelated physical products. Excluded/licence-unverified bridge records still contribute source-bound creator aliases, so they cannot split a shared family across partitions. Protected historical roots are recomputed after alias unions. No original split was reassigned to rescue a candidate.

- Repeated known source/creator components: **{len(identity['repeated_source_or_creator_groups'])}**.
- Components crossing provisional training/development: **{len(mixed)}**; all affected candidates are quarantined.
- Candidates with known historical source/creator relationships: **{len(historical)}**; quarantined.
- Exact repeated original source URLs: **{len(identity['duplicate_original_source_URL_groups'])} groups**.
- Equal image-object ETag/size suspicions: **{len(identity['equal_HEAD_etag_size_suspicions'])} groups**, flagged rather than claimed to be proven pixel duplicates.

identity-audit.json records all member photo IDs, aliases, group conflicts and preserved original partitions. Unknown aliases and cross-creator copies/rooms remain unresolved. **No perceptual duplicate, visual indoor-scene, furniture-quality or actual physical-identity screening passed**, because no new photograph bodies were acquired.

## LVIS annotation value

| Native core category | Rights-verified images / instances before identity exclusions | Proposed eligible images / instances |
|---|---:|---:|
{category_rows}

Every counted instance retains its original box and segmentation. Category image counts overlap. Native LVIS IDs and separate subcategories/parts remain intact; lampshades do not become lamps, drawers do not become cabinets, and generic Table is not silently equivalent to COCO Dining table. Source `neg_category_ids` and `not_exhaustive_category_ids` remain preserved. Missing labels are unknown; positive nonexhaustive categories cannot provide verified background negatives. These source annotation counts are not detector accuracy or human-verified furniture quality.

annotation-value.json contains exact rights-verified and acquisition-eligible counts. The source manifest still preserves all905 potential core annotations for400 candidates; unresolved rights or groups do not erase their metadata, but prevent eligibility.

## Separate registration assessment: ten existing COCO photos /33 annotations

| COCO photo ID | Existing partition | LVIS annotations | Native groups |
|---|---|---:|---|
{registration_rows}

Read-only local byte hashes of all ten photographs match their existing acquisition records. Source IDs and dimensions match LVIS metadata, and saved EXIF orientation is1. No photograph was decoded, downloaded or visually inspected in this assessment. Native polygon extents and source bounding-box deltas are measurements in existing-ten-photo-registration-assessment.json; neither small deltas nor matching dimensions prove actual visual alignment or correct object-unit semantics.

**All33 remain inactive.** Semantic Cabinet/Desk/Lamp unit checks, visual image-to-mask registration, source-specific partial-label losses and original development isolation remain unresolved. There was no dataset merge, new human label, annotation correction or training. The previously quarantined15 additional-class annotations remain excluded; this assessment covers only the ten earlier provisionally accepted COCO photos.

## Exact next approval

{next_text}

Source evidence/head sizes establish availability and documented grants only. After separate acquisition approval, screen decoding/dimensions/SHA256, exact and perceptual duplicates, creator/source overlap, indoor context, visibility/occlusion/truncation, native mask registration and ambiguous/missing furniture annotations. Do not treat unlabelled furniture as negatives. Original ground truth and partitions must remain unchanged; any uncertain image must be quarantined, not replaced.

## Verification and preservation

Final test counts and logs: test-results.json. Independent integrity checks: integrity-verification.json. All original OWLv2, selected Phase5C and SigLIP2 model hashes, historical COCO/Open Images seals, both LVIS source archives and Phase7 metadata/preparation seals are verified read-only. Existing frozen official ABO images/features stay stat-only; no frozen photograph pixels are decoded or scored. The Mac simulation project is not accessed. No old dataset or source code is modified; implementation is isolated under tools/phase7-rights-v1/.

The source URLs and evidence dates for verified photographs are in each rights record and the proposed acquisition manifest, with LVIS annotation attribution retained separately. Commercial compatibility is an evidence snapshot under the stated copyright-licence policy, not a fabricated ownership warranty or legal clearance of all possible rights.
'''
 with (OUT/'report.md').open('x') as f:f.write(body)
 print('Rights report and licence/attribution ledger written',flush=True)

if __name__=='__main__':run()
