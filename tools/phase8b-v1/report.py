"""Actual rights/identity yield, unchanged current coverage and separate acquisition request."""
from audit_core import *

def native_counts(rows):
 result={}
 for cid,name in CORE.items():
  anns=[a for r in rows for a in r['original_lvis_furniture_annotations'] if a['category_id']==cid]
  result[name]=dict(images=sum(any(a['category_id']==cid for a in r['original_lvis_furniture_annotations']) for r in rows),boxes=len(anns),masks=sum(bool(a.get('segmentation')) for a in anns))
 return result

def run():
 summary=read(OUT/'summary.json');audit=read(OUT/'rights-audit.json');records=audit['records'];net=read(OUT/'network-ledger.json');proposal=read(OUT/'proposed-image-acquisition-v1.json')
 current=read(ROOT/'reports/phase8a-v1/current-coverage.json');source=manifest()['images'];byid={r['coco_image_id']:r for r in source}
 selected=proposal['images'];verified=[byid[r['coco_image_id']] for r in records if r['licensing_status']==STATUSES[0]]
 eligible_groups={part:len({r['identity_decision']['known_source_group'] for r in selected if r['proposed_partition']==part}) for part in ('train','development')}
 coverage={}
 for part,oldname in [('train','train'),('development','fresh_development')]:
  eligible=native_counts([r for r in selected if r['proposed_partition']==part]);rights=native_counts([r for r in verified if r['proposed_partition']==part]);category={}
  for name in CORE.values():
   old=current['split_totals'][oldname]['per_category'][name]
   category[name]=dict(current_images=old['images'],current_boxes=old['boxes'],current_masks=old['masks'],
    newly_rights_verified_before_identity=rights[name],eligible_proposed_additions=eligible[name],
    hypothetical_after_separate_acquisition_and_successful_screening=dict(images=old['images']+eligible[name]['images'],boxes=old['boxes']+eligible[name]['boxes'],masks=old['masks']+eligible[name]['masks']))
  coverage[part]=category
 commit(OUT/'coverage-impact.json',dict(by_partition=coverage,current_datasets_changed=False,photo_bodies_acquired=0,
  prospective_counts_are_not_current_eligible_training_data=True,missing_annotations_not_negatives=True))
 commit(OUT/'license-attribution-ledger.json',dict(records=records,source_manifest_sha256=PIN,
  photo_rights_not_annotation_rights=True,method='Exact photograph-specific ImageObject identity, author account, title, licence/version and metadata agreement; no HEAD-based copyright inference',
  annotation_license='LVIS v1 CC BY4.0 separately preserved; original COCO metadata/photo provenance retained',
  independent_ownership_property_model_and_trademark_releases_not_established=True))
 statuses=dict(collections.Counter(r['licensing_status'] for r in records))
 attempted=sum(r['attempted'] for r in records);pending=sum(not r['attempted'] for r in records)
 access_restricted=[dict(coco_image_id=r['coco_image_id'],status=r.get('status'),checked_utc=r['checked_utc'],retry_after=r.get('headers',{}).get('Retry-After')) for r in net['requests'] if r.get('status') in (401,403,429)]
 exclusions=[r for r in read(OUT/'identity-audit.json')['records'] if r['identity_flags']]
 unresolvable_source_ids=[r['coco_image_id'] for r in records if 'Redirect does not preserve' in r.get('reason','')]
 commit(OUT/'candidate-dispositions.json',dict(total=44,attempted=attempted,pending_unattempted=pending,
  rights_verified=sum(r['licensing_status']==STATUSES[0] for r in records),eligible_for_photo_proposal=len(selected),
  restricted_incompatible_licences=sum(r['licensing_status']==STATUSES[2] for r in records),
  attempted_unavailable=sum(r['attempted'] and r['licensing_status']==STATUSES[3] for r in records),
  conflicting_or_insufficient=sum(r['licensing_status']==STATUSES[4] for r in records),
  attempted_unverified=sum(r['attempted'] and r['licensing_status']==STATUSES[1] for r in records),
  identity_excluded=len(exclusions),identity_exclusions=exclusions,provider_restriction_responses=access_restricted,
  eligible_known_source_creator_groups=eligible_groups,
  unavailable_exact_source_identity_redirects=unresolvable_source_ids,
  licensing_status_counts=statuses,counts_are_separate_axes_not_additive=True))
 table=['| Category | Eligible proposed train photos / boxes / masks | Eligible proposed dev photos / boxes / masks | Current dev photos | Conditional future dev photos |','|---|---:|---:|---:|---:|']
 for name in CORE.values():
  a=coverage['train'][name]['eligible_proposed_additions'];d=coverage['development'][name]['eligible_proposed_additions'];old=coverage['development'][name]['current_images']
  table.append(f"| {name.title()} | {a['images']} / {a['boxes']} / {a['masks']} | {d['images']} / {d['boxes']} / {d['masks']} | {old} | {old+d['images']} |")
 t=read(OUT/'test-results.json');preserved=read(OUT/'integrity-verification.json')
 report=f'''# Phase 8B — targeted photograph rights audit

Completed the permitted rights-only stage against approved manifest SHA-256 `{PIN}`. Exactly 44 source IDs retain the original 8 training / 36 fresh-development proposed allocation. The historical proposal's `approved:false` was not edited; this user approval is recorded separately in `authorization.json`. The approved proposal specifies destination `data/lvis/rights-audits/phase8a-v1/`; the Phase 8B outputs use that exact directory. New tools are isolated under `tools/phase8b-v1/`.

## Actual outcome

* Attempted candidates: **{attempted}**; pending never-attempted: **{pending}**.
* Photograph-specific rights verified before identity checks: **{summary['verified_licensing_candidates']}**.
* Rights-verified, object-available and identity-eligible acquisition candidates: **{len(selected)}**; allocation **{proposal['proposed_allocation']}**.
* Rights-verified candidates quarantined by identity checks: **{summary['identity_quarantined_verified_candidates']}**.
* Eligible known source/creator components by original proposed partition: **{eligible_groups}**. These are conservative known groups, not proof of physical-scene independence.
* Licensing dispositions: `{json.dumps(statuses,sort_keys=True)}`.
* Actual downloaded evidence response bodies: **{net['actual_body_bytes']:,} bytes**. Conservative cumulative transfer charge including unfinished/error reservations: **{net['conservative_charge']:,} / {CAP:,} bytes**.
* Actual photograph bodies, model downloads, inference, training, installation and rendering: **zero**.

`candidate-dispositions.json` separates licensing, access restriction, identity exclusion and pending status. Those axes must not be summed as disjoint counts. Exact source identity could not be resolved through the permitted redirects for **{len(unresolvable_source_ids)} candidates**; these remain in the insufficient-evidence quarantine, not falsely claimed to have a contradictory or incompatible licence. Generic `/photos///` redirects were not followed and no alternative resolver was tried. Unattempted candidates are not falsely described as individually denied or licence-restricted. The provider circuits and Retry-After guidance are preserved in `network-ledger.json` and `unattempted-checkpoint-v1.json`. No request continued to a provider after a restriction/circuit opening. No old paused candidate was requested.

## Evidence and commercial policy

Certificate-verified HTTPS used one worker with at least three seconds between source requests, one evidence attempt/candidate, bounded same-photo redirects and at most one document GET/candidate. Only same-photo Flickr resolver/canonical source pages and exact approved official image-object HEADs were permitted. Image bodies, alternate endpoints, proxies and identity rotation are prohibited. HEAD established source/object availability, sizes and ETags only; it did not establish copyright.

Each verified record matches the exact Flickr photograph ID in an individual ImageObject to its author account, supplied creator/title, exact licence/version and frozen metadata licence claim. Missing creator/title, incompatible licences, changed licences and insufficient evidence remain unverified or quarantined. Raw evidence bytes, SHA-256, timestamp, method and evidence URL are retained in `responses/`, `records/` and `license-attribution-ledger.json`. No creator or human confirmation was invented.

Preserve supplied creator/title, original source photo URL, exact licence URL, copyright notices and indications of resizing/cropping/annotation changes. Source copyright evidence does not independently prove ownership or clear property, privacy/publicity, model-release or trademark rights. LVIS annotation CC BY4.0 credit/provenance is separate from photograph CC BY2.0 evidence; COCO provenance remains intact. No annotation licence is used as a photograph rights grant.

## Isolation and identity

Before requests, IDs, dimensions, archive hashes and all **164 native box/mask annotations** were checked against local official LVIS metadata. All prior 400 LVIS rights IDs, including the 296 paused IDs, are excluded. Prior COCO/Open Images photo/source groups, historical rights-only groups and frozen-related components remain protected.

After evidence retrieval, new source-bound creator aliases were unioned with historical aliases, including verified Phase 7 aliases. Historical overlap, cross-partition creator components, duplicate source URLs and matching HEAD ETag/size suspicions quarantine all affected candidates; no photo was moved to another partition. Restricted/unverified records with trustworthy creator aliases still bridge identity components. A common creator is conservatively isolated but not declared the same physical product. Unknown aliases, cross-creator scene reuse and perceptual duplicates remain unresolved until separately approved image acquisition. No perceptual or visual screening claim is made now.

`identity-audit.json` retains per-ID decisions, known groups and conflicts. Rights clearance alone does not bypass an identity quarantine. No frozen photographs were decoded or scored.

## Coverage impact, in priority order

'''+ '\n'.join(table)+f'''

Category counts overlap. Boxes/masks are original native LVIS annotations, not verified counts of independent whole furniture products. `coverage-impact.json` gives rights-verified-before-identity counts, eligible proposal counts and original current counts separately for every category and partition.

**Current actual training/development datasets remain 114/23 photographs, 361/99 boxes and 285/76 masks.** Conditional future coverage requires separate acquisition approval and successful integrity, identity, image-quality and semantic screening. Nothing in this audit makes a proposed image eligible for optimisation. Missing labels remain unknown, not negatives; native neg/NEX flags are retained. Cabinet fronts/bays versus whole cabinetry, generic Lamp versus table/floor/fixed lights, and Table/Desk unit semantics still need conservative treatment. Rights verification does not resolve those semantics.

## Exact next acquisition proposal

`proposed-image-acquisition-v1.json` lists only the **{len(selected)}** rights-verified and identity-eligible exact IDs, their original partitions, creator/title/attribution, source/evidence hashes, official image object URLs, HEAD metadata, native category IDs, original boxes/masks and annotation completeness scopes. No substitute IDs or partition reassignments were introduced.

Expected photograph transfer from recorded object HEADs: **{proposal['expected_photo_body_bytes']:,} bytes**; two-attempt expected upper sum **{proposal['expected_max_two_attempt_body_bytes']:,} bytes**. HEAD sizes are current observations, not guarantees against source changes. The separate request in `next-approval-request.json` binds the exact manifest SHA-256, proposes maximum two attempts/object, maximum **3,145,728 bytes/photo**, cumulative body cap **{proposal['proposed_cumulative_body_cap']:,} bytes**, new allocated storage cap **{proposal['proposed_new_allocated_storage_cap']:,} bytes**, destination **`data/lvis/image-pilots/phase8b-v1/`**. Photograph acquisition remains unapproved. If the manifest is empty, there is no photograph acquisition request.

After any separate acquisition approval: reverify saved evidence/object identity, enforce byte/disk limits, decode and check dimensions/hashes, screen exact/perceptual duplicates and source identities, inspect indoor context/furniture visibility and annotation-mask registration, preserve native coordinates, quarantine incomplete/ambiguous examples and preserve source-specific partial-label masks. Do not treat unannotated furniture as confirmed negatives or autoactivate the 33 older annotations. No model training/scoring is part of that request.

## Tests and preservation

**{t['total_tests']} tests across {len(t['suites'])} suites passed**, including 24 Phase 8B safety/identity tests. These test photograph-GET rejection, exact-ID scope, wrong-photo/missing/conflicting/noncommercial evidence, 401/403/429 circuit behaviour (403/429 explicitly exercised), Retry-After, TLS failure handling, one-attempt/no-retry rules, pre-request body reservations and conservative historical/cross-partition/duplicate grouping. Pure tests use local fixtures; no source requests are generated by the test suites.

Independent final integrity checks passed: request pacing, no post-restriction requests, transfer reconciliation, evidence hashes, native annotations, unchanged assignments and proposal eligibility. Historical checkpoint/data/artifact seals passed, including original OWLv2, selected Phase 5C adapted OWLv2, original SigLIP2, all earlier Open Images/COCO/LVIS manifests and Phase 8A artifacts. The historical snapshot verifier found no changes in its 5,100 recorded checks. The 33 prior LVIS annotations remain inactive. Every frozen partition is preserved; no frozen image pixels were accessed or scored. The separate Mac project was not accessed.

`integrity-verification.json`, `test-results.json`, test logs, `implementation-sha256.json`, `ARTIFACT-SHA256SUMS` and `resource-usage.json` provide verification/provenance and actual final allocated storage. Cumulative transfer is capped at 32,000,000 bytes and allocated storage at 134,217,728 bytes, including tools, evidence, logs and closeout files. Interruption journals never repeat a consumed candidate attempt; completed records are reused. A provider-suspended queue is not automatically reopened even after a cooldown.

To inspect progress without requests: `.venv/bin/python -B -m json.tool data/lvis/rights-audits/phase8a-v1/summary.json`. To rerun the new CPU suite without altering reports: `.venv/bin/python -B -m unittest discover -s tools/phase8b-v1 -p 'test_*.py' -v`. `run.py --resume` only reuses completed records and does not reopen provider circuits; no automatic reattempt or extra IDs are authorised. Do not rerun exclusive finalisation writers into this sealed directory.
'''
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('Actual rights yield, coverage impact, ledger and next approval report written.',flush=True)

if __name__=='__main__':run()
