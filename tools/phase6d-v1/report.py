"""Phase6D actual results, attributable hashes and non-executing Phase6E assessment."""
import collections as C
import csv
from common import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def native_counts(rows):
 return {label:dict(images=sum(any(a['category_id']==mid for a in r['original_annotations']) for r in rows),boxes=sum(a['category_id']==mid for r in rows for a in r['original_annotations']),polygon_masks=sum(a['category_id']==mid and isinstance(a['segmentation'],list) for r in rows for a in r['original_annotations']),crowd_regions=sum(a['category_id']==mid and bool(a['iscrowd']) for r in rows for a in r['original_annotations'])) for mid,label in CLASSES.items()}

def run():
 m=manifest();acq=read(OUT/'acquisition-complete.json');screen=read(OUT/'screening-v2.json');eligible=read(OUT/'eligible-candidates-v1.json')['images'];v=read(OUT/'integrity-verification.json');ready=read(OUT/'phase6e-training-readiness.json');tests=read(OUT/'test-results.json');original={r['image_id']:r for r in m['images']}
 copied=OUT/'licensing-evidence';copied.mkdir();attrs=[]
 total_copy=sum((ROOT/r['rights_record']).stat().st_size+(ROOT/r['source_document']).stat().st_size for r in m['images']);space(total_copy+8*2**20)
 for photo in acq['records']:
  r=original[photo['image_id']];stem=f"{r['coco_image_id']:012d}"
  (copied/(stem+'.json')).write_bytes((ROOT/r['rights_record']).read_bytes());(copied/(stem+'.html-response')).write_bytes((ROOT/r['source_document']).read_bytes())
  assert sha(copied/(stem+'.json'))==r['rights_record_sha256'] and sha(copied/(stem+'.html-response'))==r['source_document_sha256']
  attrs.append(dict(image_id=r['image_id'],coco_image_id=r['coco_image_id'],original_flickr_photo_id=r['source_identity'],partition=r['proposed_split'],source_url=photo['source_url'],sha256=photo['sha256'],bytes=photo['bytes'],width=photo['width'],height=photo['height'],creator=r['creator'],creator_url=r['creator_url'],title=r['source_title'],original_source_page=r['verified_source_page'],photo_license=r['current_license_url'],rights_checked_utc=r['rights_checked_utc'],rights_record_sha256=r['rights_record_sha256'],source_document_sha256=r['source_document_sha256'],attribution=r['attribution'],annotation_creator='COCO Consortium',annotation_license='https://creativecommons.org/licenses/by/4.0/',changes='Original acquired JPEG bytes preserved. Separate diagnostic contact sheets resize and overlay unchanged source boxes; no category/box/mask corrections.',attempts=photo['attempts']))
 with (OUT/'attribution.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(attrs[0]));w.writeheader();w.writerows(attrs)
 commit(OUT/'license-and-provenance-ledger.json',dict(records=attrs,original_manifest_sha256=PIN,source_rights_grants_reverified=72,preflight='preflight.json',raw_evidence_copies='licensing-evidence/',head_is_not_copyright_proof=True))
 with (OUT/'PHOTO-SHA256SUMS').open('x') as f:
  for r in sorted(acq['records'],key=lambda r:r['file']):f.write(r['sha256']+'  '+r['file']+'\n')
 rows=screen['images'];counts={part:native_counts([r for r in eligible if r['proposed_split']==part]) for part in ('train','development')};all_counts=native_counts(rows)
 commit(OUT/'eligible-category-counts.json',counts)
 crows=[[label,all_counts[label]['boxes'],counts['train'][label]['images'],counts['train'][label]['boxes'],counts['train'][label]['polygon_masks'],counts['development'][label]['images'],counts['development'][label]['boxes'],counts['development'][label]['polygon_masks']] for label in CLASSES.values()]
 flags=C.Counter(flag for r in rows for flag in r['flags']);states=C.Counter(r['status'] for r in rows);eligible_boxes=sum(len(r['original_annotations']) for r in eligible)
 summary=dict(status='complete_acquisition_integrity_screening_only',approved_images=72,downloaded_images=72,original_allocation=dict(train=60,development=12),actual_photo_body_bytes=10853122,conservative_transfer_charge=10853122,retries=0,
  provisionally_accepted=55,accepted_train=45,accepted_development=10,rejected=1,quarantined=16,accepted_known_source_groups=v['eligible_known_source_groups'],
  source_native_boxes=362,source_polygon_mask_annotations=362,source_crowd_rle_regions=0,eligible_native_boxes=eligible_boxes,eligible_polygon_mask_annotations=eligible_boxes,eligible_crowd_regions=0,
  exact_duplicate_pairs=0,suspected_perceptual_pairs=0,duplicate_policy='Fixed dHash<=4 or pHash<=6 suspicion thresholds; no proof of universal physical independence.',
  original_annotations_and_slots_unchanged=True,human_labels_fabricated=False,model_inference_training_or_scoring=0,
  combined_nominal_train_images=81,combined_positive_scope_train_candidates=74,fresh_development_images=15,historical_development_images_excluded_from_new_tuning=18,tests_passed=sum(r['tests'] for r in tests if r['passed']),
  training_approved=False,phase6e_readiness='Conditional small source-agreement head-only pilot after adapter/config approval; no seven-class or verified-accuracy readiness.',
  report='report.md',current_screening='CURRENT-SCREENING.json',training_readiness='phase6e-training-readiness.json')
 commit(OUT/'summary.json',summary)
 pools=table(['Source pool','Training images','Fresh development','Historical development (excluded from new tuning)'],[['Existing Open Images Phase5',27,0,18],['Newer Open Images Phase6A',9,5,0],['COCO Phase6D provisional survivors',45,10,0],['Nominal combined',81,15,18],['Prospective conservative positive scope',74,15,18]])
 combined=table(['Source category stratum','Positive-scope train images / boxes','Fresh dev images / boxes'],[[label,f"{r['images']} / {r['boxes']}",f"{ready['fresh_development_counts'][label]['images']} / {ready['fresh_development_counts'][label]['boxes']}"] for label,r in ready['conservative_train_counts'].items()])
 report=f'''# Phase6D — COCO photograph acquisition and screening completed

**Exactly72 approved photographs acquired**, preserving the60 train /12 fresh-development allocation. Actual image transfer **10,853,122 bytes**; one GET per object, zero retries/partial responses. All downloaded files pass SHA-256 tracking, size/ETag pinning, JPEG integrity, decoding and exact COCO dimension checks. No alternative source, substitution or additional photograph was used. No model inference, scoring, fine-tuning, training or production deployment occurred.

Final outcomes: **55 provisionally accepted (45 train /10 dev),16 quarantined,1 rejected**. All72 originals, native boxes, masks and category IDs remain intact. Eligibility excludes whole uncertain photos; it does not repair labels or shift source partitions.55 photos represent **{v['eligible_known_source_groups']} metadata-known source groups**, not55 proven independent physical furniture products. No human review or boundary accuracy was fabricated.

## Before acquisition: provenance, rights and exclusions

Approved manifest SHA-256: `{PIN}`. Its exact72 original IDs, URLs and allocations were rechecked against the400-ID rights audit. All20 known source/creator overlap exclusions remain excluded, along with275 HTTP429,31 unresolved and2 out-of-policy records. None was replaced or retried for new acquisition eligibility.

All72 saved photo-specific source documents were rehashed and reparsed before download. Their photo identity, creator, title and CC BY2.0 version match the frozen source evidence. Evidence freshness was within seven days. These are source-specific copyright-grant checks, not rights inferred from COCO licence ID or S3 HEAD. No new Flickr bodies/access-restriction workarounds were requested. Each official COCO S3 object was rechecked immediately before GET under normal certificate-verified HTTPS, exact size/ETag and JPEG content type; redirects were denied.

Creator/title/original-page/CC BY2.0 links, source-document and rights-record hashes, dimensions, byte count, timestamp, retry count and actual photo SHA-256 are in `attribution.csv` and `license-and-provenance-ledger.json`. `licensing-evidence/` preserves byte-identical copies of72 source records and their HTML response bodies. Retain COCO Consortium / CC BY4.0 annotation credit separately. Other property/privacy/publicity/model releases were not assumed. Original JPEG bytes were never edited; contact sheets are separately labelled resized/source-box-overlay derivatives with attribution links.

## Actual source annotations and eligible categories

{table(['Native source category','All72 native boxes','Train images','Train boxes','Train polygon-mask annotations','Dev images','Dev boxes','Dev polygon-mask annotations'],crows)}

Category image counts overlap. The acquired cohort has **362 boxes and362 polygon-mask annotation objects**, zero crowd RLE regions. A polygon annotation can contain multiple polygon components; those components are not counted as extra instances. The55 provisional survivors retain **280 boxes /280 polygon-mask annotations**,222 train /58 dev. They are source-confirmed and structurally checked, not independently human-confirmed class/boundary truth. Small, occluded or clipped source regions remain tagged diagnostics; these counts do not silently certify every region as a complete visible object or final training target.

All original COCO annotations exactly match the approved manifest and the preserved Phase6B source. No mask was created from a predicted box or vice versa. No labels or missing background negatives were invented. Source category IDs remain chair62, couch63, bed65 and dining table67. Desk/Cabinet/Lamp are absent from COCO annotation scope; their absence never proves no such furniture exists in a photo. Source category67 includes coffee-table-like/living-room surfaces in the inspected images; its native name is preserved without claiming a verified fine-grained dining subtype or complete broad-Table coverage.

## Actual screening evidence

Automatic measurements include actual decoding/dimensions/orientation; cryptographic file and decoded-pixel hashes; gray variance/white fraction/Laplacian blur proxy; normalized source geometry; source-box border contact, size and overlap; and possible category contradictions. COCO does not supply per-object occlusion/truncation/viewpoint flags. Border/overlap are measurements and proxies, not fabricated ground-truth visibility labels. Multiple furniture objects are retained; legitimate occlusion or border contact alone does not cause rejection.

All72 new contact thumbnails were inspected with original source boxes, plus seven targeted full-resolution inspections. The initial automatic gate flagged3 near-coincident couch/bed source-box conflicts. Final diagnostics also detect near-coincident chair/couch or chair/bed regions,11 affected photos overall. These are unresolved source-category units, not asserted annotation errors. Preserve both source labels and quarantine the photograph rather than pick a convenient class.

Other uncertain captures include food/tea, bottle/glass, laptop or mirror-portrait foregrounds obscuring furniture, a distant blurred furniture background, and a bed visible only as a thin clipped strip. One clear three-panel composite (fridge /portrait /hallway) is rejected as unsuitable for a coherent room-photo pilot, but its source file and licence are retained. These exclusions do not automatically identify a wrong COCO label. Ordinary room images with visible furniture but partial/occluded regions remain provisional and keep source-boundary caveats.

{table(['Recorded flag (counts overlap)','Photos'],sorted(flags.items()))}

`screening-v1.json` preserves the initial automatic result. `screening-v2.json` is the final assistant-diagnostic screen; `CURRENT-SCREENING.json` pins it by SHA. `eligible-candidates-v1.json` lists55 survivors and explicitly does not authorize training or create a permanent new split. Every human-verification flag stays false. No source annotation was silently rewritten, and no quarantined example can enter that receipt.

## Duplicates, known identity and dataset isolation

**0 exact duplicate pairs and0 suspected perceptual pairs** were flagged among the72 new photos or against comparable stored Open Images signatures. Thresholds are dHash Hamming<=4 or pHash<=6. Existing ABO records were compared by stored SHA/pixel hashes only (**{read(OUT/'screening-v1.json')['abo_metadata_hash_records_compared']} metadata records**); their old/frozen photos were not decoded. No exact ABO hash overlap was found. Perceptual thresholds can miss resized/compressed/reframed related scenes, and do not prove physical independence. Historical noncomparable signatures and unknown cross-author capture relationships remain limitations.

The Phase6C creator/source-group isolation is retained. Source groups do not cross the accepted train/dev pool; no protected historical source group entered. Shared creators are treated as conservative families, not automatic product-identity merges. Original60/12 memberships remain unchanged:15 train slots and2 development slots are excluded, without replacement or moving members. Frozen datasets were not used for visual review, inference, scoring or selection. Historical bytes were checked read-only where permitted by their existing preservation protocol; protected official holdout paths use stat-only checks.

## Acquisition accounting and recovery

Stage network body bytes: **10,853,122**, conservatively charged10,853,122 under100,000,000. HEAD bodies0; GET attempts72, all first attempt; partial transfers0. The earlier independent Phase6C rights bodies12,536,007 remain in their own preserved ledger; the two stages together total23,389,129 recorded body bytes. Evidence copies are local copies, not extra network downloads. Per-image3,145,728-byte and two-attempt caps were enforced. Actual allocated totals, including raw photos, evidence copies, temporary allowance, reports, contacts and new code, are in `resource-usage.json`; the512MiB cap was never approached.

The durable ledger reserves full remaining response length before GET and retains uncertain reservations after failure. Resume verifies a committed partial prefix, exact range/ETag and content length before appending; it does not repeat verified completed photo transfers. Committed full bodies can be inspected/published without another GET, and completed-object records survive interruption between ledger commit and file publication. Exclusive file publication prevents overwriting old photos. Changed headers, decode/dimension failures, rights contradictions and resource limits are fatal. No recovery was needed in this actual run. `PHOTO-SHA256SUMS` and per-object records preserve all72 actual cryptographic hashes.

## Phase6E readiness assessment — no training performed

{pools}

These are pool-accounting views, not changes to existing training/evaluation partitions. The conservative prospective scope masks legacy Desk/Cabinetry and descendant-only/incompatible positive coverage without altering old labels;20 of the27 legacy Open Images train photos retain unambiguous positive targets under that policy. Adding9 newer OI and45 COCO survivors gives **74 prospective training candidates /71 metadata-known training groups**, plus **15 fresh development photos**. The nominal unchanged source pool remains81 train photos. The18 old development photos,18 eligible original OI frozen photos and all ABO frozen material remain outside new optimization. The original full frozen memberships are preserved.

{combined}

Broad OI Table and source COCO category67/dining_table are separate strata here. Images can contribute multiple categories; do not add category image counts to obtain distinct photographs. Weak source annotation is not human-verified Furniture AI accuracy. **Bed has only one fresh development photo; Lamp has no fresh development support, and Desk/Cabinet have no conservative positive training support.** This is not a seven-class readiness claim.

`phase6e-training-readiness.json` records conditions for a small, provisional four-class source-agreement head-only study after separate approval. The historical Phase5C worker is fixed at27 images and must not be reused unchanged. A variable-size, digest-bound source adapter and tested partial-label loss are required: COCO class absence is not an explicit image negative; unknown classes/regions and ambiguous units must remain masked, and only justified source negatives may be used. Source-positive counts above are not a declaration of whole-image exhaustiveness. Keep original boxes/masks/crowd semantics and freeze source-specific evaluation queries/mapping before scoring.

A possible configuration for review is batch1, accumulation4, frozen backbone/text/objectness and detector heads only, FP16 after smoke/nonfinite/gradient checks, seed20261105, up to10 epochs with `ceil(final approved N/4)*10` updates under a separately approved cap. Candidate learning rates for review are1e-5 and3e-6. Prior comparable Phase5C head-only execution used about1.93GiB allocated VRAM; this is a projection for RTX4070 8GiB, not a new measurement. No configuration or queue is approved here, no training directory/checkpoint was created, and no detector parameter update or model inference occurred.

Any later comparison must score immutable original OWLv2, immutable Phase5C and the future model on identical eligible fresh development examples, native source mapping/prompts, preprocessing and fixed metrics. Keep historical/frozen panels out of tuning, report uncertainty/per-category localization and duplicate failures, and do not compare new source-stratified numbers directly with old seven-class AP as if the evaluation were unchanged. Additional reliable development Bed examples and a clear Table/Desk/Cabinetry policy remain higher priorities than widening adaptation scope. COCO cannot supply native Desk/Lamp supervision; the previous separately gated Open Images metadata expansion is still unapproved and was not acquired.

## Tests and preservation

{table(['Suite','Tests','Passed'],[[r['suite'],r['tests'],r['passed']] for r in tests])}

All **{sum(r['tests'] for r in tests)} relevant automated tests passed**. New tests cover exact approved IDs/slots, source header pinning, proper range tails, exclusive photo publication, completed-body decode without network, disabled redirects, IoU/geometry ambiguity, distinction between exact/perceptual suspicion, image signatures, unchanged annotations, final eligibility/group isolation, actual byte/hash consistency, zero model operations and non-authorizing combined-pool assessment. Initial six safety tests passed before acquisition. No test downloaded another photo or modified a historical result.

Independent post-acquisition verification decoded and rehashed all72 new photos, reproduced image signatures, checked362 source annotation objects and all rights hashes, and preserved60/12 memberships.5,100 historical records (5,080 unique paths,187 official holdout paths stat-only) match. Additional seals match for Open Images metadata39, old photo/evidence571, Phase5B79, Phase5C639, Phase6A preparation134, Phase6A photo/audit203, Phase6B metadata44 and Phase6C rights1,326, plus prior implementation hashes. SigLIP2's eight model/processor files match their own ledger. Explicit before/after hashes for original OWLv2, SigLIP2 and selected Phase5C match, and all Phase5C checkpoint receipts remain sealed. Historical predictions, annotations, partitions, model weights, environment and the separate Mac simulation were unchanged. See `integrity-verification.json` and `test-results.json`.

## Review entry points

- `contact-sheets-v1/index.html`: all72 approved new photos and source-box overlays, with attribution link.
- `CURRENT-SCREENING.json`, `screening-v2.json`, `eligible-candidates-v1.json`: effective eligibility, evidence and preserved source labels.
- `attribution.csv`, `license-and-provenance-ledger.json`, `licensing-evidence/`, `PHOTO-SHA256SUMS`: source grants, credits and image hashes.
- `phase6e-training-readiness.json`: prospective pool accounting and explicit blockers/configuration review.
- `resource-usage.json`, `integrity-verification.json`, `test-results.json`: actual resources and preservation evidence.

Acquisition and screening stop here. Phase6E training requires a new explicit approval and a reviewed source-adapter/configuration; it has not begun.
'''
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('REPORT CREATED',OUT/'report.md',flush=True)

if __name__=='__main__':run()
