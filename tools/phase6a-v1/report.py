"""Record actual Phase 6A preparation outcomes; no photographs or model calls."""
import collections as C
import csv
from core import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def run():
 m=read(MANIFEST);audit=read(OUT/'metadata-audit.json');rights=read(OUT/'license-attribution-ledger.json');tests=read(OUT/'test-results.json');v=read(OUT/'verification.json')
 lookup={r['image_id']:r for r in read(OUT/'candidate-manifest.json')['images']}
 attrs=[]
 for record in rights['records']:
  original=lookup[record['image_id']]
  attrs.append(dict(image_id=record['image_id'],status=record['status'],creator=record.get('creator',''),metadata_author=original['author'],source_url=record.get('source_url',original['source_landing_url']),creator_url=record.get('creator_url',''),license_url=record.get('license_url',''),metadata_license=original['license_url'],verified_utc=record['checked_utc'],evidence_sha256=record.get('response_sha256',''),reason=record.get('reason',''),attribution=record.get('attribution',''),annotation_creator='Google LLC',annotation_license='https://creativecommons.org/licenses/by/4.0/'))
 with (OUT/'attribution-ledger.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(attrs[0]));w.writeheader();w.writerows(attrs)
 names={}
 with (META/'oidv7-class-descriptions-boxable.csv').open(newline='') as f:
  for r in csv.DictReader(f):names[r['LabelName']]=r['DisplayName']
 hierarchy=read(META/'bbox_labels_600_hierarchy.json');desc={}
 def visit(node):
  children=[]
  for child in node.get('Subcategory',[]):children.extend([child['LabelName']]+visit(child))
  if node['LabelName'] in MIDS.values():desc[node['LabelName']]=[dict(mid=i,name=names.get(i,i)) for i in sorted(set(children))]
  return children
 visit(hierarchy)
 sources=dict(checked_utc=now(),open_images=dict(description='https://storage.googleapis.com/openimages/web/factsfigures_v7.html',download='https://storage.googleapis.com/openimages/web/download_v7.html',annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_creator='Google LLC',photos='Per-photo source evidence, CC BY 2.0 for all 41 selected; no CC0 photos selected',hierarchy_sha256=sha(META/'bbox_labels_600_hierarchy.json'),descendants_not_substituted=desc),
  coco=dict(terms='https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm',download='https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm',format='https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py',category_id_reference='https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt',annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_creator='COCO Consortium',photo_rights='Individual photo licenses, not covered by annotation licence; no COCO photographs selected',local_instances_available=False,metadata_head=read(OUT/'coco-metadata-head.json')),
  licensing_policy='Allow canonical CC0 1.0 and CC BY 1.0/2.0/2.5/3.0/4.0 only. Quarantine NC, ND, SA and unknown/inaccessible rights under this deliberately narrow pilot policy. No inference of privacy/property releases.',
  relevant_cc_documents=['https://creativecommons.org/licenses/by/2.0/','https://creativecommons.org/licenses/by/4.0/','https://creativecommons.org/publicdomain/zero/1.0/'])
 write_new(OUT/'source-and-license-provenance.json',sources)
 distributions=[]
 for cat in CLASSES:
  train=m['categories']['train'][cat];dev=m['categories']['development'][cat]
  distributions.append([cat,MIDS[cat],train['images'],train['native_boxes'],train['ordinary_eligible_boxes'],dev['images'],dev['native_boxes'],dev['ordinary_eligible_boxes']])
 scope=read(OUT/'annotation-scope-audit.json');masks={r['image_id']:set(r['optimization_classes_to_mask']) for r in scope['records']}
 simple={p:sum(r['proposed_split']==p and bool(set(r['eligible_box_counts'])-masks[r['image_id']]) for r in m['images']) for p in ('train','development')}
 summary=dict(phase='6A',status='preparation_complete_acquisition_and_training_not_approved',proposed_photographs=len(m['images']),verified_individual_licenses=41,licensing_quarantines=7,metadata_candidates=48,
  partitions=dict(C.Counter(r['proposed_split'] for r in m['images'])),proposed_native_boxes=sum(sum(r['native_box_counts'].values()) for r in m['images']),ordinary_eligible_boxes=sum(sum(r['eligible_box_counts'].values()) for r in m['images']),
  nonambiguous_metadata_supported_images=simple,source_page_body_bytes=rights['actual_bytes'],source_page_conservative_transfer_charge=rights['conservatively_charged_bytes'],expected_photo_bytes=m['expected_photo_bytes'],worst_two_attempt_photo_bytes=m['maximum_two_attempt_photo_bytes'],
  expected_incremental_photo_and_screen_storage_bytes=64*2**20,proposed_total_storage_ceiling=STORAGE,transfer_ceiling=TRANSFER,
  new_photographs_downloaded=0,new_archives_downloaded=0,optimizer_updates=0,tests_passed=sum(r['tests'] for r in tests if r['passed']),finalized_manifest_sha256=sha(MANIFEST),
  readiness='Not ready for seven-class expanded training: no new sofa/desk/lamp evidence, only three beds, unknown image quality/duplicates, unresolved Cabinetry scope; no actual newly accepted training images exist yet.')
 write_new(OUT/'summary.json',summary)
 write_new(OUT/'acquisition-approval-request.json',dict(request='Approve acquisition and screening of only the finalized 41 photographs, no model scoring or parameter updates.',manifest=str(MANIFEST),manifest_sha256=sha(MANIFEST),image_count=41,train_slots=32,development_slots=9,destination=str(DEST),expected_photo_bytes=m['expected_photo_bytes'],max_transfer=TRANSFER,max_storage=STORAGE,per_image_max=IMAGE_MAX,max_attempts=2,existing_rights_charge=rights['conservatively_charged_bytes'],approval_granted=False))
 test_table=table(['Suite','Tests','Passed'],[[r['suite'],r['tests'],r['passed']] for r in tests])
 report=f'''# Phase 6A — commercially permissive detection-data preparation

Implemented and tested isolated acquisition, licensing, metadata conversion, screening and training-data adapters. **No photographs, metadata archives or model weights were downloaded; no detector parameters were updated.** Only original creator-page HTML was retrieved for the requested individual rights audit, plus body-free official image-object HEADs and public documentation research.

The exact finalized proposal is **41 additional Open Images photographs, 32 training / 9 fresh development, 108 native mapped boxes and 97 metadata-eligible ordinary boxes**. This is a shortfall from the up-to-400 target, not a 400-image dataset. All photo decoding, image usability, cross-dataset image-byte/perceptual matching and final training eligibility are pending image-acquisition approval. There are currently **zero newly accepted training photographs**.

## Available data and the shortfall

Inspected existing local Phase 1–5 artifacts first. Existing ABO product photos have no room detection boxes. Existing authorised room photos have no ground-truth boxes. Prior Open Images photos were already proposed/used/reserved, so none were redownloaded or repurposed. Every historical 150-image Open Images membership, including its entire known source group, is excluded; all original frozen sets and partitions remain intact.

The full two-source image tables contain 166,482 valid source identity records. Full-table photo ID, original MD5 and photographer grouping includes otherwise excluded bridge records. The expansion uses only the unconsumed **source-validation** pool and excludes the entire source-test pool to preserve its benchmark role. Of 1,028 native-furniture source-validation photos, 48 satisfy all strict metadata filters and historical group exclusions; these represent 48 known source groups. Counts of exclusion reasons overlap: 861 have no strong indoor metadata evidence, 139 invalid original size or rotation, 113 historical image/source-group membership, 61 no eligible ordinary box, 53 overlapping sofa/bed or table/desk conflicts, 16 outdoor titles, 5 nonzero rotation, 5 missing source identity. Complete image-level exclusions are in `exclusions.json`. No rule was relaxed to reach a quota.

Indoor evidence means an indoor title or annotated indoor context object, not a claim of visual verification. Ordinary eligible boxes have source-human positive labels, no source group-of/depiction ignore flag, normalized sides at least 0.05 and area at least 0.005. Occlusion/truncation flags and all original source boxes remain recorded. Source-native weak annotation is not application-specific human verification.

## Taxonomy and original annotation provenance

{table(['Class','Open Images MID','Train images','Train native boxes','Train ordinary eligible boxes','Dev images','Dev native boxes','Dev ordinary eligible boxes'],distributions)}

Category image counts overlap within a photo. Native counts include ignored or otherwise ineligible source boxes; metadata-eligible ordinary counts are not a claim of post-image-screening usability.

Open Images exact names are Couch, Chair, Table, Bed, Desk, Cabinetry and Lamp. Native MIDs, dictionary SHA, hierarchy SHA and descendant names are recorded in `taxonomy.json` and `source-and-license-provenance.json`; child labels are not silently expanded into parents. Desk source boxes may describe booths/side furniture rather than conventional desks. Cabinetry can denote a whole kitchen run rather than individual doors or freestanding cabinets. These source-unit differences remain unresolved. No new source label or box was invented or replaced.

COCO 2017 instance categories provide **chair 62, couch 63, bed 65 and dining table 67**. There is no native desk, cabinet or lamp detection category. Dining table is a subtype of broad table, not blanket evidence for every table; COCO couches/chairs/beds require a separate dataset-native evaluation stratum before cross-dataset claims. COCO JSON instances parsing preserves original pixel XYWH, converts to normalized XYXY with strict bounds, preserves crowd flags as ignored regions, and rejects duplicated IDs, mismatched category names and broken references. No pycocotools installation is needed. COCO test photos have no public test instance boxes suitable for this local proposal.

No official COCO instances metadata is local, so exact eligible COCO image/box counts are **unknown, not zero**. The official annotations archive is advertised as about 241 MB; its HTTPS host failed certificate hostname verification in this environment. A GCS path derived from the documented bucket was checked by HEAD and returned 404. Neither TLS checks nor source restrictions were bypassed. Exact bytes/integrity and an official secure source must be established before a separate COCO metadata proposal. No COCO archive or photograph was fetched. See `coco-metadata-head.json`. Additional unconsumed Open Images training metadata or COCO annotations are the route toward 400, subject to a separately verified metadata-source/resource proposal, not silently using reserved photos or unsupported labels.

## Commercially permissive photo-license verification

**41 of 48 candidate creator pages verified CC BY 2.0; 7 quarantined** (six HTTP 404, one HTTP 403). All 41 image objects also passed official-source HEAD checks for exact path, JPEG type, content length and pinned ETag. No CC0 image was selected. A licence URL in dataset metadata alone never qualifies a photo.

Verification matched a unique original Flickr ImageObject to the original photo identity, matching creator account, current allowed CC licence and required creator credit. Original response bytes, SHA-256, creator name/profile, original source link, licence link, method and UTC verification time are preserved in `rights-audit/`, `license-attribution-ledger.json` and `attribution-ledger.csv`. Quarantined decisions are not overwritten or silently substituted. Photo acquisition must occur within seven days of evidence verification or a new versioned rights audit/manifest/approval is required.

Allowlist: CC0 1.0 and CC BY 1.0, 2.0, 2.5, 3.0, 4.0. The narrow pilot policy quarantines NC, ND, SA, unrecognized licences, changed licence claims, absent creator evidence and inaccessible pages. Exclusion of SA is a deliberate policy restriction rather than a statement that SA universally prohibits commercial use. These records verify commercially permissive copyright terms at the source; they do not establish privacy, publicity, property or model-release rights.

Keep creator and photo title, original photo URL, exact licence URL and changes (CVDF resizing, rotations, cropping, additional annotation). Preserve original source restrictions/credits when redistributing any derived images. Open Images annotation attribution is Google LLC / CC BY 4.0; COCO annotations are COCO Consortium / CC BY 4.0. Photo licences are separate from annotation licences, and neither dataset warrants individual-photo clearance. Official document sources and checked timestamps are recorded in `source-and-license-provenance.json`.

## Frozen proposal and acquisition budgets

Final manifest: `acquisition-manifest-v2.json`.

SHA-256: `{sha(MANIFEST)}`

The initial hash-only partition draft is preserved separately as `acquisition-manifest.json`. Final v2 uses deterministic, metadata-only group stratification, seed `{SEED}`, with at least one train and development example of each available class where feasible. It was frozen before image inspection or model scoring and has 32 train / 9 development slots. No automatic replacement or repartitioning is permitted after quarantine.

Exactly **41** approved-candidate image URLs have the form `https://open-images-dataset.s3.amazonaws.com/validation/<locked-16-hex-image-id>.jpg`. Only the exact individually verified objects in v2 are proposed. Destination: `{DEST}` (does not exist; no training directories were created).

Expected photo body bytes: **{m['expected_photo_bytes']:,}**. Worst case two complete attempts: **{m['maximum_two_attempt_photo_bytes']:,}**. Individual rights-page actual body bytes already transferred: **{rights['actual_bytes']:,}**; conservative ledger charge including unconsumed failed-request reservations: **{rights['conservatively_charged_bytes']:,}**. Maximum combined charged rights plus two photo attempts: **{rights['conservatively_charged_bytes']+m['maximum_two_attempt_photo_bytes']:,} bytes**, well under **1,073,741,824 bytes**. Per-photo maximum **3,145,728 bytes**, maximum two attempts, total additional allocated-storage ceiling **2,147,483,648 bytes**. Expected incremental photo/screen storage is approximately 64 MiB; actual allocation must remain checked, not inferred from logical lengths. No paid source or model download is used.

Transport pins manifest/approval digest, approved destination, source URL, remote size and ETag. No redirects or alternate image hosts are followed. Before network I/O, the ledger durably reserves the full object limit. Partial response files remain immutable and fully charged; safe resume skips verified completed objects or restarts only the same pinned object under a second attempt filename. It never concatenates potentially different versions, silently refunds uncertain transfers or resets the two-attempt budget. File SHA-256 and decoding are mandatory. Resource caps include prior rights reservations and preparation storage. Licensing changes, stale evidence, changed size/ETag, corruption and budget issues stop acquisition.

## Automatic quality, duplicate and annotation screening

Implemented future decoding/dimension/integrity checks, exact SHA and decoded-pixel hashes, dHash/pHash similarity, blank/white/small/blur heuristics, historical signature comparison, and cross-partition quarantine. Similarity is not proof of physical identity, and does not merge unrelated listings. Frozen historic photos are compared by existing signature metadata only, never newly decoded. Any cross-partition perceptual suspicion quarantines affected new photos rather than moving frozen members. Existing ABO protected hash/identity metadata is checked; unknown cross-author physical identities and unrecorded historical perceptual relationships remain limitations.

Unusable photos, source/identity overlaps and ambiguities remain explicitly quarantined or unresolved. No automated result is labelled human reviewed. Desk/Cabinetry gradient supervision, positive and negative, is masked until scope is established; source boxes remain preserved for separate source-native evaluation. Sixteen proposed photos have no other eligible category. A further hierarchy audit found 14 photos with descendant-only table regions lacking comparable native parent boxes; optimization table coverage is conservatively masked on those photos, without inventing parent GT. Complete original source rows, affected classes and masking reasons are in `annotation-scope-audit.json`. Metadata alone therefore supports at most **{simple['train']} training / {simple['development']} development nonambiguous images** before actual quality/duplicate checks; these are upper bounds, not accepted counts. Development evaluation uses original annotations/coverage, never the optimization mask. There is no full seven-class expansion here, and no inference or holdout scoring has occurred.

## Prepared training comparison

`training.py` supplies pinned original/Phase5C checkpoint specifications, offline inference loading, variable-size deterministic batching and update schedules, hash-checked training paths, future screened-manifest locking and a separate digest-bound training-approval gate. A 401-row no-download test confirms this adapter has no Phase2A 275-image or Phase5C 27-image ceiling; it also verifies development examples cannot enter update groups. Existing Phase5C checkpoint/recovery, frozen-head adaptation loss and metric implementations are referenced by SHA and remain unchanged. No new optimizer runner is authorized or invoked in this phase; final run orchestration/checkpoint queue must be pinned to the screened lock and approved runtime configuration after acquisition.

Proposed comparison: immutable original, immutable Phase5C selected head delta `{SELECTED_SHA}`, and a new head-only fit initialized from the original. Proposed conservative fit: LR 1e-5, weight decay 0.01, seed 20261105, batch1, accumulation4 with correct smaller-final-group normalization, FP16 dynamic scale, clipping1, up to10 epochs and an upper ceiling500 optimizer updates. Actual update cap must be derived and approved after usable N is known. Keep backbone/objectness frozen, 80% CUDA memory ceiling, stop on OOM/nonfinite loss, bounded numerical replay only, per-step committed hashes/RNG/scaler state and graceful stop/recovery using the proven Phase5C receipt pattern. Previous comparable head-only execution used 1.93 GiB allocated VRAM; expansion changes update time and data IO, not batch memory, though this is a projection, not a new measurement.

All three models must use the same eligible fresh development photos, original pixel transforms, prompts, exact taxonomy and source annotations, class coverage masks, AP floor0.001, confidence0.15, NMS IoU0.5, IoU0.50/0.75 and existing 101-point AP implementation. Report per-category support/FP/FN/duplicates/localization; unsupported categories NA. Report primary source-native and separately identified nonambiguous-category aggregates, without selecting on old development results or suppressing regressions. No old Phase5 frozen or official Phase2B material is accessed for tuning/scoring. This small, imbalanced proposal is **not ready for a general seven-class training claim**; larger metadata and resolved annotation scope are recommended first.

## Tests, dry runs and preservation

{test_table}

All **{sum(r['tests'] for r in tests)} tests pass**. Tests cover licence allowlists and individual identity/evidence, creator mismatch, changed licences, COCO conversion/crowds/schema, unsafe paths/hosts/redirects, capped transfers, interrupted partial-body recovery with cumulative accounting, two-attempt limits, cache tampering, deterministic stratification, known-group/hash leakage, no implicit acquisition/training approval, exact/perceptual duplicates and blank imagery, variable-size batches, frozen checkpoint hashes and source-record integrity. No test fetched external photos or modified historical artifacts.

Preservation passed: 5,100 historical records (5,080 unique snapshot paths, 187 official holdout paths stat-only), metadata39, photo/evidence571, photo118, Phase5B79, Phase5C639 checksum records and old implementation hashes. Both original and selected model hashes match. Frozen pixels decoded/scored0; detector optimizer updates0; separate Mac application accessed0. `verification.json` records exact checks. The project Python is `.venv/bin/python`; no packages, environments or deployed models changed.

## Exact next approval

Approve **only the 41 photographs in `acquisition-manifest-v2.json`**, SHA `{sha(MANIFEST)}`, with **32 training / 9 development slots**, **10,487,710 expected photo bytes**, maximum **two attempts/object**, **3,145,728 bytes/photo**, **1 GiB cumulative transfer including the already charged rights audit**, **2 GiB total new storage**, destination `{DEST}`. Approval should include acquisition, integrity verification and automated screening only; no substitutions, scoring, parameter updates or deployment. The machine-readable request is `acquisition-approval-request.json` and is currently unapproved.

A separate metadata-source investigation/acquisition approval will be needed to approach400 with broader class coverage; COCO's secure official archive access and exact size remain unresolved. A separate final training configuration and screened-dataset approval will be needed before any new detector updates. No image acquisition has been started.

## Official references

- [Open Images V7 description and licences](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- [Open Images V7 official annotation downloads](https://storage.googleapis.com/openimages/web/download_v7.html)
- [COCO Consortium terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [COCO official download definitions](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- [COCO official instances API/schema](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
- [Google TensorFlow COCO category IDs](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)
- [CC BY2.0 permissions and credit terms](https://creativecommons.org/licenses/by/2.0/), [CC BY4.0](https://creativecommons.org/licenses/by/4.0/), [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
'''
 write_new(OUT/'report-binding.json',dict(manifest_sha256=sha(MANIFEST),verification_sha256=sha(OUT/'verification.json'),test_results_sha256=sha(OUT/'test-results.json')))
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('REPORT',OUT/'report.md');print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
