"""Acquisition/screening provenance and combined prospective readiness; no training activation."""
import collections as C
import csv
from pilot_core import *

def run():
 m=manifest();source={r['coco_image_id']:r for r in m['images']};photos=read(OUT/'acquisition-complete.json')['records'];screen=read(OUT/'screening-v2.json');ready=read(OUT/'combined-training-readiness.json');integrity=read(OUT/'integrity-verification.json');tests=read(OUT/'test-results.json')
 rows=[]
 for r in photos:
  orig=source[r['coco_image_id']];p=orig['verified_photo_rights'];rows.append(dict(coco_image_id=r['coco_image_id'],lvis_image_id=orig['lvis_image_id'],source_photo_id=orig['source_photo_id'],
   source_url=r['source_url'],original_photo_source_url=p['source_url'],creator=p['creator'],creator_url=p['creator_url'],title=p['title'],photo_license=p['license_url'],
   license_checked_utc=p['evidence_checked_utc'],evidence_url=p['evidence_url'],evidence_file=p['evidence_file'],evidence_sha256=p['evidence_sha256'],rights_record=p['rights_record'],rights_record_sha256=p['rights_record_sha256'],
   annotation_license=orig['annotation_license'],annotation_credit='LVIS / Gupta, Dollar, Girshick / CC BY4.0',annotation_archive_sha256=orig['annotation_archive_sha256'],
   attribution=p['attribution'],other_rights=p['other_rights'],original_partition=r['proposed_split'],identity_group=r['identity_group'],downloaded_utc=r['downloaded_utc'],
   bytes=r['bytes'],sha256=r['sha256'],width=r['width'],height=r['height'],file=r['file'],format='JPEG',attempts=r['attempts'],
   changes='Official COCO image bytes preserved. Separate contact sheets resize and overlay original LVIS boxes/polygons. No source annotation geometry or JPEG pixels changed.',live_Flickr_requests=0))
 commit(OUT/'license-and-provenance-ledger.json',dict(records=rows,manifest_sha256=PIN,photo_specific_saved_evidence_reverified=66,licensing_retries=0,unattempted_rights_resumed=False))
 with (OUT/'attribution.csv').open('x',newline='') as f:
  fields=['coco_image_id','creator','creator_url','title','original_photo_source_url','photo_license','license_checked_utc','evidence_sha256','annotation_license','annotation_credit','sha256','changes']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in rows:writer.writerow({k:r[k] for k in fields})
 with (OUT/'PHOTO-SHA256SUMS').open('x') as f:
  for r in sorted(photos,key=lambda r:r['file']):f.write(r['sha256']+'  '+r['file']+'\n')
 current={r['coco_image_id']:r for r in screen['images']};quarantined=[r for r in screen['images'] if r['status']=='quarantined'];accepted=[r for r in screen['images'] if r['status']=='provisional_source_candidate']
 raw_counts={}
 for cid,name in CLASSES.items():
  raw_counts[name]=dict(images=sum(any(a['category_id']==cid for a in r['original_annotations']) for r in screen['images']),instances=sum(a['category_id']==cid for r in screen['images'] for a in r['original_annotations']))
 newtable='\n'.join(f"| {name} ({cid}) | {screen['eligible_category_counts']['train'][name]['images']} | {screen['eligible_category_counts']['train'][name]['instances']} | {screen['eligible_category_counts']['development'][name]['images']} | {screen['eligible_category_counts']['development'][name]['instances']} |" for cid,name in CLASSES.items())
 combined='\n'.join(f"| {name} | {ready['per_category_training'][name]['images']} / {ready['per_category_training'][name]['boxes']} / {ready['per_category_training'][name]['instance_masks']} | {ready['per_category_fresh_development'][name]['images']} / {ready['per_category_fresh_development'][name]['boxes']} / {ready['per_category_fresh_development'][name]['instance_masks']} |" for name in ('sofa','chair','table','bed','desk','cabinet','lamp'))
 qlines='\n'.join(f"| {r['coco_image_id']:012d} | {r['proposed_split']} | {r['assistant_diagnostic']['observations'][0]} |" for r in quarantined)
 new_train=sum(a['id'] in r['valid_core_annotation_ids'] and a['category_id'] in CLASSES for r in accepted if r['proposed_split']=='train' for a in r['original_annotations']);new_dev=sum(a['id'] in r['valid_core_annotation_ids'] and a['category_id'] in CLASSES for r in accepted if r['proposed_split']=='development' for a in r['original_annotations'])
 body=f'''# Phase7 LVIS-associated photograph acquisition and screening

Completed {now()} (UTC evidence timestamps; user calendar date is Pacific). Workspace `/home/tristary/furniture-ai-ml`, `.venv/bin/python`. This stage performed only approved image acquisition, model-free screening, provenance and readiness preparation. No training, fine-tuning, model inference, scoring, rendering, weight/package download or Mac application access.

## Actual acquisition and licensing

All **66 exact approved photographs** were acquired, on the first attempt, preserving their **55 training /11 development** assignments. Actual downloaded response bodies: **8,368,471 bytes**, matching the frozen HEAD estimate. There were66 photo GETs,66 source-object HEADs, zero retries and no partial transfers. All source objects matched the pinned official HTTPS URL, Content-Length, ETag and JPEG type, and were decoded with the expected dimensions. No source redirects or alternate image URLs were followed. Transfer stays below the100,000,000-byte cap; each file is below3,145,728 bytes. Allocated storage including new implementation, reports, images and contacts is recorded in resource-usage.json, below512MiB. Response-body counts exclude protocol/header overhead, as in the approved acquisition proposal; no separate wire-byte measurement is claimed.

Manifest SHA256: `{PIN}`. Sources are exactly the approved `https://s3.amazonaws.com/images.cocodataset.org/train2017/<12-digit-ID>.jpg` objects, official COCO photographs associated with LVIS annotations. Timestamp, source version, byte count, format, dimensions, image SHA256, attempts and source/creator rights references are recorded in acquisition-complete.json and license-and-provenance-ledger.json. PHOTO-SHA256SUMS covers all66 unmodified JPEG files.

Before acquisition all66 saved same-photo source grants were re-parsed offline, binding the specific Flickr photo ID, creator, title and exact **CC BY2.0** URL to the recorded evidence hashes. The full licensing ledger preserves creator/source links, grant/evidence timestamps, encoded-document SHA256, attribution and separate LVIS annotation **CC BY4.0** credit. Evidence age was within the7-day replay policy; no live source-page refresh or licensing retry was made. Copyright grants do not establish property/privacy/publicity releases or a legal chain-of-title warranty. No such clearance is invented.

All nine previously identity-quarantined IDs remain excluded, including newly frozen-related COCO154613. The original five frozen-related metadata exclusions remain preserved. **None of the296 unattempted rights checks was resumed**, and no Flickr requests occurred. Source annotations/IDs match the frozen parent, and both original LVIS archive hashes match. All122 native source annotations are retained, including108 core annotations and14 separately identified subtypes/parts; none was corrected or overwritten.

The isolated downloader durably reserves possible response-body bytes before GETs, uses certificate-verified HTTPS/If-Match, enforces two attempts/object and exact range responses, verifies partial prefixes before recovery, and publishes completed files exclusively. Successful-file resume skips verified objects without re-downloading them. Uncertain or changed sources stop rather than silently replace IDs. No network interruption occurred in the actual run; the safety paths are regression-tested.

## Screening results

**48 provisional source candidates:40 training /8 development.18 photographs are quarantined.0 are classified as definitively rejected.** All66 valid acquired files are retained; the original55/11 allocation is unchanged. The40/8 figures are eligible subsets, not reassigned partitions. These are provisional source agreement/quality findings, not independent human-verified labels or model accuracy.

All66 passed file integrity, JPEG decoding, dimensions, finite/in-bounds box/polygon checks and nondegenerate mask raster diagnostics. No box/polygon extent discrepancy above2px was observed. Diagnostic rasterization uses Pillow and can differ at pixel edges from COCO mask rasterization; the original polygons remain authoritative and unchanged. Valid geometry does not establish object meaning or real image alignment.

No exact encoded/decoded-image duplicate pair or perceptual similarity suspect was found within the new66 or against stored Open Images/COCO signatures. Existing ABO hashes were also checked without opening old images. The fixed dHash<=4 /pHash63<=6 thresholds only flag suspicion and do not prove independence. Historical/frozen pixels were not decoded for this comparison. Source/creator groups remain isolated across partitions. Unknown physical scene/product identities or visual duplicates outside these checks remain possible.

The photo/signature checks cover every acquired file. Five bounded contact sheets cover all66 new images with original LVIS mask contours and boxes. The assistant also inspected15 targeted originals/overlays, including uncertain background furniture, bedside units, an integrated radio cabinet, clipped/occluded workstations, and overlapping native Desk/Table surfaces. This is assistant diagnostic work: **human-review flags remain false**. No classifier/detector was run. The48 retained images have broadly plausible source-native furniture/workstation context and boundaries at the inspected resolution; independent semantic/registration truth is not claimed.

### Provisional core annotation counts

| Native category | Train images | Train instances/boxes/masks | Dev images | Dev instances/boxes/masks |
|---|---:|---:|---:|---:|
{newtable}

Thus retained new core supervision is **{new_train} training instances/boxes/masks and{new_dev} development instances/boxes/masks**, or81 total. Counts of images overlap between categories. Separate armchairs, filing cabinets and lampshades remain in original_annotations; they are not silently converted into core Chair/Cabinet/Lamp targets.

### Quarantined photographs and reasons

| COCO photo ID | Original partition | Diagnostic reason |
|---|---|---|
{qlines}

These are uncertainty exclusions, not invented corrected categories or confirmed source-annotation mistakes. In particular, COCO000000086008 has overlapping native Desk361/Table1050 masks on a close support-surface view; it is quarantined instead of assigning a replacement category. Several source-valid Desk images predominantly show a desktop surface, foreground product or occluder, rather than an identifiable furniture unit. Foreground people can also prevent source-unit disambiguation. Geometric checks passed for these images, illustrating why geometry alone is insufficient.

Border contact, partial objects and occlusion are not automatically invalid: identifiable workstations remain provisional positives with limitations. The retained set still contains partial/close views and unconventional units (hotel writing furniture, a partly visible bed/post, an integrated radio cabinet and individual wall cabinets). These cases require an explicit source-native object-unit policy before broad cross-dataset training; no whole-object completeness guarantee is claimed. Missing furniture annotations remain unknown. LVIS negative/nonexhaustive flags are preserved, and discarded/tiny/uncertain regions may not become verified background negatives.

Authoritative versions: screening-v2.json; CURRENT-SCREENING.json pins its SHA; eligible-candidates-v1.json lists only provisional candidates. No permanent dataset lock or new partitions were created.

## Contacts and attribution

Contact index: contact-sheets-v1/index.html; train-001.jpg through train-004.jpg and development-001.jpg. Targeted unchanged-geometry overlays: targeted-overlays-v1/000000568281.jpg and000000086008.jpg. Originals: photos/<12-digit-ID>.jpg. Contact thumbnails and overlays are separate derivatives; originals are unchanged. Creator/title/photo licence/source URLs and annotation credit are in attribution.csv. The original contact captions show the pre-refinement pending status; consult CURRENT-SCREENING.json/screening-v2.json for final status.

## Ten existing COCO photos /33 LVIS annotations

existing-ten-registration-reassessment.json re-verifies all ten existing local photo hashes, LVIS/COCO IDs, dimensions and source coordinate metadata, with all33 annotations unchanged. No old photograph was decoded or visually re-reviewed in this reassessment. **Zero annotations were activated or merged.** Native Desk/Cabinet/Lamp unit semantics, actual mask/image alignment and a tested source-specific partial-label loss remain unresolved. Original development membership must be preserved. These ten photographs are not counted as new examples, and their33 inactive LVIS overlay annotations are not added to combined training totals.

## Combined real-image training readiness

After exact source-photo/SHA deduplication and known creator/source-component checks, the prospective conservative positive-target inventory has:

- **114 unique training photographs**, from **108 known source/creator components**.
- **23 unique fresh-development photographs**, from **23 known components**.
- Training: **361 source instances/boxes,285 instance masks**.
- Fresh development: **99 source instances/boxes,76 instance masks**.
- Another **18 historical-development photographs remain reserved**, outside the fresh selection counts. All60 original frozen slots and18 eligible reserved frozen references remain untouched.

There were no duplicate-photo references or new known cross-partition source-component exclusions in this combined inventory. Multiple distinct photographs from the same creator component are retained within one partition; they are not collapsed into one physical product. Component counts are not independently verified physical identity counts.

The older nominal training pool was81 photographs, of which74 retain positive targets under previously established conservative scope masks. Adding40 LVIS candidates yields121 nominal photographs but **114 with the retained positive scope**; the seven old no-target source records are unchanged and explicitly omitted prospectively. The fresh-development total is old15 plus new8. Old Open Images source masks and COCO eligibility are reused without modifying completed datasets. All photographs have saved photo-specific rights evidence; source claims are not universal release or annotation-accuracy guarantees.

| Project comparison group | Train images / boxes / masks | Fresh dev images / boxes / masks |
|---|---:|---:|
{combined}

These group totals are descriptive, not permission to merge taxonomies. combined-training-readiness.json preserves native strata: COCO67 Dining table, Open Images broad Table and LVIS1050 remain distinct; LVIS Cabinet181 is not automatically an Open Images Cabinetry kitchen run. Legacy ambiguous Desk/Cabinetry/descendant targets stay masked. Open Images boxes have no instance masks in the audited local pool, while retained COCO and LVIS targets have masks. Original source instances are counted once per photograph/annotation pool; inactive COCO/LVIS overlays are not double-counted.

**Recommendation: prioritise more diverse labelled indoor photographs and source-unit/loss-policy preparation before broad seven-class training.** Cabinet has only3 training photographs and **zero fresh-development** photographs; Lamp has10 training/one development photograph, Bed29 training/two development. Desk expansion is highly biased toward close workstation surfaces. No seven-class model selection can be justified for missing/very sparse development classes, and a new training batch is not authorized.

The296 unattempted rights candidates remain an optional separately approved future data route after provider access is available, preferably with deterministic category interleaving to improve Cabinet/Lamp coverage. They are not rights-cleared or authorized here. Do not retry the already-attempted/rate-limited sources or use existing frozen panels to compensate for sparse development classes. Before training, separately approve a versioned native-scope dataset/configuration, test federated/partial-label losses, resolve category units, and define a fair protocol excluding every frozen holdout from selection.

## Tests and preservation

**{tests['total_tests']} relevant CPU-compatible tests passed**, including source pinning, resource/attempt/range limits, exclusive publication, masks/boxes, native category granularity, duplicate flags, known-source isolation, original annotation/partition preservation, combined counts and inactive33-overlay handling. test-results.json and per-suite logs retain exact results. No actual GPU inference or training ran.

integrity-verification.json independently rehashed and decoded the66 new photos, reconciled all8,368,471 body bytes and first attempts, checked every source annotation and saved licence hash, and verified historical seals. Original OWLv2, Phase5C adapted and SigLIP2 checkpoint hashes match. COCO/Open Images datasets, LVIS source archives, rights records, Phase7 preparation code/reports and all frozen partitions remain unchanged. Official ABO holdout images/features stay stat-only; no held-out scoring occurred. All new code is isolated under tools/phase7-images-v1/. The Mac application was not accessed.

The stage stops here. No training, dataset activation, additional download or resumption of rights requests is included.
'''
 with (OUT/'report.md').open('x') as f:f.write(body)
 commit(OUT/'summary.json',dict(downloaded_photos=66,actual_photo_body_bytes=8368471,retries=0,original_allocation=dict(train=55,development=11),provisional_eligible_photos=48,
  provisional_eligible_allocation=dict(train=40,development=8),quarantined_photos=18,rejected_photos=0,eligible_training_core_instances_boxes_masks=new_train,eligible_development_core_instances_boxes_masks=new_dev,
  exact_duplicate_pairs=0,perceptual_similarity_suspects=0,combined_unique_positive_scope_training_photos=114,combined_fresh_development_photos=23,
  combined_train_instances_boxes=361,combined_train_masks=285,combined_fresh_development_instances_boxes=99,combined_fresh_development_masks=76,
  tests_passed=tests['total_tests'],existing33_annotations_inactive=True,rights_licensing_requests=0,unattempted296_not_resumed=True,training_inference_scoring_rendering=False,
  readiness='More diverse labelled data and explicit native-scope/unit policy before broad seven-class training',report='report.md'))
 print('REPORT AND LICENSING LEDGER WRITTEN',flush=True)

if __name__=='__main__':run()
