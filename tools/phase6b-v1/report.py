"""Publish factual audit tables and a blocked, reproducible future image proposal."""
import collections as C
import csv
from common import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def run():
 audit=read(OUT/'metadata-audit.json');plan=read(OUT/'proposed-image-candidates-v1.json');verify=read(OUT/'integrity-verification.json');acq=read(OUT/'acquisition-complete.json');ex=read(OUT/'extraction-complete.json');defs=read(OUT/'category-and-license-definitions.json');tests=read(OUT/'test-results.json')
 rows=[];candidate_counts={}
 for mid,(native,target) in MAP.items():
  a=audit['source_partitions']['train2017']['categories'][native];b=audit['source_partitions']['val2017']['categories'][native]
  rows.append([native,mid,a['images'],a['boxes'],a.get('segmentation_polygon',0),a.get('crowd_boxes',0),b['images'],b['boxes'],b.get('segmentation_polygon',0),b.get('crowd_boxes',0)])
  candidate_counts[target]={}
  for part in ('train','development'):
   images=[r for r in plan['images'] if r['proposed_split']==part]
   candidate_counts[target][part]=dict(images=sum(any(a['category_id']==mid for a in r['original_furniture_annotations']) for r in images),boxes=sum(a['category_id']==mid for r in images for a in r['original_furniture_annotations']),polygon_masks=sum(a['category_id']==mid and isinstance(a['segmentation'],list) for r in images for a in r['original_furniture_annotations']),crowd_regions=sum(a['category_id']==mid and a['iscrowd']==1 for r in images for a in r['original_furniture_annotations']))
 candidate_table=table(['Native target','Proposed train images / boxes','Proposed dev images / boxes'],[[target,f"{v['train']['images']} / {v['train']['boxes']}",f"{v['development']['images']} / {v['development']['boxes']}"] for target,v in candidate_counts.items()])
 license_rows=[]
 for lic in defs['licenses']:
  train=audit['source_partitions']['train2017']['license_image_counts'].get(str(lic['id']),0);val=audit['source_partitions']['val2017']['license_image_counts'].get(str(lic['id']),0)
  train_f=audit['furniture_license_image_counts']['train2017'].get(str(lic['id']),0);val_f=audit['furniture_license_image_counts']['val2017'].get(str(lic['id']),0)
  license_rows.append([lic['id'],lic['name'],train+val,train_f+val_f,'Potential only; source verification needed' if lic['id']==4 else 'Outside strict CC0/CC BY allowlist'])
 # All rows retain unverified rights; this CSV is a review queue, not proof.
 with (OUT/'rights-review-queue-v1.csv').open('x',newline='') as f:
  fields=['image_id','coco_image_id','source_identity','identity_group','selection_anchor','proposed_split','proposed_source_page_resolver','original_flickr_asset_url','metadata_license_id','metadata_license_url','rights_status','creator','license_verified','proposed_secure_image_url']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in plan['images'])
 with (OUT/'furniture-counts.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['category','id','train_images','train_boxes','train_polygon_masks','train_crowd_rle','val_images','val_boxes','val_polygon_masks','val_crowd_rle']);w.writerows(rows)
 write(OUT/'proposed-cohort-counts.json',candidate_counts)
 requests=dict(schema='phase6b-next-rights-audit-proposal-v1',approval_granted=False,candidate_manifest='proposed-image-candidates-v1.json',candidate_manifest_sha256=sha(OUT/'proposed-image-candidates-v1.json'),source_photo_id_count=400,
  scope='Resolve and verify only400 listed Flickr photo IDs/creator accounts and CC0/CC BY rights; official photo object HEAD only; no photograph bodies, archive or weight downloads.',
  source_resolver='https://www.flickr.com/photo.gne?id=<exact listed ID>',allowed_canonical_page='HTTPS Flickr /photos/<creator>/<matching numeric photo ID>; allow only explicitly inspected same-photo redirects, no arbitrary hosts or embedded images',
  per_source_page_body_limit=1048576,max_decoded_document_bytes=4194304,max_source_page_get_attempts=2,max_cumulative_body_transfer=256000000,max_new_allocated_storage=536870912,
  expected_body_transfer_scenario_bytes=400*120000,expected_bytes_are_not_verified=True,
  quarantine_policy='Inaccessible, conflicting/changed/restricted licence, mismatched photo/creator, malformed evidence, uncertain attribution or protected known family: quarantine, never fabricate or silently replace.',
  output='A current per-image rights/creator ledger plus official size/ETag HEAD preflight. Reconcile creator groups against historical source accounts before freezing any exact image acquisition manifest or partition.',
  photo_downloads_authorized=0,model_inference_or_training_authorized=False)
 write(OUT/'next-rights-audit-proposal.json',requests)
 provenance=dict(annotation_creator='COCO Consortium',annotation_license='https://creativecommons.org/licenses/by/4.0/',terms_url='https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm',
  archive_url=URL,archive_sha256=SHA,archive_etag=ETAG,independent_checksum_url=read(PROPOSAL)['objects'][0]['checksum_source'],checked_utc=now(),
  source_documents={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'provenance').iterdir()) if p.is_file()},
  attribution='Retain COCO Consortium annotation credit, CC BY4.0 link and changes; cite COCO original dataset publication. Individual creators/licences must accompany any later photograph or derived contact sheet.',
  photo_rights='All image-level licence values are historical dataset metadata claims, not individual source-page verification. No photograph is commercially cleared by this audit.')
 write(OUT/'license-and-source-provenance.json',provenance)
 summary=dict(phase='6B',status='complete_metadata_only',archive_size=SIZE,archive_sha256=SHA,archive_attempts=1,retries=0,actual_network_response_body_bytes=acq['actual_total_body_bytes'],extracted_metadata_bytes=ex['total_extracted_bytes'],
  source_images=dict(train2017=118287,val2017=5000),furniture_images=dict(train2017=24234,val2017=1048),furniture_boxes=dict(train2017=64176,val2017=2912),
  structurally_valid_polygon_annotations=66629,structurally_valid_crowd_rle_annotations=459,source_annotation_issue_count=0,
  metadata_cc_by_claim_furniture_images=dict(train2017=3949,val2017=200),individual_photo_rights_verified=0,metadata_candidates_after_filters=3132,proposed_candidate_images=400,draft_partition_counts=dict(train=320,development=80),selected_native_annotations=1662,
  proposed_photo_bytes_scenario=100000000,scenario_is_not_remote_size_verification=True,new_photographs_or_models=0,model_inference_or_updates=0,tests_passed=sum(r['tests'] for r in tests if r['passed']),
  next_step='Separately approve a bounded400-photo-ID individual-source licensing/creator audit and body-free object preflight before any image-acquisition approval.',
  unsupported_categories=['desk','cabinet','lamp'],candidate_partitions_final=False,historical_preservation_passed=True)
 write(OUT/'summary.json',summary)
 report=f'''# Phase 6B — COCO2017 metadata acquisition and audit completed

One approved official annotation ZIP acquired and verified, with no photographs, model weights, detector inference or training. Project Python: `{sys_executable()}`. No package/environment or Mac simulation changes.

## Archive acquisition and secure extraction

Source: `{URL}`. Default HTTPS certificate-chain and hostname verification succeeded; redirects were forbidden. This is the already approved path-style address for the official COCO bucket, not an unapproved mirror. The current official download page and independent Google TensorFlow Datasets checksum registry were fetched as small provenance documents before archive acquisition.

Archive size: **252,907,541 bytes**. Published expected SHA-256 and actual local SHA-256 both:

`{SHA}`

Source ETag: `{ETAG}`. Archive attempts1; retries0; failures0; partial transfers0. Archive completion timestamp: `{acq['downloaded_utc']}`. Archive body bytes252,907,541; three source-document bodies17,587; cumulative recorded response-body transfer **252,925,128 bytes**, under520,000,000. No other archive or image object was acquired. Network evidence is in `source-preflight.json`, `network-ledger.json`, `acquisition-complete.json` and saved source documents. HTTP/TLS headers are not claimed as measured wire-level traffic; the approved cap is enforced conservatively on fetched bodies with substantial remaining headroom.

All six ZIP entries were inspected before extraction for traversal/absolute/drive/backslash paths, duplicate names, links/special files, encryption, unsupported selected-member compression, declared size and extreme expansion ratio. Only two permitted JSON members were streamed out with size limits and CRC checks:

{table(['Extracted member','Bytes','SHA-256'],[[r['file'],r['bytes'],r['sha256']] for r in ex['members']])}

Combined required extraction: **489,773,314 bytes**. All archive members together would expand to834,416,290 bytes; retaining the252,907,541-byte ZIP plus full extraction would require1,087,323,831 bytes before audit output, exceeding1GiB. Captions/keypoints were therefore not extracted. The actual archive plus two required JSON files totals742,680,855 logical bytes; inventory, proposals, tests and reports remain within the100MB derived-artifact allowance and overall1GiB allocated-storage limit. Final measured allocation is in `resource-usage.json`.

Ledger reservations precede archive requests; resume only permits the same source/ETag and verified committed prefix, an exact206 Content-Range and at most two attempts. A single `.part` file is appended safely, avoiding a second complete stored archive. Changed or uncertain prefixes and checksum failures stop instead of silently restarting or overwriting. No resume was needed. Original executed acquisition/extraction code is preserved in `executed-acquisition-code-v1/`; final code additionally hardens atomic exclusive completion publication. Final verified archive/extraction receipts are atomically published in `acquisition-verified-commit.json` / `extraction-verified-commit.json`. Original first-completion records remain preserved and hash-bound by these verified receipts.

## Exact available furniture counts

The source has118,287 train2017 images /860,001 annotations and5,000 val2017 images /36,781 annotations. Furniture occurs in24,234 training-source and1,048 validation-source photos. Counts by class overlap across images.

{table(['Source class','ID','Train images','Train boxes','Train polygon masks','Train crowd RLE','Val images','Val boxes','Val polygon masks','Val crowd RLE'],rows)}

Total native furniture boxes: **67,088** (64,176 train /2,912 val). Every relevant annotation has a structurally valid mask encoding: **66,629 ordinary polygon annotations** and **459 crowd RLE regions**. Polygon entries can contain multiple components for one annotation; these are not additional instances. Crowd RLE describes grouped regions and must not be counted as independent single-object masks. Bounding boxes and masks are separate source fields, and neither was invented from the other. No raster mask files or photographs were downloaded or synthesized.

All relevant boxes passed finite positive extent / image-bound checks; polygons passed coordinate-array structure checks; uncompressed RLE runs sum to source image area with matching mask dimensions. No relevant structural annotation issue was found. This validates encoded metadata, not visual boundary correctness. All source annotations are preserved in the original ZIP/extracted JSON; the candidate proposal includes1,662 byte-value-equivalent native furniture annotation objects, independently compared with the source. Some selected images contain crowd/tiny boxes beyond the ordinary large boxes that made them candidates; retain these distinctions during future training.

## Taxonomy and unknown labels

Exact categories: chair62, couch63 (proposed sofa alias), bed65, dining table67. **Desk, Cabinet and Lamp are absent from the80-class instance taxonomy.** Dining table is a specific subtype, not a replacement for every broad Table region or a desk. No substitute category or missing annotation was fabricated. Preserve native class IDs, categories and crowd flags. Keep unsupported classes and unannotated furniture unknown, never confirmed background negatives solely because COCO omits them. Cross-dataset aggregates must distinguish this four-class scope from prior seven-class Open Images metrics.

The same category and licence definitions occur in both source partitions. Occlusion, truncation, viewpoint and same-product/capture-family labels are not supplied in these instance records. Multiple co-occurring indoor object categories and positive large masks support scene/complexity prioritization heuristically; they do not prove a real indoor scene, a clear viewpoint, independence or complete visible objects. A later photo audit is still required.

## Photograph licensing and attribution coverage

The annotation licence is **COCO Consortium / CC BY4.0**. It does not clear photo copyrights. Every one of the123,287 photo records remains **individually rights-unverified**. The instances schema provides a historical licence ID, original Flickr asset URL, official object URL, dimensions and capture-date metadata; it supplies no creator name/profile or current source-page licence evidence. `candidate-inventory.jsonl` is a per-furniture-image licence/provenance ledger with explicit unverified status and null image hashes; `rights-review-queue-v1.csv` contains the400 proposed source IDs.

{table(['Licence ID','Source licence name','All photos','Furniture photos','Strict policy disposition'],license_rows)}

There are **20,327 CC BY-claim photos overall**, including **4,149 furniture photos** (3,949 train /200 val). Only those claims potentially meet this phase's narrow CC0/CC BY profile; no CC0 licence entry is provided in this archive. SA and ND are outside the chosen allowlist, which is not a claim that every such licence is noncommercial. “No known copyright restrictions” is not automatically CC0; “United States Government Work” is declared but unused by these image records and is not silently promoted to clearance. NC records remain excluded.

Current creator/source identity, licence version, attribution and possible other rights must be independently established before a photograph can qualify. Keep historical metadata and fresh evidence separate, and quarantine inaccessible, contradictory, restricted or uncertain sources. No review decisions or human confirmations were generated. Annotation attribution and exact extracted/source hashes are recorded in `license-and-source-provenance.json`; original COCO info/license records are preserved.

## Reproducible future image candidate inventory

The complete furniture inventory has **25,282 rows**. From source train2017 only, **3,132 metadata candidates /3,132 known source groups** pass the strict licence-claim, size, mask/box and scene-context criteria. Source val2017 is audited but reserved from this expansion's optimization pool. Existing train/dev/frozen datasets were not repurposed.

Full existing Open Images information tables were used to union known original-photo, normalized photographer-account and original-MD5 relationships, including excluded bridge records. All191 historical Open Images pilot memberships and their191 known groups were protected. Four train-source and one val-source furniture record intersect a protected known source group and are excluded; other filter reason counts overlap and are recorded in `metadata-audit.json`. There is no source train/val ID overlap. Exact/perceptual image duplicates cannot be checked without later approved photographs; no old or frozen photo pixels were opened. COCO lacks product/physical-object family IDs, and creator groups outside existing metadata remain unknown.

The full proposed next image inventory is **`proposed-image-candidates-v1.json`**, SHA-256 `{sha(OUT/'proposed-image-candidates-v1.json')}`. It selects400 source-train photos, one per known metadata group, with100 selection anchors per chair/sofa/bed/dining-table stratum; it does not claim equal actual category frequencies. Multi-furniture scenes yield overlapping class counts:

{candidate_table}

Seed: `{SEED}`. Ranking requires dimensions>=256, a finite in-bounds noncrowd polygon-supported furniture box with normalized sides>=0.05 and area>=0.005, a potentially allowed metadata licence and indoor context. It favors sofa/bed anchors, multiple context categories and multiple usable furniture categories, with stable hash ties. No visual predictions or old holdout labels inform selection.

The **320 train /80 development allocation is a draft**, derived by seeded20% development assignment within anchor strata, not a final partition. Known source groups are isolated and selection was independently reproduced from the inventory. Creator/account verification and capture/byte/perceptual checks can invalidate members or require a new grouped proposal before final freeze. No photo-acquisition-ready or commercially cleared image manifest exists yet, and no permanent training split was created. Individual rights verification is0/400; all candidate creator fields and image SHA fields remain null.

## Future budgets and exact next gate

For400 photographs, **100,000,000 bytes** is a scenario at250,000bytes/photo, **not a remote-size measurement**. A later proposed image cap is1GiB transfer /2GiB storage, maximum3,145,728bytes/photo and two attempts; this is not approved. Official image paths are recorded in the candidate manifest, but their actual sizes/ETags and licences must be verified and frozen before requesting photo acquisition. No photograph HEAD/GET was issued in this metadata audit.

The concrete next approval proposal is `next-rights-audit-proposal.json`: metadata/source-page verification for **only the400 listed photo IDs**, matching canonical Flickr photo/creator/licence evidence and official object HEADs; **no image bodies**, no extra archive or models; max256,000,000 response-body bytes,512MiB new storage,1MiB per source-page body with4MiB decoded cap and at most two source-page GET attempts. Expected page traffic of400×120,000=48MB is a scenario, not a guarantee. Changed/inaccessible/uncertain sources must be quarantined without silently replacing IDs. The later exact photograph manifest can contain fewer than400 eligible examples after this audit.

For **Desk/Lamp**, COCO cannot supply native supervision. The previously prepared Open Images training-metadata proposal remains a separate unapproved route (3,273,620,121 expected bytes,6,600,000,000 transfer cap /6GiB storage). No such files were acquired, and no budget was borrowed from this phase. More metadata alone will not resolve native Desk meaning or Cabinetry unit ambiguity; those semantics remain separately tracked.

## Tests, integrity and preservation

{table(['Suite','Tests','Passed'],[[r['suite'],r['tests'],r['passed']] for r in tests])}

All **{sum(r['tests'] for r in tests)} relevant tests passed**. They cover pinned source/header/range behavior, normal certificate verification, redirects, exclusive atomic receipts and completion recovery, traversal/link/missing-member/bomb rejection, bounded JSON parsing across string/number/exponent chunk boundaries, malformed JSON, licence policy, source-photo IDs, geometry, polygon-versus-RLE semantics, reproducible known-group selection and actual source counts/hashes. Initial nine acquisition/extraction safety tests also passed before acquisition; no extra download or test model was required.

Independent verification rehashed archive/extracted files, recounted both source partitions, compared all1,662 selected source annotation objects, reproduced the400-member/320–80 proposal and checked protected source groups. Historical verification passed5,100 records (5,080 unique snapshot paths,187 official holdout paths stat-only), plus Open Images metadata39, prior photo/evidence571, Phase5B79, Phase5C639, Phase6A preparation134 and acquired photo/audit203 sealed records and prior implementation hashes.

Explicit unchanged model SHA-256:

{table(['Model','SHA-256'],list(verify['model_hashes'].items()))}

All eight pinned SigLIP2 model/processor files also match their own checksum ledger; Phase5C checkpoint receipts and all prior classifier results remain protected by the historical seals. Frozen photo decoding/scoring0, detector inference/updates0, new photos0, Mac application access0. No production deployment or environment changes. See `integrity-verification.json`, atomic verified receipts and `test-results.json`.

## References

- [COCO official download definitions](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- [Independent Google TFDS sizes and SHA-256 values](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/coco.txt)
- [COCO annotation and photograph terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [COCO official annotation API/schema](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
'''
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('REPORT CREATED',OUT/'report.md')

def sys_executable():
 import sys
 return sys.executable

if __name__=='__main__':run()
