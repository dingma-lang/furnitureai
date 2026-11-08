"""Produce auditable source counts, attribution and a rights-only next-stage request."""
import collections as C
import csv
from core import *

def run():
 counts=read(OUT/'furniture-counts.json')['splits'];overlay=read(OUT/'annotation-overlay-summary-v2.json');cross=read(OUT/'existing-photo-cross-reference-v2.json');plan=read(OUT/'proposed-photo-rights-audit-v1.json');integrity=read(OUT/'integrity-verification.json');archive=read(OUT/'archive-integrity.json')
 ledger=[]
 for r in cross['records']:
  ledger.append(dict(coco_image_id=r['coco_image_id'],kind='existing_photo_annotation_overlay',
   photograph_rights_status=r['licensing_status'],evidence=r['rights_evidence'],historical_status=r['historical_status'],historical_partition=r['historical_partition'],
   annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_credit='LVIS / Gupta, Dollar, Girshick; retain source IDs and indication of conversion/overlays',
   new_photograph_acquisition_approved=False,annotation_training_approved=False))
 for r in plan['images']:
  ledger.append(dict(coco_image_id=r['coco_image_id'],kind='proposed_new_photo_metadata_only',source_photo_id=r['source_photo_id'],source_url=r['source_page_resolver'],
   photograph_rights_status=r['commercial_rights_status'],creator=None,photo_license_verified=None,evidence=None,
   annotation_license=r['annotation_license'],annotation_archive_sha256=r['annotation_archive_sha256'],proposed_partition=r['proposed_partition'],
   known_source_group=r['known_source_group'],new_photograph_acquisition_approved=False,annotation_training_approved=False,
   required_attribution='Retrieve exact same-photo creator, title, source URL, commercial licence/version and notice requirements before approval; retain LVIS annotation credit separately'))
 commit(OUT/'licensing-and-attribution-ledger.json',dict(annotation_license='https://creativecommons.org/licenses/by/4.0/',
  annotation_source_evidence='official-lvis-site.js',annotation_source_evidence_sha256=sha(OUT/'official-lvis-site.js'),records=ledger,
  new_individually_verified_photos=0,new_creator_page_requests=0,photograph_bodies=0,human_evidence_fabricated=False))
 commit(OUT/'conditional-photo-acquisition-proposal-v1.json',dict(schema='phase7-conditional-photo-acquisition-v1',exact_source_manifest='proposed-photo-rights-audit-v1.json',
  source_manifest_sha256=sha(OUT/'proposed-photo-rights-audit-v1.json'),exact_coco_photo_ids=[r['coco_image_id'] for r in plan['images']],
  exact_lvis_ids=[r['lvis_image_id'] for r in plan['images']],proposed_allocation=plan['proposed_allocation'],category_distribution=plan['category_distribution'],
  annotation_instances_including_separate_subtypes_and_parts=sum(len(r['original_lvis_furniture_annotations']) for r in plan['images']),
  core_instances_boxes_and_masks=sum(r['boxes'] for r in plan['category_distribution'].values()),photographs_eligible_for_download_now=0,
  status='BLOCKED_PENDING_INDIVIDUAL_RIGHTS_AND_IDENTITY_EVIDENCE',approved=False,estimated_image_body_bytes_range=plan['estimated_image_bytes_range'],
  estimated_photo_and_mask_contact_audit_storage_bytes_range=[128*2**20,512*2**20],
  estimate_not_remote_HEAD_verified=True,hypothetical_worst_case_at_3MiB_per_photo=plan['hypothetical_max_image_transfer'],
  remaining_checks=['same-photo source and creator identity','commercial licence/version and attribution','availability and exact object HEAD sizes','creator aliases and historical source groups','source annotation unit and mask policy','after separate photo approval: decoding, exact/perceptual duplicates, indoor suitability, visible furniture and source-mask registration'],
  no_substitutions_or_permanent_partition_changes=True))
 with (OUT/'furniture-counts.csv').open('x',newline='') as f:
  fields=['split','native_category_id','native_name','core_project_group','images','instances','boxes','segmented_instances','polygon_instances','rle_instances','explicit_negative_images','not_exhaustive_flag_images','positive_not_exhaustive_images','unknown_presence_images']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for split,data in counts.items():
   for r in data['furniture_categories']:writer.writerow(dict(split=split,**{k:r[k] for k in fields if k!='split'}))
 tables={split:{r['native_category_id']:r for r in data['furniture_categories']} for split,data in counts.items()}
 core_rows='\n'.join(f"| {name} | {cid} | {tables['train'][cid]['images']:,} | {tables['train'][cid]['instances']:,} | {tables['val'][cid]['images']:,} | {tables['val'][cid]['instances']:,} |" for cid,name in CORE.items())
 flag_rows='\n'.join(f"| {name} | {tables['train'][cid]['explicit_negative_images']:,} | {tables['train'][cid]['positive_not_exhaustive_images']:,} | {tables['val'][cid]['explicit_negative_images']:,} | {tables['val'][cid]['positive_not_exhaustive_images']:,} |" for cid,name in CORE.items())
 sub_rows='\n'.join(f"| {cid} {name} | {tables['train'][cid]['images']:,} / {tables['train'][cid]['instances']:,} | {tables['val'][cid]['images']:,} / {tables['val'][cid]['instances']:,} |" for cid,name in SUBTYPES.items())
 hashes='\n'.join(f"- `{r['file']}`: {r['compressed_bytes']:,} bytes; SHA-256 `{r['sha256']}`; ZIP member {r['uncompressed_bytes']:,} bytes, CRC32 `{r['crc32']}`." for r in archive['archives'])
 candidate_rows='\n'.join(f"| {name} ({v['native_id']}) | {v['images']} | {v['boxes']} | {v['masks']} |" for name,v in plan['category_distribution'].items())
 body=f'''# Phase 7 LVIS metadata acquisition and audit

Completed {now()} in `/home/tristary/furniture-ai-ml` with `.venv/bin/python` (Python 3.14). This is an annotation-only audit: no photograph bodies, weights, training, detector inference, synthetic rendering, installation or official holdout scoring. No Mac application access.

## Acquisition and integrity

Exactly two approved official objects were acquired, each on its first attempt:

- https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
- https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip

{hashes}

Both hashes match the independent published TensorFlow Datasets checksum table at https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/lvis/checksums.tsv . The same-source sizes match too. The prior planning report had not located this checksum file; acquisition preflight found and preserved it. Multipart S3 ETags were checked as object versions, not misinterpreted as MD5. All ZIP member CRCs passed before annotation processing.

Archive bodies total **414,291,789 bytes**. Current official licence/source JavaScript and checksum documentation add **86,462 bytes**, so actual cumulative response-body transfer is **414,378,251 bytes**. The conservative charge equals that total; no retries or partial transfers occurred. This leaves **435,621,749 bytes** below the approved 850,000,000-byte cap. Completed-resume dry run skipped both archives without new network requests.

Full extraction would produce 1,298,390,107 bytes of JSON. Archives plus full extraction would require **1,712,681,896 bytes**, beyond 1 GiB. No full JSON extraction occurred. Verified ZIP streams use bounded decompression reads and an 8 MiB per-JSON-entry parser bound; declared expansion is constrained to 4 GiB/member and 100x ratio, and unexpected names/symlinks/encryption/layout are rejected. A compact SQLite index and selected annotation overlays replace full extraction. See resource-usage.json for final allocated-byte measurements, including new implementation files; all remain below 1 GiB with a 32 MiB safety reserve.

Certificate-verified HTTPS, no redirects, exact Content-Length/ETag/type, If-Match and exact 206 Content-Range are enforced. GET attempts reserve their entire possible response body in a durable ledger before requests. Resume appends only to a hash-verified, source-bound partial prefix; uncertain partials stop rather than being discarded. Two attempts per archive are a hard limit. Successful archives are never overwritten. Recovery is implemented and protocol-tested; no actual network interruption was necessary in this acquisition.

## Exact core furniture counts

LVIS train contains **100,170 images and 1,270,141 annotation instances**; validation contains **19,809 images and 244,707 instances** across 1,203 native categories. The seven core classes have **32,070 train** and **6,757 validation** boxes/masks. Image counts overlap across categories and must not be summed as distinct photographs.

| Native class | ID | Train images | Train instances/boxes/masks | Val images | Val instances/boxes/masks |
|---|---:|---:|---:|---:|---:|
{core_rows}

Every listed core annotation has a bounding box and polygon segmentation; no core RLE or missing masks were observed. Instance counts mean source annotation objects, not polygon components. All computed relevant category image/instance counts match the archive's declared category statistics. Train and validation native definitions, IDs, synonyms and synsets agree; source image IDs are disjoint between those LVIS splits.

### Federated annotation flags

| Class | Train explicit-negative images | Train positive nonexhaustive images | Val explicit-negative images | Val positive nonexhaustive images |
|---|---:|---:|---:|---:|
{flag_rows}

Full per-category counts, unknown-presence counts, all relevant not-exhaustive flags and source geometry proxies are in furniture-counts.json and furniture-counts.csv. In these observed classes the reported not-exhaustive flags apply to positive images. Original complete lists remain in the unchanged ZIPs; derived ledgers retain the relevant native IDs. Never infer a negative from an absent annotation. A positive nonexhaustive class cannot make unmatched objects verified background; an explicit negative for a native class does not prove absence of every broad project subtype.

Cabinet has 602/1,659 positive training images marked nonexhaustive. Chair has 314/1,927. This substantially limits ordinary all-background detection losses. Border contact and small box counts are geometry measurements/proxies only, not proof of an incomplete or unusable photograph. No visual image-quality assessment was performed.

## Preserve native granularity

Distinct tracked subcategories/parts are independently counted below; none is silently merged into the seven project classes:

| Native category | Train images / instances | Val images / instances |
|---|---:|---:|
{sub_rows}

Segmentation/box and federated coverage counts for these categories are also in furniture-counts.json. Lamppost is outdoor street furniture and lampshade is a part; neither becomes an indoor whole lamp. Sofa bed has ambiguous sofa/bed use, cupboard includes closet semantics, file cabinet is a specific unit, and dining/coffee/kitchen tables retain distinct native IDs. LVIS Table 1050 is not declared equivalent to COCO Dining table 67. LVIS Cabinet 181 is not automatically Open Images Cabinetry (which can represent an entire run); Desk must still match the intended furniture unit.

An additional 14 related native classes are separately audited in additional-unmapped-furniture-categories.json, including bench, bookcase, chaise longue, drawer, dresser, footstool, futon, music stool, ottoman, church bench, recliner, step stool, stool and wardrobe. **Drawer 390 is not a whole cabinet.** Their exact train/validation images, boxes, masks and negative/nonexhaustive flags were independently streamed and match native declared counts; they have no automatic seven-class mapping. All 1,203 original category definitions remain in native-category-definitions.json and the source ZIPs.

## The 77 existing COCO IDs

All **77/77** appear in LVIS: **63 in source LVIS train, 14 in source LVIS validation**. Existing project memberships remain 63 training and 14 development; these are different partition systems and were not reassigned. Some previously acquired project-training photographs therefore appear in LVIS validation; never assume that whole source split is an independent new project holdout.

- **45 images** have **127 annotations** in the seven core classes: 36 historical training images and nine development images.
- Including explicitly separate whole-furniture subtypes, **52 images** have **178 boxes/masks**: 43 training and nine development images.
- Another **38 annotations on 15 images are lampshades**, counted separately. Counting every tracked native category gives 216 annotations on 54 images; that larger figure must not be reported as whole-furniture supervision.
- Core classes absent from COCO's taxonomy gain **24 Cabinet + 4 Desk + 20 Lamp = 48 annotations**, on 12 existing photos. This is new source-native class supervision, not proof of 48 additional physical objects.
- **33 of those annotations** (15 Cabinet, four Desk, 14 Lamp) occur on **10 previously provisionally accepted COCO photographs**. The other 15 occur on already-quarantined material and remain excluded. No quality gate is lifted. The 33 still require native unit/mask registration/partial-label checks before any training approval.

The authoritative detailed list is existing-photo-cross-reference-v2.json, with every photo ID, original COCO annotations, available native LVIS annotations, prior status/partition, scope and rights evidence. Raw selected LVIS annotations are in existing-image-lvis-overlays.json. The v2 cross-reference resolves the older OI `eligibility_status` field as well as newer `status`; v1's four generic status placeholders are superseded, with all original files untouched. annotation-overlay-summary-v2.json is authoritative for aggregate roles/opportunity.

For native Chair/Sofa/Bed only, a deterministic same-category IoU>=0.5 greedy pairing identifies possible same-object COCO/LVIS annotations. Table and other ambiguous taxonomies are not silently equated. Unmatched boxes are additional source annotations, not verified extra physical objects. All masks remain source-specific overlays; no COCO annotation was replaced or concatenated into an optimisation dataset.

## Frozen/identity isolation and rights

The complete metadata cross-reference found **five LVIS training photographs in known creator/source components linked to existing frozen material**: IDs **226517, 247929, 374451, 412151, 449835**. These are creator/source-family relationships, not exact matches to the frozen photographs and not proof of the same physical room. They are conservatively excluded from the new proposal. Zero exact frozen photograph ID matches were found in the local COCO/LVIS identity join. frozen-and-source-group-isolation.json retains those links and all train/development/frozen overlap counts; no frozen pixels were opened or scored.

Existing photo-specific CC BY2.0 evidence for all 72 acquired COCO photographs is cross-referenced by exact COCO/source IDs and saved evidence hashes. Five OI-to-COCO source-photo matches also have historical individual-photo rights evidence, but mapping their derivative images/masks needs explicit content/coordinate registration. Existing quarantine/unresolved status remains authoritative. Current saved rights are not a claim about unrelated photos from the same creator.

LVIS annotation rights are **CC BY4.0**, verified from the current official site's licence text, with LVIS/Gupta/Dollar/Girshick attribution retained. This does **not** grant photograph rights. licensing-and-attribution-ledger.json separates annotation licences, existing photograph evidence and new unverified photos. No creators, licences, releases or human review flags were invented. Publicity, privacy, property and trademark releases are not established by a photograph's copyright licence.

## Exact proposed expansion, not approved image acquisition

There are **10,335 LVIS training images with a seven-core-class annotation** in the metadata index. After excluding prior photo/audit IDs, limiting to a strict COCO licence-ID4 claim, conservative source-group isolation, source-train membership, dimensions and large structurally valid native labels, **1,267 metadata-only candidates** remain. None has newly verified individual rights. The deterministic proposal selects **400 exact photo IDs**:

- Selection anchors: 100 Desk, 100 Cabinet, 100 Lamp; 25 each Bed, Sofa, Chair and Table. These are ranking anchors, not an equal class distribution or relabelling.
- Provisional allocation: **320 training / 80 fresh development**. Every photo is from both LVIS source-train and COCO source-train. No official LVIS validation photo is selected.
- All previous 400 COCO rights-audit photo IDs are excluded, including HTTP429, unresolved and incompatible records; no creator-page requests or rate-limit retries occurred.
- 400 distinct known source-photo components, but **399 creator aliases remain unknown** and physical scene/product independence is unverified. These are not 400 confirmed independent rooms. Unknown creator/room duplicates require later evidence, and proposal partitions remain provisional until rights/identity checks.

| Native class | Proposed images | Available source boxes | Available instance masks |
|---|---:|---:|---:|
{candidate_rows}

The proposal has **905 core boxes/masks**, or **1,080 source annotations** when separately retained subtypes/parts are included. Category image counts overlap. Raw masks, boxes, native IDs, source archive SHA, source URL, provisional partition, native completeness flags, metadata claims and unresolved risks are preserved in proposed-photo-rights-audit-v1.json. This is an exact future rights-audit manifest, not a trainable dataset. conditional-photo-acquisition-proposal-v1.json lists all exact IDs, source-manifest SHA, costs and blocking conditions; **zero photos are currently eligible for download**.

Photo transfer projection: **60–160 MB** for 400 images, assuming 150–400 KB each from the prior COCO experience. No new photo HEAD checks were authorized/executed in this stage, so this is not a measured object-size sum. At a hypothetical 3 MiB/object maximum, 400 images could require 1,258,291,200 bytes; actual eligible counts/sizes must be measured and separately budgeted before image approval. Planning storage for photos/masks/contacts is roughly 128–512 MiB, also a projection. No exact-byte photo-acquisition approval is requested now.

Realistic opportunity: the existing 10 accepted COCO photos can gain 33 class annotations after scope/registration checks without adding photographs. For new photos, verified eligible count is currently **0**, with a 400-photo metadata ceiling for this proposed audit. The previous COCO funnel retained 55/400 after rights and image checks. If that descriptive rate held, 400 new candidates would yield about **55 usable images**, not 400; that is an unvalidated planning proxy, not a guarantee or confidence interval. Rate limits or poor source evidence can yield zero. Underrepresented categories and native masks make these candidates promising, but there is no currently verified route to claim 1,000 new commercially cleared real photographs from these records alone.

## Verification and next approval

Relevant CPU-compatible regression tests and historical preservation checks pass; final counts/logs are in test-results-final.json. Original OWLv2, SigLIP2 and selected Phase5C hashes match. All nine historical artifact ledgers, historical snapshots and the Phase7 preparation seal remain intact. Official ABO holds remain stat-only. No source code from completed phases, dataset partitions, checkpoint weights or historical predictions were modified. No full JSON extraction or photograph bodies exist in this output.

**Next approval requested:** rights/provenance evidence audit of exactly the **400 IDs** in proposed-photo-rights-audit-v1.json; preserve proposed **320/80** membership. Destination **data/lvis/rights-audits/phase7-v1/**, maximum cumulative response-body transfer **128,000,000 bytes**, maximum new allocated storage **512 MiB**, maximum **one source-evidence attempt per candidate**; exact official photo-object HEAD checks permitted. No photograph bodies, models, inference, training or rendering. Stop a provider circuit on HTTP429; no proxy/endpoint rotation, inaccessible-source bypass, silent substitutions or automatic licence approval. Quarantine unresolved records. The exact signed-scope-by-hash request is next-approval-request.json. Only a later separately approved manifest of individually rights-cleared survivors may authorize image acquisition.

Sources: official LVIS site https://www.lvisdataset.org/ (source/licence text saved in official-lvis-site.js); LVIS API and federated evaluation https://github.com/lvis-dataset/lvis-api ; independent checksum documentation https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/lvis/checksums.tsv . All actual source URLs, TLS/HEAD metadata, checksums and timestamps are preserved in preflight.json, transfer-ledger.json and archive-integrity.json.
'''
 with (OUT/'report.md').open('x') as f:f.write(body)
 print('Report and separate licensing/conditional-acquisition ledgers written',flush=True)

if __name__=='__main__':run()
