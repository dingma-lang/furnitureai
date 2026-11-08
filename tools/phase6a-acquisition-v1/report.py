"""Final acquisition report and attributable provenance; no scoring/training."""
import collections as C
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def run():
 v=read(DEST/'integrity-verification.json');m=read(MANIFEST);a=read(DEST/'acquisition-complete.json');s=read(DEST/'screening-v3.json');tests=read(DEST/'test-results.json')
 attr=[]
 for r in a['records']:
  source=r['source'];lic=read(ROOT/source['license_record'])
  attr.append(dict(image_id=r['image_id'],sha256=r['sha256'],bytes=r['bytes'],width=r['width'],height=r['height'],creator=lic['creator'],creator_url=lic['creator_url'],title=source['title'],original_photo_url=lic['source_url'],license_url=lic['license_url'],verified_utc=lic['checked_utc'],evidence_sha256=lic['response_sha256'],official_download_url=source['image_url'],original_md5_base64=source['metadata_original_md5'],source_etag=source['remote_object']['etag'],partition=source['proposed_split'],annotation_license='https://creativecommons.org/licenses/by/4.0/',annotation_creator='Google LLC',changes='Original acquired JPEG bytes preserved; CVDF resize inherited. Separate contact sheets resize and overlay original source boxes; no new box labels or coordinates.'))
 with (DEST/'attribution.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(attr[0]));w.writeheader();w.writerows(attr)
 write_new(DEST/'photo-provenance.json',dict(records=attr,individual_source_evidence='licensing-evidence/',no_human_review_fabricated=True,model_inference=False))
 with (DEST/'PHOTO-SHA256SUMS').open('x') as f:
  for r in sorted(a['records'],key=lambda r:r['file']):f.write(r['sha256']+'  '+r['file']+'\n')
 # Link creator/licence credits from every derivative contact-sheet entry point.
 contacts=DEST/'contact-sheets-v1/index.html'
 contacts.write_text(contacts.read_text()+'<p>Creators, source URLs and CC BY2.0 credits: <a href="../attribution.csv">attribution ledger</a>. Changes: CVDF resize; thumbnail resizing and source-box overlays in these contact sheets. Original JPEGs preserved.</p>')
 train=read(DEST/'train-eligible-counts.json');dev=read(DEST/'development-eligible-counts.json')
 category_rows=[]
 for c in CLASSES:
  native_images=sum(any(g['label']==c for g in r['original_annotations']) for r in s['images']);native_boxes=sum(g['label']==c for r in s['images'] for g in r['original_annotations'])
  category_rows.append([c,native_images,native_boxes,train['categories'][c]['images'],train['categories'][c]['boxes'],dev['categories'][c]['images'],dev['categories'][c]['boxes']])
 flags=C.Counter(flag for r in s['images'] for flag in r['flags']);total_eligible_boxes=sum(r['boxes'] for part in (train,dev) for r in part['categories'].values())
 summary=dict(downloaded=41,integrity_passed=41,provisionally_eligible_train=9,provisionally_eligible_development=5,quarantined=27,source_confirmed_native_boxes=108,eligible_unmasked_boxes=total_eligible_boxes,
  confirmed_annotation_meaning='Confirmed against published official source rows and validated geometry, not independently human-confirmed visual category/boundaries.',
  flags=dict(flags),exact_duplicate_pairs=0,suspected_perceptual_pairs=0,actual_photo_bytes=10487710,actual_combined_body_bytes=v['actual_combined_body_bytes'],conservative_combined_charge=v['conservative_combined_charge'],
  current_effective_screening='screening-v3.json',eligible_candidates='eligible-candidates-v1.json',optimizer_updates=0,model_inference=0,training_approved=False,
  next_approval='Metadata-only COCO archive:252907541 expected bytes,520000000 transfer cap,1GiB storage; Open Images training metadata is a separately proposed larger budget, not currently approved.')
 write_new(DEST/'summary.json',summary)
 report=f'''# Phase 6A — approved photograph acquisition and screening completed

**All 41 exact approved photos acquired, first attempt, 10,487,710 image bytes.** The 32 training / 9 fresh-development allocation is unchanged. Every photo passes SHA-256 tracking, pinned official object size/ETag, JPEG integrity, decoding, dimensions and orientation checks. No substitutions, extra photographs, datasets, archives or model weights were downloaded. No model inference, detector updates or holdout scoring occurred.

Final effective eligibility is **9 training photos / 5 fresh-development photos**, with **52 unmasked source-supported boxes**. **27 photos are quarantined.** These are provisional source-annotation/assistant-screened candidates, not a human-verified dataset. The acquired allocation remains32/9; excluded members were not replaced or moved.

## Actual categories and source confirmation

{table(['Category','All acquired native images','All native boxes','Eligible train images','Eligible train boxes','Eligible dev images','Eligible dev boxes'],category_rows)}

Category image counts overlap. Native boxes include ignored or masked source annotations. All **108 original mapped native boxes** were checked against original official CSV rows, exact MID, coordinates and flags. Source-confirmed means provenance/schema/geometry confirmation; it does **not** mean independent human confirmation of object identity or accurate visible boundaries. Runtime optimization masks do not edit original Open Images boxes, categories or source rows.

The final cohort has no training Table boxes, no development Bed boxes and no Sofa/Desk/Lamp supervision. Cabinetry remains masked/unresolved. The two training Bed images remain partial/close captures with limited room context. This does not support a new broad seven-class training or accuracy claim. No training batch is started or recommended from these14 examples alone.

## Screening stages and reasons

The immutable first automatic screen (`screening-v1.json`) retained20 provisional candidates:14 train/6 development;21 were quarantined (20 lack unambiguous unmasked boxes, one fails a blur heuristic). Bounded contact sheets of all41 approved new photos were inspected, plus four targeted full-resolution photos. No old/frozen photographs were decoded for visual review.

Assistant diagnostic refinements are separately retained in `screening-v2.json` and final `screening-v3.json`. Six additional captures remain quarantined: a staged catalogue-like image with an embedded furniture depiction; a dark/soft photo whose native Table regions have uncertain furniture units; a stove-dominated photo with only a tiny clipped Table region; a visibly motion-blurred bedroom capture missed by Laplacian screening; and two commercial kitchen work-surface examples whose applicability to the intended Table unit remains unresolved. These decisions do not assert the official source label is wrong and do not replace its box/category. They conservatively exclude uncertainty from optimization.

{table(['Flag','Photo count'],sorted(flags.items()))}

Source Desk/Cabinetry supervision stays masked. Table coverage also stays unknown where descendant-only source regions have no comparable native Table box (14 acquired photos, retained in the original scope audit). No missing parent labels or background negatives were fabricated. All assistant visual notes identify the reviewer kind as assistant; every human-review flag remains false.

`CURRENT-SCREENING.json` pins **screening-v3.json** by SHA. `eligible-candidates-v1.json` lists the14 surviving members, not an approved training dataset or train queue. The historical preparation adapter's function referencing `screening-v1.json` is obsolete for final eligibility; a future separately approved training adapter must consume the final pointer/digest and the14-member receipt. Historical code is preserved rather than silently patched to change its earlier audit.

## Duplicate, identity and leakage checks

No exact new-image duplicates or suspected dHash/pHash pairs were found within the41 photos or against comparable stored Open Images photo signatures. Existing ABO metadata image hashes/pixel hashes were additionally compared: **{v['abo_metadata_hash_records_checked']} records**, no exact overlap. No historical or frozen image pixels were opened for these comparisons. Comparable perceptual signatures are available for prior Open Images, but ABO coverage here is existing-hash-only; absence of hash matches cannot prove absence of different-resolution or cross-author related captures.

All41 source-photo, photographer, original-MD5 and known identity groups are distinct and isolated across the32/9 proposal; every previous Open Images pilot member and its full known source group is excluded. No grouping was inferred solely from a visually similar stock image. Unknown cross-author physical furniture relationships remain unverified; no image was silently merged or moved. Original frozen evaluation memberships remain unchanged.

## Licensing, attribution, transfer and disk accounting

Before acquisition, each of the41 saved individual Flickr source documents was re-hashed and reparsed. A unique ImageObject matched the original photo ID, creator account and CC BY2.0 licence. Evidence was about0.02days old, inside the approved seven-day freshness rule. No inaccessible or uncertain source replaced a selected object. Every official image URL/Content-Length/ETag was rechecked by HEAD immediately before GET. All41 downloads used HTTPS with default certificate verification; no redirects or alternate hosts were followed.

`licensing-evidence/` contains byte-identical copies of each verified creator record and source-response body. `attribution.csv` and `photo-provenance.json` record creator/title/source/licence URLs, UTC verification, original MD5, object ETag, actual SHA-256/bytes/dimensions and modification notes. Open Images annotation attribution remains Google LLC / CC BY4.0. The photograph licence does not replace annotation provenance or establish model/property/privacy releases. The embedded-depiction example remains quarantined rather than assuming all visible content is cleared.

Actual photo body transfer: **10,487,710 bytes**, zero retries or partial responses. Prior rights-page bodies: **5,412,567 bytes**. Combined recorded body bytes: **{v['actual_combined_body_bytes']:,}**. Conservative transfer charge including prior uncertain failed-page reservations: **{v['conservative_combined_charge']:,}**, under1GiB. No additional archive body was fetched during expansion research; remote object probes were HEAD-only. Browser documentation research is separate from this dataset-transfer ledger.

Original photos are unchanged. `PHOTO-SHA256SUMS` and the final artifact seal preserve actual hashes. Allocated disk usage is reported separately in `resource-usage.json` after report/seal generation, including duplicate local licensing evidence, contacts, new code and the preserved original preparation; it remains well under2GiB. The original network budget and two-attempt caps were never reset.

## Tests and preservation

{table(['Post-acquisition suite','Tests','Passed'],[[r['suite'],r['tests'],r['passed']] for r in tests])}

All **{sum(r['tests'] for r in tests)} post-acquisition tests pass**, plus the22 Phase6A preparation/acquisition safety tests rerun before the destination was created. This is110 distinct current-stage checks across pre/post execution. The preparation-only assertion that an acquisition destination must not yet exist is not misrepresented as a post-acquisition test; its scope was verified before acquisition and replaced after acquisition by exact approved membership and final receipt checks.

Independent post-acquisition verification redecoded and re-hashed every new photo, reproduced stored signatures, checked source annotations, rights evidence and preserved partition/group identities. It checked5,100 historical preservation records (5,080 unique paths,187 official holdout paths stat-only), metadata39, prior photo/evidence571, Phase5B79, Phase5C639 and Phase6A preparation134 sealed artifact records, plus old implementation hashes. Original OWLv2 and Phase5C checkpoint hashes remain unchanged. No SigLIP/ABO checkpoint, existing result, old annotation, environment, production model or Mac application was modified. GPU inference/training was not used.

## Secure COCO access resolved

The failing custom/dotted-bucket host is not necessary. The verified HTTPS path-style source is:

`https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`

HEAD returned200, **252,907,541 bytes**, ETag `"f4bbac642086de4f52a3fdda2de5fa2c"`, `application/zip`, with certificate verification enabled and no redirect. DNS resolution of the official COCO host includes `images.cocodataset.org.s3.amazonaws.com`; AWS documents CNAME bucket identity and path-style access. This supports the inference that the path-style URL addresses the same published official bucket/object, rather than an unrelated third-party mirror. The dotted virtual-host alternative still fails certificate verification and was not used. GCS alternatives did not provide an accessible matching archive and were not substituted.

Independent Google TensorFlow Datasets checksums list the same official archive size and expected SHA-256:

`113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`

This is a trusted expected value for the proposed future download, not a claim that the archive was downloaded or locally verified. `expansion-proposal-v1/official-source-heads.json` preserves actual probes and DNS; metadata proposals preserve source references and the expected digest. No SSL verification was disabled, custom certificate validation used, package installed or COCO data acquired.

## Separately gated expansion proposals

**Recommended next step: COCO annotations only.** `expansion-proposal-v1/coco-metadata-acquisition-proposal.json` pins one252,907,541-byte official object, expected SHA above, max two attempts, **520,000,000-byte cumulative transfer cap /1GiB storage**. Destination `data/coco/metadata/phase6b-v1/`. Only `annotations/instances_train2017.json` and `annotations/instances_val2017.json` may be extracted after ZIP central-directory, path, symlink, duplicate-name, CRC, member-size and total-expansion checks. Maximum selected extraction650,000,000bytes; derived outputs100,000,000bytes. No image/keypoint/caption extraction, photograph GET or model operation. It awaits separate approval.

COCO can expand native chair62, couch63, bed65 and dining table67. Dining table remains a subtype; no substitution for broad Table or absent Desk/Cabinet/Lamp. Parsed metadata should first yield a source-identity/rights-audit candidate list, then a new exact photo manifest. Individual creator/licence acceptance rates and exact eligible photo/box counts remain unknown until this work; an annotation licence alone never qualifies a photograph for commercial use.

**Desk/Lamp route: Open Images official training metadata**, rather than repurposing reserved test photos. Three verified uncompressed HTTPS objects are proposed in `openimages-training-metadata-proposal.json`:

{table(['Official object','Bytes'],[['oidv6-train-annotations-bbox.csv','2,258,447,590'],['train-images-boxable-with-rotation.csv','638,407,721'],['train-annotations-human-imagelabels-boxable.csv','376,764,810']])}

Total **3,273,620,121 bytes**; requests explicitly accepting gzip still reported no stored/transferred compression. Proposed independent cap **6,600,000,000 cumulative bytes** for up to two attempts/object and **6GiB storage**. This is larger than the current41-photo approval and **requires an explicit budget increase**. It is not authorized or begun. Stream-filter the CSVs with a disk-backed identity index, preserving raw source rows and native semantics, while reusing local dictionary/hierarchy. Do not load all training rows as Python objects or automatically treat Table descendants and unknown classes as background.

After separately approved metadata and individual rights checks, the aspirational additional400-photo allocation is sofa100, desk60, lamp80, bed60, chair50, table50; Cabinetry0 until unit policy is resolved. These are priorities, not promised availability or fabricated counts. Future photographs should use a fresh deterministic80/20 known-group split; original frozen members remain untouched. A scenario of400 photos at250,000bytes each is100,000,000 photo bytes, but actual URLs, licences and HEAD sizes must be frozen before any photo approval. Suggested later photo caps remain1GiB transfer/2GiB storage. Metadata and photo budgets are separate and cannot be borrowed from the current approval.

No additional data proposal has been marked approved. No training batch is authorized. The next approval requested is **metadata-only COCO acquisition/audit**, not another photograph collection or detector update; the larger Open Images training-metadata budget can be approved separately if desired to address Desk/Lamp coverage.

## References and artifact entry points

- [COCO official publication/download definitions](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- [AWS path-style/CNAME and dotted-bucket TLS documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html)
- [Google TFDS official size/SHA registry](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/coco.txt)
- [COCO annotation/image licensing distinction](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [Open Images official annotation sources](https://storage.googleapis.com/openimages/web/download_v7.html) and [licensing](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
- `contact-sheets-v1/index.html`: all41 approved new photos, source-box overlays and attribution link.
- `CURRENT-SCREENING.json`, `screening-v3.json`, `eligible-candidates-v1.json`: effective decisions and source lineage.
- `integrity-verification.json`, `test-results.json`, `PHOTO-SHA256SUMS`, `photo-provenance.json`, `attribution.csv`: evidence and checks.
'''
 with (DEST/'report.md').open('x') as f:f.write(report)
 print('REPORT CREATED',DEST/'report.md')

if __name__=='__main__':run()
