"""Exact acquired/screened/combined readiness report and separate future approval scopes."""
import collections as C
from pilot_core import *

def run():
 screening=read(OUT/'screening-v2.json');ready=read(OUT/'combined-training-readiness.json');net=read(OUT/'network-ledger.json');tests=read(OUT/'test-results.json');verified=read(OUT/'integrity-verification.json')
 if tests['status']!='PASS' or verified['status']!='PASS':raise ValueError('Checks failed')
 source=manifest()['images'];quarantine=[r for r in screening['images'] if r['status']=='quarantined']
 new_table=['| Category | Newly retained train photos / boxes / masks | Newly retained dev photos / boxes / masks | Unresolved source targets |','|---|---:|---:|---:|']
 combined_table=['| Category | Train photos / known groups | Train boxes / masks | Fresh-dev photos / known groups | Fresh-dev boxes / masks |','|---|---:|---:|---:|---:|']
 for cid,name in CLASSES.items():
  a=screening['eligible_category_counts']['train'][name];b=screening['eligible_category_counts']['development'][name]
  unresolved=sum(x['category_id']==cid and x['id'] in r['unresolved_annotation_ids'] for r in screening['images'] for x in r['original_annotations'])
  new_table.append(f"| {name.title()} | {a['images']} / {a['boxes']} / {a['masks']} | {b['images']} / {b['boxes']} / {b['masks']} | {unresolved} |")
  t=ready['split_totals']['train']['per_category'][name];d=ready['split_totals']['development']['per_category'][name]
  combined_table.append(f"| {name.title()} | {t['images']} / {t['source_creator_components']} | {t['boxes']} / {t['masks']} | {d['images']} / {d['source_creator_components']} | {d['boxes']} / {d['masks']} |")
 source_table=['| Partition | Source | Unique photos | Boxes | Masks |','|---|---|---:|---:|---:|']
 for part in ('train','development'):
  for dataset in ('OpenImages','COCO','LVIS'):
   rows=[r for r in ready['source_rows'] if r['split']==part and any(t['native_dataset']==dataset for t in r['resolved_targets'])]
   targets=[t for r in rows for t in r['resolved_targets']]
   source_table.append(f"| {part} | {dataset} | {len(rows)} | {len(targets)} | {sum(t['has_mask'] for t in targets)} |")
 quarantine_text='\n'.join(f"* COCO {r['coco_image_id']:012d} ({r['proposed_split']}): {r['assistant_diagnostic']['observations']}" for r in quarantine)
 newer=read(OUT/'next-cabinet-rights-candidates-v1.json');a_request=read(OUT/'approval-A-next-cabinet-rights-audit.json')
 resource_note='`resource-usage.json` reports exact final allocated bytes including every new photograph, derivative, partial, tool, log, manifest and closeout file.'
 report=f'''# Phase 8C — targeted LVIS photograph acquisition and conservative readiness

Executed only the approval in `authorization.json`: exactly 27 pinned photographs, original 7 training / 20 fresh-development assignments. Outputs use the explicitly approved destination `data/lvis/image-pilots/phase8b-v1/`; the phase name is 8C. New isolated tools are under `tools/phase8c-v1/`. No additional rights requests, photograph IDs, metadata archives, models, package installation, model inference/scoring, training, rendering, production changes or Mac application access occurred.

## Acquisition and integrity

**27/27 downloaded**, each in one body attempt; **3,913,471 actual cumulative response-body bytes** against the 100,000,000-byte cap. All pinned source URLs, JPEG Content-Type, lengths and ETags matched on certificate-verified HTTPS HEAD/GET requests. Each photograph stayed below 3,145,728 bytes. No retries or partial files remain. The conservative transfer charge also equals 3,913,471 bytes.

Every JPEG decoded and matched the original metadata dimensions; SHA-256 is recorded in `PHOTO-SHA256SUMS`, and every downloaded body's MD5 matches its pinned single-part ETag. ETag/size checks establish object version/integrity, not copyright. Originals are never cropped, re-encoded or overwritten; gallery JPEGs are clearly separate derivatives. `network-ledger.json` records exact requests, attempts, bytes, timestamps and atomic completion state. Resume verifies completed hashes, prefix proofs and Range/ETag/length responses and cannot repeat successful transfers or exceed two body attempts/object. A CPU test exercised completed resume with networking disabled.

The 27 photograph-specific saved CC BY2.0 documents were rehashed and replayed offline before downloads, matching exact photograph ID, creator account/title and licence. All four historical-overlap exclusions, twelve unresolvable records and one insufficient-evidence record remained excluded. No old paused audit was restarted. Creator/source aliases remain isolated: eligible new images comprise six train components and twelve fresh-dev components, not nineteen proven independent physical rooms. Rights evidence is a specific source grant, not independent ownership, privacy/property/model-release or trademark clearance.

`attribution.csv` and `license-and-provenance-ledger.json` preserve creator/title, source and licence URLs, notices, change-attribution requirements, verification timestamps, evidence SHA-256, photograph SHA-256, native LVIS IDs, annotation archive hashes, boxes/polygons and federated flags. Photograph CC BY2.0 rights remain separate from LVIS annotation CC BY4.0 attribution and COCO provenance. No creator or human review was invented.

## Screening outcome

All 27 originals and their unchanged native box/polygon overlays were examined in ten paired diagnostic sheets; six ambiguous cases also received individual original/overlay inspection. This is assistant screening, not human-reviewed ground truth. All automated format/dimension/blank/blur/box-polygon checks passed. The low-level quality measures and thresholds are heuristics and do not establish dataset representativeness or semantic accuracy.

**Nineteen images** retain **39 provisional source-native boxes and masks**: **six training images / 12 targets**, **thirteen fresh-development images / 27 targets**. **Eight images are quarantined**; no image is automatically substituted. Across all 27 photos, **60 of the 99 original annotation targets remain unresolved/inactive**. These include targets within otherwise eligible photographs. All 99 original annotation dictionaries remain unchanged and recoverable; no labels, geometry or native IDs were repaired or invented. No example is automatically admitted to optimisation, and no final training lock or new partition is created.

'''+ '\n'.join(new_table)+'''

Category image counts overlap. Retained means a recognisable source-native unit and broadly consistent visible-mask registration at inspected resolution, not independent whole-object or pixel-perfect ground truth. Shared-project semantic mapping remains unverified. Blank/blur statistics, bounds checks and assistant visual findings are kept separately in `screening-v2.json` and `annotation-decisions.json`.

Eight complete-image quarantines:

'''+quarantine_text+'''

Additional annotation-level exclusions include fixed wall/headboard/ceiling fixtures, separate chandelier shades, reflected lamps, cropped or ambiguous support surfaces, heavily occluded units, a single-seat armchair-like Sofa label and a source Sofa mask that incorporates a separate ottoman. Source labels remain intact. Reflections are not admitted as additional physical objects. Cropping or occlusion alone is not automatic rejection where the native object remains recognisable; severe incomplete-unit evidence stays unresolved.

**No newly usable Cabinet target survived.** All five candidate Cabinet photos used uncertain kitchen/bathroom fronts, doors, bays or mixed storage units. Their 21 Cabinet targets remain inactive. Lamp and Bed source coverage improved, but portable lamps are kept distinct from fixed lighting; small Lamp regions and reflected units were not promoted merely to increase counts.

## Duplicate and isolation results

No encoded SHA-256 or decoded-pixel duplicates were found within these 27 photos or against saved historical Open Images/COCO/LVIS signatures and ABO hashes. No dHash<=4 or pHash63<=6 similarity flags were found against the stored comparable room-image signatures. This is bounded screening, not proof against all duplicates; historical pixels and frozen images were not decoded. Missing or differently generated historical signatures cannot provide complete perceptual coverage.

No new historical creator/source conflict or train/dev component overlap was found. All original assignments remain unchanged. The combined manifest verifies unique source photo IDs/encoded hashes and disjoint known components between training and fresh development. Shared creators group photographs conservatively without asserting they depict the same physical furniture. Unknown aliases, scene reuse across creators and actual capture independence remain limitations.

## Updated combined readiness: eligible native positives only

**120 prospective training photographs from 114 known source/creator components**, containing **373 boxes / 297 masks**. **36 fresh-development photographs from 35 components**, containing **126 boxes / 103 masks**. The previous 114/23 pool is unchanged in its historical files; the new derived assessment adds only 6/13 images and 12/27 retained targets. Source rows and all native annotations/scopes are versioned in `combined-training-readiness.json`. No COCO photograph is counted again because it also has LVIS annotations. The previously inactive 33 annotations on ten COCO photos remain inactive and excluded from all new totals.

'''+ '\n'.join(combined_table)+'\n\n'+ '\n'.join(source_table)+'''

Per-category source breakdown, border/unit-policy flags, completeness and saved-rights counts are available in the nested `split_totals.*.per_category` records. Category images/groups overlap and must not be summed as independent photos. The old pool was not newly visually re-audited; its previously recorded provisional and taxonomy limitations remain. Aggregate numbers are source-native descriptive counts, not validated project-wide labels or real-world accuracy.

Completeness/loss requirements:

* Open Images native image-label/box flags are preserved; unknown descendants and unannotated objects are not negatives.
* COCO positives retain original category/mask provenance; absent furniture classes are not confirmed background.
* LVIS source `neg_category_ids` and `not_exhaustive_category_ids` remain verbatim. Filtering out same-native targets downgrades the derived positive set to nonexhaustive, even when original source flags suggested exhaustive annotation. `derived_training_scope` records original/retained IDs and target-retention state.
* All unresolved image regions/targets remain ignored, never negative. No unmatched/background/objectness losses are authorised by this manifest. A later production matcher/loss must propagate these masks through matching and reductions; geometry-only tests do not implement detector training.
* Table/dining-table/Desk, cabinet/front/run, and floor/table/fixed-light mappings remain separate native strata. No false class equivalence or fully annotated seven-class image is claimed.

Cabinet still has **zero fresh-development photographs** and only three previously provisional training photos. New Bed development coverage rises from 2 to 11 photos; Lamp from 1 to 11. Bed/lighting examples are heavily bedroom/hotel biased, and some are truncated. These gains support investigation, not a reliable general seven-class benchmark.

## Next strategy and separately bounded approvals

**A — Highest data priority: genuinely whole Cabinet examples, with better scene diversity.** Repeating the same kitchen-front acquisition pattern is unlikely to fix whole-cabinet supervision. Local LVIS/COCO metadata has 310 COCO-train, CC BY-claimed images with 1,462 native Cabinet181 annotations and seven with thirteen FileCabinet438 annotations; these are overlapping source opportunities before exclusions, not rights-verified whole objects. The source-native file-cabinet definition is a document-storage container; it is a distinct specialised subtype and cannot silently substitute for every domestic cabinet.

A new offline shortlist contains **'''+str(newer['count'])+''' never-requested IDs**, **eight proposed fresh-development / two training**, with three native FileCabinet438 photos and seven Cabinet181 photos. Exact IDs, original boxes/masks, scopes and source URLs are in `next-cabinet-rights-candidates-v1.json`. No source was requested or independently rights-cleared. This deliberately explores single-class cabinet contexts and the distinct file-cabinet subtype rather than requiring the earlier multi-furniture context proxy; it changes no current eligibility rules, mappings or training labels. Indoor context and whole-unit identity still require later actual images. All old 400+44 IDs/paused IDs and known creator/frozen relationships remain excluded.

Separate **A approval request**: `approval-A-next-cabinet-rights-audit.json`, exact ten IDs only, rights evidence/optional exact-object HEADs, **10,000,000 response-body bytes**, **64 MiB allocated storage**, one evidence attempt/candidate, one worker, >=3-second pacing, stop on restriction/429, Retry-After compliance, no photo bodies or training; destination `data/lvis/rights-audits/phase8d-cabinet-v1/`. Evidence volume ~1.35 MB is a projection, and later photograph transfer is not HEAD-verified. Neither those rights requests nor image acquisition is executed here.

**B — Best immediate engineering milestone: prepare the partial-label training adapter and exact native dataset lock.** A broad seven-class experiment is not ready for an executable training approval because Cabinet development coverage is empty, whole-unit semantics remain unresolved, and the production partial/federated matcher/loss is not yet verified. `approval-B-conditional-head-study.json` is explicitly non-executable. It outlines a later two-configuration head-only comparison against immutable original and Phase5C baselines: at most five epochs/150 updates each, 300 total, batch1/accumulation4, one-hour wall bound, conditional mixed precision and soft 6.5GiB monitored VRAM target. This is a proposed cap, not measured runtime or a guaranteed hardware reservation; stop on OOM. No full-backbone/LoRA update, held-out scoring or deployment is proposed. Exact source IDs/target masks, unit policy, loss tests and training config require separate approval before GPU parameter updates. Existing performance figures cannot be transferred to this dataset.

**C — Useful supplementary synthetic engineering pilot, not a real-dev substitute.** Retain the previous six CC0 furniture assets plus procedural lamp plan: 100 views/25 seeded layouts, one asset identity per category, 640x480/32 samples. Single-asset identity/style diversity and synthetic-to-real shift limit its learning value; it does not solve Cabinet real-evaluation coverage. `approval-C-synthetic-stages.json` binds the existing requests. Asset stage: exact33 objects, expected11,026,401 bytes, 25,000,000-byte transfer cap and128MiB storage. Runtime-documentation stage: exact three URLs, 5,000,000-byte cap/20MiB storage. These need independent approvals. Exact Blender/runtime binary versions, download sizes and installation budget remain unknown until that metadata preflight; installation cannot be approved precisely now. Rendering is a later separate conditional request: five smoke views included in100, 25layouts, two hours,1GiB output, no model training. No assets, runtime, installation or render was acquired/executed here. Real development/frozen evaluation remains real-only.

Ranked recommendation: A for the missing real Cabinet supervision; B as parallel local engineering preparation, with actual training deferred; C as a later source-separate technical pilot. Do not tune on reserved/frozen datasets or declare weak/source-native agreement human-verified accuracy.

## Verification and resource accounting

**'''+str(tests['total_tests'])+''' tests across '''+str(len(tests['suites']))+''' suites passed**, including21 new Phase8C tests. Checks cover pinned HTTPS headers/range recovery, exclusive publication, decode/hash/MD5 identity, completed no-network resume, transfer/attempt corruption, geometry/mask validation, quarantine exclusion, unresolved-target completeness downgrades, train/dev identity isolation and source/native annotation immutability. Recorded logs and `test-results.json` contain results; no skipped tests were accepted.

`integrity-verification.json` confirms unchanged original OWLv2, Phase5C adapted checkpoint and SigLIP2 hashes; all earlier OI/COCO/LVIS/Phase8A/8B artifact and implementation seals; historical snapshot preservation; no frozen-image decoding/scoring; and no Mac access. The previous33 labels remain inactive. No historical report, split, photograph, annotation, model or production prediction was overwritten. New masks/boxes were not invented.

'''+resource_note+'''

The 512MiB cap includes all derivatives and code. `completion.json`, `implementation-sha256.json`, and `ARTIFACT-SHA256SUMS` seal this phase. No source archives were extracted and no new metadata/model bodies were downloaded. The original JPEGs and provenance ledger remain distinct from accepted native-positive target lists.

Commands (project `.venv/bin/python`, no package changes):

```bash
# Read derived coverage and individual dispositions.
.venv/bin/python -B -m json.tool data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json
.venv/bin/python -B -m json.tool data/lvis/image-pilots/phase8b-v1/annotation-decisions.json
# Rerun the isolated CPU regression suite; performs no downloads/training.
.venv/bin/python -B -m unittest discover -s tools/phase8c-v1 -p 'test_*.py' -v
```

Completed downloads must not be reacquired. The source runner supports explicit `--resume` with version/partial proofs and completed-object skipping, but no further acquisition is required. Report/screening writers refuse overwriting existing results; use a separately authorised new version for future changes. Stop here pending separate next-stage approvals.
'''
 with (OUT/'report.md').open('x') as f:f.write(report)
 print('Detailed acquisition, screening, native coverage and separate next-stage report written.',flush=True)

if __name__=='__main__':run()
