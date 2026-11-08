"""Assemble offline, read-only-source Phase 8A deliverables; never launch acquisition."""
import collections
import csv
import io
from prep_core import *
from ontology import OI, COCO, LVIS, mapping

_exclusive_commit = commit


def commit(path, obj):
    if Path(path).exists():
        if read(path) != obj:
            raise ValueError('Existing preparation result differs: '+str(path))
        return
    _exclusive_commit(path, obj)


def write_text(name, text):
    if (OUT/name).exists():
        if (OUT/name).read_text() != text.replace('\r\n','\n'):
            raise ValueError('Existing preparation text differs: '+name)
        return
    with (OUT/name).open('x') as f:
        f.write(text)


def run():
    coverage=read(OUT/'current-coverage.json')
    candidates=read(OUT/'targeted-rights-candidates-v1.json')
    registration=read(OUT/'existing33-registration-v2.json')
    synthetic=read(OUT/'synthetic-pilot-assessment.json')
    tests=read(OUT/'test-results.json')
    preservation=read(OUT/'preservation-verification.json')
    assert tests['status']==preservation['status']=='PASS'
    # CSVs are views of immutable audit JSON, never replacement training manifests.
    flat=[]
    for split,summary in coverage['split_totals'].items():
        for category,s in summary['per_category'].items():
            flat.append(dict(partition=split,category=category,images=s['images'],
                source_creator_groups=s['source_creator_components'],boxes=s['boxes'],masks=s['masks'],
                native_nonexhaustive_images=s['confirmed_native_nonexhaustive_images'],
                class_completeness_unverified_images=s['class_completeness_unverified_images'],
                cross_dataset_unit_policy_required_images=s['images_requiring_cross_dataset_unit_policy'],
                border_or_source_truncation_images=s['border_or_source_truncation_images'],
                saved_photo_rights_verified_images=s['rights_verified_saved_evidence_images']))
    source_rows=[]
    for split in coverage['split_totals']:
        rows=[r for r in coverage['rows'] if ('fresh_development' if r['split']=='development' else r['split'])==split]
        for category in CLASSES:
            for dataset in ('OpenImages','COCO','LVIS'):
                subset=[r for r in rows if any(t['native_dataset']==dataset and t['comparison_group']==category for t in r['resolved_targets'])]
                anns=[t for r in subset for t in r['resolved_targets'] if t['native_dataset']==dataset and t['comparison_group']==category]
                source_rows.append(dict(partition=split,category=category,dataset=dataset,
                    images=len(subset),source_creator_groups=len({r['known_source_component'] for r in subset}),
                    boxes=len(anns),masks=sum(a['has_mask'] for a in anns)))
    for name,rows in [('coverage-by-category.csv',flat),('coverage-by-category-and-source.csv',source_rows)]:
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_text(name,buf.getvalue())
    policies=[]
    for dataset,table in [('OpenImages',OI),('COCO',COCO),('LVIS',LVIS)]:
        for native,group in table.items():
            policies.append(dict(dataset=dataset,native_id=native,comparison_group=group,policy=mapping(dataset,native)))
    commit(OUT/'annotation-interoperability-policy.json',dict(
        version='phase8a-source-native-positive-v1',native_mapping=policies,
        original_coordinates_labels_and_masks_immutable=True,
        comparison_groups_are_not_automatic_training_relabels=True,
        table_desk_policy='Dining table, generic table and desk retain native IDs. No inheritance or project-level negative transfer. Shared units require explicit semantic confirmation.',
        cabinet_policy='Keep cabinet/cupboard/built-in cabinetry native; doors, drawers, fronts, bays and whole runs are not interchangeable objects. Do not sum source masks as independently confirmed whole cabinets.',
        lamp_policy='Lamp626 and TableLamp1051 do not prove floor-lamp identity. Wall/ceiling lights, lampshades and lampposts cannot silently become a portable whole-lamp class.',
        multi_object_policy='Preserve separate native objects, crowd/group flags and masks. Do not split group boxes or combine overlapping different furniture into a fabricated whole object.',
        lvis_completeness='Exact native neg_category_ids may grant native-class negatives. not_exhaustive_category_ids suppress completeness; all other missing native/project classes remain unknown.',
        coco_completeness='Present source-native positives only; missing objects or any of the other three target furniture classes are not confirmed negatives.',
        openimages_completeness='Preserve image-level positive/negative labels and group/truncated/occluded/depiction/inside flags; no hierarchy-wide completeness inference.',
        default_loss='Matched valid native positive: class/box/available mask. Unknown, group-of, ambiguous or ignored: zero. No background/objectness negatives.',
        optional_exhaustive_negative='Disabled by default. Exact LVIS native-positive-exhaustive unmatched terms require explicit opt-in AND all native targets retained. Never transfer to project aliases.',
        implementation='tools/phase8a-v1/ontology.py',
        production_training_loss_adapter_implemented=False,
        readiness='CPU scalar and metadata prototype tested; future detector matcher/loss integration needs independent tests before training.'))
    # Exact counts for proposed, still rights-unverified, images, by tentative partition.
    proposed=[]
    for split in ('train','development'):
        for group in CLASSES:
            rows=[r for r in candidates['images'] if r['proposed_partition']==split]
            selected=[r for r in rows if any(LVIS.get(a['category_id'])==group for a in r['original_LVIS_core_annotations'])]
            anns=[a for r in selected for a in r['original_LVIS_core_annotations'] if LVIS.get(a['category_id'])==group]
            proposed.append(dict(partition=split,category=group,images=len(selected),boxes=len(anns),masks=sum(bool(a.get('segmentation')) for a in anns),rights_verified_images=0))
    commit(OUT/'targeted-proposal-category-coverage.json',dict(rows=proposed,all_counts_metadata_only=True,photo_download_authorized=False))
    table=['| Category | Train photos / groups | Train boxes / masks | Fresh-dev photos / groups | Fresh-dev boxes / masks |',
           '|---|---:|---:|---:|---:|']
    for g in CLASSES:
        t=coverage['split_totals']['train']['per_category'][g];d=coverage['split_totals']['fresh_development']['per_category'][g]
        table.append(f"| {g.title()} | {t['images']} / {t['source_creator_components']} | {t['boxes']} / {t['masks']} | {d['images']} / {d['source_creator_components']} | {d['boxes']} / {d['masks']} |")
    source_table=['| Partition | Source | Unique photographs | Boxes | Masks |','|---|---|---:|---:|---:|']
    for split in ('train','fresh_development'):
        for source in ('OpenImages','COCO','LVIS'):
            rows=[r for r in coverage['rows'] if ('fresh_development' if r['split']=='development' else r['split'])==split and any(t['native_dataset']==source for t in r['resolved_targets'])]
            anns=[t for r in rows for t in r['resolved_targets']]
            source_table.append(f"| {split} | {source} | {len(rows)} | {len(anns)} | {sum(a['has_mask'] for a in anns)} |")
    report='''# Phase 8A: exact coverage and training readiness

Prepared offline in the Windows WSL model-training workspace. No requests, photograph acquisition, installations, model loading/inference, optimiser updates, rendering or frozen-image pixel access were performed. Existing source photographs were only read to inspect the ten inactive-label overlays. All new files live under `tools/phase8a-v1/` and `reports/phase8a-v1/`.

The combined inventory reconciles to **114 prospective training photographs and 23 fresh-development photographs**, with **361/99 boxes and 285/76 masks**, respectively. There are **108/23 known source/creator components**, not 108/23 proven independent physical scenes. Cabinet, Lamp and Bed fresh-development coverage is the binding limitation. A reliable seven-class training comparison is not ready.

## Exact current inventory

'''+ '\n'.join(table)+'''

Category rows overlap because photographs can contain multiple categories; do not sum them as photograph totals. Groups are connected known source/creator components and do not prove absence of unknown duplicate identities. These are source-native annotation counts, not independently verified whole-object counts or human-verified accuracy. Masks refer to available source instance polygons/RLE, not new model-generated masks.

'''+ '\n'.join(source_table)+'''

`coverage-by-category-and-source.csv` gives every category/source/partition, including zero rows, with group, box and mask counts. `coverage-by-category.csv` includes completeness, unit-policy and truncation measurements. `current-coverage.json` retains each photograph's source identity, partition, SHA-256, component and exact original annotations.

All 137 source-photo IDs and encoded SHA-256 hashes are unique in the combined inventory. The ten COCO photographs with additional LVIS overlays are already counted once under COCO; their 33 inactive labels are **not** added to current totals. Their source COCO/LVIS IDs remain linked, not treated as separate photographs. Original Open Images CSV lacks annotation IDs: the audit derives stable box/flag IDs with occurrence counters, preserving the historical repeated `native-box` placeholders and every native row unchanged.

The combined inventory includes only 20 eligible legacy Open Images training photos, 9 newer OI training photos, 45 COCO and 40 LVIS. Seven legacy nominal training photos without eligible positive targets do not re-enter the current prospective pool. Historical development and frozen subsets are not relabelled as fresh-development data.

## Completeness, rights and quarantine

None of the 137 images has independently verified full seven-class completeness. All 45 COCO training and 10 fresh-development photos have unverified class completeness. All currently included LVIS positive targets are outside their image's native NEX list, but this is source-native completeness only: absent other categories remain unknown. Open Images flags and exact native human image labels are preserved; broad parent/sibling negatives are never inferred.

Shared Table, Desk, Cabinet and Lamp unit semantics require an explicit policy on 45/12, 38/8, 3/0 and 10/1 train/dev images, respectively. This marks cross-dataset semantic risk, not proof all those annotations are wrong. Source border contact or truncation occurs on Sofa 23/6, Chair 20/7, Table 17/7, Bed 24/1, Desk 34/6, Cabinet 2/0 and Lamp 5/0 train/dev images. Border contact is a measurement/heuristic flag, not automatically a rejection or proof of an invalid box.

All 137 current photographs have preserved photograph-specific rights evidence from earlier audits. No new rights verification occurred. That evidence is distinct from an independent legal determination of current ownership, model/property releases or all downstream uses. Creator, evidence hashes, photograph IDs, licences and attribution stay in the referenced historical ledgers. New candidate licence IDs alone are not rights clearance.

Dispositions remain separate by acquired/source cohort:

| Cohort | Original recorded dispositions |
|---|---|
| Original OI 150 slots | 63 provisional candidates across original partitions; 32 licensing-quarantined not acquired; 5 scene quarantines; 50 unresolved |
| Phase 6A OI 41 | 14 provisional candidates; 27 quarantined |
| Phase 6D COCO 72 | 55 provisionally accepted; 16 quarantined; 1 rejected |
| Phase 7 LVIS 66 | 48 provisional source candidates; 18 quarantined |

These cohort totals include other historical partitions and must not be summed as the current train/dev pool. Nine rights-verified identity-quarantined records were not acquired, and 296 paused rights requests remain unattempted. `quarantine-and-rights-dispositions.json` preserves per-record statuses/flags and category counts without conflating unacquired rights records with image-quality quarantines.

## Conservative interoperability policy implemented

Native IDs remain authoritative: OI Sofa `/m/02crq1`, Chair `/m/01mzpv`, Table `/m/04bcr3`, Bed `/m/03ssj5`, Desk `/m/01y9k5`, Cabinetry `/m/01s105`, Lamp `/m/0dtln`; COCO chair 62, couch 63, bed 65, dining table 67; LVIS sofa 982, chair 232, table 1050, bed 77, desk 361, cabinet 181, lamp 626. Native LVIS subtypes remain distinct; grouping is descriptive rather than an automatic relabel.

* **Table/Desk:** COCO dining table is not a generic desk. OI Table, OI Desk, LVIS Table/Desk and their specialised subtypes retain different IDs. No parent/sibling inheritance supplies missing positives or negatives. Occluded desks and vanity/workstation units stay unresolved where unit scope is unclear.
* **Cabinet:** built-in Cabinetry runs, freestanding cabinets, cupboards, drawers and individual fronts/bays are not equivalent annotation units. Preserve cupboard/wardrobe ambiguity natively. Do not call fifteen front masks fifteen independently verified whole cabinets.
* **Lamp:** generic LVIS Lamp is not evidence of a floor lamp. TableLamp stays a subtype; wall/ceiling fixtures, lampposts and lampshades cannot be silently converted to portable whole lamps. The desired project lighting scope needs explicit confirmation before shared-class training.
* **Multiple objects:** retain each native box/mask, crowd/group flags, ignored regions and object identity. Do not manufacture a whole object from fragments, split group boxes, or invent omitted objects.
* **Coordinates:** retain OI normalised XYXY and COCO/LVIS pixel XYWH, declared dimensions and original mask encoding. The derived pixel view checks finite/in-frame boxes without silently clipping or editing native coordinates.

`ontology.py` implements and tests source-native scope and loss masks. Valid matched native positives enable class/box/available-mask terms. Ambiguous/group/ignored targets and unknown classes have zero loss. Background/objectness negatives default off. Exact explicit native negatives never widen to shared-project aliases. An optional native exhaustive-positive negative path is off by default and requires both an explicit opt-in and confirmation that all native targets were retained; NEX always blocks it. The CPU scalar loss fixture confirms that changing unknown-class logits cannot change loss.

This is a tested metadata/scalar prototype, **not a completed detector training loss/matcher integration**. A later training implementation must carry masks through matching, class aggregation and reductions and test gradients, target retention and calibration. Positive-only loss is not assumed to produce calibrated objectness or control false positives.

## Thirty-three inactive LVIS annotations: actual local inspection

All ten source JPEGs decode, match their recorded SHA-256, source identities and declared COCO/LVIS dimensions, and have compatible EXIF orientation. Native box/polygon coordinates register to their original local pixel frames; original annotation IDs and polygons are preserved. All ten overlays were inspected by the assistant at full resolution. This is recorded as assistant inspection, not a fabricated human review or independent ground truth.

| COCO ID | Partition | Additional native labels | Observations |
|---|---|---|---|
| 68162 | train | 1 Desk | Writing/workstation unit broadly plausible; shared semantic signoff absent |
| 131352 | train | 2 Lamp | Wall-mounted bedside lights, not floor lamps |
| 165742 | train | 2 Lamp | Small distant/reflection regions; physical unit/subtype unresolved |
| 199688 | train | 5 Lamp | Three pendants, one wall fixture, one apparent floor-standing lamp |
| 279122 | train | 2 Desk, 2 Lamp | Desk/storage ambiguity behind sofas; ceiling-fan light and base-occluded tall lamp |
| 295693 | fresh dev | 1 Lamp | Table lamp on storage furniture |
| 384375 | train | 1 Desk | Hotel writing/vanity desk partly chair-occluded |
| 436273 | train | 15 Cabinet | Kitchen cabinet fronts/bays; not established as fifteen whole cabinets |
| 468162 | fresh dev | 1 Lamp | Base hidden, table/floor subtype unresolved |
| 504586 | train | 1 Lamp | Wall-mounted reading fixture, not floor lamp |

The 33 comprise **15 Cabinet, 4 Desk and 14 Lamp** annotations; all remain **inactive**. There are zero independently verified activation IDs. Box/polygon agreement does not establish semantic correctness or a universal unit definition. Even complete later verification adds no new photographs and at most two existing fresh-dev photos with native Lamp supervision; it cannot fill Cabinet or Bed development gaps. Development annotations remain evaluation-only. `existing33-registration-v2.json` contains per-annotation provenance, geometry, assistant findings and explicit inactive/human-false flags. The bounded contact sheets and original-size overlays are indexed in `existing-overlay-gallery-v1/index.json`.

## A — Targeted real-photo expansion: highest priority

The offline selector screened existing local LVIS/COCO metadata only. It excludes all prior 400 LVIS rights IDs (including all 296 paused requests), prior COCO rights IDs, existing photo/known creator-source groups and frozen-related components. It requires original COCO-train photo metadata, a potentially compatible CC BY claim, minimum 256-pixel dimensions, at least two native furniture categories, suitable native box/mask metadata and a source-exhaustive Cabinet/Lamp/Bed anchor. Multiple furniture classes are an indoor-context proxy, not visual proof of an indoor scene. No unknown identity is silently declared independent.

The requested 45 Cabinet/45 Lamp/30 Bed search yields only **44 distinct new proposed photos**, with selection anchors 8 Cabinet, 14 Lamp and 22 Bed. Eligibility rules were not weakened to reach 120. One known component per candidate is retained. The proposed new allocation is **36 fresh-development / 8 training** (new proposed data only). Anchor allocation is 7 Cabinet/10 Lamp/19 Bed dev and 1/4/3 train. Previously frozen memberships do not move.

Across overlapping categories the exact candidate metadata provides: Sofa 8 photos/10 boxes; Chair 6/7; Table 2/2; Bed 30/34; Desk 2/2; Cabinet 9/27; Lamp 35/82. All 164 boxes have source segmentation annotations. These are native source units and licence-claimed opportunities, **zero newly rights-verified or photo-screened examples**. Per-partition counts are in `targeted-proposal-category-coverage.json`.

The exact 44 COCO/LVIS IDs, source/evidence and official object URLs, scopes and original annotations are in `targeted-rights-candidates-v1.json`. SHA-256 is bound by `approval-A-targeted-rights-audit.json`. Creator fields remain null where unavailable; no creator is guessed.

**Separate A approval:** rights-evidence audit only for these exact 44 new IDs, optional HEAD of their exact official objects, **32,000,000 cumulative response-body bytes**, **134,217,728 bytes additional allocated storage**, one evidence attempt per candidate, one worker, minimum three-second request interval, Retry-After compliance and stop-provider circuit on HTTP 429. Destination `data/lvis/rights-audits/phase8a-v1/`. No photograph bodies. Provider access must first be available without bypass; approval does not resume the 296 paused IDs. Expected evidence transfer 5.94 MB is a projection; a later photo acquisition might be 6.6–17.6 MB but has no HEAD-verified estimate or approval. If access remains limited, report unavailable/quarantined; do not try alternate identities or evade restrictions.

## B — Existing-label activation: conditional, limited benefit

No blanket activation approval is requested. `approval-B-existing-label-activation.json` lists zero approvable IDs and the remaining unit/semantic/registration requirements. Independently confirm only a precise future annotation list after agreeing whether Lamp includes fixed lights and Cabinet means a front, bay, freestanding unit or built-in run. Preserve source labels and keep development-only labels out of gradients. Recheck hashes/partitions before any separately approved activation. Current activation count remains zero.

## C — Synthetic technical pilot: useful after real-data coverage work

The saved proposal is reproducible: **100 views of 25 layouts**, 7 living rooms, 6 bedrooms, 6 offices, 6 studios, seed 20261106. Six CC0 asset models provide one identity each for Sofa/Chair/Table/Bed/Desk/Cabinet, plus a procedural floor lamp. Plans contain 500 object appearances before visibility/quality checks. Per-category planned view presence is Sofa 52, Chair 100, Table 52, Bed 48, Desk 48, Cabinet 100, Lamp 100; these are **not rendered/accepted labels**. All seven labels can appear, but single identities do not provide adequate model/style diversity.

Six exact cached asset bundles comprise **33 glTF/bin/1k texture objects, 11,026,401 expected bytes**. Original creators, CC0 sources, sizes/MD5 and API attribution are retained. Poly Haven API's required service credit is preserved separately from asset rights. No assets are downloaded now. Saved-catalogue candidates for later style diversity are listed in `synthetic-pilot-assessment.json`; no additional URL/file-size verification or licence audit was performed. In particular the cached desk-lamp endpoint was inaccessible and is not retried or worked around. A verified second floor-lamp asset remains a gap.

Vary camera height/focal length/pose, room dimensions/layouts, lights/exposure, object placement and partial occlusions with fixed seeds and collision/visibility rejection. Split future synthetic groups by asset identity and room/layout seed, not camera view. Real development/frozen sets remain real-only. Synthetic data must have a separate source tag, object identity, geometry/seed/asset/licence provenance, native renderer masks and depth units.

Pure fixture tests cover visible instance-ID raster to box/mask derivation and deterministic plans. Actual Blender integration is untested: verify glTF units/origins, multiple meshes per object, alpha, visibility, instance IDs, coordinate conventions, mask/box pixel equality, camera projection, finite metric depth and temporary EXR cleanup. Visible masks give visible boxes, not amodal geometry or a proven occlusion fraction. Reject empty/tiny/out-of-frame objects rather than inventing positives.

Blender and BlenderProc are not installed. Bundled Blender Python differs from project Python 3.14; dependency compatibility must be pinned without changing `.venv`. Existing launchers must not be invoked because automatic runtime download paths are not approved. RTX 4070 8GB is plausible for a small 640x480, 32-sample scene, but no local speed or VRAM benchmark exists. Soft VRAM target 6 GiB is not a hard reservation. Stop on OOM; CPU rendering is an explicitly separate slower fallback, not an automatic action.

Separate requests, none executed:

1. **C1 assets only:** exact six/33 objects in `approval-C1-exact-CC0-asset-bundle.json`, expected 11,026,401 bytes, cap **25,000,000 transfer bytes**, **128 MiB storage**, maximum two attempts/object; `data/synthetic/assets/phase8a-v1/`. Reverify official sources and expected hashes before acquisition, preserve attribution. No installer/rendering.
2. **C2 runtime metadata only:** exact three URLs in `approval-C2-runtime-metadata-preflight.json` (PyPI BlenderProc JSON, official BlenderProc install utility source, Blender release directory); cap **5,000,000 transfer bytes**, **20 MiB storage**, two attempts/object; `data/synthetic/runtime-preflight/phase8a-v1/`. No package/binary bodies or installation. This produces a later exact installer/wheel/version/hash/size proposal. Current runtime installation size and network budget are **unknown**, so installation approval is not ready.
3. **C3 conditional rendering:** `approval-C3-conditional-render-pilot.json` is not executable until assets, runtime metadata, separate exact installation and semantics/geometry checks pass. Proposed **100 views including five smoke views**, 25 layouts, 640x480, 32 samples, one process, max **two hours**, **1 GiB new render output**, `data/synthetic/image-pilots/phase8a-v1/`, no network/model training. Projected outputs 256–614 MiB and 5–60 seconds/view are unmeasured estimates, not promises; scratch use is monitored within the budget. Rendering needs its own approval after the runtime is installed.

## D — Later controlled seven-class training: defer

No optimiser updates or new training approval are requested now. `approval-D-training-status.json` records the missing Cabinet/Lamp/Bed real-dev coverage, shared-unit decisions and production partial-label matcher/loss integration. Once those are addressed, propose exact image/annotation IDs, source-native completeness masks, class mapping, original versus Phase 5C versus expanded head-only comparison, seed/configuration/update budget and deterministic fresh-dev-only selection. All frozen datasets stay untouched. Eight-GB head-only feasibility must be verified with a separately authorised smoke test; no current VRAM measurements are fabricated.

Rank: **A** real fresh-development coverage first; **B** independently verified existing native labels second, with limited incremental coverage; **C** a small synthetic engineering pilot third, never a replacement for real evaluation; **D** a later controlled training study after these gates. A/C approvals are independent; approving one does not approve the others, rendering or training.

## Tests, preservation and reproducibility

'''+f"**{tests['total_tests']} tests across {len(tests['suites'])} suites passed**, including 25 new Phase 8A tests; no skips/failures. "+'''
Tests cover native mapping/completeness, loss masking including optional exhaustive negatives, coordinates, deterministic underfilled selection, old-ID and component exclusion, count/hash consistency, inactive-label flags, synthetic fixtures and no-network pure operations. Historical checkpoint hashes, prior artifact seals, source archive hashes and partition references match. `preservation-verification.json` records 5,100 historical snapshot checks without mismatches, prior artifact ledgers, Phase 7 seals and unchanged OWLv2/SigLIP2/Phase5C hashes. No frozen pixels were decoded or scored. The Mac application was not accessed.

Rerun the new suite read-only with `.venv/bin/python -B -m unittest discover -s tools/phase8a-v1 -p 'test_*.py' -v`. The full suite recorder `run_tests.py` writes exclusively and intentionally refuses existing logs. Audit assemblers likewise refuse overwriting results (the report assembler may skip byte-equivalent results after an interruption); use a new versioned destination for future regeneration. Original inputs and completed experiments remain immutable. No test or audit command authorises a network, runtime or training stage.

All limits and candidate/asset manifests are machine-readable in this directory. Current preparation used zero network bytes, zero new photo/asset bodies and zero model operations. New local allocations and output/implementation SHA-256 records are supplied in `completion-record.json` and checksum files.
'''
    write_text('report.md',report)
    approvals=[]
    for name in sorted(OUT.glob('approval-*.json')):
        if name.name=='approval-requests.json':
            continue
        approvals.append(dict(path=name.name,sha256=sha(name)))
    commit(OUT/'approval-requests.json',dict(status='All actions pending; preparation only',requests=approvals,
        no_implied_approval_for_installation_rendering_or_training=True))
    print('Report, exact CSV views, interoperability policy and separate approvals assembled.')


if __name__=='__main__':run()
