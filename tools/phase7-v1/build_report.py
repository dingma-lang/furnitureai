"""Assemble the read-only research evidence, proposals and unified schema."""
import ast
import collections as C
import datetime as dt
from contracts import ROOT,OUT,CATEGORIES,ALLOWED_LICENSES,read,sha,write_new

def main():
    audit=read(OUT/'metadata-audit-v1.json');pilot=read(OUT/'synthetic-pilot-v3.json')
    overlap=read(OUT/'lvis-overlap-targets-v2.json');oi=read(OUT/'openimages-local-counts-v2.json')
    research=[r for f in (OUT/'sources').glob('research-ledger*.json') for r in read(f)['records']]
    heads={r['url']:r for r in research if r.get('method')=='HEAD' and r.get('status')==200}
    lvis_objects=[]
    for split in ('train','val'):
        url=f'https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{split}.json.zip';r=heads[url]
        lvis_objects.append(dict(url=url,bytes=int(r['headers']['Content-Length']),etag=r['headers']['ETag'],
            official_document='https://www.lvisdataset.org/static/js/main.18748841.chunk.js',
            source_license='https://creativecommons.org/licenses/by/4.0/',expected_local_filename=f'lvis_v1_{split}.json.zip',
            published_full_file_sha256=None,checksum_policy='ETag is multipart, NOT an MD5. Verify TLS, pinned size/version, all ZIP CRCs and compute local SHA256; no independent whole-file checksum located.'))
    proposal=dict(schema='phase7-lvis-metadata-approval-request-v1',approved=False,objects=lvis_objects,
        expected_body_bytes=sum(r['bytes'] for r in lvis_objects),max_cumulative_body_bytes=850000000,
        max_attempts_per_object=2,max_new_allocated_storage=1073741824,destination='data/lvis/metadata/phase7-v1/',
        processing='CRC-checked safe ZIP streaming; no extraction required. Reject unexpected layout, unsafe entries and expansion beyond 4GiB per member or 100x compression ratio.',
        rights='LVIS Consortium annotation attribution/CC BY4.0 retained; photograph-specific evidence remains mandatory for any subsequent image acquisition.',
        outputs=['Exact native furniture image and instance/mask counts','Image-ID-only overlap with 77 local COCO join targets and known frozen references','Source-native federated annotation coverage ledger','Proposed annotation overlays preserving existing status/partition; no automatic eligibility upgrades'],
        source_change_policy='Stop on size/ETag/TLS/integrity/licensing changes; no unapproved redirects or substitutes',
        photographs=0,model_weights=0,training=False,scoring=False)
    write_new(OUT/'next-approval-request.json',proposal)
    rights=[dict(source='LVIS',annotation_license='https://creativecommons.org/licenses/by/4.0/',
        evidence_file='sources/lvis-main.js',evidence_sha256=sha(OUT/'sources/lvis-main.js'),
        photo_rights_status='not_established_by_annotations',new_verified_photos=0),
        dict(source='Open Images V7',annotation_license='https://creativecommons.org/licenses/by/4.0/',
        evidence_file='sources/openimages-description.html',evidence_sha256=sha(OUT/'sources/openimages-description.html'),
        photo_rights_status='per_photo_verification_required_do_not_retry_rate_limited_creators',new_verified_photos=0),
        dict(source='Objects365',status='excluded_from_commercial_pool',reason='Official academic-purpose-only restriction and photograph ownership disclaimer; annotation CC BY4.0 does not resolve dataset terms.',
             evidence_file='sources/objects365-download.html',evidence_sha256=sha(OUT/'sources/objects365-download.html'))]
    for a in pilot['assets']:
        rights.append(dict(source='Poly Haven',asset_id=a['asset_id'],creator=a['creator'],license_url=a['license_url'],
            source_url=a['source_page'],evidence_file=a['license_evidence'],evidence_sha256=a['license_evidence_sha256'],
            status='CC0_asset_policy_and_catalogue_authorship_verified; asset_body_not_acquired_or_inspected',
            attribution=a['credit'],api_credit='Powered by Poly Haven'))
    for url,count,categories in [('https://universe.roboflow.com/roboflow-100/furniture-ngpea',689,['Chair','Table','Sofa']),
                                  ('https://universe.roboflow.com/vision-help/furniture-appcw',917,['Chair','Table','Door','Bed','Sofa','Window','Monitor','Shelf','Computer','Closet'])]:
        rights.append(dict(source='Roboflow Universe',source_url=url,published_images=count,published_categories=categories,
            dataset_license_claim='CC BY4.0',status='quarantined_unverified_individual_photograph_rights',
            new_verified_photos=0,evidence_method='Official publisher landing page via web research; no per-photo creator/source evidence obtained',
            note='Do not map Closet/Shelf automatically to Cabinet or count augmented version exports as independent originals',
            estimated_download_bytes=None))
    for r in rights:r['audit_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    write_new(OUT/'license-and-source-ledger.json',dict(records=rights,photo_bodies_acquired=0))
    core=[r for r in audit['lvis']['categories'] if r['mapping']=='core_native']
    ranked=[dict(rank=1,source='LVIS annotation overlays',priority='Immediate metadata pilot; reusable photo provenance, missing classes and masks',
        verified_published_native_train_image_counts={r['proposed_category']:r['published_train_images'] for r in core},
        exact_local_additional_annotation_count=None,rights_verified_novel_photos=0,expected_metadata_transfer=proposal['expected_body_bytes']),
        dict(rank=2,source='Poly Haven CC0 synthetic rooms',priority='Most scalable known-label route with lower photo-rights friction',
        asset_bundles=6,expected_asset_transfer=pilot['expected_asset_body_bytes'],planned_images=100,independent_layouts=25,
        rendered_usable_images=0,realism_or_transfer_accuracy_established=False),
        dict(rank=3,source='Open Images full training boxes',priority='Largest real-photo reservoir; rights verification remains bottleneck',
        published_train_boxed_images=1743042,published_train_boxes=14610229,per_furniture_counts=None,
        expected_metadata_transfer=3273620121,rights_verified_novel_photos=0),
        dict(rank=4,source='Roboflow Universe',status='not_in_commercial_pool_without_individual_photo_provenance',verified_eligible=0),
        dict(rank=5,source='Objects365',status='excluded_academic_only_terms',verified_eligible=0)]
    write_new(OUT/'ranked-dataset-comparison.json',ranked)
    # Structural schema plus explicit semantic validator: contracts.validate/assert_isolated.
    rights_schema=dict(type='object',required=['status','license_url','creator','source_url','evidence_sha256','verified_utc','attribution'],
        properties={'status':{'const':'verified'},'license_url':{'enum':sorted(ALLOWED_LICENSES)},
        'creator':{'type':'string','minLength':1},'source_url':{'type':'string','format':'uri'},
        'evidence_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'verified_utc':{'type':'string','format':'date-time'},'attribution':{'type':'string','minLength':1}})
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'furniture-detection-v1','type':'object',
        'required':['schema','source','source_image_id','domain','partition','width','height','image_sha256','source_groups','photo_rights','annotation_rights','annotations','coverage','native_coverage','provenance'],
        'properties':{'schema':{'const':'furniture-detection-v1'},'source':{'type':'string'},'source_image_id':{'type':'string'},
        'domain':{'enum':['real','synthetic']},'partition':{'enum':['train','development','frozen_evaluation']},
        'width':{'type':'integer','minimum':1},'height':{'type':'integer','minimum':1},
        'image_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'decoded_pixel_sha256':{'type':'string'},
        'source_groups':{'type':'array','minItems':1,'items':{'type':'string'}},'identity_groups':{'type':'array','items':{'type':'string'}},
        'photo_rights':rights_schema,'annotation_rights':rights_schema,'asset_rights':{'type':'array','items':rights_schema},
        'coverage':{'type':'object','required':list(CATEGORIES),'additionalProperties':False,'properties':{c:{'enum':['unknown','positive_nonexhaustive','positive_exhaustive','explicit_negative']} for c in CATEGORIES}},
        'native_coverage':{'type':'object','description':'Preserve neg_category_ids/not_exhaustive_category_ids or OI native human labels, hierarchy, group flags and source-native unit masks. Aggregate seven-class coverage must never manufacture broad-class negatives.'},
        'annotations':{'type':'array','items':{'type':'object','required':['source_annotation_id','category','original','mapping_rule','bbox_xywh'],
           'properties':{'category':{'enum':list(CATEGORIES)},'original':{'type':'object'},'mapping_rule':{'type':'string'},
           'bbox_xywh':{'type':'array','minItems':4,'maxItems':4,'items':{'type':'number'}},'segmentation':{},
           'segmentation_format':{'enum':['coco_polygons','coco_uncompressed_rle','coco_compressed_rle']}}}},
        'provenance':{'type':'object'}},
        'allOf':[{'if':{'properties':{'domain':{'const':'synthetic'}}},'then':{'properties':{'partition':{'const':'train'}}}}],
        '$comment':'Run contracts.validate and assert_isolated as well: finite geometry, scope contradictions, supported mask codecs and transitive identity/duplicate isolation need semantic checks. Compressed RLE is preserved raw but not eligible until a tested decoder is supplied. This schema never implies human review.'}
    write_new(OUT/'unified-detection.schema.json',schema)
    # Objects365 native vocabulary from a primary framework implementation, not guessed integer IDs.
    classes=[]
    tree=ast.parse((OUT/'sources/objects365-classes.py').read_text())
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef):continue
        for node in cls.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='METAINFO' for t in node.targets):
                try:names=ast.literal_eval(node.value)['classes']
                except ValueError:continue
                wanted={'chair','lamp','desk','cabinet/shelf','dining table','dinning table','coffee table','side table','bed','couch'}
                classes.append(dict(version=cls.name,categories=[dict(vocabulary_position_one_based=i+1,name=n) for i,n in enumerate(names) if n.lower() in wanted],
                    native_annotation_id_verified=False,note='One-based vocabulary position is NOT asserted to be native category_id without the annotation category table.'))
    write_new(OUT/'objects365-vocabulary-audit.json',dict(versions=classes,commercial_pool='excluded',source_sha256=sha(OUT/'sources/objects365-classes.py')))
    table='\n'.join(f"| {r['proposed_category']} | {r['id']} | {r['name']} | {r['published_train_images']:,} | Not acquired |" for r in sorted(core,key=lambda r:CATEGORIES.index(r['proposed_category'])))
    oi_table='\n'.join(f"| {a['category']} | `{a['source_mid']}` | {a['images']}/{a['boxes']} | {b['images']}/{b['boxes']} | Unknown |" for a,b in zip(oi['validation']['classes'],oi['test']['classes']))
    body=f'''# Phase 7 — Labelled detection data expansion preparation

Prepared {dt.datetime.now(dt.timezone.utc).isoformat()} in `/home/tristary/furniture-ai-ml` using `.venv/bin/python` (Python 3.14.4). Scope: offline implementation, local metadata audits and bounded public text/API research. No photographs, annotation archives, 3D assets or model weights acquired. No training, inference, scoring, Blender installation or Mac-project access.

## Decision and ranked strategy

1. **LVIS annotation metadata first.** It adds source-native Desk, Cabinet and Lamp classes absent from COCO, and detailed instance masks. Reusing an identical already-licensed photograph avoids another photograph transfer. Added masks do not add independent photographs. Local matches and usable additional instances remain unverified until the archives are acquired.
2. **CC0 synthetic rooms next, as a separate training domain.** Six exact asset bundles are specified (33 objects, {pilot['expected_asset_body_bytes']:,} published bytes). Plan 100 views from 25 layouts, including an original procedural lamp. No render quality or real-world transfer benefit has been measured.
3. **Full Open Images training data for real-room scale.** Official documentation reports 1,743,042 boxed training images and 14,610,229 boxes across 600 classes. Seven-class candidate counts and rights-verified yields are unknown until its large metadata is approved. These totals are not seven-class counts or commercially cleared photographs.
4. **Roboflow sources remain quarantined** without image-specific creator/source/licence evidence.
5. **Objects365 excluded** under its official academic-purpose-only terms; seek explicit clarification before considering a commercial pool.

All counts below distinguish published statistics, locally computed counts and future targets. Newly verified commercial-compatible real training photographs in this preparation phase: **0**.

## LVIS v1 categories and annotation scope

The publisher's LVIS v1 category list and training image-count table in Detectron2 were read using AST literals, without executing downloaded code. They contain 1,203 categories. Exact published **training image counts** for core native classes:

| Project group | Native ID | Native name | Published train images | Instance/mask count |
|---|---:|---|---:|---|
{table}

These counts overlap across classes; they are not eligible-photo counts. Raw instance counts were deliberately removed from the published code table, so no box/mask counts are fabricated. Official LVIS metadata carries masks and bounding boxes. The public API README describes approximately 100k training and 20k validation images; precise split/category intersections require the archives.

Subtype categories stay separate: armchair 19 (561 images), bunk_bed 170 (24), coffee_table 285 (592), sofa_bed 301 (4), cupboard/closet 346 (249), dining_table 367 (227), file_cabinet 438 (32), table_lamp 1051 (56). See metadata-audit-v1.json for definitions, synonyms and more subtypes. A sofa bed is not assigned to sofa or bed automatically; closet/cupboard is not assumed equivalent to a standalone cabinet. Lampshade 628 and lamppost 627 are not indoor whole lamps. Cabinet 181 is not automatically the same physical unit as Open Images Cabinetry. Desk 361 still needs comparable object-unit checks.

Preserve `neg_category_ids` and `not_exhaustive_category_ids`. Absence of an annotation means unknown unless explicitly verified negative for that native class. Positive nonexhaustive categories cannot supply unmatched background negatives. Broad groupings do not establish completeness for all subtypes. Never concatenate COCO and LVIS annotations blindly: keep native annotations as separate overlays; resolve coincident boxes/masks and taxonomy conflicts without inventing objects or replacing original annotations.

### Existing-photo overlap

Protected inventory: 263 historical slots = 150 original Open Images + 41 Phase 6A + 72 COCO. Five exact source-photo joins link historical Open Images images to COCO IDs: 431236, 185221, 14883, 287854, 458410. Three retain training membership and two retain development membership. Combined with 72 acquired COCO photographs, this gives **77 distinct COCO IDs to query against LVIS**. These five are not newly discovered training images; they have existing memberships/status and may still be quarantined.

Actual LVIS membership and additional furniture masks for these IDs: **unknown**, not zero. lvis-overlap-targets-v2.json is authoritative. The earlier v1 diagnostic incorrectly included COCO self-matches in its cross-source list and is superseded; a regression test now excludes self-matches. All 60 original frozen slots and 18 eligible reserved frozen references remain protected. Unknown photograph/creator aliases and unobserved physical identities remain risks; no pixel-based duplicate screening was performed in this phase.

## Open Images V7 full training audit

Exact native categories are verified from the local official dictionary. Local raw counts (images/boxes) are shown only for context, never substituted for training data:

| Group | MID | Local validation images/boxes | Local test images/boxes | Full training counts |
|---|---|---:|---:|---|
{oi_table}

The published source training bbox CSV is **2,258,447,590 bytes**, currently HEAD-verified with unchanged ETag. Required source-image metadata is 638,407,721 bytes and human boxable image labels 376,764,810 bytes, previously HEAD-verified. Combined **3,273,620,121 bytes** before any photographs. Existing unapproved proposal: `data/open-images/image-pilots/phase6a-v1/expansion-proposal-v1/openimages-training-metadata-proposal.json` (6.6 GB cumulative transfer, 6 GiB allocated ceiling, two attempts). No training CSV has been acquired.

The offline CSV audit validates schema, IDs, finite normalized boxes and native flags. Counts retain group-of/depiction/inside/occlusion/truncation flags. Image-info joins, positive/negative human label evidence and hierarchy scope are still required for eligibility. Desk booth-like objects, Cabinetry runs versus cabinets, and Table descendants stay masked/unresolved when units differ. Source human-label completeness is class-specific, not exhaustive seven-class background supervision.

Commercial scaling is annotation-rich but rights-limited. Open Images explicitly disclaims warranties for individual photo licences. Reuse saved per-photo evidence only for the identical source photo; retain creator/account aliases and SHA/MD5 provenance. Previous Phase 6C's 275 HTTP429 records must remain quarantined: no endpoint rotation, proxies or retries intended to bypass limits. Honour Retry-After, a persistent rate-limit circuit breaker, publisher access policies and normal authorised API access. A HEAD check establishes availability, not copyright.

## Synthetic pilot and implementation limits

Authoritative plan: synthetic-pilot-v3.json; v1/v2 are development drafts. Six 1k glTF assets: sofa_02, modern_arm_chair_01, dining_table, GothicBed_01, metal_office_desk, modern_wooden_cabinet. Each includes its geometry, material textures and all published dependencies, creator credits, source page, CC0 policy evidence hash and published object MD5. Expected bytes **11,026,401** across 33 unique objects; no object body has been acquired or HEAD-confirmed yet. Recheck sizes/rights at any approved acquisition and compute SHA-256 then. No HDRIs or extra material packs are needed: analytic lights and original procedural room surfaces suffice for the first technical pilot.

Poly Haven catalogue currently lists 83 furniture assets. Its broader model catalogue has lamp entries; desk_lamp_arm_01's file endpoint returned HTTP403, which was not retried or bypassed. The pilot instead identifies its lamp as original procedural geometry, not a verified downloaded CC0 asset. API use requires a Powered by Poly Haven credit separately from optional CC0 asset attribution. Catalogue rights do not prove absence of all trademark/property concerns in a rendered commercial application.

Seed 20261106; 25 deterministic layouts, four cameras each. Planned room-type views: 28 living-room, 24 bedroom, 24 office, 24 studio. Vary floor size, furniture position, camera height/FOV and lighting. Footprints do not interpenetrate. Overlap/occlusion is possible from camera placement, **not proven until rendering**. Contact and plausible arrangement, visible-object coverage, truncation, realism, texture scaling and object geometry remain post-render checks; random placement is not a guarantee of physically realistic room design.

`synthetic.py` plans scenes and converts integer instance rasters to exact **visible-mask** boxes and uncompressed COCO RLE; fully occluded objects have no visible box. This is not an amodal-box estimate. `render_worker.py` is an approval-gated BlenderProc adapter, not a GPU-tested renderer. It verifies pinned local glTF objects, dependency paths and MD5, rejects external URI resolution, joins each asset's mesh parts into one known instance, prepares RGB/instance maps/depth, hashes outputs, and marks candidates pending quality validation. Its approval JSON must bind the exact bundle SHA and at most 100 images; no approval file was created. API/import/runtime integration must be checked in a later five-frame smoke subset **within** the 100-frame bound. Raw renderer products remain candidates, not accepted examples.

RGB resolution 640x480, 32 Cycles samples, non-antialiased camera-Z depth in metres, zero object ID for room/background. Preserve valid-depth rules and distinguish camera-Z from Euclidean distance. Instance IDs map to category, asset and scene, with all original scene/camera transforms retained. Check ID-map/class consistency, finite foreground depth, box-mask containment, minimum visible area, zero-visibility exclusion and unwanted aliasing. Denoising must not change label boundaries. Do not claim occlusion fractions without separate amodal rendering.

Blender and BlenderProc are absent. BlenderProc is GPL-3.0 software; its default example uses IKEA/CCTextures and must not be run as an asset-acquisition shortcut. Blender artwork is generally not governed by Blender's GPL, while asset licences and distributed software obligations remain separate. Python 3.14 project packages are not a drop-in Blender runtime: the current setup helper references Blender Python 3.11 and can auto-install packages. Use an explicitly approved, isolated version-pinned Blender/BlenderProc runtime; never launch its auto-installer now or change `.venv`. No bpy/package installation was attempted.

RTX 4070 8GB feasibility is **projected**, not benchmarked: sequential scenes, 1k textures, 640x480 and 32 samples should be manageable; target 4–6 GiB scene allocation, monitor peaks, stop/reduce complexity only under the future approved configuration. Planning range **5–60 seconds/view**, 8–100 minutes/100 views plus load/segmentation/export overhead; not a measured throughput promise. Budget up to 2 hours for an approved smoke+pilot, stop on OOM rather than silently fall back to CPU or download anything. Approximate final RGB+uint16-ID+float32-depth outputs 0.25–0.6 GiB/100 images; intermediate EXRs can increase this. Proposed pilot output cap 1 GiB with controlled temporary-file cleanup, plus a separate dependency-runtime storage approval. Current adapter retains temporary renderer output and therefore needs that bounded cleanup/resource integration verified before a render run.

## Other datasets

Objects365 advertises 365 categories, 2 million images and 30 million boxes. Its official download page limits the dataset to academic purposes and disclaims image ownership. The annotation CC BY4.0 statement does not resolve that restriction. Excluded from this commercial-compatible pool unless clarified in writing. Official access is through the linked Biendata registration, not an unauthorised mirror; no download/storage commitment is justified while rights fail. A rough images-only projection at 0.2–0.5 MB/photo is 400 GB–1 TB (assumption, not a published transfer size).

The primary MMDetection implementation confirms v2 vocabulary positions for Chair 3, Lamp 7, Desk 10, Cabinet/shelf 13, Couch 51, Bed 76, Dinning Table 99, Coffee Table 168 and Side Table 169. **These are one-based vocabulary positions, not verified native annotation IDs.** Cabinet/shelf is mixed and cannot become Cabinet silently. See objects365-vocabulary-audit.json for both versions. Native annotation category tables, per-photo rights and archive sizes remain unacquired. Boxes are supported; instance masks are not established by this audit.

Two concrete Universe candidates: RF100 furniture-ngpea (689 images; Chair/Table/Sofa) and vision-help furniture-appcw (917 source images; Chair/Table/Bed/Sofa plus six other classes including Shelf/Closet). Both landing pages claim dataset CC BY4.0, but no individual-photo creator/source rights ledger was found there. Both remain unverified, with **0 admitted commercial-pool photographs**. Native integer class IDs require a versioned export. COCO JSON/YOLO export options do not establish ownership; model metrics and augmented counts do not establish independent examples. Exact download bytes and duplicate/overlap counts are unknown. At an explicit 0.1–0.5 MB/original planning assumption, the 689-image set would be about 69–345 MB, before annotations/versions. No export or photograph was fetched.

## Unified schema and leakage policy

`unified-detection.schema.json` + contracts.py define source ID/version, image SHA, dimensions, domain, existing partition, creator/source/identity groups, separate photograph/annotation/asset rights, source-native annotations, source-to-seven-group mapping rules, xywh pixel boxes, polygon/RLE masks, native completeness and transformation provenance. Original annotations remain intact. Unsupported compressed masks are retained raw but fail eligibility until a tested decoder is supplied. No missing human labels or review flags are filled.

Run semantic validation as well as structural checks. A transitive image-hash/source/known-family graph fails on any cross-partition component; it never reassigns records or silently merges physical products. Generic shared images indicate a relationship to investigate, not proof of a common physical product. Keep uncertain source aliases provisional. New pixel/perceptual duplicates require image bodies and later screening. Unknown furniture classes/regions cannot supply confirmed negatives. Source-specific category/unit ambiguity masks remain intact.

Future newly eligible real data: deterministic grouped 80/20 training/fresh-development allocation within rights-cleared, independent source components, prioritising rare reliable classes and indoor scenes. Freeze the allocation before experiments; known-group conflicts quarantine rather than move historical examples. All existing development/test/frozen groups remain excluded from new training and selection. Synthetic training has its own scene/asset identity groups and is never included in the real development/frozen evaluation denominator. Report real-only per-class AP/recall/IoU and source agreement separately from any synthetic diagnostics; no model was scored now.

## Concrete route to at least 1,000 additional labelled training examples

**Target, not an available dataset:** 200 newly rights-verified real training photographs + 800 separately tagged synthetic training images = 1,000 additional training examples. Reserve another 50 newly eligible real photographs for fresh development. Start with the LVIS overlay audit and the 100-view synthetic technical pilot; overlays add supervision but do not count as new photographs. After the synthetic pilot passes geometry/visibility/realism checks, extend to eight bounded 100-view batches with disjoint scene seeds/namespaces (200 layouts). More views of six assets do not mean 800 distinct furniture identities: expand independently verified asset subtypes only under a later approved manifest. All counts refer to surviving usable images, not render attempts.

For 250 total newly eligible real photos, inspect approximately 2,000–3,000 candidate records initially (planning assumption, not a guarantee). The previous COCO funnel retained 55/400 photos after rights and image checks, and 45/400 training photos; yield can differ by source/class and rate limits. Prefer already-local rights evidence for identical images and then full OI training/LVIS source-train candidates; do not recycle old development/frozen records. No unauthorised requests to previously rate-limited pages. Never weaken rights/identity rules to meet quotas. If rights verification cannot scale lawfully, render up to 1,000 training-only synthetic images across 250 new layouts after separate approval; this reaches an image-example target, not 1,000 real scenes or proven real-room improvement.

Rough future network: LVIS metadata 414.3 MB; optional OI training metadata 3.274 GB; 250 real photographs at 0.15–0.4 MB each about 38–100 MB plus bounded rights-page retrieval (unknown until a candidate manifest). Eight synthetic batches approximately 2–4.8 GiB final outputs plus assets/runtime/temporary space; obtain separate budgets before scaling. No future source count is presented as currently rights-verified availability.

## Verification and exact next approval

No-download unit tests cover literal parsing, native taxonomy, boxes/masks, federated scope, transitive identity leakage, duplicate IDs/hashes, real/synthetic isolation, immutable output behavior, safe ZIP streaming/bombs/traversal, known-source joins, deterministic scenes, no footprint intersections, visible-mask RLE round trips, unknown-instance rejection and renderer approval gating. Existing regression suites and preservation results are stored beside this report. Read-only checks verify original OWLv2, SigLIP2, Phase5C and historical seals; official ABO holds are stat-only, never opened or scored.

Recommended immediate approval: **only the two official LVIS v1 train/validation annotation ZIPs**, exact URLs/ETags in next-approval-request.json. Expected total **414,291,789 bytes** (350,264,821 train + 64,026,968 validation); cumulative cap **850,000,000 bytes**, at most two attempts per object, allocated storage cap **1 GiB**, destination **data/lvis/metadata/phase7-v1/**. Use safe ZIP streaming without extracting the full JSON to disk. Retain annotation CC BY4.0/provenance, verify CRC/SHA and reject source changes. No photographs, models, training or scoring. A whole-file independent publisher SHA was not located; multipart ETags are not whole-file MD5 checksums. This limitation is explicit in the approval manifest.

Optional later separate approvals: the exact six-asset/33-object **11,026,401-byte** CC0 bundle (suggested 25 MB cumulative cap, two attempts, 128 MiB asset storage) and a separately pinned runtime/100-frame render configuration. No Blender installation or rendering is included in the recommended LVIS-only request.

## Primary references

- LVIS publisher API/readme: https://github.com/lvis-dataset/lvis-api ; official site licence/download text preserved in sources/lvis-main.js from https://www.lvisdataset.org/ .
- Published category definitions: https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py and training counts companion lvis_v1_category_image_count.py.
- Federated semantics: https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py .
- Open Images: https://storage.googleapis.com/openimages/web/factsfigures_v7.html and download_v7.html .
- Poly Haven: https://polyhaven.com/license ; https://polyhaven.com/models/furniture ; https://github.com/Poly-Haven/Public-API .
- BlenderProc: https://github.com/DLR-RM/BlenderProc ; https://dlr-rm.github.io/BlenderProc/blenderproc.api.renderer.html .
- Blender output licensing: https://www.blender.org/about/license/ .
- Objects365: https://www.objects365.org/download.html ; https://www.objects365.org/overview.html ; https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/objects365.py .
- Universe: https://universe.roboflow.com/roboflow-100/furniture-ngpea ; https://universe.roboflow.com/vision-help/furniture-appcw ; export format support https://supervision.roboflow.com/latest/how_to/process_datasets/ .

Research response bodies saved: **1,345,699 bytes** of public text/JSON. No Flickr creator pages were requested. Source checksums/timestamps: sources/research-ledger*.json. Small-source research does not imply archive, image, asset or installation approval.
'''
    with (OUT/'report.md').open('x') as f:f.write(body)

if __name__=='__main__':main()
