# Phase 10E.8 — Automatic Chair-confusion qualification

Completed November 23, 2026. Parent `ecafe99dc8fd288da151f51dcc00ee3f724889ff`;
branch `experiment/phase10e8-chair-candidate-qualification`.

**Final counts: 0 trusted negatives, 5 unverified candidates, 16 protected/unsuitable
candidates. Additional valid high-confidence Chair-negative query coverage: 0.**
All 21 exact E7 candidates across ten optimization photographs/ten known source
groups were inspected. No manual rectangles, invented human confirmations,
model inference, training, external downloads, deletion or deployment were performed.

## What was actually inspected

Every candidate was rejoined to its existing E5 and original OWLv2 raw query,
image hash, source-photo ID, original dimensions and source group. The original
preprocessing/query layout and inverse bottom/right square-padding transform
were verified from the saved arrays; no new predictions were computed.

The audit examined every overlapping native COCO/LVIS annotation rather than
assuming the old largest-bbox label described the crop. Continuous intersection
area with each original simple polygon was measured separately. Fractions are
not summed across overlapping COCO/LVIS masks, and missing annotations never
become background negatives. Original category IDs, source files, original
annotation checksums, licence evidence and completeness flags are retained.

Twenty-one new diagnostic plates show the full photograph, native/protected
boxes, E5 query, original same-index query, original source-mask overlays and
enlarged actual crop pixels. The assistant actually viewed all 21 plates before
authoring observations. These are **assistant visual assessments plus source
geometry checks**, not human labels, an installed local vision model, or
calibrated correctness probabilities.

The existing eligibility policy was retained: at least 32 pixels on each side
of the outward-rounded original-pixel rectangle, at least two pixels of exact
interior clearance inside one original non-Chair visible polygon, no protected
overlap, actual visual support, and actual high-confidence query coverage.
Real Chairs, seating/bar stools, possible furniture fragments, ambiguous parts
and incomplete regions cannot be approved as negatives. Footstools, ottomans
and step stools are not silently assigned Chair or treated as safe background.

## Individual decisions

Sample IDs match E7. Detailed coordinates, both models' seven scores, native IDs,
all original annotations, masks, evidence and provenance are in the dashboard and
private manifest. Scores below are the E5 Chair scores; every original same-index
Chair score is below 0.15, which is a comparison result rather than absence proof.

| Sample | E5 Chair | Final decision | Observed content / reason |
|---|---:|---|---|
| V01 | 0.764 | Protected/unsuitable | Upholstery/frame/glass boundary; possible unannotated seating fragment |
| V02 | 0.763 | Unverified | Handbag rim and hardware; best mask covers 87.8%, not the full query |
| V03 | 0.728 | Protected/unsuitable | Another partial query on the same exact handbag instance as V02 |
| V04 | 0.713 | Protected/unsuitable | Refrigerator detail; rounded rectangle only 18x24 pixels |
| V05 | 0.693 | Unverified | Sink/drainer plus clutter/background; mask coverage 58.6% |
| V06 | 0.685 | Protected/unsuitable | Possible armrest or mixed body/furniture boundary; not certified person pixels |
| V07 | 0.668 | Protected/unsuitable | Narrow dark clothing fragment; 19-pixel rounded width |
| V08 | 0.647 | Protected/unsuitable | Person edge plus adjacent brown furniture/background; possible seating content |
| V09 | 0.622 | Protected/unsuitable | Repeated partial handbag/glass-edge query on V02's source instance |
| V10 | 0.582 | Protected/unsuitable | Person/wooden divider or booth-edge fragment |
| V11 | 0.574 | Protected/unsuitable | Same handbag rim; repeated partial query and below the size gate |
| V12 | 0.567 | Protected/unsuitable | Same handbag/fittings/glass-edge source instance |
| V13 | 0.564 | Unverified | Toilet fragment plus adjacent floor/wall; mask coverage 63.3% |
| V14 | 0.539 | Protected/unsuitable | Native microwave polygon, but rounded crop only 16x11 pixels |
| V15 | 0.538 | Unverified | Unidentified dark vertical object/frame beside glass; best mask supports 21.8% |
| V16 | 0.532 | Protected/unsuitable | Overexposed person/head/background patch; visual contents insufficiently resolved |
| V17 | 0.519 | Protected/unsuitable | Small reflected/depicted decoration/frame; seating cannot be excluded |
| V18 | 0.515 | Protected/unsuitable | Tiny background patch with zero person-mask support |
| V19 | 0.514 | Protected/unsuitable | Narrow person/plant/background boundary; mixed and too small |
| V20 | 0.505 | Protected/unsuitable | Frame-like structure below flowers; possible furniture, only 19.7% plant-mask support |
| V21 | 0.501 | Unverified | Standing person's jeans/hip plus background strip; mask support 69.4% |

“Protected/unsuitable” means excluded from negative supervision. It does **not**
create a new Chair-positive annotation. Several crops visibly involve other
objects, but neither a partly covered crop nor an ambiguous furniture fragment
is thereby a trustworthy whole-query Chair-negative region.

The five remaining unverified candidates cover four photographs/four known source
groups. None of the 21 targeted candidates reaches 0.80. No complete real Chair
or stool was conclusively identified among this closed candidate subset; five
possible seating/furniture-boundary cases were conservatively protected rather
than labelled non-Chair. This is not a census of all furniture in the photographs.

## Why no new hard negatives qualified

The bbox labels were frequently misleading. The best original visible mask for
five queries is **the same handbag**, even though their earlier source-bbox
contexts said laptop or person. Other person-bbox queries mostly contain
background, plant details, a possible armrest or glare. For V18 the actual visible
person-mask overlap is zero. Native category provenance alone does not certify
every pixel inside its broad bbox.

No candidate's complete query rectangle passed the original polygon-interior
and safety requirements. Nine fail the existing integer-rectangle size gate;
the selected tiny microwave mask has essentially full area overlap but lacks
the required size and boundary clearance. The other masks are partial or
semantically unsuitable. Area near 100% alone cannot replace exact containment.
No masks, boundaries, thresholds or quality gates were relaxed to force a count.

Both saved model queries for all 21 candidates were inspected and registered to
the original image. With **zero newly admitted regions**, there are **zero new
valid Chair-negative query terms**, zero newly covered photographs/source groups
and zero positive/protected queries incorrectly assigned new negatives. This
is actual saved-query/evidence qualification, not synthetic center-grid coverage.
The original same-index box can differ from E5's box; query identity is not proof
of physical-object identity. No new inference was necessary.

## Deduplication and historical evidence

All 21 candidate IDs are carried forward from E7; 13 exact pixel footprints had
already appeared in E6. They are not counted again as new mistakes.

Five visually consistent handbag queries refer to **COCO instance 2078705 on the
same photograph**. V02, the highest-scoring representative, remains unverified;
four other partial observations are not independently weighted. These need not
have pairwise box IoU above 0.5: the relation is a repeated partial source-instance
query, not an invented strict-NMS duplicate.

That same handbag instance already has source-supported region **C174** among
the existing 41 grants. Four current candidates overlap old negative rectangles,
but none is fully contained in one. The old C174 evidence therefore cannot be
counted as a new hard confuser or expanded to include uncertified pixels.
The possible LVIS handbag alias in the five inactive C21 records remains a
separate provenance relationship; it was not silently merged or activated.

The regression checks distinguish **nine native annotation-reference joins** from
the **five strongly associated handbag observations**. A reference join alone,
especially to a broad person mask, does not establish observed object identity.
The authoritative `historical-evidence-deduplication-v2.json` makes that
distinction; the earlier diagnostic version and failed regression transcript
were retained. Final decisions remain 0/5/16.

All original 41 source regions, five inactive C21 regions, 182 verified positive
boxes, training partitions and original loss implementations remain unchanged.

## Versioned dataset and local dashboard

Private root: `data/open-images/preparation/phase10e8-v1/`.

- `dataset-index-v1.json`: hashes the authoritative component manifests.
- `hard-negative-dataset-v1.json`: disjoint trusted/unverified/protected IDs;
  **no training authorization or activation**.
- `automatic-decisions-v1.json`: all 21 assistant assessments and source evidence;
  human-reviewed/human-confirmed flags remain false.
- `candidate-source-audit.json`: original coordinates, polygons, all native
  annotations, exact class scores and mask/geometry diagnostics.
- `actual-query-coverage.json`: actual zero additional accepted query coverage.
- `historical-evidence-deduplication-v2.json`: exact/reference/possible-alias
  distinctions and zero newly trusted source instances.

Dashboard: **http://localhost:8778**. An unrelated existing service already used
8768; it was left untouched. The new server binds only 127.0.0.1 and permits GET/
HEAD reads of the fixed dashboard, 21 plates and ten authorized training source
images. Mutation endpoints, path traversal, unknown images and untrusted Host
headers are rejected. No third-party scripts, fonts or image requests are used.

Filter/search the decisions; click a plate for the original photograph, zoom,
native boxes, native visible-mask overlays and original/E5 query boxes. Expand
the score/provenance tables or export the read-only decision CSV. No rectangle
drawing or human approval is required or silently recorded. Masks are rasterized
only for display; admission uses original continuous geometry.

Restart, if needed:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e8-v1/serve8.py --port 8778
```

`--port 0` chooses a free localhost port if 8778 is later occupied. HTTP tests
cover every plate, all ten original images, API counts, CSV, headers and rejected
routes/mutations. A browser automation engine is not installed; no browser
package or other dependency was added.

## Concrete licensed training-data acquisition proposal

Because this candidate audit adds no trustworthy hard coverage, **do not repeat
the same training experiments**. The next proposed step is the exact **64-source-ID
individual-photo rights audit** described in
`licensed-training-acquisition-proposal.json`. No requests have been made.

`licensed-training-rights-candidates-v1.json` preserves exact official HTTPS S3
image URLs, source-specific Flickr page resolvers, source IDs, dimensions,
native annotation IDs and the previous training-only candidate roles. It excludes
all previously attempted/paused IDs, including the 296 paused requests, and known
historical photo/source identities. Creator and capture-group independence is
currently unknown and must be established rather than assumed.

Existing metadata-only potential coverage, **not acquired/rights-cleared counts**:

| Project concept | COCO native class ID/name | Candidate photographs | Native furniture boxes |
|---|---|---:|---:|
| Chair | 62 / chair | 63 | 169 |
| Sofa | 63 / couch | 12 | 19 |
| Table | 67 / dining table | 51 | 62 |
| Bed | 65 / bed | 13 | 18 |
| Desk | Unsupported by this COCO taxonomy | 0 | 0 |
| Cabinet | Unsupported by this COCO taxonomy | 0 | 0 |
| Lamp | Unsupported by this COCO taxonomy | 0 | 0 |

The 268 boxes overlap photographs across categories and are not 268 independent
scenes. COCO's native masks are available in existing metadata, but exact usable
mask counts, Chair/stool completeness and whole-object semantics still need audit.
This shortlist targets Chair confusers and cannot solve seven-category gaps alone.

**Stage A requires separate approval:** rights/creator/source/capture evidence
for exactly these 64 IDs; at most **32,000,000 response-body bytes**, **128 MiB
new allocated storage**, one evidence attempt per ID, one worker, at least three
seconds between provider requests. Respect Retry-After; stop on 429/restriction,
never bypass access controls or resume the paused IDs. Certificate-verified HTTPS
only. Optional single HEAD on each exact official image after rights checks
establishes only object size/availability, not copyright. Destination:
`data/coco/rights-audits/phase10e8-v1/`. **No photo bodies.**

Every source needs photograph-specific acceptable CC0/CC BY evidence, actual
creator/attribution, exact licence version, source URL, evidence hash/time and
source-group clearance. The metadata's CC BY 2.0 identifier is not ownership
verification. Current independently rights-cleared new photographs: **zero**.
Current photograph-acquisition-ready manifest: **none**.

**Stage B needs another exact survivor-manifest approval:** at most 64 cleared
photos, estimated **9.6–25.6 MB** (unverified estimate, no HEAD requests made),
**100,000,000-byte cumulative cap**, **3,145,728 bytes/image**, two attempts/image,
**512 MiB storage**, destination `data/coco/image-pilots/phase10e8-v1/`.
No automatic replacements or partition reassignment. No paid service is proposed.

Prioritize real indoor scenes with people/clothing, handbags/hardware, screens,
appliances, plants and possible furniture-frame confusion alongside fully boxed
real Chairs and seating stools. Preserve original masks and category IDs; unknown
background stays ignored. Require explicit visible-instance/Chair-stool evidence
before any new negative labels. Incomplete annotation needs its own annotation
scope, not invented absence. Future inference and training require separate
approval after rights, acquisition and annotation checks; none is implied here.

Independent seven-category real-room acceptance collection remains entirely
separate. Its verified eligible count is still **zero**; no exposed development,
reserved or frozen photographs were used for this qualification or model tuning.

## Tests and preservation

**80 CPU tests passed**: 26 E8 qualification/geometry/dashboard tests, eight E7
cache-integrity tests, 26 E6 evidence safeguards and 20 E1 mask/cache regressions.
The actual localhost endpoint returned all 21 rows and the exact 0/5/16 counts.
No GPU inference, model construction, backward pass, optimizer update or external
data acquisition occurred. No environment changes or source deletions were made.

Historical checks passed for **9,973 protected earlier data/model files** and
subsequent C8/C11/C13/C17–C21/D/E1–E7 seals. These include read-only historical
checksum verification, not decoding or mining evaluation images. Original OWLv2,
SigLIP 2, E5 and other adapted checkpoints, original annotations, old review
artifacts and frozen partitions are unchanged. The Mac application was not
accessed. `main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

**One recommended next action:** approve the exact 64-ID, 32 MB / 128 MiB
rights-evidence audit. Stop before photograph acquisition, further inference,
training, loss changes or deployment.
