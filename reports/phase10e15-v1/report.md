# Phase 10E.15 — Close Wellcome pilot and prepare modern-room acquisition

Date: 2026-11-24. Project: `/home/tristary/furniture-ai-ml`.
Parent: `f9a06a6103428bee4915acbde03862c498b66c92`; initial branch
`experiment/phase10e14-synthetic-pipeline`, clean. Work branch:
`experiment/phase10e15-modern-room-intake`. No broad code audit or refactoring.

## Decision

Close the Wellcome route for modern-room furniture improvement. Prepare a
**24-photograph quality pilot: 16 training-side / 8 independently collected
evaluation-side**, with source/photographer/room components fixed before any model
performance is observed. Those are collection targets, not acquired or consented
counts. Prefer genuine creator-submitted rooms with explicit ML/commercial
permission, supplemented by specific, appropriately licensed photographs.

There are **three exact Commons leads ready for a separately approved bounded
download-and-suitability check**, not for automatic training activation. Their
native objects total **21,251,887 bytes**. Their present descriptions suggest two
living rooms and one daily-use home workspace. Visual authenticity/suitability,
privacy and room independence are still unverified. None is allocated to
independent acceptance evaluation.

**Do not start another training cycle from the unchanged 59-photo pool.** It has
supported three-category supervision, but no new qualified labels or seven-class
independent modern-room evaluation have been added. The next useful step is data
quality acquisition/qualification, not another routine software cleanup.

## 1. Repository and CPU baseline

The E14 commit and clean worktree were verified before branching. Existing E14
and E11 artifact-seal checks passed. Record E14's established baseline as:

- 425 passing CPU tests, 33 documented skips and zero test errors/failures.
- 34 passing new synthetic pipeline tests.
- 28,342 preserved historical data/model/report files.

These are recorded E14 results; its broad suite and historical model experiments
were **not rerun**. Native PDF integration remains an explicit optional-platform
limitation. This phase adds only local intake tooling, focused tests and planning
documents. No active model, classification loss, scoring threshold or production
configuration was changed. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

## 2. Wellcome E11 final status

E11 completed and is committed. Its gallery, rights ledger, inspection report,
closed-acquisition marker and seven JPEGs are present; the existing seal check
passed without reopening acquisition.

| Item | Actual result |
|---|---:|
| Photographs acquired | 7 / 7 approved |
| Actual JPEG bytes | 2,406,919 |
| File-specific source-supported CC BY 4.0 declarations | 7 |
| Modern-room-qualified images | **0** |
| Limited historical/institutional inspection images | 7 |
| Native boxes / masks | 0 / 0 |
| New activated positives / trusted negatives | 0 / 0 |
| Verified independent source groups | 0 |

Exact acquired IDs: `efqgz854`, `bpxzpgg7`, `x6v3kwb7`, `y4n8bmat`,
`vugqdsr7`, `mhnzfe6b`, `mqxh737p`. Rights evidence retains image-specific
Wellcome catalogue/IIIF records, CC BY 4.0, source works, evidence hashes and
Wellcome Collection credit. Individual photographer/holder identity was not
invented. The seven remain inspection material, not new active training data.

Assistant visual observations from the completed E11 inspection—not human labels
or exhaustive instance counts—found clear Chairs in five photos, Beds in two,
and a clear Table in one. Six have Table/workbench/counter ambiguity and six have
Cabinet/front/built-in ambiguity. No clear Sofa, Desk or standalone Lamp was
established. Two images contain people. The laboratories may share a room/source;
hospital images share institutional context. No scene is certified Chair-negative.

The reading room and hospital rooms could support limited future box review for
large furniture, but they are not representative of ordinary modern bedrooms or
offices. Grain, shadows, clipping, medical beds and built-in/front semantics limit
value. **No further Wellcome acquisition is recommended.** Preserve E11 unchanged.
Gallery: `data/licensed-rooms/image-pilots/phase10e11-wellcome-v1/gallery.html`.

## 3. Existing usable photographs, without conflating eligibility

Existing E6/E10B metadata was reconciled by image identity/content hash rather
than summing overlapping COCO and LVIS records. All **120 source-pool photos**
were available and their recorded SHA-256 values matched. No photographs were
decoded or model-scored for this inventory.

| Existing material | Photos | Status for the next experiment |
|---|---:|---|
| Active optimization pool | **59**, 56 known source groups | Usable under its existing three-category/provisional annotation protocol; not a complete modern-room benchmark |
| Exposed internal-development panel | **19**, 19 known groups | Retain existing role; never recycle as independent acceptance evaluation |
| Inactive/unassigned rights-evidenced local pool | **42**, 39 known groups | Not newly eligible; source semantics, captures and annotations still need qualification |
| E11 Wellcome inspection pilot | 7 | Historical/atypical; zero modern-qualified |
| Historical Phase4 room integration photographs | 3 | Exposed, provisional labels; not cleared as new ML training/evaluation material by this phase |
| New fully qualified modern seven-class training photos | **0** | No acquisition/activation in this phase |
| New complete independent modern acceptance photos | **0** | Must be acquired and annotated separately |

Current admitted positive boxes, from actual E10B eligibility metadata:

| Category | Training photos / boxes | Exposed internal-development photos / boxes |
|---|---:|---:|
| Sofa | 34 / 40 | 9 / 13 |
| Chair | 42 / 114 | 11 / 39 |
| Table | 0 / 0 | 0 / 0 |
| Bed | 25 / 28 | 6 / 9 |
| Desk | 0 / 0 | 0 / 0 |
| Cabinet | 0 / 0 | 0 / 0 |
| Lamp | 0 / 0 | 0 / 0 |
| Total unique photos / positive boxes | 59 / 182 | 19 / 61 |

Per-category photo counts overlap. Zero admitted boxes do not mean an object is
absent from those images. Native records on the inactive pool include furniture
and parts, but they were not silently reactivated. Known source groups are not
proof of independent physical rooms.

The Phase4 records identify three Unsplash-origin photographs. Current Unsplash
terms restrict image use in ML/AI datasets; a generic historical image licence
record is not a fresh ML permission grant. Their existing files and historical
results remain untouched; do not count them as commercially cleared additions
without purpose-specific evidence. [Unsplash current terms](https://unsplash.com/terms).

## 4. Modern acquisition channels and exact leads

**Preferred channel: creator-submitted residential/small-office photographs.**
Obtain exact photo IDs, authorship/authority, purpose-specific permission,
commercial-derived-model scope and privacy conditions. There are currently
**zero identified/consented new contributor photographs**. A blank permission
template is not evidence. No creator was contacted, and this report grants no
permission to solicit or collect private photographs automatically.

**Secondary channel: file-specific Wikimedia Commons own-work declarations.**
Commons requires checking each file's actual licence/credit, not the CC0 licence
on structured metadata or a generic platform statement. CC BY allows reuse with
its attribution/notice conditions; CC0 waives copyright rights to its extent, but
does not resolve every other right. [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).

One official MediaWiki `imageinfo` metadata request returned 6,570 bytes. No image
or thumbnail body was requested. Exact file pages, creator declarations, current
object URLs, published dimensions/bytes and source SHA-1 are frozen in
`proposed-photo-acquisition.json`, with the metadata receipt SHA-256.

| Source ID | File / creator | Declared licence | Exact bytes | Source-described content |
|---|---|---|---:|---|
| `commons-page:150832261` | Workspace Setup.jpg / FreshDev | CC0 1.0 | 4,073,349 | Daily-use home working setup, 2024 |
| `commons-page:178637758` | Living room in Winnipeg.jpg / Alexhalipchuk | CC BY 4.0 | 6,356,815 | Christmas living room, photograph dated 2022 |
| `commons-page:195256255` | Cozy living room with a view of the lake and trees in the background at midday.jpg / Shixart1985 | CC BY 2.0 | 10,821,723 | Living room, couch and coffee-table description, 2026 |

File-specific sources: [Workspace Setup](https://commons.wikimedia.org/wiki/File:Workspace_Setup.jpg),
[Winnipeg living room](https://commons.wikimedia.org/wiki/File:Living_room_in_Winnipeg.jpg),
[Lake-view living room](https://commons.wikimedia.org/wiki/File:Cozy_living_room_with_a_view_of_the_lake_and_trees_in_the_background_at_midday.jpg).
Descriptions are metadata hints, not verified category labels. Native boxes and
masks are **zero**. Uploader account declarations support copyright provenance,
not verified people/property releases or photographic authenticity. No exact
source-ID, literal group or SHA-1 match was found against the inventoried existing
120 source photos; unknown aliases, crops and room relationships remain pending.
Do not interpret this bounded check as global perceptual duplicate clearance.

**Not proposed:** more historic museum/Wellcome scenes; older home-office material
as a substitute for current variety; or automated Unsplash/Pexels collection.
Unsplash restricts ML use. Pexels' API guidance says not to build datasets/train
ML models, and its current terms prohibit unauthorized automated extraction for
ML. No authorization from these providers is assumed. [Pexels API guidance](https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions),
[Pexels current terms](https://www.pexels.com/terms-of-service/).

## 5. Complete seven-category annotation requirements

The normative new-collection specification is
[seven-category-annotation-spec.md](seven-category-annotation-spec.md). It defines:

- Sofa; Chair including **seating and bar stools**; whole Table; dedicated Desk;
  whole Bed; enclosed whole Cabinet units; standalone floor/table/desk Lamp.
  Footstools/ottomans/step stools, counter/island ambiguity, cabinet door fronts,
  open bookcases and wall/ceiling fixtures are explicitly separated or ignored.
- Oriented-pixel XYXY visible-extent boxes; native frame/conversion preservation;
  clipped/occluded/difficult flags; one physical object per image; duplicate/native
  annotation aliases, reflections and multi-object boxes handled explicitly.
- All seven per-category completeness decisions, with reviewer/date/evidence;
  missing categories remain unknown. Ignore/protected regions do not become
  negatives. Supported negatives must be class- and region-specific with evidence.
- Native annotation and photograph rights kept separate. Source proposals and
  masks stay inactive until qualified. Current source annotations are unchanged.

Source boxes avoid redraw when trustworthy, identity-bound and compatible. Where
the source supplies none—as with all three new Commons leads—new annotation is
still necessary; automated inference is a later separately approved proposal,
never a substitute for complete independent evaluation annotation.

## 6. Two collection paths and a practical private workflow

`collection-plan.json` targets 24 diverse photographs: 16 training / 8 independent
acceptance evaluation, with at least six training and four evaluation
creator/room-connected components. Scene targets are six living rooms, four
dining/stool scenes, six bedrooms, six home/small offices and two mixed cluttered
rooms, distributed across both purposes. These are goals, not observed counts.

Training prioritizes occlusion, partial objects and distractors such as plants,
TVs, appliances, decorations and people **where permission allows**. Use supported
positive furniture and genuinely verified class-specific negatives. Leave the
remaining incompletely annotated regions ignored. Choose any model-selection
development components solely from qualified training-side groups, separately
from the eight acceptance photographs.

Independent evaluation is collected through different creators/source/room/capture
components, without selecting photos based on observed model errors. Complete the
applicable seven-category review and second annotation check, then freeze groups,
image hashes, annotations, checkpoint/prompt/preprocessing and metrics before any
model output. Do not move the exposed 19-photo panel, 18/36-photo historical panels
or other previously scored material into fresh acceptance evaluation. Preserve
all reserved/frozen memberships. Eight photos assess data quality; they cannot
establish reliable 80% precision / 90% recall. Report class counts, ignored scope
and group uncertainty. Missing class coverage blocks a seven-category claim.

The lightweight interface is implemented at `tools/phase10e15-v1/`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e15-v1/serve15.py --port 8783
```

Open `http://localhost:8783/`. It has browser-local file previews/hashing,
photographer/source/room/session fields, genuine permission-evidence file hashing,
scope confirmations, source-box import/overlay, optional draw/edit/undo, explicit
per-object review and seven-category completeness. Original native annotations
remain in the export. Unique JSON exports are inactive and private. New image
selection resets prior confirmations; annotation edits reset completeness.

No photo uploads, directory listings, remote scripts, remote annotation services,
model calls or creator messages exist in the server. Only the exact static page
and script are served; CSP blocks network connections from the page. The CLI
validator reuses E11 permission checks and optionally verifies actual oriented
dimensions/hash, historical exclusions and cross-purpose group overlap. It never
certifies rights or activates training/evaluation automatically.

See [tool instructions](../../tools/phase10e15-v1/README.md). No real photograph
was imported through the new interface. Browser automation/Node are absent, so
no browser end-to-end test is claimed; perform the documented synthetic manual
walkthrough before real intake. No new dependencies were installed.

## 7. Exact next approval and retraining decision

Recommended bounded first acquisition request:

> Acquire **only the three exact original Commons photographs** in
> `reports/phase10e15-v1/proposed-photo-acquisition.json`, for training-side
> suitability inspection. Expected image bodies **21,251,887 bytes**; maximum
> cumulative response-body transfer **60,000,000 bytes**, including metadata,
> failures and retries; maximum **12,582,912 bytes per image**; at most **two
> attempts per object**; maximum **128 MiB allocated new storage** under
> `data/licensed-rooms/image-pilots/modern-room-screening-v1/`. No substitutes,
> outreach, model inference, annotation activation or training. Reverify exact
> object identity/size/licence first; stop on changed or restricted sources.

Published SHA-1 values are source integrity claims; record SHA-256 only after a
future authorized download. Decode/orientation, photographic authenticity,
privacy, modern-room suitability, perceptual duplicates and annotation feasibility
must then be screened; quarantine uncertainty, never silently replace images.
Native resolution avoids an unverified derivative URL; max decoded RGB for the
largest photo is about 69 MiB before decoder/analysis overhead. Storage budget
allows bounded temporary files and small inspection artifacts, not unrestricted
rendered caches. Actual new image transfer this phase: **zero bytes**.

The other 21 target photographs (13 training / 8 evaluation if all three leads
qualify) have no invented IDs or grants. They require separately authorized,
exact-identity creator intake or another finalized licensed manifest. No blanket
outreach/download request is made for those unknown photographs.

**Retraining readiness:** the existing 59 photos can technically replay the
preserved three-category experiment, but there is no new evidence justifying
another cycle, and no qualified expanded seven-category set yet. Do not train.
The first post-acquisition decision should be whether new data actually adds
representative rooms, supported Table/Desk/Cabinet/Lamp labels and trustworthy
negatives—not whether the software can run another optimizer loop.

## 8. Verification and preservation

Twelve new focused CPU tests pass: permission evidence/scope gates, exact source
box binding and provenance, category-completeness contradictions, group/hash
exclusions, synthetic EXIF/photo identity, no automatic label activation, no
prechecked grants, and localhost upload/path/rebinding/CSP controls. They use
synthetic images/evidence and local HTTP only. The E14 broad baseline is recorded,
not rerun. No expensive historical experiments were executed.

Existing eligibility/source manifests, exclusion registries, E11/E14 seals and
original training context have recorded before/after hashes. All 120 source-pool
photos were rehashed; the E11 seal verifies its seven images, and the three
historical integration-image hashes match. No historical dataset, annotation,
checkpoint, prediction, partition or Mac application was modified. Public Git
artifacts contain tooling, public source metadata and reports, no private indoor
photographs, grants, signatures or contributor details. Core ML code is unchanged.

Stop here. Acquisition and all model activity require separate authorization.
