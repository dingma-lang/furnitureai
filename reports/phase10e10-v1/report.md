# Phase 10E.10 — permitted-access and real-room metadata discovery

Prepared for 23 November 2026 (America/Los_Angeles). Source receipts use UTC;
requests after local evening carry 24 November UTC dates. Project and interpreter:
`/home/tristary/furniture-ai-ml`, existing `.venv/bin/python`, CPU-only stdlib.
No photograph, thumbnail, preview, model, annotation archive or GPU operation.

## Result and decision

A legitimate no-auth **Wellcome Collection metadata route** worked. It yielded
78 unique image records. Of those, **76 have image-specific CC BY declarations
with attribution supporting conditional reuse**, and **2 lack sufficient credit
or creator evidence**. These are source-supported rights findings, not a warranty
of ownership, commercial personality/property rights or model legality everywhere.
The initial search also returned **20 work-level records**, all unresolved pending
exact photograph identification and adequate rights/attribution evidence.

| Scope | Rights-supported A | Unresolved B | Unsuitable C |
|---|---:|---:|---:|
| New image-level records | 76 | 2 | 0 |
| New initial work-level records | 0 | 20 | 0 |
| New metadata records total | 76 | 22 | 0 |
| Preserved original Flickr IDs, separately | 0 | 64 | 0 |

Only **98 new metadata records** were retained against the 200-record ceiling.
Work-level records are not counted as 20 verified additional photographs.
Rights-supported inventory excludes both unresolved image records; unresolved
inventory preserves all 22 new unresolved records and references the old64 ledger.
Unclear restrictions are not relabelled as definitive prohibitions.

This does **not** establish a materially better training dataset. There are no
new verified negatives, detection boxes, instance masks or visually verified
rooms. A narrow seven-photo inspection pilot is supportable, but another training
experiment is not recommended. Independent seven-category evaluation readiness
remains zero.

## Flickr route and required user action

The [official Flickr SDK](https://github.com/flickr/flickr-sdk) documents a key-based
public API route and gives `flickr.photos.getInfo` as its metadata example. The
licence dictionary endpoint is `flickr.photos.licenses.getInfo`; resolving its
licence ID does not itself establish ownership or photo-use permission. Current
SDK README was retrieved from the provider's repository with verified TLS.

No Flickr credential was available in this process. No credentials were requested,
generated, read from secret files, printed or committed. No request to either API
method, or to any of the64photo pages, was made. Current commercial-ML access
permission was **not established**. Flickrhelp returned403; the provider was
stopped. The earlier photograph-page robots denial and all64B outcomes remain
unchanged. The SDK's mocked unauthenticated example is not access permission.

Use your own Windows browser to review the [API documentation](https://www.flickr.com/services/api/),
[API terms](https://www.flickr.com/services/api/tos/) and current platform terms.
If eligible, request the appropriate commercial application access using
[the official application process](https://www.flickr.com/services/apps/create/),
accurately explaining metadata auditing, storage and intended ML use. Ask the
provider for clarification if those terms do not clearly permit the intended use.
Do not request private third-party material. If legitimate access is granted,
configure credentials privately outside Git/chat and report only that access is
configured. A later permitted audit must still verify each exact photo, creator,
licence version, attribution and usage restrictions. A key is not a rights grant.

## Alternative-source findings and restriction handling

- **Wellcome:** the [official repository](https://github.com/wellcomecollection/catalogue-api)
  and [catalogue reference](https://developers.wellcomecollection.org/api/catalogue)
  explicitly document public search access without signup/authentication. Its
  image interface supplies photo-specific licence and credit fields. No-auth
  read-only serial metadata queries were used; no collection snapshots acquired.
- **Wikimedia Commons:** API and reuse documentation were accessible. The current
  robots policy denies `/w/` query paths for this client. No Commons queries,
  scraping or alternate endpoint were attempted. General API guidance alone was
  not used to override the user's no-robots-bypass boundary. Commons remains a
  promising source only after permitted access is clarified.
- **Library of Congress:** JSON API documentation accessible; collection-specific
  rights page403. Stopped before image discovery. Highsmith/HABS are potential
  future channels, not cleared photographs from this audit.
- **Smithsonian:** official OpenAccess repository accessible, but current rights
  page403. No S3/API fallback used to circumvent the stop, no rights upgrades.
- **Met:** policy page429. No retry or alternate collection API call; restriction
  preserved. The receipt index records Retry-After if provided.
- **NASA:** current [media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/)
  accessible. It distinguishes third-party copyright, identifiable people,
  endorsement/identifiers and SI/training conditions. No NASA photograph records
  were discovered or cleared, and no image bodies requested. Generic government
  ownership assumptions would not clear individual images or all proposed uses.

All source evidence URLs, UTC timestamps, response hashes, TLS certificate hashes,
statuses and measured transfers are in `source-evidence-index.json`; raw source
bodies remain under ignored local metadata storage. No named bot impersonation,
proxy, identity rotation, authentication bypass or endpoint fallback was used.

## Category and scenario evidence

| Target category | Image records with title/subject hints | Interpretation |
|---|---:|---|
| Sofa | 0 | Missing reliable metadata coverage |
| Chair | 2 | Restraint-chair and sedan-chair titles; neither accepted as ordinary Chair supervision |
| Table | 2 | Textual table mentions; no boxes or completeness evidence |
| Bed | 2 | Bedroom context only; no confirmed visible Bed objects |
| Desk | 0 | Missing reliable metadata coverage |
| Cabinet | 2 | Laboratory glass-front storage; whole-cabinet/front/built-in semantics unresolved |
| Lamp | 0 | Search matches did not substantiate standalone Lamp coverage |

Hints overlap, do not constitute labels, and do not establish negative-category
completeness. Hospital wards may contain beds or chairs, but no such object counts
were invented. All78image records have **zero available native detection boxes
and zero instance masks** in this metadata interface. No claim is made that masks
exist elsewhere. Missing boxes never establish Chair-free content.

Much of this collection is historical medical material, street scenes or portraits.
Office/lamp/sofa keyword searches often returned irrelevant records or none.
No visual difficulty, indoor suitability or safety passed based on keywords.
People, patients, medical context, restraints and minors require a conservative
later suitability/privacy assessment. No previews were fetched to inspect them.

## Identity and grouping

There are **8 provisional declared-contributor buckets**: seven named-contributor
buckets and one shared unknown-creator bucket. These are **not8verified independent
photographers or rooms**. Named contributors may be organisations, and their roles
and aliases are not fully verified. All unknown creators are grouped together,
rather than counted as independent source groups.

Known source IDs/URLs were compared with the historical exclusion registry and
local licence ledgers. No exact historical ID/URL or contributor-name overlap was
found. This does not rule out reuploads. Same-title/contributor clusters flag
possible reprints of the same historical negative. Exact pixel hashes, perceptual
similarity and physical room/capture independence remain unverified without images.
Native source IDs, work IDs, original image IDs and descriptive evidence survive.
No dataset partitions were assigned or changed, and no holdout images were read.

## Acquisition proposals, not current authorization

`potential-training-acquisition.json` proposes exactly seven licensed room-scene
images for **training-side visual/annotation feasibility inspection only**:

| Wellcome image ID | Native image ID | Metadata scene |
|---|---|---|
| efqgz854 | L0028138 | German Hospital main kitchen |
| bpxzpgg7 | L0014195 | Laboratory interior with bench/shelves |
| x6v3kwb7 | L0014196 | Laboratory with right-side cabinets |
| y4n8bmat | L0014193 | Laboratory with left-side cabinets |
| vugqdsr7 | V0031425 | American Room with reader's table/display |
| mhnzfe6b | L0028136 | Single-patient hospital room |
| mqxh737p | L0028137 | Two-patient hospital room |

All seven share the conservative unknown-creator bucket. No independent-group
claim is made. Exact source service IDs and original dimensions were checked via
IIIF `info.json` metadata. Source profiles support the proposed size transformation.
The future URLs request JPEG derivatives fitting1600×1600; they have **not** been
fetched. No encoded byte lengths are advertised in that metadata. Estimates are
not fabricated from pixel counts.

Separate proposed approval, bound by the manifest SHA256 in
`next-photo-approval-request.json`:

- Exactly those7source IDs; no replacements.
- Destination `data/licensed-rooms/image-pilots/phase10e11-wellcome-v1/`.
- Maximum512,000bytes/object, two attempts/object.
- Maximum8,000,000cumulative transfer bytes including retries/metadata.
- Maximum64MiB new allocated storage. One successful copy per image is capped
  at3,584,000payload bytes in total; actual JPEG sizes are unknown.
- Preserve Wellcome Collection credit, source work links, CC BY4.0 notices,
  available creator credits and resizing/modification indication. Licence and
  access must be rechecked before transfer; quarantine contradictions, sensitive
  or unsuitable content and duplicates instead of replacing images.
- **No training, inference, automatic negative-label approval or evaluation.**

The independent-evaluation acquisition manifest is **empty**. The separate plan
requires new source/creator/capture groups, all7categories, realistic clutter and
occlusion, and complete category-positive and negative-coverage annotations.
Twenty rooms from at least five independent groups is a planning target only.
These7training-side candidates must not be repurposed into that evaluation set.

## Verification workflow and training gate

The later workflow records decoded dimensions/orientation/SHA256, source licence
and attribution, duplicate/reprint/room identity, visible furniture, privacy and
annotation suitability. Automatic proposed annotations/negative regions remain
explicitly unverified. Ordinary chairs, seating stools and bar stools are protected;
footstools, ottomans and step stools remain separate. Native or newly proposed
fragments, partial objects, uncertain taxonomy and incomplete areas stay ignored.
Actual model-query coverage may be checked only after separate inference approval.
Manual rectangle drawing is not mandatory; unsupported absence claims are forbidden.

This collection alone cannot justify another training run. Preferred next action:
obtain explicit creator grants for **modern, diverse real-room photographs**,
covering commercial ML/annotation/model distribution and resolving other people's
rights, with room/capture identifiers and disjoint train/evaluation creators.
`creator-permission-template.json` is blank; it is not a signed grant.
`consent10.py` validates required fields and scopes but never manufactures a
signature, permission, human review or automatic approval. Original photograph
redistribution is optional and not silently assumed.

`modern-room-contribution-plan.json` proposes100training-side photos from10groups
and30evaluation photos from five different groups as future collection targets,
not verified availability, current permission or mandatory balanced quotas.
Freeze exact IDs, source groups, actual consent evidence and budgets before transfer.
The existing59-photo pool, all checkpoints and frozen/exposed partitions stay intact.

## Tests, resource use and preservation

**38 CPU tests passed**: access allowlists, no binary requests or credentials,
NC/ND/ShareAlike distinctions, missing/conflicting attribution, group uncertainty,
no invented negatives, exact metadata scope, manifest hashes, per-photo evidence,
transfer reserve, unexpected MIME/size, persistent429stop, interrupted transactions,
no redirect/retry, closure, and explicit consent gates. Synthetic TLS fixtures
were used; tests did not access the network or load images/models.

Read-only historical preservation and the Phase10E.9 artifact seal passed.
Model weights, source annotations, all historical partitions, the original64IDs
and their0A/64B/0Crights records remain unchanged. The Mac project was not accessed.

Source discovery used46certificate-verified HTTPS requests,589,786response-body
bytes,1,056,793observed TLS bytes, and4,071,449conservatively charged bytes, below
32,000,000. The charge includes64KiB/request allowance; physical IP/DNS/TCP
retransmissions are not directly measured. Git backup traffic is separate from
this source-request ledger. New local storage is only a few MiB, below128MiB;
post-seal verification prints allocated bytes. No image HEAD, photograph body,
thumbnail, preview, model load, inference, training or new annotation occurred.
The audit is closed to additional network requests.
