# Phase 10E.9 — Photograph rights and acquisition preflight

Completed November 23, 2026. Parent `fa9c22e7fbea0559dd57726a161c4d6e1eaf5689`;
branch `experiment/phase10e9-photograph-rights-audit`.

**64 IDs were audited locally. No individual photograph source pages could be
requested through the permitted audit path. Rights outcomes: 0 supported,
64 unresolved, 0 unsuitable.** Training and independent-evaluation photograph
acquisition lists are both empty. No photographs, models or datasets were
downloaded; no GPU work, training, new labels, partition changes or deployment ran.

## Access finding and stopping point

The first and only external request retrieved the official
[Flickr robots policy](https://www.flickr.com/robots.txt) with verified HTTPS,
HTTP 200, on `2026-11-23T08:54:38.669053+00:00`.

The audit identified itself consistently as:

```
FurnitureAI-RightsAudit/10E.9 (user-authorized source-license metadata verification; no image downloads)
```

The default rule is `User-agent: *` / `Disallow: /`. This client does not match
the specifically named agents receiving different rules; all 64 frozen resolver
paths are denied for this audit client. This finding is specific to the chosen
automated client and method, not a claim that every human/browser or every
named provider client is blocked.

No user-agent rotation, alternate endpoint, browser identity, API, proxy, cookie,
authentication bypass or retry was used. All candidate-page access stopped.
There was **no observed HTTP 429**: the restriction came from the retrieved
access policy. It is not a copyright-licence finding and does not justify marking
64 photographs commercially incompatible. A durable policy circuit and closed
audit marker prevent further requests by this completed run.

Hosting terms, individual licence pages and specific creator/ownership claims
were not fetched after the denial. Public access, a COCO licence identifier or
an official S3 URL was never substituted for photograph-level rights evidence.

## Exact scope and local verification

Frozen manifest:
`data/open-images/preparation/phase10e8-v1/licensed-training-rights-candidates-v1.json`.
SHA-256: `50e29c7b64b4ca67b695df5c941e4bb3c930ffe483e25d0de731ac9e22c86466`.

The approved proposal's rights destination is
`data/coco/rights-audits/phase10e8-v1/`. This new rights-audit directory is
separate from existing E8 preparation artifacts under Open Images.

All 64 COCO IDs and original Flickr photo IDs were checked against the existing
official `instances_train2017.json` and the prior proposal. Image dimensions,
source-asset photo IDs, metadata licence identifiers, native annotation IDs and
counts match. The source JSON was streamed with bounded memory; its preserved
SHA-256 is `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d`.

The selected photos have **786 native annotations**, including **268 furniture
annotations**. All 786 source bboxes are valid within their recorded dimensions
and all have native polygon segmentation data. Masks were not raster-decoded
against new image bodies and no visual registration, indoor-scene suitability,
annotation completeness or hard-negative usefulness was asserted.

Existing local registries and earlier rights records were used first. No exact
candidate ID or source-photo ID overlaps the 844 previous rights candidates,
including the 296 paused requests. No known project image-ID/source-photo-ID
overlap or local filename match was found. Full existing Open Images metadata
was also checked for matching historical source/profile claims; none was found
for these 64 IDs. No original photographs were decoded for this metadata audit.

Distinct COCO IDs and Flickr photo IDs do **not** prove independent physical
rooms, products or capture sessions. Creator and capture-group identities remain
unverified for all 64. No body hashes, perceptual duplicate screening or visual
quality checks were performed. No new training/development/evaluation assignment
was made; prior additional-training-candidate roles remain provisional.

## Rights dispositions and evidence by ID

| Result | Count | Meaning |
|---|---:|---|
| A — Rights-supported candidate | **0** | No individual grant/creator evidence obtained |
| B — Unresolved rights | **64** | Source evidence inaccessible through the permitted automated method |
| C — Unsuitable | **0** | No photograph-specific restriction or other disqualifying fact established |

Remote activity was **one policy-document request, zero individual source-page
attempts, zero photograph-object HEADs and zero photograph-body requests**.
“64 locally audited” must not be read as “64 individual licences verified.”

`rights-by-ID.json` and `rights-decisions.csv` contain every exact ID and its
outcome. `candidate-annotation-inventory.json` supplies the matching per-ID
dataset, original hosting asset, proposed official URL, dimensions, native IDs,
category counts and annotation licence. `rights-audit-index.json` binds these
public metadata components and the private full ledger by checksum.

For each ID, the full `license-attribution-ledger.json` preserves:

- Original COCO/Flickr identity, source resolver, original hosting asset and
  official image URL; canonical owner-page URL remains unknown.
- Historical photograph licence claim: COCO licence ID 4, CC BY 2.0 URL. This
  remains **unverified metadata**, not a current photograph grant.
- Verified creator/rightsholder, photograph licence/version, licence evidence URL,
  attribution conditions and rights-verification time: **null/unverified**.
- A real metadata-audit UTC time and shared policy-access evidence, clearly
  distinguished from missing photo-specific documentary licence evidence.
- Native annotation availability, recorded dimensions, source-group uncertainty,
  duplicate/local availability status and no final partition assignment.

The shared policy response has SHA-256
`596abe673cd61794959827e668ec2f1577bc8aafad8753867a41dfed710e1e8c`.
It explains the access stop; it is not misrepresented as 64 separate licence
documents. Raw policy bytes and response headers remain local; response-cookie
values and full native annotation bodies are not published to Git.

Annotation rights, photo rights and hosting rules remain separate:

1. Existing COCO annotations retain COCO Consortium / CC BY 4.0 attribution.
2. The photographs need acceptable individual creator/licence evidence.
3. The observed hosting access rule prohibits this automated access method.
4. Permission relevant to obtaining/storing/training, other content rights and
   any applicable conditions have not been independently established here.

No ownership, privacy/publicity/property releases or explicit AI-training consent
was invented from the historic licence identifier or the absence of a signal.

## Potential content and annotation limitations

These are **metadata availability counts**, not rights-cleared or visually
accepted training examples. Photos overlap categories.

| Project concept | Candidate photographs | Native boxes | Mask-bearing native annotations | Rights-supported photos |
|---|---:|---:|---:|---:|
| Sofa / COCO couch | 12 | 19 | 19 | 0 |
| Chair | 63 | 169 | 169 | 0 |
| Table / COCO dining table | 51 | 62 | 62 | 0 |
| Bed | 13 | 18 | 18 | 0 |
| Desk | 0 | 0 | 0 | 0 |
| Cabinet | 0 | 0 | 0 | 0 |
| Lamp | 0 | 0 | 0 | 0 |

Desk, Cabinet and Lamp are unsupported by this COCO taxonomy; Table is not
silently relabelled Desk, and lighting objects or storage fronts are not invented.
Project Chair includes ordinary seating and bar stools, whose complete coverage
is not established by native COCO Chair annotations. Missing labels never become
Chair-free or seven-class background evidence.

Promising native context includes 112 person annotations in 38 photos; 18 potted
plants in 11; six handbags in four; 12 laptops in six; seven TVs in seven; six
refrigerators in five; four microwaves in four; and six sinks in five. These are
useful source-supported content signals, not confirmed future hard negatives.
Body-part negatives are not created from person bboxes.

An additional read-only audit streamed **both existing LVIS ZIP archives** and
verified their cached index against original source annotations for these same
64 photographs. The index's federated flags intentionally retain only 23 selected
categories; the new audit preserves full original image flags and compares the
proper selected-category projection instead of mistaking it for full source scope.
LVIS native splits are 54 train / 10 validation, while all 64 photographs are COCO train2017.
These native splits do not create or alter Furniture AI partitions.

| Native LVIS class | ID | Same candidate photos | Instances / masks |
|---|---:|---:|---:|
| sofa | 982 | 7 | 8 |
| chair | 232 | 15 | 37 |
| table | 1050 | 6 | 6 |
| bed | 77 | 6 | 10 |
| desk | 361 | 1 | 2 |
| cabinet | 181 | 5 | 48 |
| lamp | 626 | 4 | 9 |

There are 120 core-category LVIS instances plus 39 separate native subtype/part instances:
12 armchair, 3 coffee_table, 1 sofa_bed, 18 lampshade, 2 rocking_chair and 3 table_lamp.
All 159 source annotations are preserved inactive. **Do not sum these with the 268
COCO boxes as independent objects**: they refer to the same photographs and may
annotate the same furniture differently. Cabinet instances may represent fronts,
parts or built-in storage;lampshades are not standalone lamps;Table/Desk and
sofa_bed mappings remain unresolved. No photograph registration/visibility check
or rights upgrade occurred. The authoritative `rights-audit-index-v2.json` adds
this existing annotation availability while retaining 0 / 64 / 0 rights outcomes.
Existing LVIS annotation credit and CC BY 4.0 evidence are retained separately
from COCO Consortium annotation credit in `annotation-provenance.json`; neither
annotation licence grants rights to the underlying photographs.

`metadata-priority.csv` ranks the same 64 IDs deterministically using only local
person/household-confuser, Bed/Sofa and possible outdoor-context metadata. The
ranking does not change rights status or partition membership. Some sources
also contain outdoor/mixed-context labels such as skateboard, airplane or zebra;
metadata does not prove every photograph is a suitable indoor room scene.

## Acquisition plans and exact next authorization

**Potential training list:** empty, because no candidate passed photograph-level
rights verification. No unsupported photo is substituted. The prior 64-photo
estimate of 9.6–25.6 MB remains a metadata projection without HEAD verification;
it is neither an approved download nor a current acquisition-ready list.

**Potential independent seven-category evaluation list:** separately empty.
None of these training candidates, existing exposed development photos or
reserved/frozen material was reassigned. A later new-room collection should use
individually authorized, source/creator/capture-disjoint material, complete blind
seven-category/Chair-stool annotations, independent completeness review and
frozen image/annotation/protocol hashes before scoring. A proposed first batch
of 20 is only a workflow target; no exact new IDs, rights, images or acquisition
approval exist for that collection yet.

`acquisition-plans.json` and the separate local training/evaluation manifests
preserve both empty lists and their prerequisites. Automatic future verification
is designed to display native masks/boxes and whole-photo/crop context, preserve
real furniture/stools and unknown regions, propose source-specific interior
regions without mandatory manual rectangle drawing, and keep assistant/model
proposals distinct from actual human confirmation. Actual model-query confidence
coverage requires a separately authorized later inference step. No labels were
generated and no threshold, model or loss function was changed now.

**No photograph-acquisition approval is ready to request.** First obtain a
provider-permitted rights-evidence route or acceptable photograph-specific
creator documentation, and complete rights, attribution and group checks.
Then prepare a nonempty exact survivor manifest with verified source URLs,
sizes and network/storage/retry caps for a separate photograph-download approval.
The current authorization cannot be used to bypass access rules or acquire any
of the unresolved photographs.

## Resource accounting

| Source-audit measurement | Bytes |
|---|---:|
| Retrieved compressed response body | **658** |
| Measured TLS bytes sent | **1,904** |
| Measured TLS bytes received | **9,820** |
| Measured TLS stream total | **11,724** |
| Conservative transfer charge | **77,260** |
| Approved maximum | **32,000,000** |

The TLS meter includes encrypted handshakes, HTTP headers/body and socket
read-ahead. Physical IP/TCP/DNS/retransmission traffic is not directly measured;
the conservative charge adds a 65,536-byte protocol allowance per transaction
and keeps full reservations for interrupted requests. Response-body counts do
not include fictional zero-byte source-page attempts. The source-data ledger is
separate from the requested small Git metadata backup.

The authoritative final network snapshot is `network-final-ledger.json`; the
earlier policy-stage ledger predates the explicit persisted robots-policy circuit.
New allocated storage remains below 128 MiB, including metadata, code, reports,
raw policy evidence and synthetic CPU fixtures, well below 128 MiB. Final
measurements are in `final-summary.json`; immutable writers retain recovery
links and count unique allocated inodes.

## Tests and historical integrity

**40 CPU tests passed:** 18 pre-network rights/transport safeguards, eleven completed-
audit checks and 11 existing geometry/decision safeguards. Synthetic fixtures
test valid/mismatched licences, source identity, AI-use signals, TLS requirements,
forbidden image GETs, 429 stops and bounded decoding. They are not real rights
evidence and never enter the actual one-request ledger or 64-ID dispositions.

Completed-audit tests confirm the observed robots denial, immutable closed-audit
stop, 0/64/0 outcomes, empty acquisition lists, exact native metadata, no invented
groups, no photo/GPU activity and reconciled transfer/storage counters. Deep
historical checks run in an isolated CPU process to prevent same-named legacy
modules from being shadowed; no historical code or environment was changed.

Preservation passed for 9,973 earlier protected data/model files and subsequent
C8/C11/C13/C17–C21/D/E1–E8 seals, including original OWLv2, SigLIP 2, E5 and other
adapted checkpoints, annotations, review records and frozen partitions. This is
read-only checksum verification; evaluation photos were not decoded, mined or
scored. The Mac application was not accessed. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.

**Recommended next action:** establish a provider-permitted rights-evidence
method or obtain acceptable creator-supplied grants for the exact pending IDs.
Do not start photograph acquisition, inference, training or another loss experiment.
