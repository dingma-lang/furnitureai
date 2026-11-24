# Phase 10E.16 — Three-photo modern-room acquisition and suitability

Completed 2026-11-24 on `experiment/phase10e16-modern-room-pilot`, based on
`03534497404bf3a8dc6c1bf49c108f336b1b28f4`. Exactly the three photographs approved
in the user attachment were acquired. The frozen proposal remained unchanged;
the new authorization record explicitly supersedes its old proposal-only flag.

## Decision

**One promising candidate (A), one further-review candidate (B), one unsuitable
furniture-positive scene (C).** Modern licensed rooms are more promising than the
closed historical Wellcome route, but titles alone are not reliable selection.
Do not expand through more generic workspace titles or train from these three
photographs. Prioritize a separately authorized, exact-ID creator-consented
batch targeting whole Beds, dedicated Desks, Cabinets and standalone Lamps.

All three remain outside active training, development and independent evaluation.
There are zero new verified boxes, masks, negative regions or model-query labels.
The observations below are assistant visual screening of actual acquired pixels,
not human ground truth or exhaustive category-completeness certification.

## 1. Actual acquisition, integrity and resources

| Source ID / photograph | Creator / licence | Original dimensions | JPEG bytes | Suitability |
|---|---|---|---:|---|
| `commons-page:150832261` — Workspace Setup | FreshDev / CC0 | 4032 x 3024 | 4,073,349 | C |
| `commons-page:178637758` — Winnipeg living room | Alexhalipchuk / CC BY 4.0 | 4608 x 3456 | 6,356,815 | B |
| `commons-page:195256255` — Lake-view living room | Shixart1985 / CC BY 2.0 | 6016 x 4016 | 10,821,723 | A |

All three JPEGs passed verification, complete decoding, native-dimension checks
and published source SHA-1 comparison. Their original bytes were not resized or
modified. SHA-256, source SHA-1, dimensions, byte sizes, URLs and credit are in
`image-and-rights-manifest.json` and `image-sha256-and-provenance.csv`. Local
1600px gallery previews are identified as derived copies; they are not additional
source photographs.

Source acquisition used exactly ten certificate-verified HTTPS transactions:
one official imageinfo metadata GET, three licence-document GETs, three exact
object HEADs and three image GETs. All metadata/rights/HEAD checks passed before
the first image GET. No redirect was followed and no source transaction retried.
The implementation uses one attempt per photograph, within the approved two.

| Transfer measurement | Actual bytes |
|---|---:|
| Image response bodies | **21,251,887** |
| All source response bodies, including metadata/licence documents | **21,354,997** |
| Observed bidirectional TLS stream bytes | **21,472,875** |
| Conservative source transfer charge | **22,128,235** |
| Source transfer cap | 60,000,000 |

The source stream counter includes TLS handshakes, HTTP headers and read-ahead.
It adds a 65,536-byte allowance per request. Physical IP/DNS/TCP retransmissions
are not directly measured and are not falsely described as measured bytes.
Failed/redirect responses would remain charged and stop the provider; none
occurred. Request intents preserve reservations across interruptions, and a
closed/interrupted pilot refuses automatic repeat acquisition.

Two local browser smoke attempts read **8,192,496 response-body bytes** from the
already acquired local files/static pages, not additional source photographs.
Those local bodies and conservative per-request allowances are also included in
the combined figure in `summary.json`; it remains below 60,000,000 bytes. Git
code/report backup traffic is separate from acquisition and localhost serving.

Allocated storage is approximately **24 MB**, including originals, previews,
evidence, retained isolated browser profiles, tools and reports. The exact final
allocated-byte count is printed by `verify16.py check`, with the pre-seal count
in `summary.json`. It is well below **134,217,728 bytes (128 MiB)**. All photographs
are below the **12,582,912-byte** per-image maximum. No extra dataset/model bodies
were downloaded. CPU image decoding used RAM, not GPU VRAM.

## 2. Rights evidence and attribution

Fresh official Commons `imageinfo` evidence was compared against all three frozen
file titles/page IDs, native URLs, bytes, dimensions, source SHA-1, creator credit,
own-work declarations and photograph-specific licences. The three exact delivery
endpoints returned successful matching JPEG HEAD responses. No access restriction,
changed licence or content identity was encountered.

Canonical source pages:

- [Workspace Setup](https://commons.wikimedia.org/wiki/File:Workspace_Setup.jpg).
- [Living room in Winnipeg](https://commons.wikimedia.org/wiki/File:Living_room_in_Winnipeg.jpg).
- [Lake-view living room](https://commons.wikimedia.org/wiki/File:Cozy_living_room_with_a_view_of_the_lake_and_trees_in_the_background_at_midday.jpg).

Fresh licence documents, metadata, peer certificate digests and response receipts
are preserved under `evidence/`, `requests/` and `rights-ledger.json`. CC BY credit
must retain the recorded author, linked source and licence/version, and note
modifications such as the resized preview. FreshDev is credited even though CC0
does not impose attribution. These are file-specific source declarations, not
inference from Commons' structured-metadata licence.

Copyright declaration support, privacy/property concerns and visual suitability
remain separate statuses. No new creator grant, ownership warranty, identifiable
person release or private-property permission was invented. The public source
declarations do not automatically resolve third-party rights. Originals and
previews remain local and are not committed to public Git.

## 3. Actual visual findings

### Workspace Setup — C: unsuitable for the intended furniture-positive objective

This is a close-up of two laptops, a large monitor, keyboard, mouse, cables,
notes and a cropped wooden work surface. Much of the image is equipment and wall.
There is insufficient whole-object/room context to establish a whole dedicated
Desk rather than a generic table/worktop. No clearly identifiable Chair or seating
stool was established. Missing categories are not certified absent.

The equipment is clear, but the furniture surface is truncated at the borders
and supports are not visible. Screen content and notes need privacy review before
reuse. Their text is not transcribed into labels or a public evidence description.
Monitors, stands, laptops, cables and surface edges might be useful confusion
probes, but no detector was run and no Chair-negative region is qualified. Do not
admit this merely because its title says workspace or home office.

### Winnipeg living room — B: potentially useful, further review required

A residential holiday scene contains an enclosed television-storage cabinet,
partial foreground ornate table-like furniture, a partial lamp and clipped drawer
storage on the right. A striped ottoman/footstool-like object has an unresolved
seating function. It is not automatically Chair and not automatically Chair-negative.

This is real residential context, unlike the historical/institutional Wellcome
pilot. However, softness, uneven light, a date imprint and border cuts complicate
precise extent annotation. Review whole Cabinet units versus fronts, truncated
Table extent and standalone Lamp type/extent. No sofa, bed or dedicated desk was
established from the visible scene.

The TV/reflections, fireplace/mantel, Christmas tree/lights, candles, cups, framed
material/cards and furniture handles provide clutter/confusion potential. No live
person is evident, but small framed personal material may still need privacy
review. No object absence, exhaustive box coverage or negative region is asserted.

### Lake-view living room — A: suitable candidate for later annotation

The strongest photograph shows a connected sectional Sofa clipped at the left
edge, a large foreground coffee Table, and two patio Chairs visible through glass.
A smaller table between the patio chairs needs separate extent review. Interior
shadow, bright outdoor light, reflective flooring, window/door frames and partial
occlusion make this a relevant modern-room challenge.

Annotate the connected sectional as one physical object, not individual sofa
parts. Distinguish actual chairs from their reflections and shadows. Retain
through-glass/outdoor context flags and visible extent/truncation. Do not relabel
coffee-table storage panels as a separate Cabinet. Candlesticks are not silently
mapped to Lamp; the outdoor tub-like structure, wine bucket/glasses/bottles,
curtains and door frames are distractors requiring their own evidence if later
used as class-specific negatives. No detector scores or model errors are claimed.

No person or plainly identifying document is evident, but source rights and
property/privacy qualification remain separate from this A suitability grade.

### Seven-category coverage from screening, not labels

| Category | Clearly observed candidate images | Further uncertainty |
|---|---:|---|
| Sofa | 1 | Connected sectional, truncated |
| Chair | 1 | Two patio chairs in the lake scene; Winnipeg ottoman-like seating unresolved |
| Table | 2 | Partial Winnipeg furniture; small patio table extent requires review |
| Bed | 0 established | No absence certification |
| Desk | 0 established | Workspace worktop lacks whole-object context |
| Cabinet | 1 | Whole TV storage candidate; clipped drawer/storage extent unresolved |
| Lamp | 0 clearly qualified | One partial Winnipeg lamp candidate needs review |

Counts overlap, are assistant observations and do not create native annotations.
**Hard-negative potential is present, but newly verified hard negatives = 0.**
Do not convert visual object lists or unmatched/missing labels into training targets.

## 4. Duplicates and independence

All three originals have distinct SHA-256s. Bounded 64-bit dHash/aHash screening
found zero within-pilot near-duplicate flags and zero exact/perceptual flags against
the existing **59 optimization + 19 already-exposed internal-development**
photographs. Source hashes were checked before decoding those references. No
reserved/frozen photograph was decoded or scored.

There are three distinct uploader-account buckets. Cross-provider creator aliases,
physical rooms, locations and capture relationships are not independently verified;
the formal count of verified independent physical/capture groups remains **zero**.
Perceptual heuristics do not establish global uniqueness. The E15 metadata check
also found no exact match against its bounded 120-photo source inventory, but
that is not a claim about every external source or modified crop.

All three retain the approved training-side inspection role, with **no actual
dataset partition assignment and zero independent-evaluation admissions**. Separate
fresh creators/rooms are still required for future acceptance evaluation.

## 5. Intake and local gallery validation

The unchanged E15 schema/validator was used to import all three actual photographs
into isolated pilot `intake/` records. Original content hashes and oriented
dimensions were verified; credit/licence/evidence hashes, account source groups
and explicitly unknown capture/room buckets were recorded. The seven-category
specification remains `modern-room-seven-v1-chair-includes-seating-stools`.

All class completeness stays unknown, object lists stay empty, and human review,
privacy/grant confirmation and activation flags stay false. E15 correctly keeps
these records unqualified for training/evaluation. The known public copyright
evidence is retained separately; no fake creator confirmation is filled to pass
the validator. Native annotation import/correction on these photos is **not
applicable: there are zero source boxes/masks**. Synthetic E15 regressions cover
that logic without inventing real boxes.

The read-only gallery is:

`data/licensed-rooms/image-pilots/modern-room-screening-v1/gallery.html`

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/phase10e16-v1/serve16.py --port 8784
```

Open **http://localhost:8784/gallery.html**. It shows the images, sources, licence
credit, observed categories, distractors, suitability, annotation requirements and
independence limitations. Click a preview for the unchanged original. The linked
intake page is E15's existing page/script, served in this isolated pilot. Only
exact approved files are accessible; uploads, directory traversal and normal
access to smoke-controller endpoints are denied.

### Browser smoke: explicitly incomplete

Existing Windows Chrome/WSL interop was available, so bounded attempts used fresh
isolated profiles, GPU disabled, no component installation/update or personal
profile, cache limits and a 30-second per-attempt bound. The first stopped on a
browser temporary disappearing during allocated-storage enumeration. A narrow
race correction was regression-tested; permission errors still propagate.

The follow-up ended with an async DOM wait timeout before it reported the first
photo hash. Total observed runtimes were approximately 2.81s and 4.55s, with no
resource limit violation. Both attempts loaded the E15 page/first local JPEG but
did not certify the complete UI sequence. No additional source GET occurred and
no actual or synthetic human annotation was exported. **Browser end-to-end
hashing/import/edit/undo is unverified**, not a claimed pass. CPU content/provenance
and source-box regression checks did pass. Manually complete the documented
browser walkthrough before relying on new human annotation exports; do not fill
permissions just to make a smoke test pass.

## 6. Tests, preservation and Git backup

- **16 focused E16 CPU tests passed**, covering scope, altered rights/identity,
  MIME/size/hash checks, transfer reservations, 429/redirect/interruption stops,
  closed-pilot protection, actual inactive intake records and read-only routes.
- **12 E15 intake CPU regressions passed**, using synthetic grants/photos for
  metadata, completeness, EXIF and source-box checks. No real human review is
  fabricated and no broad reliability suite was rerun.
- Targeted preservation rechecks the existing 130 historical-photo entries,
  eligibility/source/annotation/exclusion inputs, parent code/report seals and
  new original-image hashes. No old source code, checkpoint, dataset membership,
  prediction, annotation or model default was changed. The Mac project was not
  accessed. No inference, optimization or GPU-model call occurred.
- E15 had not been pushed at the start. Its branch was backed up and the remote
  commit verified exactly as `03534497404bf3a8dc6c1bf49c108f336b1b28f4`.
  `main` remains `e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`. The new E16
  experimental branch contains only approved code/metadata/small reports; photos,
  previews, browser profiles and local intake files remain ignored and local.

## 7. Exact next direction — not another automatic acquisition

The lake scene supports continuing modern-room collection, and Winnipeg provides
useful ambiguity/clutter review. Their gaps also show that generic living-room
searches do not solve Bed/Desk/Cabinet/Lamp supervision. Recommend **eight
creator-consented photographs** as the next targeted proposal:

- Training-side: two bedrooms with beds/bedside lamps; two home/small offices with
  whole desks and cabinets; one dining/stool scene; one cluttered living room.
- Independent-evaluation-side: one separately sourced bedroom and one office,
  with complete seven-category annotation and disjoint source/creator/room groups.
- At least four training and two evaluation creator/room components. Keep all
  related captures together and preserve every existing exposed/frozen partition.
- Planning estimate 24–64 MiB; proposed future cap 120,000,000 transfer bytes,
  12 MiB/photo, two attempts and 256 MiB storage. These are **not observed costs or
  authorization**, and the exact source sizes must be known before execution.

There are currently **no exact new IDs or creator grants for that eight-photo
proposal**. Do not invent them, contact creators or import replacements. First
obtain user-authorized identities/permissions and freeze a new exact acquisition
manifest for separate approval. The original 24-photo target remains a quality
goal; three inspected photographs are not sufficient for retraining or independent
accuracy. No next acquisition, annotation activation, model experiment or deployment
is started by this recommendation.
