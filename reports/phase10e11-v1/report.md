# Phase 10E.11 — Wellcome photograph suitability pilot

Completed using the existing `.venv/bin/python` and Pillow13.0.0. Exact user-approved
seven-photo inspection only; no GPU, model loading/inference, training, active
annotations, new collection expansion or Mac-application access.

## Decision

**Stop further Wellcome acquisition for the modern-room furniture objective.**
All7are actual room scenes with some useful furniture observations, but all7have
historical or institutional characteristics. None is qualified as a modern-room
training candidate. Retain them immutably as limited-use inspection material;
there is no evidence-based reason to launch a new training experiment from this
pilot or repeat more rounds of historical-image selection.

| Suitability, separate from rights | Count |
|---|---:|
| A — promising modern real-room candidate | 0 |
| B — limited-use historical/atypical scene | 7 |
| C — unsuitable | 0 |

These are assistant visual assessments after inspecting all acquired images,
not fabricated human reviews or source ground-truth labels. No scores or detector
errors were measured, and no category absence was certified.

## Actual acquisition and rights

Exactly7of7approved IDs acquired, one GET attempt per image, no retries or
redirects. Each image passed JPEG integrity verification, complete decoding,
source-service identity and aspect-ratio/dimension checks. No replacements,
additional photos, thumbnails, archives, model weights or other datasets acquired.

| Image ID | Native image ID | Acquired dimensions | JPEG bytes |
|---|---|---|---:|
| efqgz854 | L0028138 | 1600×1026 | 324,315 |
| bpxzpgg7 | L0014195 | 1600×1204 | 402,533 |
| x6v3kwb7 | L0014196 | 1600×1182 | 276,003 |
| y4n8bmat | L0014193 | 1600×1182 | 338,830 |
| vugqdsr7 | V0031425 | 1600×1207 | 396,908 |
| mhnzfe6b | L0028136 | 1600×1032 | 325,512 |
| mqxh737p | L0028137 | 1600×1027 | 342,818 |

Original dimensions and exact IIIF delivery URLs remain in the provenance
manifest. Only the approved derivatives fitting1600×1600were requested; no
original-resolution download or silent resize substitution. JPEG bytes were not
modified after download. Cryptographic hashes are in
`image-sha256-and-provenance.csv` and `image-and-rights-manifest.json`.

Before any image request, all7IDs were checked against the preserved76supported
E10records and their exact approved manifest SHA256
`53ce3a31a7d85dae17b374746f7705fd814f968cea3774834d63600b4c824be6`.
Their current first-party image records were re-fetched using the documented
Wellcome Catalogue API, and original IIIF service IDs/dimensions were rechecked.
All7still attach CC BY4.0 and **Wellcome Collection** credit to the same image.
All7rights checks succeeded;0rights failures or quarantines.

Both delivery hosts returned404for their robots.txt, recorded as no published
host-level robots policy rather than as copyright proof. The documented public
catalogue and IIIF interfaces from E10were used with verified TLS, an honest
identified client and serial paced requests. No blocked provider endpoints,
identity rotation, proxies, authentication or access-policy bypasses were used.
Current source receipts and hashes survive under the approved output directory.

Attribution must retain Wellcome Collection credit, linked source work and
CC BY4.0 notice, and identify the1600pxderivative modification. Photographer and
copyright-holder identities remain unknown; no names were invented. The recorded
licence declaration supports conditional reuse, not a warranty of ownership or
all privacy/publicity/property rights. People are visible in2photographs: the
kitchen and reading room. Do not infer unrestricted personality rights from CC BY.

## Visual observations and limitations

| Photograph | Observed content | Main training limitation |
|---|---|---|
| Kitchen efqgz854 | Industrial kitchen, people, prep surfaces, counters, ovens, cookware and pipes | Historical equipment; counters/island vs Table/Cabinet scope and people conditions |
| Laboratory bpxzpgg7 | Work surface, glassware, partly visible genuine seating, storage and instruments | Occlusion, shadow and clipped furniture; protect real chairs |
| Laboratory x6v3kwb7 | Long bench, drawers, glass-front storage; dark stool-like form uncertain | Small/uncertain seating, built-in/front semantics and laboratory apparatus |
| Laboratory y4n8bmat | Genuine bentwood seating, work island, cases and instruments | Partial seating, glare/shadows, repeated-source context |
| Reading room vugqdsr7 | Clear chairs and table, person, books, display case, lights and guide equipment | Best general-furniture visibility but historical; bookcase/display-case taxonomy and people conditions |
| Single hospital room mhnzfe6b | Bed, armchair, partial wooden chair, bedside furniture, flowers and sink | Hospital bed domain, clipping/occlusion and bedside-object taxonomy |
| Double hospital room mqxh737p | Two beds, two clear chairs, privacy screen, plants, bedside furniture | Foreground bed truncation, screen/frame confusions and vintage layout |

None primarily depicts a medical-body subject. Medical/institutional room context
remains atypical for the intended modern residential/office distribution. The
reading room is the closest analogue; hospital bedrooms contain clear furniture
but do not establish modern-room generalization.

| Category | Images with clear assistant visual observations | Further uncertainty |
|---|---:|---|
| Sofa | 0 | Not established; no absence certification |
| Chair | 5 | A further laboratory view has uncertain stool-like content |
| Table | 1 | Six other scenes have workbench/prep/bedside/tray mapping questions |
| Bed | 2 | Hospital beds; one foreground bed is truncated |
| Desk | 0 | Workbenches/reading table not silently relabelled Desk |
| Cabinet | 0 | Six scenes contain candidate storage/fronts/bedside units requiring ontology decisions |
| Lamp | 0 | Mounted/ceiling fixtures not silently mapped to standalone Lamp |

Counts overlap and are not complete instance counts or accepted labels. **No native
bounding boxes or masks are available for these7images.** Source scene descriptions
were broadly accurate, but they did not establish full object coverage or seating
absence. The acquired laboratory images reveal genuine/ambiguous seating that
must be protected. No photograph is approved wholesale as Chair-negative.

Large Chairs/Bed/Table in the reading/hospital scenes could be boxed after proper
annotation review. Laboratory worktops, glass-front/built-in cabinets and medical
fixtures require a conservative semantic policy first. Full-vs-visible extent,
truncation, occlusion and uncertain objects need explicit flags; no boxes were
invented here. Image dimensions are adequate, but monochrome grain, clipped
highlights, shadows and frame-edge cuts limit small-object annotation precision.

Glassware, appliances, pipes, screens, bed rails, display-case frames, plants,
telephone cords and repeated handles provide potential distractors. Their value
as actual high-confidence Chair confusions or hard-negative query coverage is
**unknown** without separate model inference and evidence qualification. Trusted
new negative regions:0. Active annotations:0. GPU forwards:0.

## Duplicates and source independence

All7JPEG SHA256hashes are distinct.64bitdHash/aHash screening produced0near-duplicate
flags within the pilot and0exact/perceptual flags against the59optimization plus
19already-exposed internal-development reference images. Historical reference
hashes were verified before decoding. No frozen/reserved image was decoded.
These bounded heuristics do not prove global image or physical-room independence.

The three laboratory photographs may show the same or related laboratory space,
despite not triggering perceptual similarity thresholds. Keep them together until
room/capture evidence proves otherwise. Hospital views also share an institutional
source context. All7retain the single conservative unknown-creator bucket; no
verified independent source group, new dataset split or evaluation membership is
asserted. Native work/image IDs and possible relationships were not overwritten.

## Gallery and practical modern-room workflow

Read-only gallery:
`data/licensed-rooms/image-pilots/phase10e11-wellcome-v1/gallery.html`.
It shows every acquired image, source/credit/licence, visual candidate classes,
quality, distractors, suitability and evidence limits. Click an image for its
full acquired resolution. No boxes or inference scores are fabricated.

Start/restart locally if needed:

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e11-v1/serve11.py --port 8782
```

Open `http://localhost:8782/gallery.html`. The server binds127.0.0.1only, permits
only the exact gallery, blank intake page, review JSON and7images, and offers
no uploads or directory listing. Photos stay out of Git.

The linked **blank future creator-intake page** exports metadata locally only
after explicit fields and checkbox confirmations. It does not acquire images,
send messages, sign on anyone's behalf or activate labels. Public code and a
workflow manifest are committed; permission evidence and contact details should
remain private. No browser automation runtime is installed in this environment;
HTML scope and exported-schema consent gates were checked by CPU tests, not a
claimed end-to-end browser test.

The short permission template covers storage, annotations/modifications,
training/evaluation for the agreed purpose, and commercial derived-model use.
It does not assume public redistribution of original photographs. Store creator
credit preferences, exact photo IDs, permission evidence/date and source IDs.
Resolve other people's privacy, documents, screens and private-property issues.
Use opaque creator/source/physical-room/capture groups, not published street
addresses. Check source IDs, original/modified hash trails, perceptual duplicates,
room aliases and related sessions. Freeze disjoint training/evaluation groups
before model use; unknown identities stay unresolved.

The seven-class annotation checklist requires actual object/occlusion/truncation
and category-completeness review, protecting ordinary chairs and seating/bar stools
while keeping ottomans, footstools and step stools separate. An optional later
annotation interface can overlay proposed boxes/masks for accept/edit/ignore;
manual coordinate entry need not be required. The current page describes that
workflow, not an installed automatic annotator or approved future inference run.
Unverified proposals remain inactive; absence of source boxes is never a negative.

**Next development action:** collect explicitly consented modern, diverse room
photographs through this intake workflow, with separate creators/physical rooms
for independent evaluation. No third parties were contacted and no new photos
were obtained through intake. Do not launch training from this7photo count.

## Resource use, tests and preservation

- Acquired image bodies: **2,406,919bytes**.
- All source response bodies including preflight metadata: **2,436,491bytes**.
- Observed bidirectional TLS stream transfer: **2,644,717bytes**.
- Conservative transfer charge: **3,398,381bytes**, below8,000,000.
-23requests total; seven image requests, each one attempt;0retries/redirects.
- Every image below512,000bytes. Allocation below64MiB; final seal verification
  prints exact new allocated storage across data, code and reports.

TLS measurements include handshakes, HTTP headers and read-ahead.32KiB/request
additional allowance is charged separately; physical IP/DNS/TCP retransmissions
are not directly measured. Small Git backup and local browser-serving traffic
are separate from the source-request ledger.

**30 CPU tests passed**, covering approved IDs/URLs, rights changes, corruption,
size/transfer limits, unexpected MIME,429and redirects, interrupted/repeated
transactions, closed-pilot protection, fresh evidence hashes, exact-file gallery,
consent and group conflicts, no missing-category negatives and inactive reviews.
No real network operations occurred in tests; transport replies were synthetic.

Historical read-only preservation passed, as did the previous E10/E9seals.
Original/adapted OWLv2, SigLIP/classifier weights, historical experiments,
annotations/partitions, the64Flickr statuses and main remain unchanged. Checkpoint
hashes and9973historical data/model records are covered by preservation output.
No Mac project was accessed. Original historical files were neither overwritten
nor deleted. Acquisition is closed; no expansion, training or deployment follows.
