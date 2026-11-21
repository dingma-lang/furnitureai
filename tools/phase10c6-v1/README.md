# Phase 10C.6 — automatic visible-object negative evidence

All execution uses `.venv/bin/python`. No local model inference, GPU work,
training, downloads or installations are needed. The assistant actually inspected
36 private plates containing208 candidate regions from50 original optimization
photographs. There are208 automatic observations, never human confirmations.

Open the results dashboard:

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B tools/phase10c6-v1/server6.py --port 8770
```

<http://localhost:8770/> shows aggregate/category/group counts, all three evidence
tiers, photograph overlays, reasons and exact source geometry. No repeated review
clicks are required. Optional concerns create append-only correction files;
exclude/recheck flags block future authorization until a new evidence version
resolves them. They do not rewrite the sealed result or manufacture approval.

The automatic pipeline retains59 optimization photographs,56 known source
groups and182 existing native positive boxes. The initial123 annotations include
77Person,35Plant/Flower and11Drawer/Furniture-part source records. Eight further
families contributed64 diverse candidates. All remaining eligible local native
geometry was scanned;21 additional sufficiently large interiors were inspected.
The full local inventory contains643 raw non-furniture annotation records and625
geometric object hypotheses. Those hypotheses and208 reviewed candidates are not
proven independent physical-object counts. Remaining geometry exclusions were
not silently recorded as visually reviewed or accepted.

Tier A consists of41 non-overlapping rectangles across24 photos/source groups.
Each requires exact provenance, an eligible native COCO/LVIS visible instance
polygon, a simple in-frame polygon, strict full-rectangle containment with≥2px
continuous-boundary clearance, at least32px per inspected side, actual visual
agreement, no protected or visually uncertain overlap, and no duplicate weighting.
Pillow/soft seating accessories, mixed screen content, sparse mask envelopes,
reflections with insufficient detail and small/blurred crops remain untrusted.

Pillow rasterization and erosion merely search for proposals. Acceptance uses
the original continuous polygon; it does **not** claim bit-identical COCO RLE
raster decoding. Unsupported crowd/RLE/unsafe geometry is rejected. Source
annotations can be imperfect: “source-verified” means verified under this
documented policy, not human-reviewed ground truth, physical absence behind an
occluder, or guaranteed detector accuracy. Confidence is qualitative evidence
strength; no calibrated probability was invented.

Tier B has37 visually screened but independently insufficient candidates; Tier C
has130 uncertain/rejected records. Neither tier enters trusted negative masks.
Original boxes, masks, category IDs, photo rights/creator/attribution evidence,
image hashes and group assignments remain in the ignored local artifacts:

- `automatic-evidence.json`: all automatic observations and outcomes.
- `source-negative-evidence.json`: only accepted source-polygon evidence.
- `training-evidence-manifest.json`: original rows plus inactive additive evidence.
- `mask-comparison.json`: per-photo CPU coordinate-probe differences.
- `observations/`, `inspection/`, `inspection-expansion/`: actual visual lineage.

The exact Chair mask differs by352 terms on the fixed CPU grid/interior probes;
all41 interior probes receive a new Chair-negative target. Existing positives,
legacy mask terms, normalization and other categories are preserved. This is not
actual-model query coverage. `masks6.source_supervision()` has a separate disabled
automatic-source gate. It cannot masquerade as C.3 human evidence, and existing
manual/verified evaluators remain unchanged. No generic Other/background output
is introduced; other furniture is never declared background.

The bounded training proposal stays inactive. Before any optimizer update, a
separately approved zero-update smoke must prove actual model-query coverage and
CUDA/AMP/memory safety. If real queries receive no added targets, the experiment
remains blocked despite the synthetic coordinate contrast. Independent real-room
evaluation remains entirely separate; no reserved/frozen scoring occurred.

CPU and interface verification:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c6-v1/verify6.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c6-v1/browser6.py
```

Preparation scripts require fresh versioned destinations. Immutable outputs are
not overwritten on reruns. `record6.py` records actual assistant observations only
after direct image inspection; never use it to pre-fill fabricated visual review.
The browser test uses installed Windows Chrome, synthetic images and an in-memory
correction handler. Photos and automatic/manual review records never enter Git.
Stop/restart only the recorded C.6 server; previous review services are preserved.
