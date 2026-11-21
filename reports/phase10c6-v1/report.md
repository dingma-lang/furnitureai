# Phase 10C.6 — automatic visual review and source-mask negatives

The automatic review was carried out: **208 annotated object candidates in50 optimization photographs**, using36 private inspection plates. All123 initial Person/Plant/Drawer-part annotations were inspected, followed by85 additional native candidates. Counts refer to annotation candidates; overlapping source records are not necessarily independent physical objects.

**41 source-mask-supported Chair-negative regions in24 photographs/24 known source groups;37 automatic-only candidates;130 uncertain/rejected records.** No human confirmations were created. No model inference, GPU work, training, downloads, installations or production changes occurred.

## Evidence and scope

The preserved local inventory contains643 raw non-furniture records and625 geometric object hypotheses. The initial123 and64 diverse additional objects were visually reviewed. Another447 remaining native records underwent geometry search;21 further qualifying interiors were inspected.426 native records did not yield a qualifying≥32px interior and were not object-wise visually reviewed. Nine alias records did not become additional independent targets. The search is conservative and does not prove that every mathematically possible interior rectangle was found.

The actual inspected set spans50 photos and47 known creator/source groups. The dataset remains **59 optimization photos,56 source groups and182 existing native positive boxes**. The remaining42 inactive prospective-training photos were not activated. No internal-validation, reserved or frozen material became training evidence. The historical Chair-FP human review queue is entirely internal-validation diagnostics; those saved predictions were excluded from training. Previous review decisions/flags remain untouched.

| Native reporting family | Inspected | A | B | C | A source groups |
|---|---:|---:|---:|---:|---:|
| animals | 8 | 0 | 0 | 8 | 0 |
| clothing bags | 14 | 2 | 6 | 6 | 2 |
| furniture parts | 11 | 0 | 7 | 4 | 0 |
| kitchen appliances | 8 | 2 | 1 | 5 | 2 |
| people | 77 | 17 | 3 | 57 | 10 |
| people parts | 8 | 0 | 4 | 4 | 0 |
| plants flowers | 35 | 4 | 4 | 27 | 4 |
| screens | 20 | 8 | 2 | 10 | 8 |
| tableware small objects | 9 | 1 | 4 | 4 | 1 |
| wall art | 8 | 0 | 5 | 3 | 0 |
| windows doors | 10 | 7 | 1 | 2 | 6 |

A=source-supported visible-region negative under the stated policy. B=automatically screened, excluded from trusted masks. C=uncertain/unsuitable, ignored. The windows/doors family contributes **curtain** masks, not newly invented wall/window/door labels. Plants include flowers and decorated trees. Furniture parts include drawers and tiny door hardware. There are no accepted Drawer/Bed-rail/Wall/Animal/Artwork negatives in this version. Category and actual source-group ledgers are local; `coverage.json` gives public aggregate counts.

## Why these41 are admissible, and their limitations

There are **zero explicit source image-level Chair-negative labels** in this optimization pool. All41 use existing COCO/LVIS *visible instance polygons*, not missing labels or the furniture detector's confidence. Each combines:

1. Exact original photograph/hash, native annotation/category ID, original geometry and source metadata checksum, photograph-specific rights/creator/attribution lineage.
2. A simple, finite, in-frame native non-Chair polygon. Rasterization/erosion only proposes a region; acceptance certifies continuous original-polygon containment, rectangle-edge nonintersection and≥2px boundary clearance. No claim of bit-identical official COCO raster decoding is made; unsupported RLE/crowd/unsafe shapes are not invented.
3. A≥32px-per-side inner rectangle actually inspected in photograph context and native/inner crops.
4. Zero intersection with known Chair/stool/ignored source regions or any newly observed mixed/uncertain native region. Overlapping source targets are not double-weighted.
5. Explicit `human_reviewed:false` and `human_confirmed:false`, qualitative evidence strength, and a separate automatic-source policy hash.

Accepted examples include shirt/shorts/head interiors, dense flower/tree interiors, blank/inspected non-Chair screen pixels, refrigerator panels, curtain fabric, handbag/backpack fabric and a decorated bowl. Evidence applies to **observed visible pixels only**; it does not certify physical absence behind an occluder or whole-photo completeness. No generic Other/background label is introduced; Sofa/Bed/other furniture classes do not become negatives.

The automatic visual component and source annotations can be imperfect. “Source-verified” here means the original annotation/spatial evidence passed this documented policy, **not human-reviewed ground truth, perfect pixel labels or independently validated detector accuracy**. Native masks may be coarse; source/creator grouping is a known independence proxy, not proof of different physical scenes. No calibrated model confidence/probability was fabricated.

## What was excluded

Low resolution, blur, occlusion and mixed furniture dominate. A plant mask containing obvious window gaps remained B. A laptop region including fingers rather than isolated device pixels remained B. Pillow/cushion masks stayed B because they may be seating components. Tiny animal wallpaper motifs, face crops on screens, overexposed objects and ambiguous bed rails stayed untrusted. A television showing a seated talk-show presenter was C: a TV label cannot certify absence of *depicted* Chairs. One otherwise plausible hair region overlapped an uncertain partial-person box; five overlapping native source targets were excluded from duplicate weighting.

No missing furniture annotations were converted to negatives. The37 B and130 C outcomes are not trusted training regions. Unknown surrounding background stays ignored. Source photographs, original boxes/masks, licenses and all historical artifacts remain unchanged.

## Loss-mask comparison

The unchanged Phase10B supervision is the baseline. Adding the41 accepted regions gives **352 extra Chair-negative terms** on a fixed CPU synthetic query grid plus strictly interior coordinate probes, including **41/41 interior probes**. Existing positive targets, existing valid masks, loss normalization/coefficients and every non-Chair term are preserved. No objectness or encoder updates occur.

This demonstrates a genuine *prepared supervision* contrast. It is not actual OWLv2 query coverage. If a separately authorized zero-update smoke shows no additional valid targets on real predicted boxes, training remains blocked. CPU tensors/gradients and image inspection are not a new model forward pass.

The new `masks6.source_supervision()` uses a disabled, separately approved automatic-source gate with pinned manifest/evidence/policy hashes. It cannot pass automatic records through old C.3/C.2 human-evidence gates; those gates and the verified Phase2B evaluator were preserved. Optional user exclude/recheck notes block future use until a new checked version resolves them.

## Dashboard and files

Open **http://localhost:8770/**. It shows counts by tier/category/group, all automatic outcomes, image overlays and mask differences. Default view lists the41 supported regions; choose All/B/C to inspect the rest without repeatedly approving photographs. Optional correction notes are append-only and never constitute training authorization. Older interfaces remain unchanged.

Private artifacts in `data/open-images/preparation/phase10c6-v1/` include:

- `automatic-evidence.json`:208 source/visual outcomes and observations.
- `source-negative-evidence.json`:41 accepted regions with native polygon, exact coordinates, source/photo hashes and policy binding.
- `training-evidence-manifest.json`: unchanged59 training rows plus inactive evidence; B/C lists and unknown-background policy.
- `observations/`, `inspection/`, `inspection-expansion/`:208 automatic observation records and36 actual inspection plates.
- `source-group-coverage.json`, `mask-comparison.json`: detailed per-group and per-photo accounting.
- `proposed-smoke-photos.json`: exact24 eligible training photographs for a future smoke; not executed.

Only small code and aggregate reports are suitable for GitHub. Photographs, rights evidence, review notes, native full metadata, contact sheets and manifests stay ignored/local.

## Is a new GPU experiment justified?

**A controlled diagnostic comparison is technically justified by source-backed regions and nonzero coordinate-mask contrast, conditional on a successful real-query smoke. It is not currently authorized or independently validated.**

Next proposed approval: zero-update GPU smoke on the exact24 listed optimization photographs, at most24 forward passes,10minutes,6.5GiB allocated VRAM stop target. Measure actual added negative-query counts, positive preservation, CUDA/AMP/loss/memory safety. Stop if no meaningful query contrast, nonfinite values, hash drift or resource breach. Do not automatically continue into training.

After successful smoke, a separate bounded training authorization could compare immutable original, saved historical Phase10B behavior, and identical-initialization legacy/source-mask arms: LR1e-5, fixed seed6106, batch1/accumulation4, up to5epochs and75updates per arm,150total,60minutes,6.5GiB stop. Keep the same12 class/box-head tensors trainable and freeze vision/text/objectness/remaining parameters. Only the mask changes between arms. Anchoring shared shift/scale, furniture-only residuals and named auxiliary classes must be later separate ablations; freezing encoders alone does not prove the shared head cannot forget.

Use identical19 source-group-isolated internal-validation photos for a locked diagnostic comparison, clearly marked already exposed. Report fixed0.15 P/R/TP/FP/FN, matched-recall precision, AP50/AP75, PR curves, Chair FP distributions, misses, per-category localization/regressions, duplication/fragmentation, latency and memory. Original missing/incomplete annotation caveats remain. No80%precision/90%recall claim is made. Independently evaluated, unexposed real rooms remain a separate future acquisition/annotation/evaluation protocol; all reserved/frozen sets stay untouched.

## Verification and recovery

**317 CPU tests passed;17 synthetic Windows Chrome browser checks passed.** Actual Windows localhost rendering and photograph overlays were also checked read-only. Historical verification passed9,973 dataset/model files,160 Phase10B snapshots,24 result artifacts,27 prior implementation files,78 eligible photo hashes and earlier C.2–C.5 seals. All182 source positives and partition assignments are unchanged.


Restart `.venv/bin/python -B tools/phase10c6-v1/server6.py --port 8770`. Stop only the recorded C.6 PID/Ctrl+C; older tools remain available. Training is disabled by default. No model/annotation rollback or deletion is needed because no historical artifact was modified.
