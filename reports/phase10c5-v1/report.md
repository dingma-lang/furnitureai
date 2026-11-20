# Phase 10C.5 — object awareness and fast review

The new local review tool is implemented. No GPU training/inference, model or dataset downloads, production changes, or Phase 10D activation occurred. All new supervision is inactive. **Zero new human-approved negative regions; zero added mask terms.**

## What the model already knows

Original OWLv2 is a text-conditioned open-vocabulary detector, not a seven-class model. Its pretrained image/text encoders and classification/box heads support arbitrary text queries. People and plants can be requested; their actual performance in these rooms is untested. The saved local prediction caches contain furniture queries only. [Official model card](https://huggingface.co/google/owlv2-base-patch16-ensemble), [Transformers OWLv2 documentation](https://huggingface.co/docs/transformers/model_doc/owlv2).

Local adaptations updated 12 class/box-head tensors (1,579,526 parameters). Vision/text encoders and objectness stayed frozen. The shared classification projection and image-dependent shift/scale changed. Class scores are sigmoid of shifted/scaled image–text cosine; local postprocessing does not multiply by objectness. Consequently frozen objectness does not prevent class-score inflation. A change in general-object recognition is plausible, **not demonstrated** by the furniture-only caches.

Saved CPU weight analysis shows LR1e-5 E5 classification projection relative L2 change 1.50% and shift-weight change 1.23%; LR3e-5 E5 changes 2.53% and 2.08%. Small weight changes can affect calibration. Head changes alone do not prove catastrophic forgetting. Original and adapted checkpoint hashes were verified against historical records; see `model-audit-v2.json`.

No Person/Plant/household-object categories were in the local optimization loss. The seven furniture query slots remained unchanged. The matched-loss coefficients, focal weights and threshold 0.15 were not deliberately changed to favor recall. Phase10B's Chair-positive-only unmatched queries lacked reliable negative coverage. This is a supervision/calibration problem to test, not evidence that the original encoder lacks object knowledge.

Historical exposed 19-photo results: original P62.69%/R68.85%; LR1e-5 E5 P6.83%/R90.16%; LR3e-5 E5 P4.11%/R88.52%. These are source-native diagnostic agreement, not independent accuracy. Earlier human review found genuine Chairs among unmatched predictions; missing labels and localization errors remain important caveats.

## Already-local evidence

The native-label join audited 101 historical prospective-training photos: the unchanged **59 optimization photos / 56 known source groups / 182 active furniture boxes**, plus **42 inactive expansion photos**. Fifty-five project development/internal-validation photos were excluded from training-evidence extraction. The UI separately serves eight already-exposed internal-validation photos with 56 saved prediction diagnostics; the API forbids their conversion into training negatives.

COCO, LVIS and Open Images labels retain native IDs, names, original XYWH/XYXY coordinates, masks where provided, annotation-source hashes, photo hashes, rights lineage and source groups in the private inventory. A native dataset's original split name does not replace this project's previously fixed partition assignments. No photographs were reassigned.

| Reporting family | Raw native annotations on 59 | Photos on 59 | Known groups on 59 | Raw on all 101 | Geometric hypotheses on all 101 |
|---|---:|---:|---:|---:|---:|
| people | 77 | 20 | 20 | 81 | 81 |
| people parts | 12 | 3 | 3 | 12 | 11 |
| plants flowers | 35 | 20 | 18 | 36 | 36 |
| animals | 8 | 4 | 4 | 17 | 17 |
| wall art | 12 | 3 | 3 | 21 | 21 |
| windows doors | 30 | 14 | 14 | 36 | 36 |
| screens | 61 | 22 | 21 | 301 | 260 |
| kitchen appliances | 29 | 11 | 11 | 33 | 30 |
| tableware small objects | 287 | 38 | 35 | 465 | 457 |
| clothing bags | 81 | 26 | 26 | 84 | 81 |
| furniture parts | 11 | 5 | 5 | 47 | 47 |
| walls floors ceilings | 0 | 0 | 0 | 0 | 0 |
| fragments reflections shadows | 0 | 0 | 0 | 0 | 0 |

These are native annotation candidates, **not approved negatives or independent physical-object counts**. COCO/LVIS may annotate the same object. Conservative same-family, within-photo IoU≥0.95 hypotheses help report duplicates; all native variants remain separate and no automatic label activation occurs. Raw primary annotations total more than UI tasks because some native records cluster or represent ignored/protected subtypes. “Screens” includes TV/monitor/electronics/peripheral native labels, preserved distinctly. “Plants/flowers” includes Christmas trees and flower arrangements; flowerpots are containers. Clothing/bags includes soft household objects with exact labels retained. Drawer/furniture-part evidence does not justify treating whole furniture as background.

The review queue contains **934 source-grounded object tasks** plus 56 saved diagnostic predictions. There are **369 non-protected object tasks without a known protected-box intersection**. This is geometric eligibility for review, not proof of Chair absence. Plants, people and drawers are prioritized because they match documented confusion patterns. No native instance labels matched walls/floors/ceilings, reflections or shadows in this audited pool; table-edge confusion also lacks an exact confirmed non-Chair object label. Additional targeted review or suitable annotation is required for those gaps.

The private inventory replays existing photograph-specific CC BY 2.0 creator/source/evidence hashes and annotation licensing. It does not newly establish copyright ownership, releases or blanket commercial clearance. COCO/LVIS rights are not inferred from annotation licenses. Source-group identity is a known grouping proxy, not a guarantee of independent physical scenes. ABO catalogue classification data and unannotated Phase4 room photos cannot supply trustworthy room-level non-furniture boxes or Chair absence merely from their existence.

LVIS federated negative/not-exhaustive flags remain native metadata. Missing labels, COCO absent labels and partially annotated Open Images scopes never imply verified negative objects. Newly joined native seating is protective only: previously inactive LVIS furniture labels are not silently promoted into the optimization positives. Other furniture remains protected, including footstools, ottomans and step stools.

## Fast local review

Open **http://localhost:8768/** in the Windows browser. Set reviewer/date once. Review the single highlighted object with **1 Correct Chair**, **2 Other object / Not Chair**, **3 Missing annotation**, **4 Uncertain**. Use N/P or arrows, Undo/Ctrl+Z, box-category toggles, Zoom and Open full photograph. Purple means native non-furniture; blue furniture; red saved prediction; amber uncertainty/proposal; green explicitly Chair-free; gray ignored. The source is always labelled dataset annotation or saved model prediction.

The Other action confirms the displayed native label where applicable; it does not declare its bounding rectangle Chair-free. When the whole exact box visibly contains no ordinary Chair, seating stool, bar stool, ambiguous or partial seating, use the **single explicit Chair-free button**. Known seating/ignored boxes block approval. New missing/uncertain proposals block overlaps too. The rest of the photo stays ignored for new negatives. Footstools, ottomans and step stools stay separate.

No repeated taxonomy checkboxes or normal coordinate entry are needed. Optional notes preserve the user's words; otherwise evidence is a truthful log of the specific explicit action, not an invented visual explanation. Draw/correct only when necessary. Original-image pixel-edge conversion accounts for resizing, scrolling, zoom and all eight EXIF orientations. Geometry edits revoke approvals. New positive boxes remain inactive proposals requiring later annotation activation review.

Check, then **Save versioned review**. A new local folder contains decisions, provenance, validation, additive negative evidence, unchanged-positive lineage, a CPU coordinate-fixture mask comparison and SHA-256 seals. The browser also exports the review JSON for Restore. No old CSV, JSON, image or source label is overwritten. Private files stay under `data/open-images/preparation/phase10c5-v1/`; human-review exports never enter Git. The old C.4 tool is preserved on port8766. Its 24 clarification items remain unresolved unless the human explicitly addresses them; diagnostic history is context, never automatically reclassified or converted into training absence.

## Inactive dataset and loss strategy

`hard-negative-dataset-v1.json` retains every existing optimization positive and source-group assignment. New negative regions and auxiliary labels are empty/unactivated. Native labels supply provisional named distractor candidates; no generic “Other” softmax class is introduced. Sampling proposals: cap contributions per creator/photo/object hypothesis, balance reviewed Person/Plant/Drawer confusion groups, and retain all positive furniture and incomplete-label masks. Counts must be recomputed from actual exports before approval; no 934-task completion requirement is imposed.

The minimal primary A/C ablation changes only the additive trusted Chair-negative mask. Use identical original initialization, trainable class/box heads, data/order/seed, optimizer updates, loss coefficients/normalization and evaluation. Baseline original and historical Phase10B predictions provide context. Six confirmed regions across at least three optimization source groups are the minimum pilot gate; actual predicted-query coverage must be demonstrated by the separately authorized zero-update smoke. The current empty overlay produces zero changed terms, so running A/C now would not test a meaningful difference.

Always freeze vision/text/objectness. A later *separate* ablation can freeze/anchor shared classification shift/scale, add named auxiliary object positives with class-specific masks, or adapt a furniture-only residual on frozen original features/logits. `g5_objective.py` implements isolated CPU-tested masked sigmoid focal and zero-initialized seven-slot residual components; they are not wired into a GPU runner. Positive Person evidence never creates negative Chair targets. Unknown labels have zero gradients. The residual's named non-furniture logits remain bit-identical by construction, which protects outputs but does not prove improved recognition. Teacher distillation would require additional authorized inference and independent calibration controls. Do not mix architecture, sampling and objective changes into the first mask ablation.

## Proposed bounded GPU authorization, not execution

See **`experiment-proposal-v2.json`**. After reviewing actual evidence and hashes: at most two arms, LR1e-5, batch1/accumulation4, five epochs and 75 updates each (150 total), 60-minute queue, stop at6.5GiB allocated VRAM. Start with a zero-update CUDA/AMP/memory/mask-contrast smoke. Stop on nonfinite values, frozen-parameter drift, leakage, safety-limit breach or no real query-mask contrast. Checkpoint versions, immutable initialization, resumable update budget and approved training-only group protocol must be pinned before launch. No GPU action is authorized now; the next request must include the exact accepted evidence/config hashes. This is more restrictive than the earlier 300-update ceiling.

Primary diagnostic criterion: precision improvement at original matched recall with recall loss ≤2 percentage points, paired source-group uncertainty and explicit per-category/localization regressions. Report fixed0.15 P/R, AP50/AP75, full PR curves, per-photo/class TP/FP/FN, duplicates/fragmentation, Person/Plant confusions, score/calibration distributions with label coverage, latency and memory. Do not describe incomplete-label unmatched predictions as confirmed errors or report “80/90 achieved” from exposed panels.

Independent protocol is a **separate proposal**:120 previously unexposed rights-verified real rooms, target120 distinct source/creator groups, seven-class whole-object/completeness review including stools, clutter/occlusion and named distractors. Exclude all training/negative-review sources, exposed19/18/36 panels, existing frozen/reserved sets and known duplicate groups. Freeze membership/annotations/prompts/threshold/model choice before one approved comparison; use group-bootstrap uncertainty and no tuning after inspection. Synthetic/catalogue data cannot replace this evaluation. No images selected/acquired/scored here.

## Verification and recovery

CPU regression suites: **255 passed**. Synthetic Windows Chrome browser fixture: **37 checks passed**, including four actions, shortcuts/Undo, EXIF1–8, optional drawing/correction, explicit absence, validation isolation and versioned export. No synthetic evidence was saved as a real user review. Actual Windows localhost rendering was checked read-only.


Preservation verified 9,973 historical dataset/model files,160 Phase10B checkpoint snapshots,24 Phase10B result artifacts,27 historical implementation files and78 eligible image hashes, plus prior Phase10C/C.2/C.3/C.4 seals. Full details in `preservation-summary.json`. All original/adapted OWLv2, SigLIP2/classifiers, historical masks/partitions and Mac project remain untouched.

Restart with `.venv/bin/python -B tools/phase10c5-v1/g5_server.py --port 8768`, then Restore the last exported review JSON. Process information and logs remain private in the new preparation folder. Stop only the recorded C.5 PID or Ctrl+C; C.4 remains independent. Code rollback: switch to `experiment/phase10c4-visual-negative-regions`; do not delete exports or restore historical data/model files. No code rollback can undo a future human decision: use a new versioned review instead.
