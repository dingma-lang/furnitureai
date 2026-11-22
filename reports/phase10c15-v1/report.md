# Phase 10C.15 — Focal versus BCE negative-gradient diagnostic

**Result D: inconclusive, stopped partial execution.** One GPU process performed five forwards and 28 gradient calculations, completing four of the authorized 24 photographs. A new diagnostic equality guard failed before gradient calculation on photo five. There was no optimizer, no optimizer step, no model training, no retry, no evaluation scoring, no downloads and no Phase 10D execution. The partial evidence is informative but does not authorize a training experiment.

## What was verified and run

Parent commit `109275b31cdc26ce0e2793ca6464971d08f35b6a`; branch `experiment/phase10c15-negative-gradient-smoke`. Existing `.venv/bin/python`, unchanged packages and drivers.

The exact 24 optimization-training photographs, 24 source groups and 41 source-supported Chair-negative regions were bound to the C14 proposal. Original photo bytes, orientation, source evidence, native annotations, protected regions and supervision configuration were verified. No validation or reserved photograph was a diagnostic input. The full requested cached cohort has 79 active GT objects; the four completed photos contain 16.

Initialization followed the C14 approved proposal: immutable original OWLv2 base plus the exact saved **C11 final head**, not a new initialization or continuation of training. This deliberately measures an already inflated classification state.

| Immutable input | SHA-256 |
|---|---|
| Original OWLv2 base | `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7` |
| C11 final head | `44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f` |
| Phase 10B eligibility manifest | `ec55e01d968b584819c790de2c2a843bd87d9af960642601d65187807800b937` |
| Source-negative evidence file | `48f26dd93572bd166a31270479df11accc1899d9896304eaba5028aaf7cd1448` |

The eight differentiation-enabled tensors were the same classification projection weight/bias and six box-head tensors as C11. Calibration scale/shift, vision/text backbones, objectness and every other parameter were frozen. No optimizer was even constructed. All saved model `.grad` fields remained `None` after every completed photo; `autograd.grad` returned isolated derivatives without accumulating them. Each completed actual forward was bit-identical to that photo's saved C11 epoch-five logits and boxes.

## Controlled loss definitions

The installed `Owlv2ClassPredictionHead` returns **pre-sigmoid** shifted/scaled normalized image/text similarity logits. Thus stable `binary_cross_entropy_with_logits` is appropriate. Both variants use the same model forward, output boxes, Hungarian assignment, positive annotations, active masks, protected regions and per-image active-GT divisor. No labels, negative weight, thresholds or generic background classes were added.

```text
A source negative = 0.75 × sigmoid(z)^2 × softplus(z) / nGT
B source negative = 0.75 × BCEWithLogits(z, target=0) / nGT
```

Positive and native-negative focal losses stay unchanged at alpha 0.25/0.75 and gamma two. BCE applies only to the *additional* trusted Chair-negative mask; native negatives and ignored terms are not converted. The original 5×L1 + 2×GIoU objective remains the reference for both variants.

There were seven `autograd.grad` calls per completed photo: positive, native-negative, source focal, source BCE, total classification A, total classification B and shared box loss. The same FP16 model graph was reused. Loss calculations were FP32, gradients were scaled by the historical factor 128 and unscaled for reporting. No model re-forward was used to compare variants. No alternate loss weight or experimental arm was explored.

## Why execution stopped

The GPU trace confirms `Box objective differs from executed loss` on photo five (`coco2017:000000074638`, five active GT boxes). The diagnostic reconstruction had written `5 * L1_sum / nGT`; the original loss writes `5 * (L1_sum / nGT)`. Those are algebraically equivalent but can round differently in FP32. The strict equality guard correctly refused to continue with a nonidentical reconstructed objective.

This arithmetic-order discrepancy is confirmed in the executed source. A CPU scalar test demonstrates the FP32 non-associativity. The fifth-photo CPU replay did not reproduce the GPU guard failure, and its failed GPU component values were not persisted, so the exact GPU numerical delta is unavailable. There is no evidence here of corrupted model weights or a CUDA allocation failure. The immediate code defect is in this new diagnostic, not a demonstrated defect in the historical training loss.

The CPU-only repair normalizes L1 first and then multiplies by five, exactly as the executed loss. The strict equality check remains. A regression now checks the corrected helper against the historical loss on **all 24 saved C11 optimization outputs**, including multi-object photos. The repaired helper has **not** been executed on GPU.

The precise executed sources were preserved under private `executed-source-v1/` and bound to their original authorization hashes. The current repair intentionally fails the old configuration's code-hash check; it cannot silently reuse the consumed approval. An earlier CPU test assertion comparing separate BCE/softplus kernels was also changed to a declared numerical tolerance before launch; the actual focal replay and box-equality guards remain strict. The initial failing and passing test logs are preserved.

## Partial measured gradients

Only four photos/groups completed, with **16 positive terms, 11 trusted Chair-negative terms, 100,773 ignored class-query terms, and zero protected-target violations**. Four source regions had coverage in this partial sample. These figures do not replace the full-cohort C14 coverage counts.

| Four-photo aggregate | Existing focal A | Trusted-negative BCE B |
|---|---:|---:|
| Mean source-negative loss | 0.0000001901 | 0.00239159 |
| Source-negative output-gradient L1, summed over photos | 0.00000226047 | 0.00952691 |
| Positive output-gradient L1, same for A/B | 0.0189732 | 0.0189732 |
| Positive / source-negative output-gradient ratio | 8,393.5 | 1.9915 |
| Mean source-negative projection-gradient L2 | 0.000101775 | 0.445601 |
| Mean total classification projection-gradient L2 | 0.712862 | 0.444549 |
| Mean total classification loss | 0.00209090 | 0.00448230 |
| Maximum individual source-negative output-gradient magnitude | 0.00000140367 | 0.00254707 |

BCE supplies approximately **4,214.6×** the summed negative output-gradient mass and **4,378.3×** the mean per-photo negative projection norm. One photo had zero measured focal projection gradient but BCE projection norm 0.007519. The seven-category output gradients outside the active/source mask remain exactly zero as applicable.

These parameter norms are **means of per-photo norms**, not the norm of the average signed gradient vector. Output L1 is the sum of each photo's GT-normalized derivative magnitudes. The in-memory cohort accumulator was not written before failure, so no aggregate signed direction is invented.

| Photo | Positive projection L2 | Focal-negative projection L2 | BCE-negative projection L2 | Cosine: positive vs BCE | Cosine: classification A vs B |
|---|---:|---:|---:|---:|---:|
| `000000002270` | 1.19459 | 0.00008842 | 0.55270 | -0.8450 | 0.9265 |
| `000000002347` | 0.05016 | 0 | 0.007519 | -0.7347 | 0.9935 |
| `000000030347` | 1.05609 | 0.00030860 | 1.02086 | -0.8836 | 0.3067 |
| `000000037367` | 0.55099 | 0.00001008 | 0.20133 | -0.6548 | 0.9400 |

Cosines are in the actual projection-parameter gradient space, not output-logit space. A negative cosine means the source-negative gradient opposes the positive gradient. On photo three BCE is already about 97% of the positive projection norm and substantially rotates the total classification gradient. This is evidence of a meaningful change, and also a warning about potentially suppressing legitimate Chair learning through the shared projection. Correct region-local negative labels do not guarantee harmless changes elsewhere.

Mean shared box loss is 0.343497 and mean box-head gradient L2 is 59.1353. Classification derivatives do not reach the detached box/mask path; box regression does not reach the classification projection in these measurements. Calibration derivatives are not requested because those tensors are frozen, and their `.grad` remains `None`. The box-head versus projection norm magnitudes are not a prediction of relative Adam updates or localization quality.

## What the partial signal does and does not establish

- BCE removes a real focal attenuation effect and supplies nonzero actual projection gradients without expanding negative regions.
- All 11 measured source negatives are still easy: Chair scores range from **8.92e-7 to 0.01358**, median 0.00438. None exceeds 0.15. Stronger gradients on these regions may still fail to constrain high-confidence confusions outside their content/scale coverage.
- BCE's direction opposes the aggregate positive pressure on every completed photo, but this is not proof of better furniture discrimination. The strong cancellation and shared projection could reduce true Chair recall.
- No numerical instability, invalid negative mask or protected overlap was observed in completed records. That does not establish stability on the missing 20 photos or through optimizer updates.
- Full 24-photo/group, 41-region coverage and cohort-wide gradient-direction conclusions remain unavailable. New source data and labels were not fabricated to fill that gap.
- No precision, recall, AP, validation or frozen-set score was calculated. Neither the 60%-precision development milestone nor the independent 80%-precision/90%-recall objective was tested.

## Resource and zero-update accounting

| Item | Recorded result |
|---|---|
| GPU process launches | 1 |
| Actual model forwards | 5 / maximum 24 |
| Completed photo summaries | 4 / 24 |
| Gradient calculations | 28 |
| Worker elapsed time at failure | 10.7187 seconds |
| Enclosing launcher wall time | 14.2213 seconds |
| Optimizers / optimizer steps / model updates | 0 / 0 / 0 |
| Exact peak allocated/reserved/observed VRAM | **Unavailable** |
| Terminal live-model/buffer fingerprint | **Unavailable** |

The worker enforced a 6.5 GiB limit, a stricter allocator fraction, sampled whole-device memory and had a 180-second parent deadline. However, the initial failure handler omitted peak-memory samples and the final live-model hash. The process exited, so those readings cannot now be reconstructed honestly. No post-exit idle GPU reading is substituted for peak VRAM. All four completed receipts show `.grad` remained `None`, and code contains no optimizer or update path, but a terminal whole-model hash is not claimed. Future failure telemetry and per-photo peak records were repaired on CPU only.

Diagnostic allocation remains below the 128 MiB cap, including retained atomic pending-file links, logs, executed-source copies and reports. Exact captured allocation is in `preservation.json`; `artifact-seal.json` records the final sealed allocation. No checkpoint file was written or replaced.

## Verification, preservation and next approval

Post-stop CPU suites and historical-preservation results are in `tests.json` and `preservation.json`. The local repair passes the all-24 cached box-loss regression. Original OWLv2, C11/C8/C13 checkpoints, original image bytes, source evidence, historical partitions, package source files and C14 caches remain bound to their historical hashes. No source annotation or review decision was changed. Models, photographs and raw query caches remain outside Git.

**Recommended next action:** separately authorize recovery of the missing **20** gradient summaries, reusing the four complete immutable records. This includes repeating photo five because its graph and gradients were not persisted. It would make **25 total forwards across the failed attempt plus recovery**, exceeding the prior 24-total-forward allowance by one; that revision must be explicit. No recovery has been launched, no recovery destination exists, and there is no automatically executable recovery configuration.

The exact proposed 20 IDs, source hashes, frozen C11 initialization, 180-second / 6.5 GiB / 128 MiB ceilings, zero-update conditions and telemetry requirements are in `recovery-proposal.json`. No future training configuration is prepared because result A has not been established for the full cohort. Do not rerun the old launcher; it is consumed and rejects repaired code. If a later complete diagnostic supports training, that will require another explicit approval and an independent precision/recall evaluation plan.
