"""Compact versioned report from the fixed endpoint; no model execution."""
from commonD import *


def percent(v):return 'unscored' if v is None else f'{100*v:.2f}%'


def run():
    result=read(REPORT/'results.json');done=result['resources'];arch=read(REPORT/'architecture.json');decision=result['decision'];comparison=result['comparison']
    lines=['# Phase 10D — Original-classifier-preserving box-only experiment','',
           'One authorized diagnostic arm. No model/data downloads, production replacement, new labels, negative supervision, reserved scoring or automatic follow-up experiment. Historical C21 configuration remains disabled and unchanged; this phase has its own explicit execution binding.','',
           '## Actual execution','',
           f"Completed {done['epochs']} epochs, {done['updates']} optimizer updates, {done['forwards']} durable forward intents / {done['actual_forwards']} observed detector calls and {done['training_backwards']} training backwards. GPU-process wall time including startup/smoke/final diagnostic: {done['seconds']:.3f} seconds. Peak allocated VRAM: {done['peak_allocated_bytes']} bytes ({done['peak_allocated_bytes']/2**30:.3f} GiB); peak reserved: {done['peak_reserved_bytes']} bytes.",'',
           'FP32 throughout; batch 1, accumulation 4 (last group 3); fixed seed 6106 and unchanged five-epoch C8/C20 data ordering. AdamW LR 1e-5, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01, global norm clip 1. One fixed endpoint; no epoch/threshold/hyperparameter selection.','',
           'The 322 calls were 4 zero-update smoke, 295 training, 4 post-training FP32 invariance checks and 19 final internal-development calls. No additional original-baseline inference was performed; saved original/C20 outputs were replayed with identical metrics.','',
           '## Actual parameter and data scope','',
           f"Original base `models/owlv2-original-baseline-20261103/model.safetensors`, SHA-256 `{BASE_SHA}`. All {arch['persisted_tensors_exact_original']} persisted model tensors matched this base before GPU execution and again on GPU initialization. No adapted heads were loaded.",'',
           'Exactly six trainable tensors, 1,184,260 scalars:','']
    for name in sorted(arch['trainable']):lines.append(f"- `{name}` — shape {arch['trainable'][name]['shape']}")
    lines.extend(['',f"All other {len(arch['frozen'])} named parameters remain frozen, including classification projection, shift/scale, the full vision/text model, shared layer normalization and objectness. The complete exact frozen-name/shape/hash list and registered buffers are in `architecture.json`. OWLv2's derived `box_bias` is an **unregistered plain tensor**; the executor protects it with an additional original-computation/hash check.",'',
                  f"Frozen parameter/buffer fingerprint: `{done['frozen_sha256']}`. It was checked at initialization, all six snapshots and completion; each epoch has a saved fingerprint and per-box-tensor statistics/hashes. Exact box parameter values and optimizer state are in local checkpoints. All four repeated FP32 training-photo checks were bit-identical for classification logits.",'',
                  'Optimization used exactly 59 photos / 56 known source groups / 182 source-supported positive boxes: Chair 114, Sofa 40, Bed 28. Table, Desk, Cabinet and Lamp gained no supervised examples. Chair includes seating/bar stools; native annotations and unresolved seating remain unchanged. The five newly prepared Chair-negative regions, earlier source-negative regions, all ignored regions, new annotations and generic background labels were not activated.','',
                  'The objective is the unchanged detached original-classification Hungarian assignment with 5×L1 + 2×GIoU on retained positives only. Classification/objectness/negative losses are zero. The frozen classifier can affect matching; localization can alter NMS and final retained detections despite unchanged raw logits.','',
                  '## Matched internal-development comparison','',
                  'Exact exposed 19-photo / 61-positive Sofa–Chair–Bed panel; confidence 0.15, IoU 0.50, NMS 0.50, score floor 0.001, cap 300, unchanged 101-point macro AP. Native annotation completeness controls FP scoring. Unknown unmatched predictions are not confirmed mistakes. This panel is not independent acceptance.','',
                  '| Model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN | Precision at original recall |','|---|---:|---:|---:|---:|---:|---:|'])
    for name,v in comparison.items():
        h=v['headline'];matched=v['precision_at_original_recall']
        lines.append(f"| {name} | {percent(h['precision'])} | {percent(h['recall'])} | {percent(h['ap50'])} | {percent(h['ap75'])} | {h['tp']} / {h['fp']} / {h['fn']} | {percent(matched['precision'] if matched else None)} |")
    lines.extend(['','Matched-recall values are descriptive PR-envelope values, not newly adopted thresholds. Recomputed original and C20 baseline metrics exactly matched their saved historical results. Full PR curves are in `precision-recall.json`.','',
                  '| Category / model | Precision | Recall | AP50 | AP75 | TP / scored FP / FN | Unknown unmatched | Duplicate FP |','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for c in ('chair','sofa','bed'):
        for name,v in comparison.items():
            m=v['categories'][c];lines.append(f"| {c} / {name} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['ap50'])} | {percent(m['ap75'])} | {m['tp']} / {m['fp']} / {m['fn']} | {m['unknown_unmatched']} | {m['duplicate_fp']} |")
    lines.extend(['','Table, Desk, Cabinet and Lamp remain unscored for positive-category performance; zero scored FPs in uncovered categories is not evidence of correctness. All seven native category fields, unknown counts and FP-score distributions are retained in `results.json` and `category-metrics.csv`.','',
                  '## Localization and confidence attribution','',
                  '| Category | Shared matched GT | Original mean IoU | Adapted mean IoU | Delta | IoU 0.75 up / down | Original-only / adapted-only matches |','|---|---:|---:|---:|---:|---:|---:|'])
    for c,v in result['common_GT_localization'].items():
        val=lambda n:'n/a' if v[n] is None else f'{v[n]:.5f}'
        lines.append(f"| {c} | {v['common_matched_GT']} | {val('mean_original_IoU')} | {val('mean_adapted_IoU')} | {val('mean_delta_IoU')} | {v['IoU75_crossings_up']} / {v['IoU75_crossings_down']} | {v['original_only_matches']} / {v['adapted_only_matches']} |")
    for name,v in result['logit_drift'].items():
        lines.extend(['',f"{name}: aligned 19-photo logits bit-identical to saved original = {v['raw_logits_bit_identical']}; mean logit difference {v['mean_all_categories']:.9g}, maximum absolute difference {v['maximum_absolute_difference']:.9g}. Per-category means and score/logit quantiles are in `results.json`."])
    for name,v in comparison.items():
        lines.extend(['',f"{name}: raw Chair query counts at 0.15 / 0.50 / 0.80: {v['raw_Chair_at_least']['0.15']} / {v['raw_Chair_at_least']['0.5']} / {v['raw_Chair_at_least']['0.8']}; postprocessed heuristics: {v['headline']['diagnostic_heuristics']}."])
    lines.extend(['','Four identical before/after training inputs establish numerical classification preservation independently of historical cache precision. If the aligned development logits also match original, retained detection/FP/FN differences are attributable to changed boxes and downstream matching/NMS, rather than trained calibration or classification projection. Fragmentation and mixed-object counts remain heuristics under partial annotation.','',
                  '## Learning curve','',
                  '| Epoch | Updates | Mean box loss | Mean L1 | Mean GIoU loss | Mean preclip box-gradient norm | Matched positives |','|---|---:|---:|---:|---:|---:|---:|'])
    for e in result['training_epochs']:
        lines.append(f"| {e['epoch']} | {e['updates']} | {e['mean_box_loss']:.6f} | {e['mean_L1']:.6f} | {e['mean_GIoU']:.6f} | {e['mean_preclip_box_gradient_norm']:.6f} | {e['matched_positive_targets']} |")
    lat=read(RUN/'evaluation/raw-complete.json')['records'];values=sorted(r['latency_seconds'] for r in lat)
    lines.extend(['',f"Final inference forward mean: {sum(values)/len(values):.4f}s; median: {values[len(values)//2]:.4f}s. Synchronized forward-only timing excludes CPU decode/preprocessing and transfers. No new original-model timing was collected, so a matched baseline latency claim is unavailable.",'',
                  '## Decision and limits','',
                  f"Outcome **{decision['outcome']}**. Fixed-threshold 60% precision plus original 68.85% recall milestone reached: **{decision['development_60_precision_original_recall_milestone']}**. Stronger C21 localization/precision/recall gates all passed: **{decision['C21_stronger_gates_all_pass']}**. Exact per-gate values are in `decision.json`; the 60% milestone alone is not the complete predeclared success criterion.",'',
                  'Outcome A: localization improved with classification preserved. B: classification preserved without demonstrated localization improvement. C: localization improved but precision/recall regressed. D would indicate insufficient execution/evaluation evidence. These are development diagnostics on an exposed, partially annotated three-category panel, not independent seven-class accuracy.','',
                  'Recommended next direction: '+('collect and independently complete source-disjoint all-seven-class room evaluation before any production decision;retain localization-only development as a hypothesis' if decision['outcome']=='A' else 'investigate frozen-original-teacher distillation as a separately prepared future experiment;do not start it or return to loss-coefficient sweeps without new evidence')+'.','',
                  'No fully independently annotated all-seven-class photographs currently qualify (C21 audit). The ultimate 80% precision / 90% recall target remains untested. No reserved/frozen images were used here.','',
                  '## Checkpoint, preservation and recovery','',
                  f"Final local six-box checkpoint: `{str(RUN.relative_to(ROOT))}/{done['final_checkpoint']['file']}`; SHA-256 `{done['final_checkpoint']['sha256']}`. This is a delta over the immutable original base, not a deployed full model or a twelve-head checkpoint.",'',
                  '`preservation.json` records historical image/model/annotation/seal checks, original and new checkpoint integrity, frozen fingerprints and unchanged main. `postrun-tests.json` records the relevant CPU suites. GPU receipts and raw predictions were saved atomically before the separate Torch-free CPU metric process. No interpreter/package change was made.','',
                  '`resource-accounting.json` records launcher wall time, allocated/reserved VRAM and unique-inode/conservative storage. The 600-second launcher deadline includes initialization and diagnostics; the worker stops cooperatively earlier. Durable forward intents and optimizer intents distinguish attempted operations from committed results. No retry or resume was launched.','',
                  'Rollback requires no model replacement: production and original weights never changed. Keep local epoch checkpoints, manifests and telemetry. Any future recovery needs separate authorization and must restore the six-tensor optimizer/cursor/RNG binding without replaying committed updates. Legacy twelve-head workers are incompatible. Code/configurations/reports are pushed on the dedicated experimental branch; weights, photos and raw caches stay local.',''])
    atomic(REPORT/'report.md','\n'.join(lines).encode())
    print('Phase10D actual-result report written',flush=True)


if __name__=='__main__':run()
