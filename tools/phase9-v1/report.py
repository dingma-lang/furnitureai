"""Local detailed evidence and a small sanitised Git-safe comparison summary."""
import csv
import subprocess
from core import *
from data import exposed_rows

def fmt(v):return '—' if v is None else f'{100*v:.2f}'
def table(results):
 rows=['| Method | P % | R % | F1 % | AP50 % | AP75 % | TP/FP/FN | Duplicate FP | Seconds/photo | Peak GiB |',
  '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
 for name,m in results.items():rows.append(f"| {name} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['ap50'])} | {fmt(m['ap75'])} | {m['tp']}/{m['fp']}/{m['fn']} | {m['duplicate_fp']} | {m['mean_inference_pipeline_seconds']+m['mean_analysis_seconds']:.3f} | {m['peak_allocated_bytes']/2**30:.3f} |")
 return '\n'.join(rows)

def run():
 selection=checked(OUT/'selection.json');protocol=checked(OUT/'protocol.json');preserve=read(OUT/'preservation-after.json');tests=read(OUT/'tests.json')
 exposed={p.stem:checked(p)['metrics'] for p in sorted((OUT/'evaluations/exposed').glob('*.json'))}
 final={p.stem:checked(p)['metrics'] for p in sorted((OUT/'evaluations/final').glob('*.json'))};selected=selection['selected']
 # Local full curves and figures never contain copied photograph bodies.
 for part,results in [('exposed',exposed),('final',final)]:
  directory=OUT/'curves'/part;directory.mkdir(parents=True)
  for name,m in results.items():
   with (directory/(name+'.csv')).open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['threshold','precision','recall','tp','fp']);writer.writeheader();writer.writerows(m['pr_curve'])
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(10,7))
  for name,m in results.items():
   c=m['pr_curve'];ax.plot([r['recall'] for r in c],[r['precision'] for r in c],label=name,alpha=.85 if name in ('original-baseline','adapted-baseline',selected) else .4)
  ax.set(xlabel='Source-annotated recall',ylabel='Precision in FP-observable native scope',xlim=(0,1),ylim=(0,1.02),title=part+' — native partial-label agreement, not verified real-world accuracy');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(directory/'precision-recall.png',dpi=160);plt.close(fig)
 with (PUBLIC/'summary.csv').open('x',newline='') as f:
  fields=['partition','method','images','gt_boxes','precision','recall','f1','ap50','ap75','tp','fp','fn','duplicate_fp','mean_inference_pipeline_seconds','peak_allocated_bytes'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for part,res in [('exposed',exposed),('final',final)]:
   for name,m in res.items():writer.writerow(dict(partition=part,method=name,**{k:m[k] for k in fields if k in m}))
 save(PUBLIC/'protocol.json',protocol);save(PUBLIC/'tests.json',tests)
 chosen=exposed[selected];ref=exposed['adapted-baseline'];matched=chosen['precision_at_reference_recall'].get(str(round(29/41,8)))
 cat=['| Category | Adapted baseline TP/FP/FN | Selected TP/FP/FN | Baseline AP75 % | Selected AP75 % |','|---|---|---|---:|---:|']
 for c in CLASSES:
  a=ref['categories'][c];b=chosen['categories'][c];cat.append(f"| {c} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {fmt(a['ap75'])} | {fmt(b['ap75'])} |")
 text=f'''# Phase9 — inference-only furniture precision experiments

## Backup and scope

Starting commit **{protocol['starting_commit']}**, verified equal to GitHub main before changes. Experimental branch **experiment/phase9-multiview-detection**; local baseline tag **phase9-baseline-20261118**. Production detector defaults and main were not changed. Git backs up tracked code only. The original OWLv2 (619,918,824 bytes), SigLIP2 (1,500,800,904 bytes), expanded classifier and selected Phase5C delta were directly SHA-256 verified before/after. No independent off-workstation backup of these ignored weights or source datasets was verified. No large model copies were made. Raw photos, licensed datasets, model weights and unsanitised experiment artifacts are excluded from Git.

## Fixed experiment protocol

Fifteen predeclared arms, using only **18 previously exposed development photographs /41 native boxes** for selection. Both base and selected Phase5C models remain frozen. CUDA smoke used the first exposed photo; its16 cached passes were reused. Eight passes/image/model: full-frame768/960/1152, horizontal-flip960, four overlapping62.5%-extent tiles960. These are transformations of one photograph, **not new physical viewpoints or independent captures**. Upsampling creates no new source detail. Positional-embedding/box-bias interpolation gives actual48/60/72 patch grids; recorded pixel tensors and hashes verify genuine model-resolution changes. No vertical flips.

Confidence.15, class-aware NMS IoU.5, AP floor.001, max300 per-view detections, one-to-one same-native-category matching IoU.5/.75 and101-point AP. Historical original/Phase5C baselines were exactly reproduced at operating counts/AP level. All original predictions remain intact; new per-view/fused predictions are separate. Custom confidence-weighted averaging uses complete-link same-class IoU>=.55 and at most one member/view; it is not claimed to be an external WBF implementation. Scores remain uncalibrated. Ambiguous cluster assignments are flagged rather than chained. Adjacent-object tests are included; unknown physical duplicates remain possible.

The tile method excludes internal-boundary fragments, preserves full-frame boxes, and adds only unmatched interior proposals with score>=.25 and area<=20% of the image. Scale/flip consensus requires at least two views. Original-versus-adapted agreement is correlated model agreement, not independent verification.

The optional frozen SigLIP cosine head checks only crops with minimum32-pixel side, aspect<=4, away from photo edges. Unsuitable/uncertain crops remain advisory, not forced categories. A different category with relative softmax margin>=.10 may abstain from that proposal; detector labels are never replaced. The head/encoder/preprocessor fingerprints are pinned. No encoder/head/detector updates occurred.

Selection is fixed-confidence precision improvement >=3pp over adapted baseline, loss of at most one matched GT object, AP75 loss<=2pp, and no matched-object loss for tiny-support Sofa/Chair/Lamp. Rank by precision, thenAP75/F1. Full PR curves and precision envelopes at historical reference recalls are supplementary diagnostics, not additional threshold searches. **Selected: {selected}.** Qualified improvement on the exposed protocol: **{selection['robust_improvement_on_exposed_protocol']}**. If none qualifies, the baseline is retained and no robust winner is declared. Qualification failures are recorded in the immutable selection receipt.

## Exposed development results

'''+table(exposed)+'\n\n'+ '\n'.join(cat)+f'''

Matched-recall envelope for selected method at recall>=29/41: **{matched}**. This envelope describes the observed PR curve; it was not used to retune the final operating threshold. Final confidence remains.15. AP50 is not operating precision.

Desk/Cabinet sensitivity (excluded categories, same predictions): adapted baseline **{ref['exclude_desk_cabinet_sensitivity']}**; selected **{chosen['exclude_desk_cabinet_sensitivity']}**. These are diagnostic aggregations, not repaired labels or replacement selection protocols. Per-category JSON contains AP50/AP75, P/R/F1, matched IoU, duplicates, FP-observable images and unknown ignored counts. Mixed-object/fragmentation counts are geometric candidates, not human-confirmed errors.

Runtime is the sum of measured required pass/verification stages plus measured cache-read/fusion overhead per photograph, excluding model loading, scoring, stage writes and galleries. Shared caches were used across ablations; this is not15 independently rerun end-to-end timing benchmarks. Cache-reading/fusion overhead is also recorded separately in each metric JSON. Model load records and smoke cold-start costs are retained. Peak VRAM means PyTorch allocated memory, not total/reserved GPU memory. Batch one, sequential models,75% allocator fraction,6.5GiB stop target and30-minute invocation bounds were enforced. CUDA/model errors are fatal; no CPU fallback or recipe weakening. Caches are immutable/checksummed and resumable. Actual detector passes: **{preserve['actual_detector_passes']}**.

## One locked comparison on36 fresh-development photographs

No recorded prior model use was found in inspected local model-evaluation evidence; the18 exposed IDs/hashes are disjoint. Unrecorded external use cannot be ruled out. Exact36 membership, native126 boxes, source groups and hashes were locked as metadata before selection; no fresh pixels entered tuning. Selected configuration and non-test code hashes were sealed before any final pixels were opened. Original baseline, adapted baseline and the one selected recipe were compared once. **No configuration/threshold was retuned from these results.**

'''+table(final)+'''

**The fresh-set precision/AP are not directly comparable to the18-photo values.** Source taxonomies and annotation completeness differ. COCO completeness-unverified and LVIS NEX/filtered subsets do not turn unmatched predictions into FPs. Exact native coverage with all source targets retained is required for unmatched-FP observability; source scopes and exclusions remain immutable. Retained annotated positives still contribute recall. Ignored/unresolved objects and other unlabelled furniture remain unknown. Results therefore measure **conditional native-source annotation agreement**, not exhaustive real-room precision. Cabinet has no fresh positives; no seven-class independent performance claim is possible. The fresh set is now exposed by this one comparison and must not be called untouched for future searches.

## Visual evidence

Local gallery: `galleries/exposed/index.html`, all18 exposed photographs, blue native GT, green source matches, red source-relative FPs, orange unscored proposals. Panels include original/adapted baselines, multi-scale NMS, guarded tiles, weighted fusion and the selected/cross-model method. Duplicates and mixed-object candidates are labelled; missed source annotations are counted. Creator/source/license evidence is retained in the gallery attribution JSON. Plausible appearance is not proof of correctness; native Cabinetry run/front and Desk/Table ambiguities remain. No photo/gallery is pushed to Git.

Full per-method predictions and case diagnostics: `evaluations/exposed/`; one final comparison: `evaluations/final/`. PR curves/CSVs: `curves/exposed/`, `curves/final/`. JSON events identify new/lost matches and source FPs; localization means use matched objects and can have survivor bias. Raw per-view source coordinates, crop transforms, tensor sizes, inference times and memory are under `passes/`.

## Genuine multi-photograph views

No independently verified grouped same-room/capture set was identified in the local authorised room data. Different creators/photos or matching product IDs do not prove physical identity/capture independence. The existing ABO capture/product evidence does not establish registered real-room views. `crossview.py` therefore implements only a guarded interface: require same-scene provenance, reject duplicate captures, retain predictions in their own image coordinates, and require independent object correspondences before object-level consistency. Cross-camera box fusion without verified registration is deliberately rejected. No unseen viewpoint was generated, no external image was downloaded, and no genuine-view accuracy is claimed.

## Recommendation and conditional next training proposal

Read exposed and final results together: an exposed-set qualification is not proof of broad improvement. Do not promote any method into the application's default. Prefer lower false positives only when recall, Bed/Table localization and rare-category support are maintained; distinguish Cabinet-driven source agreement and unmatched/unknown annotation gaps. Keep SigLIP as advisory unless its matched comparison qualifies, and retain safe abstention for mixed/partial crops. A method adding boxes or agreeing with itself is not automatically stronger evidence.

If further training is approved later, propose a source-native, partial-label **head-only** study using the prepared120-photo training pool and training-only grouped inner validation, original and Phase5C immutable references, at most two configs, five epochs/150 optimizer updates each (300 total), batch1/accumulation4, conditional FP16, soft6.5GiB VRAM target and one-hour wall cap. Exact target IDs, Cabinet unit semantics and LVIS/COCO completeness masks must be locked/tested first. No full-backbone/LoRA updates or new weights/data. The36-photo panel cannot be repeatedly optimised after this comparison; reserve a genuinely untouched, separately approved final evaluation. Cabinet still has zero fresh native positives, so a broad seven-class claim needs better real Cabinet evidence. This is a proposal only; **no training was performed or authorised by the implementation**.

## Verification, preservation and rollback

'''+str(tests['passed'])+''' tests passed. Historical stat snapshot and strong critical dataset/registry/model hashes passed; frozen photographs received stat-only preservation, no byte reads, decoding or scoring. The previous33 inactive LVIS labels remain inactive. Original model/configuration/source files and historical predictions were not overwritten. There were no installations, dataset/model downloads, production changes or Mac project modifications. New disk allocation is recorded in `completion.json`;512MiB phase budget includes raw artifacts, galleries, source and sanitised report.

Only new Phase9 source/tests, `.gitignore` report allowlisting, and the small sanitised report/protocol/summary/test files are committed. The data directory remains ignored. Git commit and push receipts are reported in chat after completion; no force-push or merge into main.

After the experimental commit is saved, rollback code with `git switch main`. The original production detector has not changed; there is no weight rollback or dataset deletion to perform. The baseline tag is a code reference, not a model/data backup. Keep ignored experiment artifacts for provenance.
'''
 with (OUT/'report.md').open('x') as f:f.write(text)
 # Same report contains no original photo IDs, raw GT rows, images or credentials.
 with (PUBLIC/'report.md').open('x') as f:f.write(text)
 print('Local full report and sanitised shareable metrics/protocol generated',flush=True)

if __name__=='__main__':run()
