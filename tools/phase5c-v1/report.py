"""Produce documentation and figures from completed development-only artifacts."""
import csv
import html
import os
import re
from common import *
from control import RUN_ROOT, FIT_IDS

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def csv_new(name,rows):
 with (RUN_ROOT/name).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def pct(v):return f'{100*v:.2f}'

def run():
 comparison=read(RUN_ROOT/'comparison-v1/comparison.json');names=('original',*FIT_IDS)
 primary=comparison['primary_metrics'];selected=comparison['selected_experiment'];best=comparison['selected_checkpoint']
 category_rows=[];curves=[];updates=[];runtime={}
 for name in names:
  for cat,m in primary[name]['categories'].items():category_rows.append(dict(model=name,category=cat,**m))
 for name in FIT_IDS:
  fit=RUN_ROOT/name;done=read(fit/'complete.json');start=min(read(p)['unix_ns'] for p in fit.glob('execution-*.json'))
  runtime[name]=dict(successfully_checkpointed_active_seconds=done['seconds'],orchestration_wall_seconds=(fit.joinpath('complete.json').stat().st_mtime_ns-start)/1e9,
    peak_allocated_bytes=done['peak_allocated_bytes'],amp_replays=len(list(fit.glob('amp-overflow-*.json'))))
  for f in sorted(fit.glob('validation-epoch-*.json')):
   v=read(f);m=v['metrics'];op=m['micro_operating']
   curves.append(dict(fit=name,epoch=v['epoch'],optimizer_updates=v['step'],mean_training_loss=v['mean_training_loss'],ap50=m['macro_ap50_supported'],ap75=m['macro_ap75_supported'],precision=op['precision'],recall=op['recall']))
  for f in sorted(fit.glob('update-*.json')):
   r=read(f);updates.append(dict(fit=name,step=r['step'],epoch=r['epoch'],loss=r['loss'],grad_norm=r['grad_norm'],scaler_scale=r['scaler_scale'],numeric_replays=r.get('numeric_replays',0),images=';'.join(r['images']),checkpoint_sha256=r['sha256']))
 csv_new('category-metrics.csv',category_rows);csv_new('training-development-curves.csv',curves);csv_new('optimizer-updates.csv',updates)
 csv_new('sensitivity.csv',[dict(model=name,scope=scope,**m) for name in names for scope,m in comparison['sensitivity'][name].items()])
 csv_new('localization.csv',[dict(model=name,category=cat,**v) for name in FIT_IDS for cat,v in read(RUN_ROOT/'comparison-v1'/(name+'-localization.json')).items()])
 csv_new('source-confusion.csv',[dict(model=name,source_category=g,predicted_category=p,count=count) for name in names for g,row in read(RUN_ROOT/'comparison-v1'/(name+'-confusion.json'))['matrix'].items() for p,count in row.items()])
 save(RUN_ROOT/'runtime-summary.json',dict(fits=runtime,note='Active seconds cover successful checkpointed attempts, including loading/checkpointing/development inference. They exclude the first fit failed pre-update-2 attempt and debugging/wait time. Orchestration wall time includes smoke pause and recovery. Neither is GPU kernel-only duration.'))
 save(RUN_ROOT/'selected-checkpoint-registry.json',dict(schema='phase5c-experimental-head-delta-v1',**best,experiment=selected,base_checkpoint=str(OWL),base_checkpoint_sha256=OWL_SHA,dataset_sha256=comparison['dataset_sha256'],deployment_authorized=False,frozen_evaluation_used=False,model_format='12 class_head/box_head parameter tensors plus optimizer/scaler/RNG recovery state; requires immutable original model',limitations='Selected using 18 development photographs with source-native provisional annotations; not independently validated.'))
 os.environ['MPLCONFIGDIR']=str(RUN_ROOT/'matplotlib-cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for name in FIT_IDS:
  rows=[r for r in curves if r['fit']==name];xs=[r['epoch'] for r in rows]
  for ax,key,title in zip(axes.flat,('mean_training_loss','ap50','ap75','precision'),('Training loss','Development macro AP50','Development macro AP75','Development precision at confidence 0.15')):
   ax.plot(xs,[r[key] for r in rows],marker='o',label=name);ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.25)
 for ax,key in zip((axes[0,1],axes[1,0],axes[1,1]),('macro_ap50_supported','macro_ap75_supported','precision')):
  ref=primary['original']['micro_operating'][key] if key=='precision' else primary['original'][key]
  ax.axhline(ref,color='gray',linestyle='--',label='Immutable original')
 for ax in axes.flat:ax.legend(fontsize=8)
 fig.suptitle('Phase 5C — source-annotation development agreement (18 photos; selection data)')
 fig.tight_layout();fig.savefig(RUN_ROOT/'training-development-curves.png',dpi=160);fig.savefig(RUN_ROOT/'training-development-curves.svg');plt.close(fig)
 totals=table(['Model','AP50 %','AP75 %','Precision %','Recall %','TP / FP / FN','Duplicate FP'],[
  [name,pct(primary[name]['macro_ap50_supported']),pct(primary[name]['macro_ap75_supported']),pct(primary[name]['micro_operating']['precision']),pct(primary[name]['micro_operating']['recall']),f"{primary[name]['micro_operating']['tp']} / {primary[name]['micro_operating']['fp']} / {primary[name]['micro_operating']['fn']}",primary[name]['micro_operating']['duplicate_fp']] for name in names])
 cats=table(['Category','GT','Original AP50 / AP75 %','1e-5 AP50 / AP75 %','3e-5 AP50 / AP75 %','Original TP/FP/FN','1e-5 TP/FP/FN','3e-5 TP/FP/FN'],[
  [cat,primary['original']['categories'][cat]['gt_boxes'],*[f"{pct(primary[name]['categories'][cat]['ap50'])} / {pct(primary[name]['categories'][cat]['ap75'])}" for name in names],*['/'.join(str(primary[name]['categories'][cat][k]) for k in ('tp','fp','fn')) for name in names]] for cat in CLASSES])
 sensitivity=table(['Aggregation','Original AP50 / AP75 %','1e-5 AP50 / AP75 %','3e-5 AP50 / AP75 %'],[[scope,*[f"{pct(comparison['sensitivity'][name][scope]['macro_ap50'])} / {pct(comparison['sensitivity'][name][scope]['macro_ap75'])}" for name in names]] for scope in comparison['sensitivity']['original']])
 loc=table(['Category','Common original/selected matches','Mean paired IoU change','New matches','Lost matches'],[[cat,v['common_matched_gt'],f"{v['paired_mean_iou_delta']:+.4f}" if v['paired_mean_iou_delta'] is not None else 'NA',v['newly_matched_gt'],v['lost_matches']] for cat,v in read(RUN_ROOT/'comparison-v1'/(selected+'-localization.json')).items()])
 diagnostics=table(['Model','Duplicate FP','Fragmentation candidates','Multi-object candidates','Category-mismatch candidates','Missed annotated objects'],[[name,primary[name]['micro_operating']['duplicate_fp'],*[len(primary[name]['diagnostics'][k]) for k in ('fragmentation_candidates','multi_object_box_candidates','category_mismatch_candidates','missed_annotated_objects')]] for name in names])
 fits=table(['Fit','Epochs / updates','Best epoch / update','Loss epoch 1 → 10','Active seconds','Orchestration seconds','Peak allocated GiB'],[[name,'10 / 70',f"{read(RUN_ROOT/name/'complete.json')['best']['epoch']} / {read(RUN_ROOT/name/'complete.json')['best']['step']}",f"{[r for r in curves if r['fit']==name][0]['mean_training_loss']:.4f} → {[r for r in curves if r['fit']==name][-1]['mean_training_loss']:.4f}",f"{runtime[name]['successfully_checkpointed_active_seconds']:.2f}",f"{runtime[name]['orchestration_wall_seconds']:.2f}",f"{runtime[name]['peak_allocated_bytes']/2**30:.3f}"] for name in FIT_IDS])
 tests=[]
 for path in sorted(RUN_ROOT.glob('tests-*.log')):
  text=path.read_text();match=re.search(r'Ran (\d+) tests',text)
  tests.append(dict(log=path.name,tests=int(match[1]) if match else None,passed=text.rstrip().endswith('OK')))
 save(RUN_ROOT/'test-summary-final.json',dict(runs=tests,historical_tests_passed=63,phase5c_tests_passed=12,total_current_passing_tests=75,initial_phase5c_failure='Test expected numeric_replays in the initial paused smoke receipt created before that instrumentation. Test now explicitly treats absent historical replay count as zero; historical receipt untouched. See original failed and r2 passing logs.'))
 report=f'''# Phase 5C — completed controlled head-only OWLv2 training

Both approved experiments completed, with exactly **70 optimizer updates each, 140 total**, and 10 epochs each. Selected experimental checkpoint: **{selected}, epoch {best['epoch']}, update {best['step']}**. This is provisional source-annotation agreement on 18 development photographs, not independently verified real-room accuracy. No model was deployed and no frozen-evaluation photograph was decoded, tuned on or scored.

## Immutable data, annotations and adaptation scope

27 approved training photos, 18 development photos, 18 eligible reserved frozen photos unused; all 60 original frozen slots unchanged. All 32 licensing quarantines excluded. Each training photo contributed exactly 10 times per fit, 270 presentations each. Known image hash, image ID, source identity and identity-group overlap checks passed; metadata cannot establish absence of every unknown physical relationship. No official Phase 2B images were scored.

Both configs are the byte-identical prepared `heads-lr1e-5.json` and `heads-lr3e-5.json` under `../phase5b-v1/configs/`. Seed 20261105; AdamW weight decay 0.01; batch 1; accumulation 4 (last group 3, normalized to 3); 960-square original processor; FP16 autocast with dynamic loss scaling; 80% CUDA memory ceiling. Only 12 class-head/box-head tensors (1,579,526 parameters) were trainable and in the optimizer. Backbone, text tower, objectness and other non-head parameters remained unchanged, fingerprint checked every epoch. No LoRA, last-layer or backbone adaptation occurred. Historical config `updates_approved:false` remains preserved as a preparation-time status; `authorization.json` records this phase's explicit approval separately.

The approved partial-label loss body is unchanged from Phase 5B. Unknown class targets and group/depiction regions are masked; objectness receives no global background supervision. Only published explicit negatives and positive classes with native boxes supervise classification. Unmatched candidates within those covered classes receive the approved native-class negatives, so missing instances within a supposedly covered class remain a source-label risk. Unannotated/ambiguous classes are not promoted to verified negative classes. No new labels, manual evidence or annotation corrections were invented.

## Execution and numerical protection

{fits}

The first fit paused after one real update, its committed checkpoint and optimizer state were independently verified, and it resumed without repeating that update. An FP16 nonfinite gradient occurred before update 2 on the first attempt; the optimizer was never called for that attempt. A minimal isolated runner correction bounds loss-scale reduction and RNG-preserving replay of the same accumulation group. Each completed fit recorded one numerical replay, reducing scale 1024 to 512, with finite-loss checks, all-head-gradient checks and gradient clipping retained. Nonfinite loss, persistent nonfinite gradients, CUDA allocation errors, scope mismatches and optimizer failures remain fatal. The replay used no extra optimizer updates or extra training configuration. The first fit's complete checkpoints were verified before the second started.

Active runtime totals {sum(v['successfully_checkpointed_active_seconds'] for v in runtime.values()):.2f} seconds. This includes loading, validation and checkpoint work in successful checkpointed execution; it excludes the first pre-update-2 failed attempt and debugging. The separately reported orchestration time includes smoke-pause and recovery delays, and neither timing is GPU kernel-only. Maximum allocated VRAM was {max(v['peak_allocated_bytes'] for v in runtime.values()):,} bytes, approximately 1.93 GiB. Independent final comparison adds inference time beyond the fit runtimes.

## Identical development protocol and results

{totals}

Same 18 photos, 41 ordinary native GT boxes, prompts, exact MID mapping, float32 inference, 960-square original preprocessing, class NMS IoU 0.5, AP score floor 0.001, maximum 300 detections, 101-point interpolated AP at IoU 0.50/0.75, and fixed operating confidence 0.15 / IoU 0.5. Class coverage masking and group/depiction ignore treatment are unchanged. These are the project's frozen native-box protocol, not the official Open Images hierarchy/group-of challenge metric. Original saved development predictions were checksum-verified and rescored identically. Both selected delta checkpoints were independently reloaded and reproduced saved predictions and metrics exactly. The two fits' pre-update first-group order, loss, components and gradients also matched exactly. No extra training reruns were used to claim reproducibility.

{cats}

Selected AP50 rises **7.41 percentage points**, AP75 rises **1.68 points**, and operating FP falls 101 → 67, but this masks important regressions. Cabinetry AP50 rises 8.37 → 52.59 and FP falls 78 → 21, while its TP stays 6/10 and the identity of matched objects changes. Table FP grows 8 → 27 and AP75 falls 47.92 → 36.80. Bed FP grows 4 → 9, AP50 falls 98.14 → 96.70 and AP75 falls 74.59 → 58.81. Desk recall rises 1/9 → 2/9, while AP75 slightly declines. Chair, sofa and lamp have only 2, 2 and 3 GT boxes respectively; high AP on these tiny supports does not demonstrate generalization. The lower LR has higher operating recall and fewer missed source boxes than the selected fit, but the pre-existing selection criterion was maximum development macro AP50, earliest epoch for ties, then prepared fit order for cross-fit ties. It was not changed after viewing results. The higher LR's last epoch regresses from its selected epoch, reinforcing overfitting concern.

## Localization and ambiguity diagnostics

{loc}

Paired IoU is computed only for source objects matched by both original and selected models at the fixed operating threshold. New/lost matches are reported separately, preventing survivor counts from concealing unmatched changes. Bed paired IoU declines about 0.096; Cabinetry loses three original matches and gains three others. Full per-fit localization and source-category confusion matrices are in `localization.csv`, `source-confusion.csv` and `comparison-v1/*-localization.json` / `*-confusion.json`. The auxiliary confusion matrix uses class-agnostic one-to-one IoU 0.5 matching and is explicitly separate from primary class-aware scoring; an unmatched prediction is not a fabricated confirmed-background label.

{diagnostics}

Duplicate FP is a measured repeat class-aware match under the existing scorer. Fragmentation, mixed/multi-object crops and category-mismatch candidates are geometry-based review heuristics, not independently confirmed mistakes. Mixed-object candidates increase 4 → 13, category-mismatch candidates 5 → 11, while fragmentation candidates decrease 9 → 7; these are not evidence of uniform improvement. There are 27 source-truncated/border diagnostic entries in each run. Inspect development-only galleries before interpreting visual behavior.

## Desk / Cabinetry sensitivity, separate from primary selection

Desk source boxes include booth/bench/side furniture that does not always match the ordinary desk concept. Cabinetry sometimes annotates an entire kitchen run, whereas predictions may represent doors or individual units. Original source labels and box units were preserved. An apparent false positive, false negative or class disagreement can reflect those semantics; no automatic relabeling or box repair was performed.

{sensitivity}

Excluding both ambiguous categories, selected AP50 is **96.47% versus original 96.76%**, AP75 is **63.83% versus 68.56%**, and precision is **33.33% versus 52.50%**, with identical recall 95.45%. The aggregate gain is therefore heavily dependent on Desk/Cabinetry source agreement and is not credible evidence of broad localization improvement. These post-hoc category aggregations reuse the exact same predictions, photos and matching. They do not adjudicate ambiguous boxes, alter the primary protocol or influence checkpoint selection.

## Checkpoint registry, preservation and verification

Selected head-delta checkpoint:

`{best['checkpoint']}`

SHA-256: `{best['sha256']}`

Original base SHA-256: `{OWL_SHA}`.

Load the original local `Owlv2ForObjectDetection` with its pinned processor, call isolated `adaptation.configure(model, 'heads', False)` to establish the exact 12 head keys, then load `torch.load(path, map_location='cpu', weights_only=True)['weights']` through isolated `adaptation.load_delta`; this is a detector-head delta/recovery checkpoint, not a standalone pretrained model directory. Use the matching `binding.json`, `scope.json` and registry. Do not substitute a different base. `compare.py` is the executed reference for safe loading; its existing output directory is immutable and cannot be reused for another run.

All 75 current relevant tests pass: 18 Phase 5 preparation, 14 metadata, 13 image acquisition, 18 Phase 5B and 12 Phase 5C. Initial new-test failure and corrected passing rerun logs are retained; it concerned an optional instrumentation field absent from the preserved first smoke receipt. Final integrity verification passed for all 172 committed checkpoint receipts, optimizer state steps, selected checkpoint hashes, original model/processor hashes, source/partition/config hashes and exact independent prediction replay. Historical verification checked 5,100 records, including 5,080 unique snapshot paths, 187 protected official holdout paths by stat only, all 79 Phase 5B artifacts and historical implementation hashes. Original acquisition metadata/photo hash ledgers also pass. See `final-verification.json` and `test-summary-final.json`. No original OWLv2/SigLIP checkpoint, ABO classifier, historical result, source annotation, dataset split, production application or Mac simulation was modified. No data/model downloads or package changes occurred.

## Artifacts, monitoring and recovery

`training-development-curves.csv`, `.png`, `.svg`: every epoch's loss and development metrics. `optimizer-updates.csv`: every real update, group membership, numerical replay and checkpoint SHA. `category-metrics.csv`, `sensitivity.csv`, `source-confusion.csv`, `localization.csv`: complete numerical comparisons. `selected-checkpoint-registry.json`: immutable experimental selection. `comparison-v1/<fit>/gallery.html`: all 18 development photos with GT and selected predictions. `comparison.html`: gallery entry points. Per-fit console logs, AMP warnings, execution-code fingerprints, RNG/optimizer/scaler states and receipts preserve recovery provenance.

Inspect `update-budget.json` and `<fit>/complete.json` to monitor. Both fits are now complete: **do not launch further optimization**. For an incomplete fit in this same authorized run only, the prepared recovery command was `.venv/bin/python -B tools/phase5c-v1/worker.py --fit-id <approved-fit-id> --resume`; the durable ledger refuses replay of uncertain optimizer calls and cannot exceed 70/fit or 140 globally. Completed workers skip without updating. Historical checkpoints and recovery artifacts were retained.

## Interpretation and recommended next phase

These two bounded fits support further investigation of head adaptation, particularly reduction of Cabinetry over-detection. They do not justify production replacement: evaluation uses only 18 model-selection photos, source boxes have semantic/unit ambiguities, multiple epochs/configs were selected on this same development set, category support is tiny and localization/crop ambiguity regressed in important classes. No confidence interval on this selected development estimate would remove its selection bias.

Next, define an annotation-scope policy for whole cabinets versus runs and actual desks versus source-native Desk before collecting a larger independent annotated indoor training/development set. Target beds, tables, partial/occluded furniture, mixed crops and repeated detections. Retain unknown-class masking, deterministic identity grouping and the original detector reference. Use a separately approved larger head-only experiment with an early-stopping rule fixed in advance; do not expand adaptation scope or tune confidence/NMS against these results without a new protocol. Leave the 18 reserved frozen photos untouched until a separately approved one-time independent evaluation of a frozen candidate. Do not score them now.
'''
 with (RUN_ROOT/'report.md').open('x') as f:f.write(report)
 gallery='<!doctype html><meta charset="utf-8"><title>Phase 5C development comparison</title><h1>Phase 5C development-only comparison</h1><p>Selection data; provisional source-annotation agreement. Frozen evaluation unused.</p><p><a href="report.md">Detailed report</a> · <a href="training-development-curves.png">Training/development curves</a></p>'
 for name in FIT_IDS:gallery+=f'<p><a href="comparison-v1/{html.escape(name)}/gallery.html">{html.escape(name)}: native GT and selected-checkpoint predictions</a></p>'
 gallery+='<p><a href="../phase5b-v1/original-development-v1/gallery.html">Preserved original detector development gallery</a></p>'
 with (RUN_ROOT/'comparison.html').open('x') as f:f.write(gallery)
 print('REPORT CREATED',RUN_ROOT/'report.md')

if __name__=='__main__':run()
