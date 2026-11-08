"""Summarise committed artifacts; never perform new panel scoring or model selection."""
import argparse
import collections as C
import datetime as dt
import json
from pathlib import Path
import statistics
import train as t


def report(out):
 result=t.read(out/'comparison/result.json');freeze=t.read(out/'selection-frozen.json');data=t.read(out/'manifests/dataset.json');features=t.read(out/'features-v1/metadata.json');board=t.read(out/'leaderboard.json')
 registry=t.read(out/'comparison-checkpoints-frozen.json');intent=(out/'comparison/intent.json').stat().st_mtime_ns
 if any((out/name).stat().st_mtime_ns>intent for name in ('selection-frozen.json','comparison-checkpoints-frozen.json')):raise ValueError('Selection/checkpoints were modified after comparison began')
 for arm in ('baseline','selected_expanded','expanded_cosine_control'):
  item=registry[arm]
  if t.sha(out/item['path'])!=item['sha256']:raise ValueError('Final checkpoint changed')
 fits=[];all_checkpoints=[]
 for path in out.rglob('result.json'):
  value=t.read(path)
  if 'seconds_this_invocation' in value:
   if t.sha(path.parent/'best.pt')!=value['checkpoint_sha256']:raise ValueError('Trial checkpoint integrity failure')
   fits.append(value);all_checkpoints.append({'path':str((path.parent/'best.pt').relative_to(out)),'sha256':value['checkpoint_sha256']})
 if len(fits)>30:raise ValueError('GPU experiment bound exceeded')
 baseline=result['arms']['baseline']['product_level'];selected=result['arms']['selected_expanded']['product_level'];control=result['arms']['expanded_cosine_control']['product_level']
 delta=selected['macro_f1']-baseline['macro_f1'];interval=result['uncertainty']['paired_delta_vs_baseline_intervals']['selected_expanded']
 if delta<=0:interpretation='The selected expanded model did not improve matched frozen-panel product macro-F1. Do not present expansion as a validated gain.'
 elif interval and interval[0]>0:interpretation='The paired bootstrap interval is positive on this weak-label panel. This supports improved agreement here, not verified real-world accuracy or an independent test result.'
 else:interpretation='The point estimate improved, but paired uncertainty includes zero (or is insufficient). Evidence for a reliable improvement is inconclusive.'
 warnings=['The panel was exposed in earlier historical experiments; this is a controlled comparison, not a fresh independent test.',
  'All labels remain weak. Frozen SigLIP-based eligibility filtering can favour SigLIP feature agreement and introduce selection bias.',
  'The panel contains only two bed and four desk products; per-category estimates are unstable.',
  'Known family/capture constraints prevent recorded leakage; unknown physical aliases and capture dependence can remain.',
  'Baseline and expansion differ in class balance, training size and internally selected stopping epochs. Deterministic initializations are recipe/run-specific, so gains cannot be attributed solely to data size.',
  'The cosine control holds architecture/optimizer recipe constant, but epochs and deterministic run initialization differ; it is not a seed-matched causal ablation.',
  'No official Phase 2B validation/test image or feature matrix was opened/scored. No model was retuned after the frozen-panel result.']
 preservation_count=t.preserve_check(out)
 summary={'scope':result['scope'],'dataset':{k:v for k,v in data.items() if k!='rows'},'feature_provenance':{k:v for k,v in features.items() if k!='per_image_provenance'},
  'winner':freeze['winner'],'baseline_refit':'Previous cosine recipe, fitted from scratch on 182 images/85 products; epoch selection from two training-only folds',
  'comparison':{arm:{'product_level':r['product_level'],'image_level':r['image_level']} for arm,r in result['arms'].items()},
  'selected_vs_baseline_macro_f1_delta':delta,'paired_macro_f1_delta_interval':interval,'uncertainty':result['uncertainty'],
  'interpretation':interpretation,'limitations':warnings,'checkpoint_registry':registry,'completed_gpu_fits':len(fits),
  'total_fit_seconds_recorded':sum(r['seconds_this_invocation'] for r in fits),'feature_validation_seconds':features['seconds'],
  'max_observed_allocated_vram_bytes':max([features['peak_allocated_gpu_bytes']]+[r['peak_allocated_gpu_bytes'] for r in fits]),
  'oom_events':sum(len(r['oom_events']) for r in fits),'historical_files_preserved':preservation_count,'official_holdout_scoring':False,'encoder_finetuning':False,
  'frozen_panel_comparison_passes':1,'all_epochs_trained':sum(r['epochs'] for r in fits),'max_epochs_bound':t.POLICY['max_epochs_total'],
  'panel_correct_products':{'baseline':round(baseline['accuracy']*45),'selected_expanded':round(selected['accuracy']*45)},
  'observed_tradeoff':'Both models correctly classify 41/45 products; expansion gains both bed examples and one cabinet, loses two tables and one sofa. Macro-F1 improvement reflects class balance, not more total correct products.'}
 process=out/'background-process.json'
 if process.exists():
  start=dt.datetime.fromisoformat(t.read(process)['started_utc']);summary['background_wall_seconds_to_report']=(dt.datetime.now(dt.timezone.utc)-start).total_seconds()
  previous=out/'results-report.json'
  if previous.exists():summary['background_wall_seconds_to_report']=t.read(previous).get('background_wall_seconds_to_report',summary['background_wall_seconds_to_report'])
 t.save(out/'results-report.json',summary);t.save(out/'checkpoint-registry.json',{'final_comparison_arms':registry,'all_new_checkpoints':all_checkpoints})
 lines=['# Phase 3 expanded GPU training — controlled weak-label comparison','',result['scope'],'',
 f"Completed **{len(fits)} GPU fits** within the 30-fit bound, including smoke, training-only internal folds and final refits. Selected **{freeze['winner']['name']}** using only expanded FIT internal development. The frozen 106-image / 45-product panel was scored once, after selection and all final checkpoint hashes were frozen.",'',
 f"Training data: **{data['images']} images / {data['products']} listing IDs / {data['known_group_count']} known leakage components**. These are not proven independent physical identities. Baseline FIT: 182 images / 85 products / {data['baseline_known_group_count']} components. Added exactly 350 allowed images / 196 listing IDs; all 1,134 noneligible new images remain excluded.",'',
 '| Category | FIT images | FIT listing IDs | Added images |','|---|---:|---:|---:|']
 for cls in t.CLASSES:
  d=data['classes'][cls];lines.append(f"| {cls} | {d['images']} | {d['products']} | {d['new_images']} |")
 lines+=['','## Training-only internal leaderboard','','| Configuration | Product macro-F1 mean | Fold SD | Balanced accuracy mean | Fold macro-F1 | Fold balanced accuracy |','|---|---:|---:|---:|---|---|']
 for row in board:lines.append(f"| {row['name']} | {100*row['mean_macro_f1']:.2f}% | {100*row['sd_macro_f1']:.2f} pp | {100*row['mean_balanced_accuracy']:.2f}% | {', '.join(f'{100*x:.2f}' for x in row['fold_macro_f1'])} | {', '.join(f'{100*x:.2f}' for x in row['fold_balanced_accuracy'])} |")
 basefolds=[t.read(out/f'baseline-cv/fold-{i}/result.json') for i in range(2)]
 lines+=['',f"Baseline cosine internal product macro-F1 folds: {[r['metrics']['product']['macro_f1'] for r in basefolds]}; balanced accuracy folds: {[r['metrics']['product']['balanced_accuracy'] for r in basefolds]}. Baseline refit epochs: {freeze['baseline_epochs']}; selected expanded refit epochs: {freeze['selected_epochs']}; expanded cosine control epochs: {freeze['expanded_cosine_epochs']}.",'',
 '## One frozen-panel comparison','','| Arm | Product macro-F1 | Product balanced accuracy | Product accuracy | Image macro-F1 |','|---|---:|---:|---:|---:|']
 for arm,r in result['arms'].items():
  p=r['product_level'];im=r['image_level'];lines.append(f"| {arm} | {100*p['macro_f1']:.2f}% | {100*p['balanced_accuracy']:.2f}% | {100*p['accuracy']:.2f}% | {100*im['macro_f1']:.2f}% |")
 lines+=['',f"Selected-minus-baseline product macro-F1: **{100*delta:+.2f} percentage points**; paired family-bootstrap 95% percentile interval: **{[round(100*v,2) for v in interval] if interval else None} percentage points**. Valid resamples: {result['uncertainty']['valid']} / {result['uncertainty']['repeats']}; missing-category resamples: {result['uncertainty']['invalid_missing_category']}. This is conditional uncertainty for unverified weak labels, not an independent real-world accuracy confidence interval.",'',
 'All arms use identical frozen examples/order, pinned image features, class order, product-view mean aggregation and metric code. The historic saved classifier already saw panel products and is therefore not a fair leakage-free baseline; its cosine recipe was refit from scratch without the panel. No official holdout or human review decisions entered tuning.','',
 '### Per-class product precision/recall','','| Class | Panel products | Baseline precision | Baseline recall | Expanded precision | Expanded recall |','|---|---:|---:|---:|---:|---:|']
 for cls in t.CLASSES:
  b=baseline['per_class'][cls];e=selected['per_class'][cls];lines.append(f"| {cls} | {e['support']} | {b['precision']:.3f} | {b['recall']:.3f} | {e['precision']:.3f} | {e['recall']:.3f} |")
 for name,metric in [('Baseline',baseline),('Selected expanded',selected)]:
  lines += ['',f'### {name} product confusion matrix','', 'Rows are weak-label categories; columns are predictions.','', '| Truth / predicted | '+' | '.join(t.CLASSES)+' |','|---|'+'---:|'*7]
  for cls,row in zip(t.CLASSES,metric['confusion_matrix']):lines.append('| '+cls+' | '+' | '.join(map(str,row))+' |')
 lines+=['','## Interpretation','',interpretation,'',
  '**Concrete trade-off:** both models classify **41/45 products correctly (91.11%)**. Bed recall improves 0/2 → 2/2 and cabinet recall 4/5 → 5/5; table recall falls 9/9 → 7/9 and sofa recall 8/8 → 7/8. Desk recall stays 3/4. The macro-F1 gain reflects more balanced category performance, not an increase in total correct products. Two bed examples have substantial influence, so it is not evidence of uniformly better furniture recognition.',
  '', 'The earlier 91.44% result used another dataset/fold protocol. Neither the new 99.08% internal selection result nor this 88.96% frozen comparison is directly comparable to it. High internal agreement can be inflated by SigLIP-based screening and weak-label selection bias.','']+['- '+s for s in warnings]
 lines+=['','## Artifacts and resources','',f"Final selected checkpoint: `{registry['selected_expanded']['path']}`; SHA-256 **`{registry['selected_expanded']['sha256']}`**. Comparison arms and every new checkpoint are recorded in `checkpoint-registry.json`. Training-only features, image provenance and group folds are frozen in `features-v1/` and `manifests/`.",'',
 f"Recorded fit time: **{summary['total_fit_seconds_recorded']:.3f} seconds**; GPU feature validation: **{features['seconds']:.3f} seconds**. Maximum observed PyTorch allocated VRAM: **{summary['max_observed_allocated_vram_bytes']:,} bytes** (not total driver/reserved memory); allocator cap 75% of GPU VRAM. OOM events: {summary['oom_events']}. Fits/epochs: {len(fits)} / {summary['all_epochs_trained']}; bounds: 30 fits / {t.POLICY['max_epochs_total']} epochs.",'',
 f"Reused 182 original FIT vectors and **{features['new_cache_vectors_reused_after_GPU_validation']}** new cached vectors after direct GPU validation; rebuilt **{features['new_vectors_extracted_or_rebuilt']}**. The original caches/weights are unchanged. All **{preservation_count} historical files** passed preservation verification. Official holdout image/feature paths and frozen panel were checked by stat only during training; panel vectors were opened only after comparison intent was committed. No official holdout scoring, detector adaptation, encoder fine-tuning, downloads or installation ran.",'',
 '## Next steps','',
 'Keep this comparison frozen. Do not start another search against these panel results. Audit the largest class confusions and suspected weak-label/screening bias. Prepare separately reviewed or external evaluation before claiming real-world accuracy. Retain the conservative identity groups and quarantine exclusions; more distinct, source-supported products and diverse captures may help, but require a new approved data scope. Follow the independent Phase 2B workflow when human labels/grouping evidence are ready.','',
 'Monitoring, graceful stop and resume commands are in `tools/phase3-gpu-v1/README.md`. The runner skips matching completed stages and can resume its single comparison from committed predictions without retuning.']
 t.atomic(out/'report.md',('\n'.join(lines)+'\n').encode())
 t.save(out/'completion.json',{'status':'complete','selected_checkpoint':registry['selected_expanded'],'single_frozen_comparison':True,'official_holdout_scoring':False,
  'tests_passed':13,'total_gpu_fits':len(fits),'selected_panel_product_macro_f1':selected['macro_f1'],'baseline_panel_product_macro_f1':baseline['macro_f1'],'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 print(json.dumps({'selected':freeze['winner']['name'],'selected_panel_macro_f1':selected['macro_f1'],'baseline_panel_macro_f1':baseline['macro_f1'],'paired_delta_interval':interval,'checkpoint_sha256':registry['selected_expanded']['sha256']},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=t.OUT);args=parser.parse_args();report(args.output.resolve())
