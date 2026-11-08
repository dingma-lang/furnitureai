"""Report only the new provisional train-derived experiment; never open official holdout arrays."""
import argparse
import collections as C
import json
from pathlib import Path
import numpy as np
import torch
import pipeline as p


def report(out):
 array,rows,folds=p.load_data(out)
 completion=p.read(out/'completion.json');metadata=p.read(out/'features-v1/metadata.json');cfg=completion['winner']
 confusion=np.zeros((7,7),dtype=np.int64);oof=[];fold_reports=[]
 for fold,indices in enumerate(folds):
  path=out/'trials'/cfg['name']/f'fold-{fold}'/'best.pt'
  result=p.read(path.parent/'result.json')
  if p.sha(path)!=result['checkpoint_sha256']:raise ValueError('Winning fold checkpoint changed')
  saved=torch.load(path,map_location='cpu',weights_only=True)
  if cfg['kind']=='centroid':logits=array[indices]@saved['centers'].numpy().T
  else:
   model=p.network(cfg,array.shape[1]);model.load_state_dict(saved['state_dict']);model.eval()
   with torch.inference_mode():logits=model(torch.from_numpy(array[indices])).numpy()
  # CPU forward of lightweight heads on train-derived development only, no encoder or official holdout.
  byproduct=C.defaultdict(list)
  for pos,i in enumerate(indices):byproduct[rows[i]['abo_product_id']].append(pos)
  for product,positions in sorted(byproduct.items()):
   source=rows[indices[positions[0]]];truth=p.CLASSES.index(source['weak_label']);prediction=int(logits[positions].mean(axis=0).argmax());confusion[truth,prediction]+=1
   oof.append({'abo_product_id':product,'fold':fold,'weak_label':p.CLASSES[truth],'internal_development_prediction':p.CLASSES[prediction],'human_verified':False})
  fold_reports.append({'fold':fold,'images':len(indices),'products':len(byproduct),'class_products':dict(C.Counter(rows[indices[pos[0]]]['weak_label'] for pos in byproduct.values())),'selected_epoch':result['best_epoch'],'product_macro_f1':result['metrics']['product']['macro_f1']})
 perclass={}
 for i,cls in enumerate(p.CLASSES):
  tp=int(confusion[i,i]);support=int(confusion[i].sum());predicted=int(confusion[:,i].sum());recall=tp/support if support else None;precision=tp/predicted if predicted else 0
  perclass[cls]={'products':support,'images':sum(r['weak_label']==cls for r in rows),'provisional_precision':precision,'provisional_recall':recall,'human_verified':False}
 prior=p.read(p.OLD/'run-meta.json');prior_best=max(float(r['dev_macro_f1']) for r in __import__('csv').DictReader((p.OLD/'leaderboard.csv').open()))
 summary={'scope':p.SCOPE,'selected_recipe':cfg,'selection_mean_product_macro_f1':completion['selection']['product_macro_f1'],
  'fold_standard_deviation':completion['selection']['fold_std'],'folds':fold_reports,'class_distribution_and_internal_only_metrics':perclass,'oof_product_confusion_matrix':confusion.tolist(),
  'label_source':'ABO metadata weak labels; fixed frozen-SigLIP text agreement/negative-prompt filtering; no human review used',
  'confidence_note':'Cosine scores/margins are uncalibrated screening signals, not probabilities of correct labels',
  'images':len(rows),'products':len(oof),'leakage_groups':len({r['leakage_group_id'] for r in rows}),
  'new_views_retained':sum(not r['original_cached_training_view'] for r in rows),'new_images_embedded':metadata['new_image_features'],
  'original_train_vectors_retained':metadata['cached_feature_vectors_reused'],
  'original_batch_best_internal_f1':prior_best,'original_comparison_note':'Different folds, screening and dataset; not a controlled improvement over 83.43%. Prior adaptive development choices also inform this queue.',
  'final_checkpoint':'final-selected/best.pt','final_checkpoint_has_independent_metric':False,
  'warm_start_companion':'final-warm-start/best.pt','warm_start_has_dev_score':False,
  'model_fingerprint':metadata['model_processor_fingerprint'],'frozen_backbone_license':'Apache-2.0, local pinned checkpoint','dataset_license':'ABO CC BY 4.0; retain source attribution',
  'limitations':['Internal development labels are unverified and noisy.','Early stopping and model selection reuse development folds; metrics are optimistic.','Four bed products and eleven desks give limited category support.','Only known identity/model/family/capture links are grouped; unknown physical aliases remain possible.','View expansion adds no new product identities.','Whole-object suitability and capture independence remain unverified.','The official validation/test partitions and review decisions are excluded from selection.','Final refit and warm-start checkpoints have not been independently evaluated.'],
  'encoder_adaptation':'Deferred: first expand distinct products and assess simple heads; no encoder parameters trained',
  'source_licence_sha256':p.sha(p.META/'LICENSE-CC-BY-4.0.txt'),'out_of_fold_predictions':oof}
 p.save(out/'provenance-report.json',summary)
 leaderboard=p.read(out/'leaderboard.json');lines=['# Provisional weak-label furniture development experiment','',p.SCOPE,'',
 f"Completed eight configurations, each with three training-derived group folds. Selected **{cfg['name']}**, mean product macro-F1 **{100*summary['selection_mean_product_macro_f1']:.2f}%**, fold standard deviation **{100*summary['fold_standard_deviation']:.2f} percentage points**. These are model-selection metrics, not human-verified accuracy.",'',
 f"Retained {len(rows)} images / {len(oof)} product IDs / {summary['leakage_groups']} conservative identity groups. Reused {summary['original_train_vectors_retained']} original vectors; retained {summary['new_views_retained']} additional local views after computing {summary['new_images_embedded']} new image embeddings. No images or model weights downloaded.",'',
 '| Category | Images | Products |','|---|---:|---:|']
 for cls,d in perclass.items():lines.append(f"| {cls} | {d['images']} | {d['products']} |")
 lines+=['','| Configuration | Mean internal product macro-F1 | Fold SD |','|---|---:|---:|']
 for r in leaderboard:lines.append(f"| {r['experiment']} | {100*r['product_macro_f1']:.2f}% | {100*r['fold_std']:.2f} pp |")
 lines+=['','Labels come from official ABO product/category/title metadata. Automated image checks and frozen SigLIP positive/negative prompt cosine screening provide uncalibrated agreement/confidence signals. Extra pending views require agreement; strong contradictions, prior hard exclusions, shared generic assets, mixed scenes, textures and known redundant captures are excluded. Original review status/decisions are never upgraded or changed.','',
 'Known product, group, family, model and capture relationships are reserved/grouped before folds. Whole-product identities remain separate records; generic shared assets are excluded rather than used to merge unrelated products. Class balancing and centroids are fitted from each internal training fold only. Full-training previous classifiers are never used to initialise internal-development folds.','',
 'Official validation/test vectors and human review decisions were not used for training, screening calibration, early stopping or model selection. Their bytes can be hashed for preservation, and reserved identifiers are used only for leakage prevention. The verified Phase 2B evaluator was not edited or invoked.','',
 'The previous 83.43% result uses a different development protocol. The new result is not a controlled improvement claim. Final refit uses the selected recipe for the median of three fold-selected epoch counts (7 epochs in this run; actual value in completion.json); the separate prior-checkpoint warm-start companion has no assigned CV metric.','',
 'Final selected checkpoint: `final-selected/best.pt`; prior classifier continuation: `final-warm-start/best.pt`; registry: `checkpoint-registry.json`; immutable manifests: `manifests-v1/`; local expanded embeddings: `features-v1/`. Per-epoch model/optimizer restart state is retained in each `progress.pt`.','',
 '## Limits and next data stage','']
 lines += ['- '+x for x in summary['limitations']]
 expansion=p.read(out/'manifests-v1/metadata-only-expansion.json')
 lines += ['',f"Metadata-only expansion queue: {len(expansion['products'])} screened, nonlocal, nonreserved candidates; class counts {expansion['by_class']}. These are NOT available training photographs. No acquisition was performed. More unique products, particularly beds/desks, requires a separately approved bounded image acquisition with exact official object URLs, CC BY attribution, remote size checks and transfer/storage caps. Metadata alone does not establish independent images or physical identities.",'',
 'Encoder adaptation is deferred until distinct-product expansion and lightweight head comparisons justify it. Any later adaptation should use a fresh versioned experiment, small batches, frozen early blocks and an explicit VRAM/runtime budget. This queue modifies only small heads.','',
 'Training resources: RTX 4070 Laptop GPU; 75% per-process CUDA allocator cap; one new image feature at a time; head batches of 32; eight configurations × three folds, maximum 80 epochs/fold, patience 12, 30-minute limit per invocation, 1 GiB new experiment storage cap. Graceful stops preserve epoch checkpoints and resumable feature shards.','',
 'Run/monitor/resume commands are in `tools/weak-training-v1/README.md`. Source labels, thresholds, prompts, code and model fingerprints are frozen. An incompatible resume fails instead of replacing results.']
 p.atomic(out/'report.md',('\n'.join(lines)+'\n').encode());print(json.dumps({k:summary[k] for k in ['scope','images','products','selection_mean_product_macro_f1','fold_standard_deviation']},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=p.OUTPUT);args=parser.parse_args();report(args.output.resolve())
