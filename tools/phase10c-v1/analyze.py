"""Read historical aggregates/weights; no model construction or image decoding."""
import argparse
from collections import Counter
import csv
import math
import os
import platform
import subprocess
import sys
from c10 import *
import metric_cpu as metrics

def best_at_recall(curve,recall):
    xs=[x for x in curve if x['recall']>=recall]
    return max(xs,key=lambda x:(x['precision'],x['recall'],x['threshold'])) if xs else None

def best_at_precision(curve,precision):
    xs=[x for x in curve if x['precision']>=precision]
    return max(xs,key=lambda x:(x['recall'],x['precision'],x['threshold'])) if xs else None

def quantiles(curve):
    # This reconstructs only the pooled SCORED TP/FP event distribution, not
    # per-image/category or unknown predictions. Ties keep their true weight.
    pairs=[];old=0
    for x in curve:
        current=x['tp']+x['fp'];pairs.append((x['threshold'],current-old));old=current
    out={}
    for q in (.1,.5,.9):
        target=max(1,math.ceil(old*q));cumulative=0
        for value,count in sorted(pairs):
            cumulative+=count
            if cumulative>=target:out[str(q)]=value;break
    return dict(scored_events=old,quantiles=out,scope='pooled scored TP/FP only; unknown/ignored proposals absent; no category identity')

def run():
    target=RAW/'analysis.json'
    if target.exists():raise FileExistsError('Analysis already exists; use a new version, never overwrite')
    old=read(RUN/'comparison-v1/results.json');entries={name:v['metrics'] for name,v in old['entries'].items()}
    for fit,epochs in old['epochs'].items():
        for e in epochs:
            if e['metrics'] is not None:entries[f'{fit}-epoch{e["epoch"]}']=e['metrics']
    comparisons={};base=entries['original'];target_recall=base['recall']
    for name,m in entries.items():
        comparisons[name]=dict(fixed={k:m[k] for k in ('precision','recall','ap50','ap75','tp','fp','fn')},categories=m['categories'],
            diagnostics=m['diagnostic_heuristics'],matched_original_recall=best_at_recall(m['pr_curve'],target_recall),
            best_precision_at_90_recall=best_at_recall(m['pr_curve'],.9),best_recall_at_80_precision=best_at_precision(m['pr_curve'],.8),
            simultaneous_80_precision_90_recall=any(x['precision']>=.8 and x['recall']>=.9 for x in m['pr_curve']),
            scored_event_confidence=quantiles(m['pr_curve']),
            duplicate_fp=sum(v['duplicate_fp'] for v in m['categories'].values()))
    counts={}
    for partition in ('optimization','inner_development'):
        rows=[r for r in manifest()['rows'] if r['internal_partition']==partition];native=[metrics.scopes(r) for r in rows]
        counts[partition]=dict(images=len(rows),groups=len({r['identity_group'] for r in rows}),sources=dict(Counter(r['source'] for r in rows)),
            classes={c:dict(positive_images=sum(any(a['label']==c and not a['ignore'] for a in r['annotations']) for r in rows),
                positive_boxes=sum(sum(a['label']==c and not a['ignore'] for a in r['annotations']) for r in rows),
                ignored_boxes=sum(sum(a['label']==c and a['ignore'] for a in r['annotations']) for r in rows),
                explicit_negative_images=sum(r['explicit_native_negative'].get(c,False) for r in rows),
                FP_observable_images=sum(r['fp_coverage'][c] for r in native)) for c in CLASSES})
    fitloss={}
    for fit in ('heads-original-lr1e-5','heads-original-lr3e-5'):
        updates=[read(p) for p in sorted((RUN/fit).glob('update-*.json'))];epochs={}
        for ep in range(1,6):
            us=[u for u in updates if u['epoch']==ep];samples=sum(u['samples'] for u in us)
            epochs[str(ep)]=dict(updates=len(us),sample_weighted_total_loss=sum(u['loss']*u['samples'] for u in us)/samples,
                mean_update_total_loss=sum(u['loss'] for u in us)/len(us),components_available=False)
        fitloss[fit]=epochs
    # No model instantiation/forward; weights-only CPU checkpoint inspection.
    import torch
    from safetensors import safe_open
    from box_only import BOX_KEYS
    original=ROOT/'models/owlv2-original-baseline-20261103/model.safetensors'
    baseline={}
    with safe_open(original,framework='pt',device='cpu') as f:
        for name in [f'{head}.{layer}.{suffix}' for head,layers in [('class_head',('dense0','logit_shift','logit_scale')),('box_head',('dense0','dense1','dense2'))] for layer in layers for suffix in ('weight','bias')]:baseline[name]=f.get_tensor(name)
    paths={}
    for name in ('heads-original-lr1e-5','heads-original-lr3e-5'):
        paths[name]=old['entries'][name]['checkpoint_path'];paths[name+'-final']=old['entries'][name]['final_recovery_checkpoint']
    paths['phase5c']=old['immutable_model_hashes_verified']['adapted']['path']
    deltas={}
    for label,path in paths.items():
        state=torch.load(ROOT/path,map_location='cpu',weights_only=True);deltas[label]={}
        for name,v in state['weights'].items():
            a=baseline[name];diff=v.float()-a.float();norm=float(torch.linalg.vector_norm(a.float()))
            deltas[label][name]=dict(l2_change=float(torch.linalg.vector_norm(diff)),relative_l2=float(torch.linalg.vector_norm(diff))/norm if norm else None,
                mean_change=float(diff.mean()),max_absolute_change=float(diff.abs().max()),frozen_in_box_only=name not in BOX_KEYS)
        del state
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only audit unexpectedly initialized CUDA')
    validation_files=sorted(RUN.glob('*/validation-*.json'))
    record=dict(scope='Provisional source-native internal validation; no independent accuracy claim',training_or_inference_run=False,
        saved_epoch5_predictions_found=False,epoch5_metrics_recovered=False,GPU_recovery_required=True,
        prediction_persistence='worker.validation collects transient dictionary then evaluate; checkpoints/validation JSONs contain no coordinates/logits/events',
        validation_records=len(validation_files),validation_hashes={str(p.relative_to(ROOT)):sha(p) for p in validation_files},
        completed_metric_epochs={k:sum(e['metrics'] is not None for e in v) for k,v in old['epochs'].items()},comparisons=comparisons,coverage=counts,loss=fitloss,
        head_weight_changes=deltas,per_category_confidence_distribution_available=False,raw_box_error_review_available=False,
        historical_component_loss_values_available=False,phase5c_prior_training_overlap_count=len(old['phase5c_prior_training_overlap_images']),
        threshold_changed=False,source_result_sha256=sha(RUN/'comparison-v1/results.json'),code_hashes=code_hashes(),
        runtime=dict(python=sys.version,platform=platform.platform(),gil_enabled=sys._is_gil_enabled(),jit_available=sys._jit.is_available(),jit_enabled=sys._jit.is_enabled(),
            torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),trace_active=sys.gettrace() is not None,profile_active=sys.getprofile() is not None),
        confirmed_root_cause=False)
    save(target,record)
    # Sanitized aggregate summary only; never put identities/photos/predictions in Git.
    summary={k:record[k] for k in ('scope','training_or_inference_run','saved_epoch5_predictions_found','epoch5_metrics_recovered','validation_records','completed_metric_epochs','coverage',
        'phase5c_prior_training_overlap_count','historical_component_loss_values_available','per_category_confidence_distribution_available','confirmed_root_cause','runtime','loss')}
    summary['comparisons']=comparisons;save(REPORT/'analysis-summary.json',summary)
    with (REPORT/'comparison.csv').open('x',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['model','precision','recall','AP50','AP75','TP','FP','FN','duplicate_FP','matched_recall_precision','matched_recall_threshold','P_at_R90','R_at_P80'])
        for name,r in comparisons.items():
            m=r['fixed'];a=r['matched_original_recall'];b=r['best_precision_at_90_recall'];c=r['best_recall_at_80_precision']
            writer.writerow([name,*[m[k] for k in ('precision','recall','ap50','ap75','tp','fp','fn')],r['duplicate_fp'],a['precision'] if a else None,a['threshold'] if a else None,b['precision'] if b else None,c['recall'] if c else None])
    print(json.dumps(dict(results=str(target.relative_to(ROOT)),models_run=0,checkpoints_CPU_inspected=len(paths),CUDA_initialized=False)))

if __name__=='__main__':run()
