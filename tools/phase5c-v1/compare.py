"""Reproduce selected-checkpoint development predictions and frozen-protocol metrics."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import collections as C
import copy
import gc
import json
from common import *
from control import RUN_ROOT,FIT_IDS
from checkpoints import commit_json
from metrics import evaluate,iou,inter,area


def subset_summary(primary,excluded):
 cats={c:v for c,v in primary['categories'].items() if c not in excluded}
 out={key:sum(v[key] for v in cats.values()) for key in ('tp','fp','fn','gt_boxes','duplicate_fp')}
 out.update(macro_ap50=sum(v['ap50'] for v in cats.values() if v['ap50'] is not None)/sum(v['ap50'] is not None for v in cats.values()),
   macro_ap75=sum(v['ap75'] for v in cats.values() if v['ap75'] is not None)/sum(v['ap75'] is not None for v in cats.values()),
   precision=out['tp']/(out['tp']+out['fp']) if out['tp']+out['fp'] else None,recall=out['tp']/out['gt_boxes'] if out['gt_boxes'] else None)
 return out


def confusion(rows,predictions):
 matrix={g:{p:0 for p in (*CLASSES,'missed_annotation')} for g in (*CLASSES,'unmatched_prediction')}
 for row in rows:
  gs=[g for g in row['annotations'] if not g['ignore']];used=set()
  ps=[p for p in predictions[row['image_id']] if p['score']>=.15 and row['coverage'][p['label']]]
  for p in sorted(ps,key=lambda p:-p['score']):
   options=[(iou(p['box'],g['box']),i) for i,g in enumerate(gs) if i not in used and row['coverage'][g['label']]];v,index=max(options,default=(0,-1))
   if index>=0 and v>=.5:used.add(index);matrix[gs[index]['label']][p['label']]+=1
   elif any(g['ignore'] and g['label']==p['label'] and inter(p['box'],g['box'])/area(p['box'])>=.5 for g in row['annotations']):continue
   else:matrix['unmatched_prediction'][p['label']]+=1
  for i,g in enumerate(gs):
   if i not in used:matrix[g['label']]['missed_annotation']+=1
 return dict(matrix=matrix,scope='Auxiliary class-agnostic one-to-one IoU .5 source confusion at .15; not the primary class-aware matching. Unmatched predictions are not invented background labels.')


def localization(primary,reference,rows):
 def events(m,c):return {(e['image_id'],e['gt_index']):e['iou'] for e in m['diagnostics']['events'][c] if e['outcome']=='tp'}
 result={}
 for c in CLASSES:
  before=events(reference,c);after=events(primary,c);common=set(before)&set(after)
  result[c]=dict(common_matched_gt=len(common),paired_mean_iou_delta=sum(after[k]-before[k] for k in common)/len(common) if common else None,
    newly_matched_gt=len(set(after)-set(before)),lost_matches=len(set(before)-set(after)),source_relative_fn_delta=primary['categories'][c]['fn']-reference['categories'][c]['fn'])
 return result


def run():
 import torch
 from PIL import Image,ImageOps
 from transformers import Owlv2ForObjectDetection,Owlv2Processor
 from adaptation import configure,load_delta
 from baseline import predict,gallery
 root=RUN_ROOT/'comparison-v1'
 if root.exists():raise FileExistsError('Fresh comparison root required')
 rows=load_rows('development');baseline_dir=OUT/'original-development-v1'
 original={r['image_id']:read(baseline_dir/(r['image_id']+'.json')) for r in rows}
 for r in rows:
  p=baseline_dir/(r['image_id']+'.json')
  if sha(p)!=read(baseline_dir/(r['image_id']+'.checksum.json'))['sha256'] or original[r['image_id']]['image_sha256']!=r['sha256']:raise ValueError('Original saved predictions changed')
 base=evaluate(rows,{k:v['predictions'] for k,v in original.items()})
 if base!=read(baseline_dir/'metrics.json'):raise ValueError('Primary scoring definition changed')
 root.mkdir();metrics={'original':base};all_predictions={'original':{k:v['predictions'] for k,v in original.items()}};checks={};registry={}
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for independent adapted prediction replay')
 torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(20261105);torch.use_deterministic_algorithms(True)
 processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
 for name in FIT_IDS:
  done=read(RUN_ROOT/name/'complete.json');best=done['best']
  if sha(best['checkpoint'])!=best['sha256']:raise ValueError('Selected checkpoint checksum mismatch')
  state=torch.load(best['checkpoint'],map_location='cpu',weights_only=True)
  if state['step']!=best['step'] or state['binding']['dataset_sha256']!=sha(OUT/'dataset-lock.json'):raise ValueError('Selected data/update state differs')
  model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True).to('cuda');configure(model);load_delta(model,state['weights']);model.eval()
  out=root/name;out.mkdir();records={}
  for row in rows:
   with Image.open(image_path(row,'development')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
   raw,filtered=predict(model,processor,image);rec=dict(image_id=row['image_id'],image_sha256=row['sha256'],predictions=filtered,raw_candidates=raw,split='development')
   commit_json(out/(row['image_id']+'.json'),rec);records[row['image_id']]=rec
  current=evaluate(rows,{k:r['predictions'] for k,r in records.items()});expected=read(RUN_ROOT/name/f"validation-epoch-{best['epoch']:03d}.json")
  if current!=expected['metrics']:raise ValueError('Independent selected-checkpoint metrics do not exactly reproduce saved validation')
  exact=all(records[k]['predictions']==expected['predictions'][k]['predictions'] for k in records)
  if not exact:raise ValueError('Selected-checkpoint predictions fail exact reproducibility')
  commit_json(out/'metrics.json',current);gallery(rows,records,current,out)
  metrics[name]=current;all_predictions[name]={k:v['predictions'] for k,v in records.items()};registry[name]=dict(**best,fit_complete_sha256=sha(RUN_ROOT/name/'complete.json'))
  checks[name]=dict(predictions_bitwise_json_equal=True,metrics_exactly_reproduced=True,development_images=18,frozen_images=0)
  del model,state;gc.collect();torch.cuda.empty_cache();print('REPRODUCED',name,current['macro_ap50_supported'],flush=True)
 sensitivity={}
 for name,m in metrics.items():
  sensitivity[name]={label:subset_summary(m,excluded) for label,excluded in [('all_seven_primary',set()),('exclude_desk',{'desk'}),('exclude_cabinet',{'cabinet'}),('exclude_desk_and_cabinet',{'desk','cabinet'})]}
  commit_json(root/(name+'-confusion.json'),confusion(rows,all_predictions[name]))
  if name!='original':commit_json(root/(name+'-localization.json'),localization(m,base,rows))
 selected=max(FIT_IDS,key=lambda name:(registry[name]['score'],-FIT_IDS.index(name)))
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original checkpoint changed')
 commit_json(root/'comparison.json',dict(primary_metrics=metrics,sensitivity=sensitivity,primary_protocol_changed=False,
   sensitivity_scope='Post-hoc category aggregation only; same photos/predictions/matching. Exclusions do not adjudicate ambiguous boxes and do not influence selection.',
   independent_replay=checks,selected_experiment=selected,selected_checkpoint=registry[selected],experimental_registry=registry,
   original_prediction_cache_verified=True,dataset_sha256=sha(OUT/'dataset-lock.json'),optimizer_updates_during_comparison=0,frozen_photographs_used=0))
 print('SELECTED',selected,registry[selected],flush=True)

if __name__=='__main__':run()
