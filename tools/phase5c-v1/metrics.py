"""Partial-label native-class pilot metrics. Not official Open Images mAP."""
import collections as C
import math
import numpy as np
from common import CLASSES,SETTINGS

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def iou(a,b):
 v=inter(a,b);return v/(area(a)+area(b)-v) if area(a)+area(b)-v else 0.
def check_box(b):
 if len(b)!=4 or not all(math.isfinite(float(x)) and 0<=x<=1 for x in b) or area(b)<=0:raise ValueError('Invalid normalized xyxy')

def nms(predictions,threshold=.5,max_detections=300):
 kept=[]
 for p in sorted(predictions,key=lambda x:-x['score']):
  check_box(p['box'])
  if p['label'] not in CLASSES or not math.isfinite(p['score']) or not 0<=p['score']<=1:raise ValueError('Invalid prediction')
  if not any(q['label']==p['label'] and iou(q['box'],p['box'])>threshold for q in kept):kept.append(p)
 return kept[:max_detections]

def match(rows,predictions,c,threshold,score_floor):
 gt={r['image_id']:[g for g in r['annotations'] if g['label']==c and not g['ignore']] for r in rows if r['coverage'][c]}
 ignored={r['image_id']:[g for g in r['annotations'] if g['label']==c and g['ignore']] for r in rows if r['coverage'][c]}
 used={key:set() for key in gt};ranked=[];unknown=0
 for r in rows:
  for j,p in enumerate(predictions[r['image_id']]):
   if p['label']!=c or p['score']<score_floor:continue
   if r['image_id'] not in gt:unknown+=1;continue
   ranked.append((p['score'],r['image_id'],j,p))
 ranked.sort(key=lambda t:(-t[0],t[1],t[2]));events=[]
 for score,ident,j,p in ranked:
  choices=[(iou(p['box'],g['box']),k) for k,g in enumerate(gt[ident]) if k not in used[ident]]
  best,k=max(choices,default=(0.,-1))
  if k>=0 and best>=threshold:
   used[ident].add(k);events.append(dict(image_id=ident,prediction_index=j,outcome='tp',iou=best,gt_index=k,score=score));continue
  if any(inter(p['box'],g['box'])/area(p['box'])>=.5 for g in ignored[ident]):
   events.append(dict(image_id=ident,prediction_index=j,outcome='ignored_region',score=score));continue
  duplicate=any(iou(p['box'],g['box'])>=threshold for k,g in enumerate(gt[ident]) if k in used[ident])
  events.append(dict(image_id=ident,prediction_index=j,outcome='fp',duplicate=duplicate,score=score))
 missed=[dict(image_id=ident,category=c,gt_index=k,box=g['box'],source_mid=g['source_mid'],truncated=g['flags'].get('IsTruncated')==1,occluded=g['flags'].get('IsOccluded')==1) for ident,gs in gt.items() for k,g in enumerate(gs) if k not in used[ident]]
 return events,missed,sum(map(len,gt.values())),unknown

def ap101(events,num_gt):
 if num_gt==0:return None
 e=[x for x in events if x['outcome'] in ('tp','fp')]
 if not e:return 0.
 tp=np.cumsum([x['outcome']=='tp' for x in e]);fp=np.cumsum([x['outcome']=='fp' for x in e]);rec=tp/num_gt;prec=tp/(tp+fp)
 return float(np.mean([np.max(prec[rec>=t],initial=0.) for t in np.linspace(0,1,101)]))

def evaluate(rows,predictions):
 if not rows or any(r['split']!='development' for r in rows):raise ValueError('Metrics restricted to development')
 if set(predictions)!={r['image_id'] for r in rows}:raise ValueError('Incomplete/extra predictions')
 result=dict(scope='Published exact-native annotation agreement on eligible development photos; flat partial-label pilot, not official Open Images mAP or verified real-world accuracy',settings=SETTINGS,images=len(rows),categories={},diagnostics={})
 all_events={};missed=[]
 for c in CLASSES:
  ev50,_,n,unk=match(rows,predictions,c,.5,SETTINGS['ap_score_floor']);ev75,_,_,_=match(rows,predictions,c,.75,SETTINGS['ap_score_floor'])
  op,miss,_,ignored=match(rows,predictions,c,.5,SETTINGS['operating_score']);missed+=miss
  tp=sum(e['outcome']=='tp' for e in op);fp=sum(e['outcome']=='fp' for e in op);fn=len(miss)
  vals=[e['iou'] for e in op if e['outcome']=='tp']
  result['categories'][c]=dict(gt_boxes=n,covered_images=sum(r['coverage'][c] for r in rows),unknown_class_images=sum(not r['coverage'][c] for r in rows),
    ap50=ap101(ev50,n),ap75=ap101(ev75,n),tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/n if n else None,
    mean_matched_iou=sum(vals)/len(vals) if vals else None,duplicate_fp=sum(e.get('duplicate',False) for e in op),ignored_unknown_predictions=ignored,ignored_region_predictions=sum(e['outcome']=='ignored_region' for e in op))
  all_events[c]=op
 for suffix in ('50','75'):
  vals=[x['ap'+suffix] for x in result['categories'].values() if x['ap'+suffix] is not None];result['macro_ap'+suffix+'_supported']=sum(vals)/len(vals) if vals else None
 total={k:sum(v[k] for v in result['categories'].values()) for k in ('tp','fp','fn','gt_boxes','duplicate_fp')}
 total['precision']=total['tp']/(total['tp']+total['fp']) if total['tp']+total['fp'] else None;total['recall']=total['tp']/total['gt_boxes'] if total['gt_boxes'] else None
 result['micro_operating']=total
 incorrect=[];fragments=[];mixed=[];truncated=[]
 for r in rows:
  ident=r['image_id'];gs=[g for g in r['annotations'] if not g['ignore']];ps=[(j,p) for j,p in enumerate(predictions[ident]) if p['score']>=SETTINGS['operating_score']]
  for g in gs:
   if g['flags']['IsTruncated']==1 or min(g['box'])<=.002 or max(g['box'])>=.998:truncated.append(dict(image_id=ident,category=g['label'],box=g['box'],source_truncated=g['flags']['IsTruncated']==1))
   parts=[j for j,p in ps if p['label']==g['label'] and iou(p['box'],g['box'])<.5 and inter(p['box'],g['box'])/area(p['box'])>=.8 and area(p['box'])<.6*area(g['box'])]
   if len(parts)>=2:fragments.append(dict(image_id=ident,category=g['label'],box=g['box'],prediction_indices=parts,heuristic=True))
  for j,p in ps:
   if not r['coverage'][p['label']]:continue
   wrong=[g for g in gs if r['coverage'][g['label']] and g['label']!=p['label'] and iou(p['box'],g['box'])>=.5]
   if wrong and not any(g['label']==p['label'] and iou(p['box'],g['box'])>=.5 for g in gs):
    g=max(wrong,key=lambda g:iou(p['box'],g['box']));incorrect.append(dict(image_id=ident,prediction_index=j,predicted=p['label'],annotated=g['label'],iou=iou(p['box'],g['box']),interpretation='category mismatch against published boxes, not independently confirmed model error'))
   contains=[k for k,g in enumerate(gs) if inter(p['box'],g['box'])/area(g['box'])>=.7]
   if len(contains)>=2:mixed.append(dict(image_id=ident,prediction_index=j,annotated_objects=contains,heuristic=True))
 result['diagnostics']=dict(events=all_events,missed_annotated_objects=missed,category_mismatch_candidates=incorrect,fragmentation_candidates=fragments,multi_object_box_candidates=mixed,border_or_source_truncation=truncated)
 return result
