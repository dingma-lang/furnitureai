"""Partial-native annotation metrics, exhaustive-only FP accounting and PR curves."""
import numpy as np
from core import CLASSES,SETTINGS
from fusion import iou,inter,area,valid

def events(rows,preds,category,threshold=.5,floor=.001):
 gs={r['image_id']:[g for g in r['annotations'] if g['label']==category and not g['ignore']] for r in rows if r['coverage'][category]}
 ignore={r['image_id']:[g for g in r['annotations'] if g['label']==category and g['ignore']] for r in rows};byid={r['image_id']:r for r in rows};used={k:set() for k in gs};ranked=[];result=[];unknown=0
 for r in rows:
  for j,p in enumerate(preds[r['image_id']]):
   valid(p)
   if p['label']==category and p['score']>=floor:ranked.append((p['score'],r['image_id'],j,p))
 for score,ident,j,p in sorted(ranked,key=lambda v:(-v[0],v[1],v[2])):
  record=dict(image_id=ident,index=j,label=category,score=score)
  if ident not in gs:unknown+=1;continue
  choices=[(iou(p['box'],g['box']),k) for k,g in enumerate(gs[ident]) if k not in used[ident]];best,k=max(choices,default=(0.,-1))
  if k>=0 and best>=threshold:used[ident].add(k);result.append(dict(record,outcome='tp',iou=best,gt_index=k));continue
  if any(inter(p['box'],g['box'])/area(p['box'])>=.5 for g in ignore[ident]):continue
  if not byid[ident]['fp_coverage'][category]:unknown+=1;continue
  result.append(dict(record,outcome='fp',duplicate=any(iou(p['box'],g['box'])>=threshold for k,g in enumerate(gs[ident]) if k in used[ident])))
 return result,sum(len(v) for v in gs.values()),unknown

def curve(ev,n,group_ties=True):
 out=[];tp=fp=0
 ranked=sorted(ev,key=lambda e:(-e['score'],e['image_id'],e['index'],e['label']))
 for i,e in enumerate(ranked):
  tp+=e['outcome']=='tp';fp+=e['outcome']=='fp'
  if group_ties and i+1<len(ranked) and ranked[i+1]['score']==e['score']:continue
  out.append(dict(threshold=e['score'],precision=tp/(tp+fp),recall=tp/n if n else None,tp=tp,fp=fp))
 return out

def ap(ev,n):
 if not n:return None
 c=curve(ev,n,group_ties=False) # Preserve the existing101-point ranked AP convention.
 return float(np.mean([max((x['precision'] for x in c if x['recall']>=r),default=0.) for r in np.linspace(0,1,101)]))

def evaluate(rows,preds):
 if set(preds)!={r['image_id'] for r in rows} or any(r['phase9_partition'] not in ('exposed','final') for r in rows):raise ValueError('Metric membership mismatch')
 classes={};opall=[];apall=[];nall=0
 for c in CLASSES:
  low,n,unk=events(rows,preds,c);high,_,_=events(rows,preds,c,.75);op,_,opunk=events(rows,preds,c,floor=.15)
  tp=sum(e['outcome']=='tp' for e in op);fp=sum(e['outcome']=='fp' for e in op);fn=n-tp;val=[e['iou'] for e in op if e['outcome']=='tp']
  classes[c]=dict(gt_boxes=n,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/n if n else None,
   f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,ap50=ap(low,n),ap75=ap(high,n),duplicate_fp=sum(e.get('duplicate',False) for e in op),
   mean_matched_iou=float(np.mean(val)) if val else None,unscored_or_partial_unmatched_at_operating=opunk,
   fp_observable_images=sum(r['fp_coverage'][c] for r in rows),events=op)
  opall+=op;apall+=low;nall+=n
 tp=sum(v['tp'] for v in classes.values());fp=sum(v['fp'] for v in classes.values());fn=nall-tp
 diagnostics=dict(mixed_object_candidates=0,fragmentation_candidates=0,category_disagreement_candidates=0)
 for r in rows:
  gs=[g for g in r['annotations'] if not g['ignore']];ps=[p for p in preds[r['image_id']] if p['score']>=.15]
  for p in ps:
   if sum(inter(p['box'],g['box'])/area(g['box'])>=.7 for g in gs)>=2:diagnostics['mixed_object_candidates']+=1
   if any(g['label']!=p['label'] and iou(p['box'],g['box'])>=.5 for g in gs) and not any(g['label']==p['label'] and iou(p['box'],g['box'])>=.5 for g in gs):diagnostics['category_disagreement_candidates']+=1
  for g in gs:
   if sum(p['label']==g['label'] and iou(p['box'],g['box'])<.5 and inter(p['box'],g['box'])/area(p['box'])>=.8 and area(p['box'])<.6*area(g['box']) for p in ps)>=2:diagnostics['fragmentation_candidates']+=1
 pr=curve(apall,nall);matched={}
 for target in (28/41,29/41):
  available=[p for p in pr if p['recall']>=target];matched[str(round(target,8))]=max(available,key=lambda p:p['precision']) if available else None
 return dict(images=len(rows),gt_boxes=nall,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/nall if nall else None,
  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,ap50=float(np.mean([v['ap50'] for v in classes.values() if v['ap50'] is not None])),
  ap75=float(np.mean([v['ap75'] for v in classes.values() if v['ap75'] is not None])),duplicate_fp=sum(v['duplicate_fp'] for v in classes.values()),
  mean_matched_iou=float(np.mean([e['iou'] for e in opall if e['outcome']=='tp'])) if tp else None,categories=classes,diagnostics=diagnostics,
  pr_curve=pr,precision_at_reference_recall=matched,operating_threshold=.15,ap_score_floor=.001,matching_iou=.5,ap75_matching_iou=.75,
  counts_are_source_native_not_verified_real_world_accuracy=True,unknowns_never_confirmed_false_positives=True)
