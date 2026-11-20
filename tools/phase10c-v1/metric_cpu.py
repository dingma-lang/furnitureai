"""Standalone NumPy/Python CPU implementation of the existing native protocol.

No Torch import, image loader, model, score calibration or new matching policy.
The control flow avoids generator expressions; equivalence is regression-tested.
"""
import math
import numpy as np
from c10 import CLASSES

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def iou(a,b):
    overlap=inter(a,b);union=area(a)+area(b)-overlap
    return overlap/union if union else 0.
def valid(p):
    b=p['box']
    if p['label'] not in CLASSES or not math.isfinite(p['score']) or not 0<=p['score']<=1 or len(b)!=4:raise ValueError('Malformed normalized prediction')
    for x in b:
        if not math.isfinite(x) or not 0<=x<=1:raise ValueError('Malformed normalized prediction')
    if area(b)<=0:raise ValueError('Malformed normalized prediction')
def nms(predictions,threshold=.5,limit=300):
    result=[]
    for p in sorted(predictions,key=lambda p:-p['score']):
        valid(p);suppress=False
        for q in result:
            if p['label']==q['label'] and iou(p['box'],q['box'])>threshold:suppress=True;break
        if not suppress:result.append(dict(p))
        if len(result)==limit:break
    return result

def scopes(row):
    coverage={c:False for c in CLASSES};fp={c:False for c in CLASSES}
    for c in ('sofa','chair','bed'):
        positive=[a for a in row['annotations'] if a['label']==c and not a['ignore']]
        negative=row['explicit_native_negative'].get(c,False)
        if positive and negative:raise ValueError('Contradictory native scope')
        coverage[c]=bool(positive or negative);fp[c]=negative
        if not positive:continue
        dataset=positive[0]['native_dataset'];native=positive[0]['native_id'];retained=True
        for a in row['annotations']:
            if a['native_dataset']==dataset and a['native_id']==native and a['ignore']:retained=False
        if dataset=='LVIS':fp[c]=retained and native not in row['source_native_scope'].get('not_exhaustive_category_ids',[])
        elif dataset=='OpenImages':fp[c]=retained and row['source_native_scope'].get('native_image_level_labels',{}).get(native)==1
    return dict(row,coverage=coverage,fp_coverage=fp)

def events(rows,predictions,category,threshold=.5,floor=.001):
    gs={r['image_id']:[a for a in r['annotations'] if a['label']==category and not a['ignore']] for r in rows if r['coverage'][category]}
    ignored={r['image_id']:[a for a in r['annotations'] if a['label']==category and a['ignore']] for r in rows}
    byid={r['image_id']:r for r in rows};used={k:set() for k in gs};ranked=[];result=[];unknown=0
    for r in rows:
        for j,p in enumerate(predictions[r['image_id']]):
            valid(p)
            if p['label']==category and p['score']>=floor:ranked.append((p['score'],r['image_id'],j,p))
    for score,ident,j,p in sorted(ranked,key=lambda v:(-v[0],v[1],v[2])):
        record=dict(image_id=ident,index=j,label=category,score=score)
        if ident not in gs:unknown+=1;continue
        choices=[]
        for k,g in enumerate(gs[ident]):
            if k not in used[ident]:choices.append((iou(p['box'],g['box']),k))
        best,k=max(choices,default=(0.,-1))
        if k>=0 and best>=threshold:
            used[ident].add(k);result.append(dict(record,outcome='tp',iou=best,gt_index=k));continue
        ignore=False
        for g in ignored[ident]:
            if inter(p['box'],g['box'])/area(p['box'])>=.5:ignore=True;break
        if ignore:continue
        if not byid[ident]['fp_coverage'][category]:unknown+=1;continue
        duplicate=False
        for k,g in enumerate(gs[ident]):
            if k in used[ident] and iou(p['box'],g['box'])>=threshold:duplicate=True;break
        result.append(dict(record,outcome='fp',duplicate=duplicate))
    return result,sum([len(v) for v in gs.values()]),unknown

def curve(ev,n,group_ties=True):
    result=[];tp=fp=0;ranked=sorted(ev,key=lambda e:(-e['score'],e['image_id'],e['index'],e['label']))
    for i,e in enumerate(ranked):
        tp+=e['outcome']=='tp';fp+=e['outcome']=='fp'
        if group_ties and i+1<len(ranked) and ranked[i+1]['score']==e['score']:continue
        result.append(dict(threshold=e['score'],precision=tp/(tp+fp),recall=tp/n if n else None,tp=tp,fp=fp))
    return result
def ap(ev,n):
    if not n:return None
    points=curve(ev,n,False);samples=[]
    for recall in np.linspace(0,1,101):
        best=0.
        for p in points:
            if p['recall']>=recall:best=max(best,p['precision'])
        samples.append(best)
    return float(np.mean(samples))

def evaluate(rows,predictions):
    if set(predictions)!={r['image_id'] for r in rows}:raise ValueError('Exact metric membership required')
    cats={};all_low=[]
    for c in CLASSES:
        low,n,unknown=events(rows,predictions,c);high,_,_=events(rows,predictions,c,.75);op,_,_=events(rows,predictions,c,floor=.15)
        tp=sum([e['outcome']=='tp' for e in op]);fp=sum([e['outcome']=='fp' for e in op]);ious=[e['iou'] for e in op if e['outcome']=='tp']
        cats[c]=dict(tp=tp,fp=fp,fn=n-tp,precision=tp/(tp+fp) if tp+fp else None,recall=tp/n if n else None,
            ap50=ap(low,n),ap75=ap(high,n),duplicate_fp=sum([e.get('duplicate',False) for e in op]),
            mean_matched_iou=sum(ious)/tp if tp else None,unknown_unmatched=unknown)
        all_low+=low
    supported=[v for v in cats.values() if v['ap50'] is not None]
    tp=sum([v['tp'] for v in cats.values()]);fp=sum([v['fp'] for v in cats.values()]);fn=sum([v['fn'] for v in cats.values()])
    diagnostic=dict(mixed_object_candidates=0,fragmentation_candidates=0,category_disagreement_candidates=0)
    for r in rows:
        gt=[a for a in r['annotations'] if not a['ignore']];ps=[p for p in predictions[r['image_id']] if p['score']>=.15]
        for p in ps:
            mixed=0;different=same=False
            for a in gt:
                mixed+=inter(p['box'],a['box'])/area(a['box'])>=.7
                if iou(p['box'],a['box'])>=.5:
                    if p['label']==a['label']:same=True
                    else:different=True
            diagnostic['mixed_object_candidates']+=mixed>=2
            diagnostic['category_disagreement_candidates']+=different and not same
        for a in gt:
            fragments=0
            for p in ps:
                if p['label']==a['label'] and iou(p['box'],a['box'])<.5 and inter(p['box'],a['box'])/area(p['box'])>=.8 and area(p['box'])<.6*area(a['box']):fragments+=1
            diagnostic['fragmentation_candidates']+=fragments>=2
    return dict(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,tp=tp,fp=fp,fn=fn,
        ap50=float(np.mean([v['ap50'] for v in supported])) if supported else None,
        ap75=float(np.mean([v['ap75'] for v in supported])) if supported else None,
        categories=cats,diagnostic_heuristics=diagnostic,pr_curve=curve(all_low,tp+fn),
        confidence=.15,nms_iou=.5,independent_verified_accuracy=False)
