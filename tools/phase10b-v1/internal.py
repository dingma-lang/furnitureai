"""Native partial-label internal evaluation. No external panels or frozen loaders."""
import importlib.util
from p10 import *
from evaluation import metric_row
sys.path.insert(0,str(ROOT/'tools/phase9-v1'))
from metrics import events,ap,curve
from fusion import nms,iou,inter,area

def evaluate(rows,predictions):
    if not rows or any(r['historical_partition']!='train' or r['internal_partition']!='inner_development' for r in rows):raise PermissionError('Internal validation only')
    if set(predictions)!={r['image_id'] for r in rows}:raise ValueError('Prediction membership mismatch')
    rows=[metric_row(r) for r in rows];cats={};all_events=[]
    for c in CLASSES:
        low,n,unknown=events(rows,predictions,c);high,_,_=events(rows,predictions,c,.75);op,_,_=events(rows,predictions,c,floor=.15)
        tp=sum(e['outcome']=='tp' for e in op);fp=sum(e['outcome']=='fp' for e in op)
        cats[c]=dict(tp=tp,fp=fp,fn=n-tp,ap50=ap(low,n),ap75=ap(high,n),
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/n if n else None,
            duplicate_fp=sum(e.get('duplicate',False) for e in op),
            mean_matched_iou=sum(e['iou'] for e in op if e['outcome']=='tp')/tp if tp else None,
            unknown_unmatched=unknown)
        all_events+=low
    supported=[v for v in cats.values() if v['ap50'] is not None]
    tp=sum(v['tp'] for v in cats.values());fp=sum(v['fp'] for v in cats.values());fn=sum(v['fn'] for v in cats.values())
    diagnostics=dict(mixed_object_candidates=0,fragmentation_candidates=0,category_disagreement_candidates=0)
    for row in rows:
        gt=[a for a in row['annotations'] if not a['ignore']];ps=[p for p in predictions[row['image_id']] if p['score']>=.15]
        for p in ps:
            diagnostics['mixed_object_candidates']+=sum(inter(p['box'],a['box'])/area(a['box'])>=.7 for a in gt)>=2
            diagnostics['category_disagreement_candidates']+=any(a['label']!=p['label'] and iou(p['box'],a['box'])>=.5 for a in gt) and not any(a['label']==p['label'] and iou(p['box'],a['box'])>=.5 for a in gt)
        for a in gt:
            diagnostics['fragmentation_candidates']+=sum(p['label']==a['label'] and iou(p['box'],a['box'])<.5 and inter(p['box'],a['box'])/area(p['box'])>=.8 and area(p['box'])<.6*area(a['box']) for p in ps)>=2
    return dict(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,tp=tp,fp=fp,fn=fn,
        ap50=sum(v['ap50'] for v in supported)/len(supported),ap75=sum(v['ap75'] for v in supported)/len(supported),
        categories=cats,diagnostic_heuristics=diagnostics,diagnostic_flags_are_not_verified_errors=True,
        pr_curve=curve(all_events,tp+fn),scope='provisional source-native partial annotation agreement',
        confidence=.15,nms_iou=.5,independent_verified_accuracy=False)

def predict(model,processor,row,manifest):
    import torch
    from PIL import Image
    from expanded import from_square,cxcywh_to_xyxy
    p=image_path(row,'inner_development',manifest)
    with Image.open(p) as im:im=im.convert('RGB');batch=processor(text=[['a photo of a '+c for c in CLASSES]],images=im,return_tensors='pt')
    model.eval()
    with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
    boxes=from_square(cxcywh_to_xyxy(out.pred_boxes[0].float()),row['width'],row['height']).clamp(0,1).cpu().tolist()
    scores,labels=out.logits[0].float().sigmoid().max(-1);pred=[]
    for box,s,c in zip(boxes,scores.cpu().tolist(),labels.cpu().tolist()):
        if s>=.001 and box[2]>box[0] and box[3]>box[1]:pred.append(dict(box=box,score=s,label=CLASSES[c]))
    return nms(pred,.5,300)
