"""Saved-prediction CPU audit. No model imports, inference or training entry."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
from c10 import read,save,sha,internal_rows,source_path,CLASSES,RUN
import metric_cpu as metric

OUT=ROOT/'reports/chair-fp-investigation-v1'
REC=ROOT/'data/open-images/experiments/phase10c-recovery-v1'
SEED=20261119

def geometry(pred,row,event=None):
    b=pred['box'];area=metric.area(b);chairs=[a for a in row['annotations'] if a['label']=='chair' and not a['ignore']]
    other=[a for a in row['annotations'] if a['label']!='chair']
    ci=[metric.iou(b,a['box']) for a in chairs];maximum=max(ci,default=0.)
    fragment=any(metric.inter(b,a['box'])/area>=.8 and area<.6*metric.area(a['box']) and metric.iou(b,a['box'])<.5 for a in chairs)
    broad=any(metric.inter(b,a['box'])/metric.area(a['box'])>=.7 and area>1.8*metric.area(a['box']) for a in chairs)
    othermax=max([metric.iou(b,a['box']) for a in other],default=0.)
    duplicate=bool(event and event.get('duplicate',False))
    flags=[]
    if duplicate:flags.append('protocol_duplicate')
    if fragment:flags.append('contained_chair_fragment')
    if broad:flags.append('broad_box_covering_chair')
    if .1<=maximum<.5:flags.append('near_chair_localization')
    if othermax>=.3:flags.append('other_native_furniture_overlap')
    primary=next((x for x in ('protocol_duplicate','contained_chair_fragment','broad_box_covering_chair','near_chair_localization','other_native_furniture_overlap') if x in flags),'no_chair_overlap_unresolved')
    return dict(area_fraction=area,width_pixels=(b[2]-b[0])*row['width'],height_pixels=(b[3]-b[1])*row['height'],
        max_chair_iou=maximum,max_other_native_iou=othermax,geometric_flags=flags,primary_geometry_bucket=primary,
        semantic_error_confirmed=False)

def score_bin(score):
    for lo,hi in ((.15,.3),(.3,.5),(.5,.75),(.75,.9),(.9,1.000001)):
        if lo<=score<hi:return f'{lo:g}–{min(hi,1):g}'
    raise ValueError('Not an operating prediction')
def area_bin(area):
    if area<.01:return '<1%'
    if area<.05:return '1–5%'
    if area<.2:return '5–20%'
    return '>=20%'

def write_csv(name,fields,records):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in records:w.writerow({k:json.dumps(r[k]) if isinstance(r.get(k),(dict,list)) else r.get(k,'') for k in fields})

def run():
    if (OUT/'analysis.json').exists():raise FileExistsError('No overwrite; choose a new output version')
    OUT.mkdir(exist_ok=True)
    # Seal every pre-existing recovery artifact, including the final storage record.
    original={str(p.relative_to(ROOT)):sha(p) for p in REC.rglob('*') if p.is_file()}
    for line in (REC/'SHA256SUMS').read_text().splitlines():
        digest,rel=line.split('  ',1)
        if sha(REC/rel)!=digest:raise ValueError('Saved recovery changed: '+rel)
    rows=[metric.scopes(r) for r in internal_rows()];byid={r['image_id']:r for r in rows};preds=read(REC/'post-nms-predictions.json')
    if set(preds)!=set(byid):raise ValueError('Exact19-photo membership required')
    outcomes={};fp=[];records=[]
    for c in CLASSES:
        events,_,_=metric.events(rows,preds,c,floor=.15)
        for e in events:outcomes[e['image_id'],e['index']]=e
    for row in rows:
        for j,p in enumerate(preds[row['image_id']]):
            if p['score']<.15:continue
            e=outcomes.get((row['image_id'],j));status=e['outcome'] if e else 'unscored_incomplete'
            if not e and row['coverage'][p['label']]:
                if any(a['label']==p['label'] and a['ignore'] and metric.inter(p['box'],a['box'])/metric.area(p['box'])>=.5 for a in row['annotations']):status='ignored_region'
            r=dict(image_id=row['image_id'],index=j,category=p['label'],score=p['score'],box=p['box'],status=status,
                source=row['source'],native_FP_observable=row['fp_coverage'][p['label']],confidence_bin=score_bin(p['score']),
                area_fraction=metric.area(p['box']),area_bin=area_bin(metric.area(p['box'])))
            if p['label']=='chair':r.update(geometry(p,row,e))
            records.append(r)
            if status=='fp':fp.append(r)
    old=read(RUN/'comparison-v1/results.json');new=read(next(REC.glob('metrics-cpu-*.json')))['metrics']
    from collections import Counter
    for c in CLASSES:
        if sum(r['category']==c for r in fp)!=new['categories'][c]['fp']:raise ValueError('FP reconciliation failed')
    photo=[]
    for row in rows:
        r=[x for x in records if x['image_id']==row['image_id']];chair=[x for x in r if x['category']=='chair'];cfp=[x for x in chair if x['status']=='fp']
        photo.append(dict(image_id=row['image_id'],source=row['source'],chair_FP=len(cfp),chair_TP=sum(x['status']=='tp' for x in chair),
            chair_unscored=sum(x['status']=='unscored_incomplete' for x in chair),chair_predictions=len(chair),native_chair_FP_observable=row['fp_coverage']['chair'],
            GT_chair_boxes=sum(a['label']=='chair' and not a['ignore'] for a in row['annotations']),
            chair_FP_confidence_bins=dict(Counter(x['confidence_bin'] for x in cfp)),chair_FP_area_bins=dict(Counter(x['area_bin'] for x in cfp))))
    photo.sort(key=lambda r:(-r['chair_FP'],r['image_id']))
    chairfp=[r for r in fp if r['category']=='chair'];sample=[]
    for rank,photo_row in enumerate(photo[:4]):
        candidates=[r for r in chairfp if r['image_id']==photo_row['image_id']];chosen={};rng=random.Random(SEED+rank)
        def take(r,reason):
            if r['index'] not in chosen:chosen[r['index']]=dict(r,selection_reason=reason)
        for bucket in sorted({r['primary_geometry_bucket'] for r in candidates}):
            candidates_b=[r for r in candidates if r['primary_geometry_bucket']==bucket]
            take(max(candidates_b,key=lambda r:(r['score'],-r['index'])),'highest confidence in geometry stratum')
        for b in ('0.15–0.3','0.3–0.5','0.5–0.75','0.75–0.9','0.9–1'):
            v=[r for r in candidates if r['confidence_bin']==b]
            if v:take(sorted(v,key=lambda r:r['score'])[len(v)//2],'median-ranked confidence stratum')
        for b in ('<1%','1–5%','5–20%','>=20%'):
            v=[r for r in candidates if r['area_bin']==b]
            if v and len(chosen)<12:take(max(v,key=lambda r:r['score']),'highest confidence in box-area stratum')
        shuffled=list(candidates);rng.shuffle(shuffled)
        for r in shuffled:
            if len(chosen)>=12:break
            take(r,'seeded coverage fill')
        sample.extend(list(chosen.values())[:12])
    # Eight unscored controls illustrate incomplete annotations; never call them FPs.
    unknown_photos=['coco2017:000000424683','coco2017:000000217546','coco2017:000000253212','coco2017:000000195911']
    for ident in unknown_photos:
        xs=[r for r in records if r['image_id']==ident and r['category']=='chair' and r['status']=='unscored_incomplete']
        xs.sort(key=lambda r:(-r['score'],r['index']))
        if xs:sample.append(dict(xs[0],selection_reason='highest-confidence unscored control'))
        if len(xs)>1:sample.append(dict(xs[len(xs)//2],selection_reason='median-confidence unscored control'))
    for i,r in enumerate(sample):r['sample_id']=f'chair-{i+1:03d}'
    save(OUT/'review-sample.json',dict(seed=SEED,scored_FP_sample=48,unscored_controls=8,not_a_probability_sample=True,
        automatic_human_review_flags=False,records=sample))
    write_csv('review-template.csv',['sample_id','image_id','index','status','score','box','selection_reason','reviewer','review_date','human_reviewed','decision','suspected_failure','evidence'],sample)
    write_csv('predictions-operating.csv',['image_id','index','category','status','score','confidence_bin','box','area_fraction','area_bin','width_pixels','height_pixels','primary_geometry_bucket','geometric_flags','max_chair_iou','max_other_native_iou','native_FP_observable'],records)
    write_csv('photograph-summary.csv',list(photo[0]),photo)
    aggregates={c:dict(FP=sum(r['category']==c for r in fp),confidence_bins=dict(Counter(r['confidence_bin'] for r in fp if r['category']==c)),
        area_bins=dict(Counter(r['area_bin'] for r in fp if r['category']==c))) for c in CLASSES}
    analysis=dict(active_scope='Same19 internal photos/61 native boxes; source-native FP accounting, not confirmed visual mistakes',
        available_per_image_models=['Phase10B LR3e-5 E5 recovered'],unavailable_per_image_models=['original OWLv2 on exact19','Phase10B LR1e-5 E5'],
        original_category_counts=old['entries']['original']['metrics']['categories'],lr1_category_counts=old['entries']['heads-original-lr1e-5']['metrics']['categories'],
        recovered_category_counts=new['categories'],photographs=photo,category_aggregates=aggregates,
        chair_primary_geometry_counts=dict(Counter(r['primary_geometry_bucket'] for r in chairfp)),
        chair_geometry_nonexclusive_counts=dict(Counter(g for r in chairfp for g in r['geometric_flags'])),
        chair_scored_FP=len(chairfp),chair_unscored_operating=sum(r['category']=='chair' and r['status']=='unscored_incomplete' for r in records),
        geometry_rules='Ordered primary: duplicate IoU>=.5; fragment pred-inside-GT>=.8 & area<.6GT; broad covers>=.7GT & area>1.8GT; near .1<=chairIoU<.5; other native furniture IoU>=.3; otherwise unresolved. Geometry is not semantic proof.',
        input_hashes=original,predictions_sha256=sha(REC/'post-nms-predictions.json'),manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        seed=SEED,GPU_forwards=0,training_updates=0,downloads=0)
    save(OUT/'analysis.json',analysis)
    for path,digest in original.items():
        if sha(ROOT/path)!=digest:raise ValueError('Historical recovery artifact modified')
    print(json.dumps(dict(chairFP=len(chairfp),chair_unscored=analysis['chair_unscored_operating'],top_photos=photo[:4],geometry=analysis['chair_primary_geometry_counts'],sample=len(sample)),indent=2))

if __name__=='__main__':run()
