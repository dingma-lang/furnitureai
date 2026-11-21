"""Padding/clipping and mask coverage audit from saved tensors; no forwards."""
from collections import Counter
import numpy as np
from common9 import *

def run():
    records=read(OUT/'prediction-records.json')['records'];_,dev=subsets();rows={r['image_id']:r for r in dev};result={}
    for model,(arm,epoch) in MODELS.items():
        stats=Counter();bins={};folder=C8/arm/'evaluation'/('epoch-'+str(epoch));done=read(folder/'raw-complete.json')
        raw={rec['image_id']:load_npz(folder/rec['file']) for rec in done['records']}
        for p in records:
            if p['model']!=model or p['label']!='chair':continue
            r=rows[p['image_id']];q=p['query_index'];w,h=r['width'],r['height'];s=max(w,h);anchor=((q%60+.5)/60,(q//60+.5)/60);pad=anchor[0]>=w/s or anchor[1]>=h/s
            box=registered_box(raw[p['image_id']]['square_cxcywh'][q],w,h,clip=False);clip=bool((box< -1e-6).any() or (box>1+1e-6).any())
            for threshold in THRESHOLDS:
                if p['score']<threshold:continue
                key=str(threshold);c=bins.setdefault(key,Counter());c[p['status']]+=1
                if p['status']=='fp':
                    c['fp_padding_anchor']+=pad;c['fp_box_was_clipped']+=clip;c['fp_tiny_area_lt1pct']+=p['geometry']['area_fraction']<.01;c['fp_fragment_heuristic']+=p.get('fragment_heuristic',False);c['fp_near_known_Chair_IoU_ge01']+=p.get('near_Chair_heuristic',False);c['fp_duplicate']+=p['duplicate']
        result[model]={key:dict(v) for key,v in bins.items()}
    occurrences={}
    for arm in ARMS:
        perimage=[r for i in range(1,76) for r in read(C8/arm/('update-'+str(i).zfill(3)+'.json'))['images']];scores=[s for r in perimage for s in r['added_Chair_scores']]
        occurrences[arm]=dict(training_photo_occurrences=len(perimage),positive_Chair_target_occurrences=114*5,all_query_occurrences=len(perimage)*3600,added_Chair_negative_occurrences=sum(r['added_active_Chair_terms'] for r in perimage),
            actual_additional_gradient_applied=arm=='B-corrected',max_added_Chair_score=max(scores),above_015=sum(s>=.15 for s in scores),above_05=sum(s>=.5 for s in scores),above_08=sum(s>=.8 for s in scores),scores=quantiles(scores))
    save(REPORT/'query-space-audit.json',dict(thresholds=result,training_loss_target_occurrences=occurrences,
        query_grid='Observed60x60; anchor flags refer to patch centers before inverse square padding, not to proof of predicted-box validity',
        clipping='Compare saved unclipped predicted boxes with original frame; flag is diagnostic, not automatically a false positive',
        high_score_Chair_negatives_received=0,model_forwards=0))

if __name__=='__main__':run()
