"""CPU-only matched-box diagnostics; no matching-policy changes or inference."""
from collections import defaultdict
import numpy as np
from common14 import *
from mining14 import read_derived
from audit14 import overlap_arrays
from common9 import registered_box


def run():
    _,train,dev,_,_=training_context();rows={r['image_id']:r for r in train};buckets=defaultdict(list)
    for r in read(RAW14/'query-summaries.json')['records']:
        row=rows[r['image_id']];require_training_row(row,train,dev)
        a=load_npz(ROOT/r['query_file']);d=read_derived(ROOT/r['derived_file'])
        boxes=registered_box(a['square_cxcywh'],row['width'],row['height'],clip=False)
        unmatched=d['assignment']<0
        if np.count_nonzero(d['grad_box'][unmatched]):raise ValueError('Unmatched box received regression gradient')
        for match in r['matches']:
            q=match['query_index'];ann=row['annotations'][match['annotation_index']]
            iou=float(overlap_arrays(boxes[q:q+1],[ann['box']])[0][0,0])
            buckets[(r['state'],ann['label'])].append(dict(IoU=iou,score=match['positive_score'],box_gradient_L2=float(np.linalg.norm(d['grad_box'][q].astype(float)))))
    result={}
    for (state,label),values in buckets.items():
        result.setdefault(state,{})[label]=dict(positive_matches=len(values),IoU_quantiles=quantiles([v['IoU'] for v in values]),matched_IoU_below05=sum(v['IoU']<.5 for v in values),matched_IoU_below025=sum(v['IoU']<.25 for v in values),positive_score_quantiles=quantiles([v['score'] for v in values]),box_output_gradient_L2_quantiles=quantiles([v['box_gradient_L2'] for v in values]))
    save(REPORT14/'positive-matching-quality.json',dict(by_state=result,unmatched_box_output_gradients_all_zero=True,IoU_is_standard_union_not_GIoU=True,assignment='CPU reconstruction verified against saved losses/counts;not original saved GPU indices',training_only=True,model_forwards=0,policy_changes=0))
    print('C14 positive matching/box-gradient audit PASS',flush=True)


if __name__=='__main__':run()
