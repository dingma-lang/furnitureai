"""Float64 CPU signal-strength summary; no model loading, gradients or scoring."""
import numpy as np
from scipy.special import expit
from common7 import *

def run():
    complete=read(RUN/'raw-complete.json');scores=[];losses=[];per_image=[]
    for receipt in complete['records']:
        item=read(RUN/'cpu-analysis-v1'/(str(receipt['index']).zfill(2)+'.json'));ids=item['added_Chair_mask_query_indices'];a=load_arrays(RUN/receipt['raw_file'])
        logits=a['logits'][ids,1].astype(np.float64);p=expit(logits);terms=.75*p*p*np.logaddexp(0,logits)/item['normalization_positive_GT_count']
        scores.extend(p.tolist());losses.extend(terms.tolist())
        per_image.append(dict(index=receipt['index'],new_queries=len(ids),above_015=int((p>=.15).sum()),above_001=int((p>=.01).sum()),additional_loss_float64=float(terms.sum())))
    arr=np.array(scores,dtype=np.float64);loss=np.array(losses,dtype=np.float64)
    result=dict(new_queries=len(arr),Chair_score_quantiles={str(q):float(np.quantile(arr,q)) if len(arr) else None for q in (0,.25,.5,.75,.9,.95,1)},
        above_015=int((arr>=.15).sum()),above_005=int((arr>=.05).sum()),above_001=int((arr>=.01).sum()),
        additional_focal_loss_sum_float64=float(loss.sum()),mean_image_additional_focal_loss_float64=float(loss.sum()/24),
        scope='Signal magnitude on added negative queries only; not detection precision/recall or performance improvement',
        CPU_only=True,model_forwards=0,backward_calls=0,code_sha256=sha(HERE/'signal7.py'))
    save(REPORT/'signal-summary.json',result);save(RUN/'cpu-analysis-v1/signal-per-image.json',per_image)
    print('NEW QUERIES',len(arr),'CHAIR SCORES>=.15',result['above_015'],'FLOAT64 ADDED CLASS LOSS',result['additional_focal_loss_sum_float64'])

if __name__=='__main__':run()
