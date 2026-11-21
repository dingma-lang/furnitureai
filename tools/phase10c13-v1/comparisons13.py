"""Supplemental saved-result comparisons; no image/model inference."""
import numpy as np
from common13 import *
from analyze11 import envelope


def calculate_localization(rows,m,a,b):
    result={}
    for category in ('chair','sofa','bed'):
        aa={(e['image_id'],e['gt_index']):e for e in m.events(rows,a,category,floor=.15)[0] if e['outcome']=='tp'};bb={(e['image_id'],e['gt_index']):e for e in m.events(rows,b,category,floor=.15)[0] if e['outcome']=='tp'};shared=sorted(set(aa)&set(bb));delta=[bb[k]['iou']-aa[k]['iou'] for k in shared]
        result[category]=dict(common_GT_matches=len(shared),C11_only_GT_matches=len(set(aa)-set(bb)),C13_only_GT_matches=len(set(bb)-set(aa)),mean_IoU_delta_on_common_GT=float(np.mean(delta)) if delta else None,mean_C11_IoU_on_common_GT=float(np.mean([aa[k]['iou'] for k in shared])) if shared else None,mean_C13_IoU_on_common_GT=float(np.mean([bb[k]['iou'] for k in shared])) if shared else None,IoU75_crossings_down=sum(aa[k]['iou']>=.75>bb[k]['iou'] for k in shared),IoU75_crossings_up=sum(bb[k]['iou']>=.75>aa[k]['iou'] for k in shared))
    return dict(scope='Same GT identities matched atIoU0.5/confidence0.15;descriptive,not fixed prediction boxes or an independent test',categories=result)


def run():
    _,_,dev=verify_inputs13();m=metric_module();rows=[m.scopes(r) for r in dev];a=read(C11_RUN/'cpu-diagnostics-v1/predictions-E5.json');b=read(FUTURE/'cpu-evaluation-v1/predictions-E5.json')
    loc=calculate_localization(rows,m,a,b);path=REPORT13/'localization-comparison.json'
    if path.exists():
        if read(path)!=loc:raise ValueError('Saved localization comparison differs')
    else:save(path,loc)
    result=read(REPORT13/'results.json');entries=read(C8/'metric-results.json')['entries'];metrics={'original':read(C11_RUN/'cpu-diagnostics-v1/metrics-E0.json')['metrics'],'corrected_B':next(e['metrics'] for e in entries if e['arm']=='B-corrected' and e['epoch']==5),'frozen_calibration':read(C11_RUN/'cpu-diagnostics-v1/metrics-E5.json')['metrics'],'projection_regularized':read(FUTURE/'cpu-evaluation-v1/metrics-E5.json')['metrics']}
    targets={k:v['recall'] for k,v in metrics.items()};gradient=result['gradient_records'];ratios=[r['regularization_to_data_ratio'] for r in gradient]
    save(REPORT13/'operating-point-comparison.json',dict(targets=targets,precision_envelopes={target:{model:envelope(metric['pr_curve'],value) for model,metric in metrics.items()} for target,value in targets.items()},maximum_recall_under_frozen_protocol={model:max(p['recall'] for p in metric['pr_curve']) for model,metric in metrics.items()},no_threshold_adopted=True,not_an_intrinsic_all_query_recall_ceiling=True,
        gradient_ratio_quantiles_all75=quantiles(ratios),updates_regularization_norm_exceeds_data=sum(x>1 for x in ratios),updates_regularization_norm_exceeds_half_data=sum(x>.5 for x in ratios),gradient_ratio_caveat='Data gradient obtained by subtracting analytic L2-SP gradient from unscaled accumulated gradient before clipping;FP32/AMP roundoff and directions limit interpretations'))
    print('Supplemental matched recall/localization/gradient comparisons verified',flush=True)


if __name__=='__main__':run()
