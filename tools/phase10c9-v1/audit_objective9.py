"""Inspect local source and CPU head tensors; no real-model forward is executed."""
import numpy as np
from collections import Counter
from common9 import *

def negative_focal_from_probability(p,n=1):
    if not 0<p<1 or n<1:raise ValueError('Interior probability and positive normalization required')
    return .75*p*p*(-math.log1p(-p))/n

def run():
    import torch
    from safetensors import safe_open
    binding=read(C8/'binding.json');model_file=ROOT/'models/owlv2-original-baseline-20261103/model.safetensors';initial={}
    with safe_open(model_file,framework='pt',device='cpu') as f:
        for k in f.keys():
            if k.startswith(('class_head.','box_head.')):initial[k]=f.get_tensor(k)
    tensor_changes={}
    for name,arm in (('control','A-control'),('corrected','B-corrected')):
        c=read(C8/arm/'complete.json');path=C8/c['final_checkpoint']['file']
        if sha(path)!=c['final_checkpoint']['sha256']:raise ValueError('Checkpoint changed')
        head=torch.load(path,map_location='cpu',weights_only=True)['weights'];changes={}
        for k,t in head.items():
            a=initial[k].double();delta=t.double()-a;changes[k]=dict(original_L2=float(a.norm()),delta_L2=float(delta.norm()),relative_delta_L2=float(delta.norm()/a.norm()) if a.norm() else None,
                mean_delta=float(delta.mean()),max_abs_delta=float(delta.abs().max()))
        tensor_changes[name]=changes
    train,dev=subsets();evidence=read(SOURCE/'source-negative-evidence.json')['records'];candidates=read(SOURCE/'all-candidates.json')['candidates'];source_index={c['candidate_id']:c for c in candidates}
    negative_kinds=Counter(source_index[e['candidate_id']]['native_label'] for e in evidence);negative_area=[(e['box'][2]-e['box'][0])*(e['box'][3]-e['box'][1]) for e in evidence]
    sources={
        'transformers_architecture':ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/modeling_owlv2.py',
        'transformers_postprocessor':ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/image_processing_owlv2.py',
        'legacy_loss':ROOT/'tools/phase10a-v1/expanded.py','corrected_loss':ROOT/'tools/phase10c8-v1/loss8.py','executed_worker':ROOT/'tools/phase10c8-v1/worker8.py','saved_metric_postprocessing':ROOT/'tools/phase10c8-v1/metrics8.py'}
    save(REPORT/'objective-audit.json',dict(source_files={k:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for k,p in sources.items()},
        formula='z[q,c] = (cosine(dense0(image_feature[q]), text[c]) + logit_shift(image_feature[q])) * (ELU(logit_scale(image_feature[q])) + 1); score[q] = sigmoid(max_c z[q,c])',
        query_selection='All60x60=3600 patch features; per-query max over7 prompts, not DETR object queries and not top-objectness filtering',
        objectness='Computed by separate frozen head but unused by the executed loss and final score/NMS. Saved arrays omit objectness and intermediate feature vectors; no retrospective objectness-gating or internal-layer decomposition can be claimed',
        loss='Matched class positives + explicit-native-negative masks + B source-region Chair negatives. Other classes on a matched query and unmatched unknown regions remain ignored. No softmax, objectness loss or universal background target.',
        normalization='Sum of active focal terms / positive GT count per photo. Not divided by3600queries, valid negatives or active terms. Positive and negative terms share the same divisor; sparse coverage and p^gamma suppress easy negatives independently of that divisor.',
        matching='Hungarian cost -sigmoid(class_logit)+5L1+2(1-GIoU), recomputed each image; matched positives always override negatives. Added region membership is nondifferentiable and never directly a box regression target.',
        broad_confidence_mechanism='Positive-only Chair supervision can lower loss via shared shift/scale as well as semantic projection. Shared parameters affect ignored logits. This is permitted by the mask, not explicit all-query positive labels. Attribution specifically to calibration layers requires a controlled freeze ablation.',
        preserved_parameters=binding['settings'],head_tensor_changes=tensor_changes,training_positive_counts=dict(Counter(a['label'] for r in train for a in r['annotations'] if not a['ignore'])),
        native_global_negative_photos=dict(Counter(c for r in train for c,v in r['explicit_native_negative'].items() if v)),
        negative_inventory=dict(regions=len(evidence),photos=len({e['image_id'] for e in evidence}),source_groups=len({e['identity_group'] for e in evidence}),native_label_counts=dict(negative_kinds),area_fraction_quantiles=quantiles(negative_area)),
        analytic_focal_example=dict(p_easy=.0001579642770307742,p_hard=.5,easy_loss=negative_focal_from_probability(.0001579642770307742),hard_loss=negative_focal_from_probability(.5),hard_to_easy_loss_ratio=negative_focal_from_probability(.5)/negative_focal_from_probability(.0001579642770307742),same_positive_count_divisor_cancels_in_ratio=True),
        parameter_tensors_loaded_CPU_only=True,model_forwards=0,optimizer_updates=0))
    print('CPU head tensor and installed architecture audit complete;0model forwards')

if __name__=='__main__':run()
