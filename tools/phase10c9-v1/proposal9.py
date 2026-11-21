"""Preparation only: freeze pretrained per-query shift and scale calibration.

No training launcher. This ablation is unauthorized until separately approved.
"""
from common9 import *

def configure_candidate(model):
    from expanded import configure_heads,HEAD_KEYS
    configure_heads(model)
    frozen={n for n in HEAD_KEYS if n.startswith(('class_head.logit_shift.','class_head.logit_scale.'))}
    for n,p in model.named_parameters():
        if n in frozen:p.requires_grad_(False)
    expected=HEAD_KEYS-frozen;actual={n for n,p in model.named_parameters() if p.requires_grad}
    if actual!=expected or len(frozen)!=4 or len(actual)!=8:raise ValueError('Proposed gradient scope violation')
    return sorted(actual)

def run():
    save(REPORT/'next-experiment-proposal.json',dict(status='NOT_AUTHORIZED',training_authorized=False,Phase10D_authorized=False,
        single_change='Freeze original class_head.logit_shift and class_head.logit_scale weights+biases while retaining the C8 corrected objective',
        hypothesis='Pretrained shared per-query calibration may resist the observed common-category logit inflation; dense projection drift remains a competing explanation, so an ablation is required',
        initialization='Exact immutable original OWLv2; not continuation of an inflated checkpoint',trainable_parameters=1577988,trainable_tensors=8,
        trainable=['class_head.dense0.weight','class_head.dense0.bias','box_head.dense0.weight','box_head.dense0.bias','box_head.dense1.weight','box_head.dense1.bias','box_head.dense2.weight','box_head.dense2.bias'],
        frozen='Vision/text encoders, objectness, both class-head calibration modules, and all remaining parameters',
        data=dict(optimization_photos=59,known_source_groups=56,positive_boxes=182,negative_regions=41,new_negative_candidates_activated=0,source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),development_photos=19,development_role='Exposed internal diagnostic comparison only; never gradient updates or negative mining'),
        settings=dict(seed=6106,learning_rate=1e-5,optimizer='AdamW',weight_decay=.01,batch_size=1,gradient_accumulation=4,max_epochs=5,max_optimizer_updates=75,max_gpu_seconds=600,max_vram_gib=6.5,loss_weights='Exactly C8: classification1/L1=5/GIoU=2/focal alpha.25 gamma2',sampling='Same C8 deterministic order',precision='Same C8 FP16 autocast, FP32 loss, GradScaler128'),
        comparison='Previously fixed C8 B and original on the same19-photo diagnostic protocol; final epoch endpoint fixed before results, no threshold tuning, no additional control rerun',
        predeclared_evidence_criteria=dict(Chair_FP_relative_reduction_at_015_at_least=.2,Chair_recall_absolute_drop_at_most=.03,aggregate_recall_drop_at_most=.03,matched_recall_precision_drop_at_most=.02,per_category='No concealed Sofa/Bed recall/AP75 regressions; report uncertainty and counts',not_an_independent_success_claim=True),
        stop_conditions=['Any nonfinite loss/gradient or unsafe memory/time/update event','Frozen calibration/backbone drift or checkpoint integrity failure','Positive/ignored masks changed or protected negative overlap','No material precision gain at comparable recall: do not extend epochs, tune thresholds, add weights or launch another variant'],
        preservation='Unique new outputs, original weights and all prior artifacts immutable; predictions saved before separate CPU metrics',approval_required='Explicit approval for exactly this one bounded calibration-frozen diagnostic. No GPU execution is authorized by C9.'))

if __name__=='__main__':run()
