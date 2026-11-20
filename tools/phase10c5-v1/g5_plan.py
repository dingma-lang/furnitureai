"""Write inactive research protocol, never launch training or inference."""
from g5_common import *

def run():
    protocol=dict(schema='phase10c5-inactive-experiment-proposal-v1',enabled=False,training_authorized=False,
        initialization='original immutable OWLv2',target_categories=list(CLASSES),source_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),
        data=dict(optimization_photos=59,source_groups=56,positive_boxes=182,internal_validation_photos=19,
            current_human_confirmed_Chair_negative_regions=0,minimum_regions=6,minimum_training_groups=3,
            inactive_expansion_pool=42,auxiliary_annotations_not_activated=True,unknowns_ignored=True),
        primary_ablation=dict(name='identical model/data: legacy supervision versus additive trusted Chair-negative regions',
            arms=[dict(id='A',negative_objective='legacy Phase10B'),dict(id='C',negative_objective='Phase10C.2 trusted regions; disabled unless reviewed')],
            baseline_context=['immutable original','historical Phase10B saved predictions'],
            trainable='same 12 class_head and box_head tensors; 1,579,526 parameters',
            frozen='vision, text, objectness and all remaining parameters',
            loss='legacy matcher/focal alpha0.25 gamma2; cls1 L1=5 GIoU2, same GT normalization; only trusted negative mask differs',
            seed=6105,learning_rate=1e-5,max_epochs_per_arm=5,max_updates_per_arm=75,max_total_updates=150,
            batch_size=1,gradient_accumulation=4,queue_seconds=3600,allocated_VRAM_stop_GiB=6.5,
            safety=['Zero-update CUDA/AMP/memory smoke first','Verify nonzero actual-query mask difference before training',
                'Stop on nonfinite loss, frozen-parameter change, leakage or memory/runtime limit',
                'Do not silently change negative coefficient or sampling to obtain improvement'],
            candidate_selection='training-only source-grouped internal development; 19 exposed photos diagnostic, not independent evidence',
            success='Higher precision at original matched recall; original recall within 2 percentage points; report bootstrap paired group interval and all per-category regressions',
            failure='No meaningful mask difference, Chair score inflation/recall collapse, or material per-category/localization regression'),
        later_separate_ablations=[dict(id='D-named-aux',status='not scheduled',
            hypothesis='Explicit Person/Plant etc positives and only verified class-specific negatives; no generic Other output',
            prerequisites='Separate authorization of exact native labels, dedup, rights and masks; freeze data/loss first; then isolate auxiliary weight/sampling'),
            dict(id='D-residual',status='CPU component only; not connected to runner',
                module='g5_objective.FurnitureResidual',freeze='all original parameters and logits',
                hypothesis='Only seven prompt slots adapted; named non-furniture output invariant by construction',
                limitations='Does not teach new general categories; no performance claim; prompt mapping and full pipeline still need authorized GPU verification'),
            dict(id='D-anchor',status='design only',hypothesis='Freeze or regularize class logit shift/scale, or teacher distillation on named general-object prompts',
                prerequisites='Independent architecture/calibration ablation; teacher inference needs separate approval; no existing general-object prediction cache')],
        evaluation=dict(threshold=.15,IoU=[.5,.75],
            metrics=['AP50','AP75','fixed-threshold P/R/TP/FP/FN','precision at matched recall','full PR curves',
                'per-photo/class unmatched predictions with completeness caveat','localization IoU','duplicates/fragmentation',
                'score distributions/calibration where labels complete','latency','peak VRAM'],
            exposed_panels='19 internal, 18 historical, 36 fresh: never independent success proof; no reserved/frozen scoring',
            independent_proposal='120 previously unexposed, rights-verified real rooms, target120 source/creator groups; indoor clutter and seating/plant/person distractors; seven-class exhaustive box review with stools taxonomy; incomplete categories ignored',
            freeze_before='photo IDs, source groups, annotation guide, primary threshold and selected checkpoint',
            independent_metrics='one controlled comparison on eligible identical examples; paired creator-group bootstrap; no threshold tuning or retraining after inspection',
            synthetic_or_catalogue='separate sources, never substitutes for independent real-room evaluation'),
        future_GPU_approval_requires=['versioned human export and CPU validation','at least six Chair-free regions in three optimization groups',
            'unchanged source/checkpoint hashes','exact configuration and data overlay hashes','separate explicit training authorization'])
    destination=save_versioned(REPORT/'experiment-proposal.json',protocol)
    print('INACTIVE experiment proposal;0 optimizer updates;',destination.name)

if __name__=='__main__':run()
