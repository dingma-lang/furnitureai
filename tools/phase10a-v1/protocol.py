"""Prepare bounded proposals; an authorization gate always denies Phase 10A training."""
import math
from common import *

FIT_IDS = ('heads-original-lr1e-5','heads-original-lr3e-5')

def check_bounds(config):
    if config['fit_id'] not in FIT_IDS or config['mode']!='heads': raise ValueError('Unapproved adaptation scope')
    if config['epochs']!=5 or config['max_updates']!=150 or config['batch_size']!=1 or config['accumulation']!=4:
        raise ValueError('Update/epoch/batch bound mismatch')
    if config['total_queue_max_updates']!=300 or config['queue_wall_seconds']!=3600 or config['allocated_vram_stop_bytes']!=int(6.5*2**30):
        raise ValueError('Queue resource limit mismatch')
    if config['learning_rate'] != dict(zip(FIT_IDS,(1e-5,3e-5)))[config['fit_id']]: raise ValueError('Config LR mismatch')
    if config['training_authorized'] is not False: raise PermissionError('Phase 10A cannot authorize training')
    return True

def require_training_authorization(*_):
    raise PermissionError('Phase 10A is preflight only. Separate approval and a reviewed execution worker are required.')

def update_count(images, epochs=5, accumulation=4):
    return min(150, math.ceil(images/accumulation)*epochs)

def run():
    m = read(OUT/'prepared-manifest.json'); original = read(ROOT/'data/open-images/experiments/phase9-multiview-v1/model-inventory.json')
    for fit_id, lr in zip(FIT_IDS,(1e-5,3e-5)):
        c = dict(fit_id=fit_id,training_authorized=False,mode='heads',epochs=5,max_updates=150,
          total_queue_max_updates=300,queue_wall_seconds=3600,per_fit_seconds=1800,batch_size=1,accumulation=4,
          allocated_vram_stop_bytes=int(6.5*2**30),memory_fraction=.8,learning_rate=lr,weight_decay=.01,seed=20261118,
          initialization='original_owlv2',base=original['weights']['original'],
          preserved_comparator=original['weights']['adapted'],
          manifest_sha256=sha(OUT/'prepared-manifest.json'),classes=list(CLASSES),
          prompts=['a photo of a '+c for c in CLASSES],supported_positive_categories=m['supported_positive_categories'],
          precision='fp16_if_independently_verified_else_stop',gradient_clip=1.,backbone='frozen_eval',objectness='frozen_unsupervised',
          unmatched_negative_policy='Only explicit safe-core native source negatives; no absent/NEX/filtered-instance negatives',
          expected_optimizer_updates=update_count(m['internal_partitions']['optimization']['images']),
          checkpoint_policy='New immutable head delta + optimizer/scaler/RNG/group cursor + binding + SHA256 receipt; reserve-before-call ledger',
          recovery_policy='Resume only a committed matching receipt; an uncertain optimizer boundary blocks replay',
          run_destination='data/open-images/experiments/phase10b-heads-v1/'+fit_id)
        check_bounds(c); save(OUT/'configs'/(fit_id+'.json'),c)
    save(OUT/'evaluation-protocol.json',dict(training_authorized=False,model_inference_authorized=False,
      selection_partition='training-derived inner_development only',
      exposed18='Historical reference only; not a hyperparameter selection pool',
      exposed36='Now exposed; no repeated selection or unrestricted tuning',frozen='No scoring or image decoding in Phase10A',
      primary='Maximize precision envelope at both original and Phase5C baseline recalls measured on inner-development; unreachable recall fails qualification',
      qualification=dict(min_precision_gain_pp=3,max_recall_loss_objects=1,max_ap75_loss_pp=2,
                         category_ap75_drop_over_pp_requires_failure=5,no_per_category_recall_loss_for_supported_classes=True),
      tie_break=['macro_AP75_supported','F1_at_fixed_0.15','lower_latency','fit_id'],
      metric_scope='Native partial annotations with source-specific observable FP scope; unknown is unscored',
      metrics=['full_PR_curves','matched_recall_precision','AP50','AP75','per_category_P_R','TP_FP_FN',
               'matched_IoU','duplicate_FP','fragmentation_and_mixed_object_heuristics','latency','allocated_and_reserved_VRAM'],
      operating_confidence=.15,nms_iou=.5,matching_iou=[.5,.75],ap_floor=.001,
      unsupported_categories='Table/Desk/Cabinet/Lamp inactive positives; no seven-class performance claim. Native-only diagnostics must remain separate.',
      independent_plan='Select using internal grouped development, lock code/config/checkpoint/threshold. Request separate permission for one reserved-set comparison. Audit seven-class native coverage first; Cabinet unit labels and exhaustive FP coverage remain missing. Do not broaden frozen membership or revise labels without approval.',
      model_companions=['immutable_original','immutable_phase5c_adapted'],per_category_reporting_required=True))
    print('PREPARED two unauthorized configs; expected updates per config',update_count(m['internal_partitions']['optimization']['images']),flush=True)

if __name__=='__main__': run()
