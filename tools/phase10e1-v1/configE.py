"""Explicit separate approval gates; neither preparation nor import can execute GPU."""
import math
from commonE import *
TRAIN_LIMITS=dict(epochs=5,updates=75,forwards=373,seconds=600,VRAM_bytes=6979321856,output_bytes=2147483648)
SMOKE_LIMITS=dict(updates=0,forwards=28,gradient_VJPs=48,seconds=180,VRAM_bytes=6979321856,output_bytes=134217728)

def require_execution(approval,mode):
    if mode not in ('training','gradient_smoke') or approval.get('approved') is not True or approval.get('mode')!=mode or not approval.get('approval_reference'):
        raise PermissionError('No GPU/training authorization: E1 is preparation only')
    proposal=REPORT/('gpu-training-disabled.json' if mode=='training' else 'gradient-smoke-disabled.json')
    if approval.get('proposal_sha256')!=sha(proposal) or approval.get('input_binding_sha256')!=sha(BINDING):
        raise PermissionError('Exact immutable proposal/input approval binding required')
    expected=TRAIN_LIMITS if mode=='training' else SMOKE_LIMITS
    if approval.get('limits')!=expected:raise PermissionError('Approved bounds must match prepared proposal')
    if mode=='training':
        coefficient=approval.get('coefficient')
        if type(coefficient) not in (int,float) or not math.isfinite(coefficient) or coefficient<=0 or not approval.get('strength_justification'):
            raise PermissionError('Fixed distillation coefficient has not been justified/approved')
        evidence=SMOKE/'complete.json'
        if not evidence.exists() or approval.get('strength_evidence_sha256')!=sha(evidence) or read(evidence).get('status')!='PASS':
            raise PermissionError('Actual approved gradient-smoke evidence required before training')
    binding=read(BINDING)
    for rel,h in binding['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Immutable E1 input changed: '+rel)
    return approval

def proposals():
    context=read(PREP/'training-context.json');audit=read(REPORT/'teacher-and-mask-audit.json')
    common=dict(approved=False,enabled=False,training_authorized=False,GPU_authorized=False,base_sha256=BASE_SHA,initialization='original pretrained OWLv2 only',teacher='Immutable original C17 FP16 logits/boxes on59optimization photos',method='Selective Chair-logit mean-squared consistency on source-supported visible negative regions;no seven-class/background inference',coefficient=None,coefficient_reason='Output gradients measurable on CPU,but projection Jacobians/gradient directions require actual approved zero-update measurements',learning_rate=1e-5,seed=6106,batch_size=1,gradient_accumulation=4,last_group_size=3,optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01),training_precision='FP16 autocast;FP32loss;GradScaler128initial/growth2000,matchinghistoricalcache',evaluation_precision='FP32',trainable_keys=[f'{module}.{suffix}' for module in ('class_head.dense0','box_head.dense0','box_head.dense1','box_head.dense2') for suffix in ('weight','bias')],trainable_parameters=1577988,frozen='All other parameters,calibration shift/scale,vision/text/shared features,objectness,buffers,derivedboxbias',base_objective='Unchanged C11/C8corrected focalclassification+5L1+2GIoU with original41Chair-negative source grants;noBCE/rebalancing/L2SP',closest_control='C11 calibration-frozen8tensor run;C8/13/20/10D are contextual comparisons,not controlled distillation ablations',positive_boxes=182,photos=59,known_source_groups=56,source_Chair_negative_regions=41,distill_admitted_regions=audit['distill_admitted_regions'],new_labels=0,new_C21_regions_active=False,augmentations=False,training_rows_sha256=context['training_rows_sha256'],development_rows_sha256=context['development_rows_sha256'])
    training=dict(common,limits=TRAIN_LIMITS,destination=str(FUTURE.relative_to(ROOT)),initial_numeric_teacher_checks=59,training_forwards=295,final_diagnostic_forwards=19,early_epoch_scoring=False,fixed_endpoint=75,training_ready=False,gradient_smoke_required=True,evaluation=dict(confidence=.15,IoU=[.5,.75],NMS=.5,cap=300,score_floor=.001,internal_photos=19,threshold_or_hyperparameter_tuning=False,frozen_or_reserved_scoring=False),failure_conditions=['Teacher/source/frame/prompt/cache mismatch','AnyprotectedormatchedpositiveKDterm','Nonfiniteorunauthorizedgradient/weightchange','Resource/update/forwardlimit','AMPskippedstep:noautomaticretry','Missinggradient-evidenceorfixedcoefficient'])
    emit(REPORT/'gpu-training-disabled.json',training)
    smoke=dict(common,training_authorized=False,optimizer_constructed=False,coefficient=None,unit_coefficient_is_gradient_measurement_only=True,limits=SMOKE_LIMITS,destination=str(SMOKE.relative_to(ROOT)),original_replay_photos=4,gradient_reference='ExistingimmutableC11E5checkpoint,loadedinmemoryforgradientdiagnosticsonly;notfuturestudentinitialization',reference_sha256='44690ee1b148e1815d97cd25ba9aa693c8d7a582a2b13ac30d3c0676d74e365f',gradient_photos=24,photo_ids=[r['image_id'] for r in read(PREP/'cached-gradient-audit.json')['records']],measure=['PositiveversusunitdistillationprojectiongradientL2/cosine/outputmass','Coverageandteacherconflicts','Source-groupvariability','SharedlogitresponseandChair-positivepreservationrisk'],automatic_coefficient_selection=False,no_holdout_access=True,no_cache_repair_or_extra_forward_on_failure=True)
    emit(REPORT/'gradient-smoke-disabled.json',smoke)
    emit(REPORT/'cache-repair-contingency.json',dict(approved=False,currently_required=False,missing='Historicalper-photo processedpixel/tokenhashes andnew-runtimeFP16numericreplay',first_step='28-forwardgradient-smokeincludes4originalreplays;futuretrainingrequiresall59initialreplays',if_replay_fails='Stop,identifyprocessor/precisioncause;do notauto-refreshteacher',conditional_original_only_repair_limits=dict(forwards=59,updates=0,seconds=180,VRAM_bytes=6979321856,output_bytes=134217728),new_separate_approval_required=True))
    print('E1 one training proposal disabled;coefficient unset;separate28forwardgradient-smoke proposal prepared',flush=True)

if __name__=='__main__':proposals()
