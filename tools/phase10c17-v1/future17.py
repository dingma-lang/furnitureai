"""Prepare exactly one disabled trial only after supported completed smoke."""
import copy
from common17 import *
from spec16 import validate_spec


def run():
    result=read(REPORT17/'results.json')
    if result['decision']!='A_QUALIFIED_PROMISING_FOR_ONE_DIAGNOSTIC_TRAINING_TRIAL' or result['cap_violations'] or result['model_updates'] or result['methods']['C_post_cap']['classification_vs_positive_reversals']:raise PermissionError('Smoke does not support preparing a training trial')
    cfg=copy.deepcopy(read(REPORT16/'future-experiment-disabled.json'));validate_spec(cfg)
    cfg.update(schema='phase10c17-disabled-balanced-training-trial-v1',training_authorized=False,GPU_authorized=False,Phase10D_authorized=False,arm_id='C17-source-balanced-diagnostic',smoke_evidence_sha256=sha(REPORT17/'results.json'),smoke_configuration_sha256=sha(RUN17/'authorized-config.json'),baseline_configuration_sha256=sha(C8/'binding.json'),destination='data/open-images/experiments/phase10c17-balanced-training-v1/',training_runner_implemented=False,approval_required='Separate explicit training authorization plus immutable binding of final trainer/config;no automatic follow-on')
    cfg['integration']['metadata_adapter']='Use unchanged C16.compose on original accumulation buffers;C17 measure17.finish_accumulator records cap scale and AMP scale separately'
    cfg['integration']['metadata_adapter_sha256']=sha(HERE17/'measure17.py');cfg['integration']['positive_buffer_preservation']='Check bitwise unchanged positive/base box buffers before and after source composition;source cap on all6class-head vectors after actual accumulation group'
    cfg['limits'].update(max_model_forwards=467,max_training_VJP_calls=885,max_diagnostic_VJP_calls=480,max_new_allocated_output_bytes=4*2**30,another_arm_authorized=False)
    cfg['diagnostic_cautions']=dict(cap_activated_in_smoke=False,initialization_source_ratio_very_small=True,source_all_below_confidence015=True,no_independent_accuracy_established=True,only_three_classes_have_active_training_positives=True,smoke_does_not_predict_learning_trajectory=True)
    cfg['predeclared_endpoint']='Final epoch5/update75;milestones0/1/3/5 descriptive,no adaptive best-epoch or threshold selection'
    cfg['success_assessment']='Compare with immutable C8 B using identical19-photo historical partial-label diagnostics:precision at matched recall,AP50/AP75,ChairFP/misses,Sofa/Bed regressions and source-gradient evolution. No independent accuracy claim;do not achieve precision by discarding recall.'
    cfg['stop_conditions']=['Nonfinite loss/gradients','Source mask or positive/ignored target violation','Cap violation or altered positive/base-box buffers','Unexpected trainable scope or backbone/objectness gradients','Near-zero verified-positive anchor with nonzero source gradient','Uncertain optimizer update or exhausted update/time/memory/storage budget']
    cfg['estimate_not_measurement']=dict(runtime_seconds=[250,500],peak_allocated_GiB=[2,4],checkpoint_storage_GiB=[1.5,2],requires_GPU_validation_after_final_trainer_implementation=True)
    validate_spec(cfg)
    if (ROOT/cfg['destination']).exists():raise FileExistsError('Future training destination must remain absent')
    emit17(REPORT17/'future-training-disabled.json',cfg);emit17(REPORT17/'next-approval-request.json',dict(approved=False,training_authorized=False,scope='One diagnostic head-only trial,original pretrained initialization,C16 source-aware balancing only intentional objective change',baseline='Immutable C8 corrected Arm B',max_epochs=5,max_optimizer_updates=75,learning_rate=1e-5,seed=6106,training_photos=59,known_source_groups=56,positive_boxes=182,source_regions=41,all12heads_trainable=True,calibration_trainable=True,backbone_and_objectness_frozen=True,L2SP=False,max_GPU_seconds_including_evaluation=600,max_allocated_VRAM_bytes=6979321856,max_model_forwards=467,max_output_bytes=4*2**30,source_policy_rho=.25,configuration=str((REPORT17/'future-training-disabled.json').relative_to(ROOT)),configuration_sha256=sha(REPORT17/'future-training-disabled.json'),destination=cfg['destination'],destination_created=False,prerequisites='Separate explicit approval,final trainer implementation/tests and immutable code/config binding;no rerun of current GPU smoke without new authorization',evaluation='19already-exposed internal diagnostic photos only under unchanged C8 definitions,no frozen/reserved/official scoring',automatic_follow_on=False))
    print('One future75-update diagnostic trial configured DISABLED;no training directory or launcher created',flush=True)


if __name__=='__main__':run()
