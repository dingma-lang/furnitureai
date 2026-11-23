"""One disabled candidate derived from training gradients,never development metrics."""
from common4 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only proposal')
    decision=read(REPORT/'decision.json');evidence=read(REPORT/'gradient-findings.json');cfg,*_=inputs()
    if not decision['training_ready'] or decision['coefficient'] is None:raise ValueError('Finite complete recovery required')
    files=dict(cfg['files'])
    for path in (HERE/'future4.py',HERE/'precision4.py',HERE/'proposal4.py',REPORT/'gradient-findings.json',REPORT/'decision.json',RUN/'complete.json'):
        files[str(path.relative_to(ROOT))]=sha(path)
    plan=dict(schema='phase10e4-one-disabled-teacher-distillation-training',approved=False,enabled=False,training_authorized=False,
         coefficient=decision['coefficient'],coefficient_formula=evidence['formula'],configurations=1,
         destination='data/open-images/experiments/phase10e-distillation-fp32-v1',
         limits=dict(epochs=5,updates=75,seconds=600,VRAM_bytes=6979321856,output_bytes=2147483648,forwards=373,head_contractions=354),
         training_photos=59,known_source_groups=56,positive_boxes=182,classes=list(CLASSES),
         initialization='Original immutable OWLv2 only;C11 checkpoint diagnostic reference only',base_sha256=BASE_SHA,
         trainable_keys=read(ROOT/'reports/phase10e1-v1/gpu-training-disabled.json')['trainable_keys'],
         frozen='Calibration shift/scale,vision/text/shared feature representations,objectness,buffers,derived grid;all other410parameters',
         learning_rate=1e-5,optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01),
         seed=6106,batch_size=1,gradient_accumulation=4,last_group_size=3,gradient_clip_norm=1.,
         precision=dict(forward='FP16 CUDA autocast',loss='FP32 unchanged C11/C8corrected base+fixedcoefficient*selectiveMSE',
                        classification_backward='E3 FP32 observed-node adjoint for ALL classification terms;exact observed forward;not legacy rounded FP16 adjoint',
                        box_backward='Native FP16 box branch with FP32 loss/master gradients',loss_scale=1.,GradScaler=False,TF32=False,
                        implementation='tools/phase10e4-v1/precision4.py:combined',
                        classification_source='tools/phase10e3-v1/stable3.py',
                        CPU_combined_verified=True,GPU_positive_KD_components_verified=True,GPU_native_box_verified_photos=1,
                        GPU_full_combined_training_backward_executed=False),
         numerical_differences_from_C11='Widened classification adjoints and unscaled native box backward instead of original GradScaler128. Objective/data/scope unchanged;comparison is not a pure loss-only controlled ablation.',
         mandatory_preupdate_gate='59original cache correspondences,then finite combined eight-parameter VJP on first original forward before optimizer construction;stop on failure,do not retry',
         training_orders_sha256=sha(PREP/'training-context.json'),teacher_index_sha256=sha(PREP/'teacher-index.json'),
         evaluation=dict(rows=19,partition='Existing exposed training-derived internal development only',precision='FP32 as historical protocol',
                         threshold=.15,IoUs=[.5,.75],official_or_reserved_scoring=False,
                         coefficient_or_threshold_tuning=False,
                         required=['Precision/recall at0.15,precision at matched recall,TP/FP/FN,AP50/AP75',
                                   'ChairFP/misses,Sofa/Bed regressions,localization andduplicates',
                                   'Seven-class shared logit drift,teacher/student disagreements',
                                   'Separate unmatched predictions from confirmed errors under incomplete annotations'],
                         baselines=['Original immutable OWLv2','Existing C11 calibration-frozen adapted model','Other saved adapted results as contextual comparisons'],
                         independent_holdout_used=False,predictions_atomic_before_metrics=True),
         success_target='>=60%precision with recall meaningfully above68.85% under identical diagnostic protocol;report all category regressions and uncertainty',
         long_term_target='80%precision/90%recall on independent completely annotated seven-class rooms;not established by19exposed partially annotated photographs',
         checkpoints='Five immutable epoch checkpoints with head weights,optimizer,RNG,update cursor,precision andcoefficient;no original/previous checkpoint replacement',
         stop_conditions=['Nonfinite loss/gradient/weight,invalid supervision orparameter scope','Teacher/source/coordinate/hash mismatch',
                          'Any resource/update boundary orfailed combined zero-update smoke'],automatic_retry=False,
         estimated_runtime_seconds=[180,450],estimated_peak_allocated_GiB=[2,4],estimates_not_measurements=True,
         remaining_preupdate_condition='Full combined GPU smoke is intentionally not run in E4;it is included before the first update in the separately authorized future queue.',
         files=files)
    emit(REPORT/'training-disabled.json',plan)
    print('One disabled75-update proposal pinned;coefficient',plan['coefficient'],';no training authorized',flush=True)


if __name__=='__main__':run()
