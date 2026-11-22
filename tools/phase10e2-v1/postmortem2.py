"""CPU-only recovery of already-saved outputs;no model or parameter-gradient replay."""
import numpy as np
from common2 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('Postmortem is CPU-only')
    cfg,context,byid,index,refs=inputs();failure=read(RUN/'failure.json');ident=failure['state']['photo'];entry=byid[ident]
    if ident!=cfg['photo_ids'][1] or failure['state']['forwards']!=6 or failure['state']['gradient_calculations']!=7:
        raise ValueError('Unexpected failure context requires separate audit')
    from measure2 import replay
    sys.path.insert(0,str(ROOT/'tools/phase10c8-v1'))
    from common8 import load_npz
    from teacherE import load_teacher
    from objectiveE import selective_mask,objective
    import torch
    teacher=load_teacher(refs[ident],entry['row'],index['processor_contract_sha256'])
    with np.load(RUN/'adapted/02.npz',allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
    previous=load_npz(ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/probes/epoch-5/02.npz')
    numerical=replay(raw,previous)
    z=torch.tensor(raw['logits']);b=torch.tensor(raw['square_cxcywh']);tz,tb=teacher.tensors()
    mask,mask_stats=selective_mask(z,b,tz,tb,entry['row'],entry['distill_regions'],entry['distill_protected'])
    _,_,scalars=objective(z,b,entry['row'],tz,tb,entry,1.)
    selected=mask.numpy();delta=(raw['logits']-teacher.logits)[selected].astype(float)
    derivative=2*delta/len(delta) if len(delta) else delta
    sigmoid=lambda a:1/(1+np.exp(-np.asarray(a,dtype=float)))
    summary=dict(status='STOPPED_NONFINITE_GRADIENT',classification='D — Inconclusive',coefficient=None,training_ready=False,
        actual_forwards=6,actual_detector_hook_calls=failure['state']['actual_forwards'],actual_VJP_attempts=7,successful_VJP_returns=6,failed_VJPs=1,
        persisted_positive_parameter_measurements=5,successful_positive_measurement_not_persisted=1,persisted_nonzero_KD_parameter_measurements=0,
        analytic_zero_KD_derivatives=5,original_completed_photos=4,C11_completed_measurement_photos=1,C11_forward_only_partial_photos=1,
        failure_photo=ident,failure_source_group=entry['row']['identity_group'],failure_component='First nonzero unit-distillation VJP,with fixed loss scale128',
        last_raw_replay=numerical,original_model_state_unchanged=read(RUN/'state-A-before.json')==read(RUN/'state-A-after.json'),
        C11_model_state_unchanged=read(RUN/'state-B-before.json')==failure['model_fingerprint'],
        failure_scalar_recomputed_from_saved_outputs=scalars['distillation'],failed_terms=len(delta),teacher_selected_logits=teacher.logits[selected].astype(float).tolist(),student_selected_logits=raw['logits'][selected].astype(float).tolist(),
        teacher_selected_scores=sigmoid(teacher.logits[selected]).tolist(),student_selected_scores=sigmoid(raw['logits'][selected]).tolist(),
        selected_logit_delta=delta.tolist(),terminal_output_derivative_closed_form=derivative.tolist(),scaled_terminal_derivative_closed_form=(derivative*128).tolist(),
        terminal_derivative_is_not_recovered_projection_gradient=True,FP16_max_finite=float(np.finfo(np.float16).max),
        failure_scalar_and_terminal_derivative_finite=bool(np.isfinite(delta**2).all() and np.isfinite(derivative*128).all()),
        projection_norm_for_failed_photo=None,positive_norm_for_failed_photo=None,gradient_cosine_for_failed_photo=None,
        exact_nonfinite_return_tensor_not_recorded=True,exact_first_overflow_operation_not_recorded=True,
        confirmed='The finite matched-cache forward reached the scaled KD model VJP,which returned at least one nonfinite requested gradient and stopped before updates.',
        likely_explanation='Mixed-precision backward overflow under unit MSE loss scaled128;exact intermediate/parameter not established from saved artifacts.',
        not_established=['A stable nonzero distillation projection norm','Whether any fixed coefficient is safe/effective','Whether unscaled or FP32 backward would succeed','Broad seven-class confidence control','Any detection precision/recall/localization improvement'],
        box_routing='No box-route gradient in completed positive VJPs;failed KD VJP stopped before its complete finite/routing summary,so no successful nonzero KD routing measurement is claimed.',
        raw_outputs_persisted=6,completed_measurement_records=5,optimizer_constructed=False,optimizer_steps=0,model_weight_updates=0,
        forward_runtime_at_failure=failure['seconds'],launcher_wall_seconds=read(RUN/'launch-result.json')['wall_seconds'],peak_allocated_bytes=failure['peak_allocated_bytes'],peak_reserved_bytes=failure['peak_reserved_bytes'],
        protected_overlaps_in_failed_output_mask=0,failed_mask=mask_stats,additional_GPU_attempts=0)
    if not summary['original_model_state_unchanged'] or not summary['C11_model_state_unchanged']:raise ValueError('Stage model-state preservation failed')
    emit(REPORT/'failure-analysis.json',summary)
    emit(REPORT/'decision.json',dict(classification=summary['classification'],coefficient=None,training_authorized=False,training_ready=False,
        reason='The first nonzero adapted-state KD VJP was nonfinite under fixed scale128. No finite nonzero KD projection-gradient norm or direction was recovered;choosing a coefficient would invent the required evidence.',
        next_action='Do not train. Prepare a CPU-tested numerical-measurement/overflow-telemetry repair,then obtain separate approval for any further zero-update GPU attempt. No retry or alternate precision was executed.',
        new_training_configuration_prepared=False,existing_E1_training_configuration_still_disabled=True,metrics_improvement_claimed=False))
    print('E2 CPU postmortem:6forwards,7VJPattempts;finite scalar',summary['failure_scalar_recomputed_from_saved_outputs'],';coefficient remainsunset',flush=True)

if __name__=='__main__':run()
