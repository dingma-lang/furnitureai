"""CPU checkpoint/state audit and disabled proposal. No training executor."""
import random
import platform
import torch
from common19 import *
from telemetry19 import tensor_hash,model_fingerprint


def validate_payload(payload,cfg,initial):
    required={'binding_sha256','arm','updates','epoch','next_group','frozen_sha256','weights','optimizer','scaler','rng'}
    if set(payload)!=required or payload['arm']!='C18-source-balanced' or payload['binding_sha256']!=sha(RUN18/'binding.json') or payload['updates']!=28 or (payload['epoch'],payload['next_group'])!=(1,13):
        raise ValueError('Wrong arm, binding, committed count or accumulation cursor')
    names=read(RUN18/'initialization.json')['trainable_keys'];reference=original_heads()
    if list(payload['weights'])!=names or set(names)!=HEAD_KEYS or payload['frozen_sha256']!=initial['frozen_sha256']:
        raise ValueError('Parameter order, scope or frozen provenance mismatch')
    for k,v in payload['weights'].items():
        if v.dtype!=torch.float32 or v.shape!=reference[k].shape or v.device.type!='cpu' or not torch.isfinite(v).all():raise ValueError('Invalid head state: '+k)
    optimizer=payload['optimizer'];groups=optimizer['param_groups'];expected=initial['optimizer']['param_groups']
    if groups!=expected or len(groups)!=1 or groups[0]['params']!=list(range(12)) or set(optimizer['state'])!=set(range(12)):
        raise ValueError('AdamW options, ordering or state count mismatch')
    for index,name in enumerate(names):
        state=optimizer['state'][index]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or not torch.is_tensor(state['step']) or state['step'].numel()!=1 or not torch.isfinite(state['step']) or float(state['step'])!=28:raise ValueError('Optimizer counter/schema mismatch')
        for moment in ('exp_avg','exp_avg_sq'):
            v=state[moment]
            if v.dtype!=torch.float32 or v.shape!=payload['weights'][name].shape or not torch.isfinite(v).all() or (moment=='exp_avg_sq' and (v<0).any()):raise ValueError('Optimizer moments invalid')
    expected_scale=dict(initial['scaler'],_growth_tracker=28)
    if payload['scaler']!=expected_scale or payload['scaler']['scale']!=128:raise ValueError('AMP state differs')
    rng=payload['rng']
    if set(rng)!={'python','cpu','cuda'}:raise ValueError('Missing RNG state')
    local=random.Random();local.setstate(rng['python'])
    generator=torch.Generator(device='cpu');generator.set_state(rng['cpu'])
    if not torch.equal(generator.get_state(),rng['cpu']):raise ValueError('CPU RNG restore mismatch')
    if len(rng['cuda'])!=1 or len(initial['rng']['cuda'])!=1:raise ValueError('Single GPU RNG state required')
    for v,baseline in zip(rng['cuda'],initial['rng']['cuda']):
        if not torch.is_tensor(v) or v.dtype!=torch.uint8 or v.ndim!=1 or v.shape!=baseline.shape or v.numel()==0 or v.device.type!='cpu':raise ValueError('CUDA RNG representation changed')
    return dict(optimizer_states=12,optimizer_step=28,parameter_order=names,head_shapes={k:list(v.shape) for k,v in payload['weights'].items()},python_rng_restored_locally=True,CPU_rng_restored_locally=True,CUDA_rng_available=True,CUDA_rng_bytes=int(rng['cuda'][0].numel()),CUDA_rng_sha256=tensor_hash(rng['cuda'][0]),CUDA_restore_executed=False,scheduler_present=False,scheduler_required=False,accumulation_boundary=True,epoch_zero_based=1,next_group_zero_based=13,next_update=29)


def remaining_sequence(cfg,payload):
    cursor=payload['epoch']*15+payload['next_group']
    if cursor!=payload['updates'] or cursor!=28:raise ValueError('Cursor not a committed accumulation boundary')
    flat=[dict(update=epoch*15+index+1,epoch_zero_based=epoch,group_zero_based=index,image_ids=group) for epoch,groups in enumerate(cfg['orders']) for index,group in enumerate(groups)]
    remaining=flat[cursor:]
    if len(remaining)!=47 or [r['update'] for r in remaining]!=list(range(29,76)) or sum(len(r['image_ids']) for r in remaining)!=184:raise ValueError('Remaining budget/order mismatch')
    return remaining


def restore_cpu_fixture(payload):
    """Load actual heads/moments into an isolated shape-correct CPU fixture.

    The fixture backbone is synthetic; this tests mapping/restoration, not
    a real encoder, forward pass, GPU RNG setter or optimized GPU step.
    """
    from test_preflight10 import fixture
    from expanded import configure_heads,load_delta
    m=fixture();configure_heads(m);params={k:p for k,p in m.named_parameters() if p.requires_grad}
    if list(params)!=list(payload['weights']):raise ValueError('Restore parameter ordering differs')
    before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    load_delta(m,payload)
    o=torch.optim.AdamW(list(params.values()),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);o.load_state_dict(payload['optimizer'])
    scaler=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=2000);scaler.load_state_dict(payload['scaler'])
    if scaler.state_dict()!=payload['scaler']:raise ValueError('CPU scaler restore differs')
    for i,(k,p) in enumerate(params.items()):
        if not torch.equal(p,payload['weights'][k]):raise ValueError('Loaded weights differ')
        for field in ('step','exp_avg','exp_avg_sq'):
            if not torch.equal(o.state[p][field],payload['optimizer']['state'][i][field]):raise ValueError('Loaded moment/counter differs')
    if any(not torch.equal(dict(m.named_parameters())[n],v) for n,v in before.items()):raise ValueError('Restore changed synthetic frozen state')
    return dict(status='PASS',actual_head_tensors_loaded=12,actual_optimizer_states_loaded=12,CPU_scaler_loaded=True,synthetic_frozen_parameters_unchanged=True,forward_calls=0,optimizer_steps=0,fixture_model_fingerprint=model_fingerprint(m),scope_caveat='Shape-correct CPU head fixture,not real backbone or GPU restoration')


def audit_checkpoint():
    cfg,*_=inputs18()
    if sha(CHECKPOINT28)!=CHECKPOINT28_SHA256:raise ValueError('Checkpoint28 bytes changed')
    payload=torch.load(CHECKPOINT28,map_location='cpu',weights_only=True)
    initial_refs=list((RUN18/'checkpoints').glob('update-000-*.receipt.json'))
    if len(initial_refs)!=1:raise ValueError('Unique initial state required')
    ref=read(initial_refs[0]);p=RUN18/ref['file']
    if sha(p)!=ref['sha256']:raise ValueError('Initialization snapshot changed')
    initial=torch.load(p,map_location='cpu',weights_only=True)
    result=validate_payload(payload,cfg,initial);sequence=remaining_sequence(cfg,payload)
    worker=(ROOT/'tools/phase10c18-v1/worker18.py').read_text()
    if 'lr_scheduler' in worker:raise ValueError('Unexpected scheduler implementation needs audit')
    result.update(checkpoint=str(CHECKPOINT28.relative_to(ROOT)),checkpoint_sha256=CHECKPOINT28_SHA256,binding_sha256=sha(RUN18/'binding.json'),training_manifest_sha256=cfg['training_rows_sha256'],source_evidence_sha256=sha(SOURCE/'source-negative-evidence.json'),remaining_updates=47,remaining_photo_presentations=184,remaining_order_sha256=digest(sequence),RNG_CPU_equal_to_initial=torch.equal(payload['rng']['cpu'],initial['rng']['cpu']),RNG_CUDA_equal_to_initial=torch.equal(payload['rng']['cuda'][0],initial['rng']['cuda'][0]),CPU_fixture_restore=restore_cpu_fixture(payload),checkpoint_optimizer_step_counts_exact=True,gradient_buffers_not_required='Snapshot taken after a committed step; original next group clears .grad. Four failed-group forwards/VJPs were uncommitted and must be replayed from checkpoint28 RNG.',GPU_bitwise_replay_certified=False,limitations=['CUDA RNG bytes are present and structurally match the saved initial runtime; no GPU set/get or equality roundtrip performed in this CPU phase.','Reconstruct frozen backbone from preserved original base,not from an absent full-backbone delta.','Apply model/head/optimizer/scaler load first,then restore Python/CPU/CUDA RNG LAST before first resumed forward.','No GPU recovery executor was added or invoked. A future worker must retain original scopes/order/math and bind repaired guard/telemetry.'])
    emit19(REPORT19/'checkpoint28-audit.json',result);emit19(DATA19/'remaining-sequence.json',dict(groups=sequence,sha256=digest(sequence)))
    return result


def disabled_proposal(audit,assessment):
    if not assessment['diagnostic_completion_conditionally_justified'] or audit['CPU_fixture_restore']['status']!='PASS':
        emit19(REPORT19/'recovery-not-recommended.json',dict(approved=False,reason='No justified and reproducible diagnostic continuation'));return
    cfg,*_=inputs18();spent=read(RUN18/'launch-result.json')['wall_seconds'];seconds=int(600-spent)
    proposal=dict(schema='phase10c19-disabled-checkpoint28-recovery-v1',approved=False,enabled=False,training_authorized=False,GPU_authorized=False,Phase10D_authorized=False,diagnostic_completion_only=True,performance_gain_supported=False,source_checkpoint=str(CHECKPOINT28.relative_to(ROOT)),source_checkpoint_sha256=CHECKPOINT28_SHA256,source_binding_sha256=sha(RUN18/'binding.json'),guard_sha256=sha(HERE19/'guard19.py'),telemetry_sha256=sha(HERE19/'telemetry19.py'),recovery_helpers_sha256=sha(HERE19/'recovery19.py'),remaining_sequence=str((DATA19/'remaining-sequence.json').relative_to(ROOT)),remaining_sequence_sha256=sha(DATA19/'remaining-sequence.json'),unchanged_C18_settings=cfg['unchanged_C8_settings'],unchanged_source_method=cfg['method'],unchanged_evaluation=cfg['evaluation'],trainable_keys=cfg['trainable_keys'],calibration_trainable=True,loss_rho=.25,updates=list(range(29,76)),first_cursor=dict(epoch_zero_based=1,next_group_zero_based=13,committed_updates=28),limits=dict(additional_updates=47,cumulative_updates=75,additional_GPU_seconds=seconds,cumulative_queue_seconds=600,allocated_VRAM_bytes=MAX_VRAM17,additional_forward_passes=270,cumulative_actual_forward_passes=452,training_forward_passes=184,training_VJPs=552,probe_forward_passes=48,probe_VJPs=144,evaluation_forward_passes=38,max_new_checkpoints=47,max_checkpoint_bytes=20000000,new_allocated_bytes=int(2.5*2**30),cumulative_C18_plus_recovery_allocated_bytes=MAX_OUTPUT18),prior_consumed=dict(forwards=182,training_VJPs=345,probe_VJPs=144,updates=28,launcher_seconds=spent),forwards_accounting='452=original448 planned plus4failed/uncommitted training forwards replayed once. This is a separately requested452forward allowance,not compliance with the old448forward binding.',destination='data/open-images/experiments/phase10c19-recovery-v1/',destination_created=False,runner_implemented=False,restoration_order=['Load preserved original base; configure exact12trainable heads','Load checkpoint28 head tensors,AdamW moments,scaler,cursor','Verify frozen/original/head/scaler/order identity before inference','Restore Python/CPU/CUDA RNG LAST;verify exact device RNG roundtrip without any forward','Continue group14 of epoch2;do not replay committed updates or epoch0/1probes'],diagnostics=dict(probe_epochs=[3,5],probe_photos=24,evaluation_epochs=[3,5],evaluation_photos=19,checkpoint28_scoring=False,threshold=.15,independent_or_frozen_scoring=False,endpoint='Fixedepoch5/update75; no adaptive duration,threshold or checkpoint selection'),sole_optimization_code_change='Replace only C18 subtractive guard with C19 direct-component verification;C16 compose/normalization and all update tensors unchanged. Add failure telemetry and restart control flow.',strictness='Intended optimization-state continuation,not certified binary replay or an uninterrupted-run equivalent. Mechanical restart and telemetry are new;4uncommittedforwards repeat.',approval_requirements=['Separate explicit recovery authorization;this configuration remains disabled','Final bounded executor/code binding and CPU tests before any GPU call','GPU RNG restore and frozen/head fingerprint checks under future authorization','Respect remaining515second window;stop on any new failure,no automatic retry','No variants,new weights/data,Phase10D or production replacement'])
    if (ROOT/proposal['destination']).exists():raise FileExistsError('Recovery destination must remain absent')
    emit19(REPORT19/'recovery-disabled.json',proposal)
