"""CPU checkpoint/data/order/telemetry audit. No model forward or updates."""
from collections import Counter
from common5 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only audit')
    import torch
    from scopeE import original_heads,HEAD_KEYS,TRAIN_KEYS,CALIBRATION_KEYS
    cfg,ctx,_=inputs();done=read(RUN/'complete.json');original=original_heads();checkpoints=[]
    old=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt'
    previous=torch.load(old,map_location='cpu',weights_only=True)['weights']
    final_projection={};all_changes={}
    for epoch in range(1,6):
        ref=read(RUN/'checkpoints'/f'epoch-{epoch}.json')['checkpoint'];path=RUN/ref['file']
        if sha(path)!=ref['sha256']:raise ValueError('Checkpoint checksum changed')
        p=torch.load(path,map_location='cpu',weights_only=True)
        if p['updates']!=15*epoch or set(p['weights'])!=HEAD_KEYS or set(p['trainable_keys'])!=TRAIN_KEYS or len(p['optimizer']['state'])!=8:
            raise ValueError('Checkpoint schema/update/scope invalid')
        if p['coefficient']!=COEFFICIENT or any(int(s['step'])!=15*epoch for s in p['optimizer']['state'].values()):raise ValueError('Coefficient or optimizer counter changed')
        if any(not torch.equal(p['weights'][n],original[n]) for n in CALIBRATION_KEYS):raise ValueError('Frozen calibration changed')
        if p['immutable_state']!=done['initial_frozen_sha256']:raise ValueError('Frozen state binding changed')
        checkpoints.append(dict(epoch=epoch,file=ref['file'],sha256=ref['sha256'],updates=p['updates']))
        if epoch==5:
            for n in sorted(TRAIN_KEYS):
                d=(p['weights'][n]-original[n]).double();base=float(original[n].double().norm())
                all_changes[n]=dict(delta_L2=float(d.norm()),maximum_absolute=float(d.abs().max()),relative_L2=float(d.norm())/base if base else None,
                                    difference_from_C11_L2=float((p['weights'][n]-previous[n]).double().norm()))
                if n.startswith('class_head.'):final_projection[n]=all_changes[n]
    train=[e['row'] for e in ctx['entries']];dev=ctx['development'];tids={r['image_id'] for r in train};dids={r['image_id'] for r in dev}
    forward_records=[read(p) for p in sorted((RUN/'telemetry').glob('forward-*.json'))]
    pre=[r for r in forward_records if r['phase']=='preupdate-gate'];tr=[r for r in forward_records if r['phase']=='training'];ev=[r for r in forward_records if r['phase']=='evaluation']
    if len(pre)!=59 or len(tr)!=295 or len(ev)!=19 or {r['photo'] for r in pre+tr}!=tids or {r['photo'] for r in ev}!=dids:
        raise ValueError('Actual executed source memberships invalid')
    if any(v!=5 for v in Counter(r['photo'] for r in tr).values()):raise ValueError('Training presentation frequency changed')
    updates=[read(RUN/'updates'/f'{i:03}.json') for i in range(1,76)];images=[r for u in updates for r in u['images']]
    if any(u['coefficient']!=COEFFICIENT or u['loss_scale']!=1. for u in updates):raise ValueError('Actual coefficient/scaling drift')
    for r in images:
        if r['stats']['coefficient']!=COEFFICIENT or r['weighted_KD_loss']!=COEFFICIENT*r['stats']['distillation'] or r['stats']['base']['protected_violations']:
            raise ValueError('Actual loss/positive protections changed')
    if (RUN/'gate/complete.json').stat().st_mtime_ns>=(RUN/'updates/001.intent.json').stat().st_mtime_ns:raise ValueError('Optimizer update intent preceded safety gate')
    if done['initial_frozen_sha256']!=done['final_frozen_sha256'] or sha(BASE/'model.safetensors')!=BASE_SHA or torch.cuda.is_initialized():raise ValueError('Preservation/CPU boundary failed')
    emit(REPORT/'integrity.json',dict(status='PASS',checkpoints=checkpoints,calibration_tensors_unchanged=4,
        frozen_sha256=done['initial_frozen_sha256'],original_sha256=BASE_SHA,training_photos=59,training_presentations=295,
        internal_development_photos=19,actual_partition_membership_verified=True,gate_before_first_update=True,
        coefficient_constant_all295_presentations=True,loss_scale_constant_all75_updates=True,
        trainable_changes=all_changes,projection_changes=final_projection,
        nonzero_KD_VJPs=sum(r['gradients']['method']=='actual FP32 observed-node VJP' for r in images),
        positive_KD_gradient_conflicts=sum(r['gradients']['KD_positive_cosine'] is not None and r['gradients']['KD_positive_cosine']<0 for r in images),
        above_half_bound=sum(r['gradients']['above_recovery_half_bound'] for r in images),KD_dominance_flags=sum(r['gradients']['KD_dominant'] for r in images),
        new_GPU_calls=0,new_optimizer_updates=0,original_and_historical_checkpoints_modified=False))
    print('CPU integrity PASS:5checkpoints,295presentations,75exact updates,4frozen calibration tensors',flush=True)

if __name__=='__main__':run()
