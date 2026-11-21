"""CPU-only post-run snapshot, budget, reference and preservation checks."""
import importlib.util,math
import torch
from common13 import *


def run():
    done=read(FUTURE/'complete.json');queue=read(FUTURE/'queue-complete.json');cfg=read(FUTURE/'authorized-config.json');initial=read(FUTURE/'initialization.json')
    from config12 import require_training_approval
    require_training_approval(cfg,read(FUTURE/'approval-record.json'))
    if (done['updates'],done['epochs'],done['forwards'],done['training_backwards'])!=(75,5,467,295):raise ValueError('Unexpected execution counts')
    if max(done['seconds'],queue['seconds'])>600 or max(done['peak_allocated_bytes'],done['peak_reserved_bytes'],done['peak_device_used_bytes'])>6.5*2**30:raise ValueError('Resource boundary exceeded')
    original=original_heads();calibration={n:tensor_sha(original[n]) for n in sorted(CALIBRATION_KEYS)};ref={n:tensor_sha(original[n]) for n in PROJECTION_KEYS}
    control_initial=read(C11_RUN/'initialization.json')
    if initial!=control_initial:raise ValueError('C11 initialization/scope hashes differ')
    init=read(FUTURE/'initial-comparability.json')
    if len(init['records'])!=43 or init['updates_before_check']!=0 or any(not v['bit_identical'] for r in init['records'] for v in r['arrays'].values()):raise ValueError('Initial actual query arrays not bit-identical')
    if len(list((FUTURE/'checkpoints').glob('*.pt')))!=76:raise ValueError('Expected precisely76 versioned snapshots')
    payloads=[];displacements=[]
    for i in range(76):
        paths=list((FUTURE/'checkpoints').glob('update-'+str(i).zfill(3)+'-*.pt'))
        if len(paths)!=1:raise ValueError('Ambiguous/missing checkpoint')
        p=paths[0];receipt=read(p.with_suffix('.receipt.json'));data=torch.load(p,map_location='cpu',weights_only=True)
        if receipt['updates']!=i or sha(p)!=receipt['sha256'] or data['updates']!=i or data['binding_sha256']!=sha(FUTURE/'authorized-config.json'):raise ValueError('Snapshot hash/config/cursor mismatch')
        if data['trainable_keys']!=sorted(TRAIN_KEYS) or data['calibration_hashes']!=calibration or data['frozen_sha256']!=initial['frozen_sha256'] or data['original_backbone_sha256']!=initial['original_backbone_sha256']:raise ValueError('Snapshot parameter scope mismatch')
        if any(not torch.equal(data['weights'][n],original[n]) for n in CALIBRATION_KEYS) or any(not torch.isfinite(v).all() for v in data['weights'].values()):raise ValueError('Calibration changed or nonfinite weight')
        state=data['projection_preservation']
        if not state['enabled'] or state['lambda_value']!=cfg['projection_preservation']['lambda_value'] or state['reference_hashes']!=ref or state['selected_keys']!=list(PROJECTION_KEYS) or state!=done['projection_preservation']:raise ValueError('Original reference/objective binding differs')
        opt=data['optimizer'];groups=opt['param_groups'];g=groups[0]
        if len(groups)!=1 or g['param_names']!=initial['trainable_keys'] or set(g['param_names'])!=TRAIN_KEYS or g['params']!=list(range(8)):raise ValueError('Optimizer contains unintended parameters')
        if (g['lr'],g['weight_decay'],tuple(g['betas']),g['eps'])!=(1e-5,.01,(.9,.999),1e-8):raise ValueError('Optimizer settings changed')
        if (not i and opt['state']) or (i and (set(opt['state'])!=set(range(8)) or any(int(v['step'])!=i for v in opt['state'].values()))):raise ValueError('Optimizer update accounting differs')
        displacement={n:float((data['weights'][n].double()-original[n].double()).norm()) for n in PROJECTION_KEYS};distance2=sum(v*v for v in displacement.values())
        displacements.append(dict(update=i,parameter_displacements=displacement,joint_L2=math.sqrt(distance2),unweighted_LSP=.5*distance2,weighted_LSP=.5*distance2*state['lambda_value']))
        if i:
            u=read(FUTURE/('update-'+str(i).zfill(3)+'.json'));control=read(C11_RUN/('update-'+str(i).zfill(3)+'.json'))
            if u['receipt']!=receipt or u['source_order']!=control['source_order']:raise ValueError('Training order or snapshot receipt differs')
            if any(u['named_preclip_gradients'][n] is not None for n in CALIBRATION_KEYS) or any(not math.isfinite(u['named_preclip_gradients'][n]) for n in TRAIN_KEYS):raise ValueError('Gradient scope/nonfinite violation')
            previous=displacements[i-1]['unweighted_LSP']
            for r in u['images']:
                if r['protected_violations'] or not r['existing_supervision_and_box_loss_unchanged'] or r['logging_only_original_loss_returned']:raise ValueError('Mask/objective provenance failure')
                if not math.isclose(r['projection_penalty'],previous,abs_tol=1e-7,rel_tol=1e-5) or not math.isclose(r['weighted_projection_penalty'],previous*state['lambda_value'],abs_tol=1e-6,rel_tol=1e-5):raise ValueError('Incorrect L2-SP normalization/coefficient')
                if not math.isclose(r['total_loss'],r['classification_loss']+r['box_loss']+r['weighted_projection_penalty'],abs_tol=2e-6,rel_tol=1e-5):raise ValueError('Changed total loss definition')
            if i==1:
                old_paths=list((C11_RUN/'checkpoints').glob('update-001-*.pt'));old=torch.load(old_paths[0],map_location='cpu',weights_only=True)
                if any(not torch.equal(old['weights'][n],data['weights'][n]) for n in HEAD_KEYS):raise ValueError('First update differs despite zero initial regularizer')
        payloads.append(dict(update=i,sha256=receipt['sha256'],calibration_hashes=calibration,projection_reference_hashes=ref))
        del data
    verify_inputs13();old=read(ROOT/'reports/phase10c12-v1/artifact-seal.json');seal=ROOT/old['private_seal']
    if sha(seal)!=old['private_seal_sha256']:raise ValueError('C12 seal changed')
    c12private=verify_seal(seal);c12public=verify_seal(ROOT/'reports/phase10c12-v1/SHA256SUMS');c11=verify_history()[0];c8=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));spec=importlib.util.spec_from_file_location('c13_historical_verify',ROOT/'tools/phase10c-v1/verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);historical=v.preservation()
    save(OUT13/'checkpoint-verification.json',dict(status='PASS',receipts=payloads,displacements=displacements))
    save(REPORT13/'integrity.json',dict(status='PASS',snapshots=76,updates=75,initial_queries_and_first_update_bit_identical_to_C11=True,calibration_hashes=calibration,original_projection_reference_hashes=ref,all_snapshots_have_unchanged_reference_and_calibration_hashes=True,optimizer_exact_eight_tensors=True,training_order_identical_to_C11=True,source_mask_violations=0,displacement_by_update=displacements))
    save(REPORT13/'preservation.json',dict(status='PASS',C12_public_files=c12public,C12_private_files=c12private,C11=c11,C8_private_files=c8,historical=historical,new_downloads=0,reserved_or_frozen_scored=False,Mac_project_modified=False,Phase10D=False))
    print('C13 integrity PASS:76 snapshots,75 updates,immutable references/calibration,matched C11 order/first update;historical files preserved',flush=True)


if __name__=='__main__':run()
