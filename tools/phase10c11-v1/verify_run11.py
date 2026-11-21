"""Read-only, CPU-only verification of the one completed C11 execution."""
import importlib.util
import math
import sys
import torch
from common11 import *


def run():
    done=read(FUTURE/'complete.json');initial=read(FUTURE/'initialization.json')
    if (done['updates'],done['epochs'],done['forwards'])!=(75,5,467):raise ValueError('Execution budget/count mismatch')
    queue=read(FUTURE/'queue-complete.json')
    if done['seconds']>600 or queue['seconds']>600 or max(done['peak_allocated_bytes'],done['peak_device_used_bytes'])>6.5*2**30:raise ValueError('Resource limit violation')
    if not read(FUTURE/'initial-comparability.json')['all_arrays_bit_identical']:raise ValueError('Initial predictions not bit-identical')
    original=original_heads();hashes={n:tensor_sha(original[n]) for n in sorted(CALIBRATION_KEYS)}
    if initial['calibration_hashes']!=hashes:raise ValueError('Calibration initialization differs')
    # Keep actual tensor values private. Public reports contain hashes only, never weights.
    save(OUT11/'calibration-original-values.json',dict(values={n:original[n].tolist() for n in sorted(CALIBRATION_KEYS)},sha256=hashes,encoding='Exact FP32 values as round-trippable JSON numbers; present unchanged in all76 full-head snapshots'))
    checkpoints=sorted((FUTURE/'checkpoints').glob('*.pt'))
    if len(checkpoints)!=76:raise ValueError('Expected precisely initialization plus75 updates')
    receipts=[]
    for i in range(76):
        matches=list((FUTURE/'checkpoints').glob('update-'+str(i).zfill(3)+'-*.pt'))
        if len(matches)!=1:raise ValueError('Missing/ambiguous checkpoint cursor')
        path=matches[0];receipt=read(path.with_suffix('.receipt.json'))
        if sha(path)!=receipt['sha256'] or receipt['updates']!=i:raise ValueError('Checkpoint receipt/hash mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['updates']!=i or payload['trainable_keys']!=sorted(TRAIN_KEYS) or set(payload['weights'])!=HEAD_KEYS:raise ValueError('Snapshot scope mismatch')
        if payload['calibration_hashes']!=hashes or any(not torch.equal(payload['weights'][n],original[n]) for n in CALIBRATION_KEYS):raise ValueError('Calibration changed at update '+str(i))
        if any(tensor_sha(payload['weights'][n])!=hashes[n] for n in CALIBRATION_KEYS):raise ValueError('Tensor hash mismatch')
        if payload['frozen_sha256']!=initial['frozen_sha256'] or payload['original_backbone_sha256']!=initial['original_backbone_sha256']:raise ValueError('Frozen parameter seal mismatch')
        if any(not torch.isfinite(t).all() for t in payload['weights'].values()):raise ValueError('Nonfinite head weights')
        opt=payload['optimizer'];groups=opt['param_groups']
        if len(groups)!=1 or set(groups[0]['param_names'])!=TRAIN_KEYS or groups[0]['params']!=list(range(8)):raise ValueError('Calibration entered optimizer')
        g=groups[0]
        if (g['lr'],g['weight_decay'],tuple(g['betas']),g['eps'])!=(1e-5,.01,(.9,.999),1e-8):raise ValueError('Optimizer settings changed')
        if (not i and opt['state']) or (i and (set(opt['state'])!=set(range(8)) or any(int(v['step'])!=i for v in opt['state'].values()))):raise ValueError('Optimizer step mismatch')
        if i:
            update=read(FUTURE/('update-'+str(i).zfill(3)+'.json'));old=read(C8/'B-corrected'/('update-'+str(i).zfill(3)+'.json'))
            if update['receipt']!=receipt or update['source_order']!=old['source_order']:raise ValueError('C8B training order differs')
            norms=update['named_preclip_gradients']
            if any(norms[n] is not None for n in CALIBRATION_KEYS) or any(not math.isfinite(norms[n]) for n in TRAIN_KEYS):raise ValueError('Gradient scope/finiteness violation')
            if any(r['protected_violations'] or not r['logging_only_original_loss_returned'] for r in update['images']):raise ValueError('Mask/logging integrity violation')
        receipts.append(dict(updates=i,sha256=receipt['sha256'],calibration_hashes=hashes))
        del payload
    verify_inputs()
    c10public=verify_seal(ROOT/'reports/phase10c10-v1/SHA256SUMS');seal=read(ROOT/'reports/phase10c10-v1/artifact-seal.json');p=ROOT/seal['private_seal']
    if sha(p)!=seal['private_seal_sha256']:raise ValueError('C10 seal modified')
    c10private=verify_seal(p);c8private=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('historical_preservation11',ROOT/'tools/phase10c-v1/verify.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    historical=module.preservation()
    save(REPORT11/'calibration-verification.json',dict(status='PASS',calibration_hashes=hashes,snapshots_checked=76,committed_updates=75,original_tensor_equality_at_every_snapshot=True,calibration_gradients_None_at_every_update=True,optimizer_contains_only_eight_permitted_tensors=True,training_order_identical_to_C8B=True,all_frozen_parameter_seals_consistent=True,initial_prediction_arrays_bit_identical=True,private_exact_values_path=str((OUT11/'calibration-original-values.json').relative_to(ROOT)),private_exact_values_sha256=sha(OUT11/'calibration-original-values.json'),checkpoint_receipts=receipts))
    save(REPORT11/'preservation.json',dict(status='PASS',historical=historical,C10_public_files=c10public,C10_private_files=c10private,C8_private_files=c8private,original_checkpoint_sha256=sha(BASE/'model.safetensors'),training_and_development_source_hashes_checked=78,training_or_inference_in_this_verifier=0,phase10D_activated=False))
    print('C11 PASS:76 snapshots,75 identically ordered updates,4 immutable calibration tensors;historical seals intact',flush=True)


if __name__=='__main__':run()
