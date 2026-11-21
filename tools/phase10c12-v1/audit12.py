"""Sealed C11 audit and a single training-only, endpoint-energy lambda rule."""
import math
import numpy as np
import torch
from common12 import *


def run():
    if OUT12.exists() or REPORT12.exists() or FUTURE.exists():raise FileExistsError('Fresh C12 destinations required')
    preservation,prepared,train,dev=verify_history();OUT12.mkdir(parents=True);REPORT12.mkdir(parents=True)
    source=ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/modeling_owlv2.py'
    heads=original_heads();initial=read(C11_RUN/'initialization.json');files={str(source.relative_to(ROOT)):sha(source)}
    for name,shape in zip(PROJECTION_KEYS,PROJECTION_SHAPES):
        if tuple(heads[name].shape)!=shape:raise ValueError('Verified architecture no longer matches')
    receipts=[];deltas={};projection={};previous=None;gradient=[];all_updates=[]
    for i in range(76):
        paths=list((C11_RUN/'checkpoints').glob('update-'+str(i).zfill(3)+'-*.pt'))
        if len(paths)!=1:raise ValueError('Missing/ambiguous C11 checkpoint')
        path=paths[0];receipt=read(path.with_suffix('.receipt.json'))
        if sha(path)!=receipt['sha256']:raise ValueError('C11 checkpoint checksum mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['updates']!=i or payload['trainable_keys']!=sorted(TRAIN_KEYS) or payload['frozen_sha256']!=initial['frozen_sha256'] or payload['original_backbone_sha256']!=initial['original_backbone_sha256']:raise ValueError('C11 frozen/trainable scope differs')
        if any(not torch.equal(payload['weights'][n],heads[n]) for n in CALIBRATION_KEYS):raise ValueError('C11 calibration changed')
        group=payload['optimizer']['param_groups']
        if len(group)!=1 or group[0]['param_names']!=initial['trainable_keys'] or set(group[0]['param_names'])!=TRAIN_KEYS:raise ValueError('C11 optimizer group mismatch')
        g=group[0]
        if (g['lr'],g['weight_decay'],tuple(g['betas']),g['eps'])!=(1e-5,.01,(.9,.999),1e-8):raise ValueError('C11 optimizer settings differ')
        if i and (len(payload['optimizer']['state'])!=8 or any(int(s['step'])!=i for s in payload['optimizer']['state'].values())):raise ValueError('C11 update cursor mismatch')
        drift={n:payload['weights'][n].double()-heads[n].double() for n in PROJECTION_KEYS};half_sum=.5*sum(float(t.square().sum()) for t in drift.values());deltas[i]=half_sum
        if i:
            update_path=C11_RUN/('update-'+str(i).zfill(3)+'.json');u=read(update_path);old=read(C8/'B-corrected'/update_path.name)
            if u['source_order']!=old['source_order'] or any(u['named_preclip_gradients'][n] is not None for n in CALIBRATION_KEYS):raise ValueError('C11 order/calibration-gradient mismatch')
            if any(not math.isfinite(u['named_preclip_gradients'][n]) for n in TRAIN_KEYS):raise ValueError('Nonfinite C11 gradient')
            receipts.append(receipt);all_updates.append(u)
            gradient.append(dict(update=i,epoch=u['epoch'],existing_projection_preclip_norm=math.sqrt(sum(u['named_preclip_gradients'][n]**2 for n in PROJECTION_KEYS)),preupdate_projection_distance=math.sqrt(2*previous)))
            files[str(update_path.relative_to(ROOT))]=sha(update_path)
        for n in PROJECTION_KEYS:
            if i==75:projection[n]=dict(shape=list(heads[n].shape),scalars=heads[n].numel(),original_sha256=tensor_sha(heads[n]),trained_sha256=tensor_sha(payload['weights'][n]),original_L2=float(heads[n].double().norm()),drift_L2=float(drift[n].norm()),relative_drift_L2=float(drift[n].norm()/heads[n].double().norm()),max_abs_drift=float(drift[n].abs().max()))
        files[str(path.relative_to(ROOT))]=receipt['sha256'];previous=half_sum
    # Strength selection reads only fixed optimization-photo probes and parameter states.
    # Development logits/features/metrics are not strength-calibration inputs.
    byid={r['image_id']:r for r in train};pairs=[]
    for index in range(1,25):
        records=[]
        for epoch in (0,5):
            path=C11_RUN/'probes'/('epoch-'+str(epoch))/(str(index).zfill(2)+'.json');r=read(path)
            if r['image_id'] not in byid or byid[r['image_id']]['internal_partition']!='optimization' or r['protected_violations']:raise ValueError('Strength evidence must be exact optimization-training photos')
            files[str(path.relative_to(ROOT))]=sha(path);records.append(r)
        if records[0]['image_id']!=records[1]['image_id']:raise ValueError('Paired probe identities differ')
        pairs.append(dict(image_id=records[0]['image_id'],identity_group=byid[records[0]['image_id']]['identity_group'],source_sha256=byid[records[0]['image_id']]['sha256'],initial_classification=records[0]['classification_loss'],final_classification=records[1]['classification_loss']))
    c0=float(np.mean([r['initial_classification'] for r in pairs]));c5=float(np.mean([r['final_classification'] for r in pairs]));gain=c0-c5;distance=deltas[75]
    if not math.isfinite(gain) or gain<=0 or not math.isfinite(distance) or distance<=0:raise ValueError('No defensible positive energy-matching strength; stop rather than invent lambda')
    strength=gain/distance
    for r in gradient:
        r['counterfactual_regularizer_gradient_norm']=strength*r['preupdate_projection_distance'];r['norm_ratio_to_existing']=r['counterfactual_regularizer_gradient_norm']/r['existing_projection_preclip_norm']
    last=[r for r in gradient if r['epoch']==5];last_images=[r for u in all_updates if u['epoch']==5 for r in u['images']]
    strength_record=dict(method='single_training_probe_endpoint_energy_match_v1',formula='lambda=(mean_C11_optimization_probe_classification_E0-mean_E5)/(0.5*sum_projection_squared_drift_at_C11_update75)',
        rationale='At the observed C11 endpoint, preservation costs the classification improvement on the same fixed training-photo probes. Unit ratio1 is an explicit diagnostic energy-balance design choice, not validation-selected or a claim of optimality.',
        lambda_value=strength,penalty_form='0.5*sum((W-W0)^2)+0.5*sum((b-b0)^2)',normalization='One joint half-squared sum across393728 projection scalars; no parameter-count, positive-box or category normalization',
        initial_probe_classification=c0,final_probe_classification=c5,training_probe_classification_reduction=gain,unweighted_endpoint_penalty=distance,weighted_endpoint_penalty=strength*distance,
        endpoint_regularizer_gradient_norm=strength*math.sqrt(2*distance),last_epoch_existing_projection_gradient_norm_quantiles=quantiles([r['existing_projection_preclip_norm'] for r in last]),last_epoch_counterfactual_preupdate_gradient_ratio_quantiles=quantiles([r['norm_ratio_to_existing'] for r in last]),
        all59_last_epoch_mean_classification=float(np.mean([r['classification_loss'] for r in last_images])),all59_last_epoch_mean_box=float(np.mean([r['box_loss'] for r in last_images])),
        source_photos=len(pairs),known_source_groups=len({r['identity_group'] for r in pairs}),source_role='optimization_training_only',development_metrics_used=False,development_activations_used=False,hyperparameter_search=False,number_of_proposed_strengths=1,
        limitations=['24 fixed evidence-containing training photos are a biased subset of59; endpoint energy matching is a transparent diagnostic convention, not an empirical optimum','Box-head movement can change discrete Hungarian assignment and region coverage; the observed classification gain is not a projection-only causal gain','At the historical endpoint the penalty is strong relative to residual classification loss and can reduce recall; future behavior is unmeasured','Saved gradient norms lack direction; comparison is counterfactual along C11 trajectory, not an AdamW step prediction; clipping and adaptive moments couple updates'],
        pairs=pairs,per_update_counterfactual_gradient_scales=gradient)
    save(REPORT12/'lambda-proposal.json',strength_record)
    save(REPORT12/'parameter-audit.json',dict(preservation=preservation,classification_projection=projection,frozen_calibration_hashes=initial['calibration_hashes'],calibration_unchanged_in_all76_snapshots=True,optimizer=group[0],optimizer_steps=75,trainable_keys=initial['trainable_keys'],trainable_scalars=TRAIN_COUNT,
        original_backbone_sha256=initial['original_backbone_sha256'],whole_frozen_sha256=initial['frozen_sha256'],architecture_source=str(source.relative_to(ROOT)),architecture_source_sha256=sha(source),verified_implementation='Owlv2ClassPredictionHead: dense0(768->512), normalized image/text similarity, then (similarity+logit_shift(image_features))*(ELU(logit_scale(image_features))+1). Four frozen calibration tensors are separate from dense0.',
        historical_functional_attribution=read(ROOT/'reports/phase10c11-v1/confidence-attribution.json')['functional_attribution']['candidate'],attribution_used_only_to_verify_previous_phase_not_to_select_lambda=True,no_gpu_calls=True))
    for path in (REPORT12/'lambda-proposal.json',BASE/'model.safetensors',C11_RUN/'complete.json',ROOT/'reports/phase10c11-v1/authorized-config-v2.json',ROOT/'reports/phase10c9-v1/independent-evaluation-protocol.json'):
        files[str(path.relative_to(ROOT))]=sha(path)
    save(OUT12/'input-contract.json',dict(files=files,strength_selection_scope='75 C11 optimizer records,76 parameter snapshots and48 JSON probe records on24 optimization-only photographs; no development scores/logits/features used',GPU_forwards=0,optimizer_updates=0))
    print('C12 CPU audit PASS;lambda=',strength,'unweighted endpoint penalty=',distance,flush=True)


if __name__=='__main__':run()
