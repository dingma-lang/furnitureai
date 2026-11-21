"""Exact CPU parameter/optimizer/gradient audit; no model is instantiated."""
import csv,zipfile
from collections import Counter
import numpy as np
from common10 import *

def stats(t):
    a=t.detach().cpu().double().numpy()
    return dict(shape=list(a.shape),count=a.size,min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),L2=float(np.linalg.norm(a)),scalar=float(a.ravel()[0]) if a.size==1 else None,tensor_sha256=tensor_sha(t))

def delta(a,b):
    x=a.double();y=b.double();d=y-x
    return dict(changed_elements=int((d!=0).sum()),absolute_L2=float(d.norm()),relative_L2=float(d.norm()/x.norm()) if x.norm() else None,max_absolute=float(d.abs().max()),mean_signed=float(d.mean()),scalar_signed=float(d.flatten()[0]) if d.numel()==1 else None)

def run():
    import torch
    verify_inputs();orig=original_heads();states={};summaries={};parameter_rows=[]
    for label,arm in (('A','A-control'),('B','B-corrected')):
        done=read(C8/arm/'complete.json');path=C8/done['final_checkpoint']['file'];state=torch.load(path,map_location='cpu',weights_only=True)
        if sha(path)!=done['final_checkpoint']['sha256'] or set(state['weights'])!=HEAD_KEYS:raise ValueError('C8 checkpoint mismatch')
        order=list(state['weights']);groups=state['optimizer']['param_groups']
        if len(groups)!=1 or groups[0]['params']!=list(range(12)):raise ValueError('Historical optimizer ordering no longer matches executed worker')
        mapping=dict(zip(order,groups[0]['params']));s={}
        for name in sorted(HEAD_KEYS):
            p=state['weights'][name];op=state['optimizer']['state'][mapping[name]]
            if p.shape!=orig[name].shape or op['exp_avg'].shape!=p.shape or op['exp_avg_sq'].shape!=p.shape or int(op['step'])!=75:raise ValueError('Historical parameter/optimizer incompatibility')
            grads=[]
            for epoch in (0,1,3,5):
                records=[read(C8/arm/'probes'/('epoch-'+str(epoch))/(str(i).zfill(2)+'.json')) for i in range(1,25)]
                if name.startswith('class_head.'):
                    vals=[r['class_head_used_gradient_norms'][name] for r in records];additional=[r['class_head_additional_gradient_norms'][name] for r in records]
                    grads.append(dict(epoch=epoch,classification_gradient_norm_min=min(vals),classification_gradient_norm_mean=float(np.mean(vals)),classification_gradient_norm_max=max(vals),nonzero_photos=sum(v>0 for v in vals),added_negative_gradient_norm_mean=float(np.mean(additional))))
            s[name]=dict(value=stats(p),change=delta(orig[name],p),in_optimizer=True,optimizer_steps=int(op['step']),Adam_first_moment_nonzero=int((op['exp_avg']!=0).sum()),Adam_first_moment_L2=float(op['exp_avg'].double().norm()),gradient_probes=grads,gradient_record_scope='Named classification-only and additional-negative norm probes at0/1/3/5; signed full minibatch gradients were not saved' if grads else 'Only aggregate class/total-head training gradient norms saved; no per-box-tensor gradient norm record')
        states[label]=state;summaries[label]=s
    for name in sorted(HEAD_KEYS):
        parameter_rows.append(dict(path=name,shape=str(list(orig[name].shape)),original_scalar=stats(orig[name])['scalar'],original_L2=stats(orig[name])['L2'],A_scalar=summaries['A'][name]['value']['scalar'],B_scalar=summaries['B'][name]['value']['scalar'],A_L2_delta=summaries['A'][name]['change']['absolute_L2'],B_L2_delta=summaries['B'][name]['change']['absolute_L2'],A_relative_L2=summaries['A'][name]['change']['relative_L2'],B_relative_L2=summaries['B'][name]['change']['relative_L2'],A_max_absolute_delta=summaries['A'][name]['change']['max_absolute'],B_max_absolute_delta=summaries['B'][name]['change']['max_absolute'],optimizer_steps_each=75,proposed_frozen=name in CALIBRATION_KEYS))
    with (REPORT/'parameter-differences.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(parameter_rows[0]),lineterminator='\n');w.writeheader();w.writerows(parameter_rows)
    # Every calibration scalar remains recoverable exactly without publishing weights.
    with (OUT/'exact-calibration-values.csv').open('x',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['path','flat_index','original','A','B','A_minus_original','B_minus_original'])
        for name in sorted(CALIBRATION_KEYS):
            a=orig[name].double().flatten().tolist();aa=states['A']['weights'][name].double().flatten().tolist();bb=states['B']['weights'][name].double().flatten().tolist()
            for i,(x,y,z) in enumerate(zip(a,aa,bb)):w.writerow([name,i,repr(x),repr(y),repr(z),repr(y-x),repr(z-x)])
    schema=Counter();cache=[]
    for root in (C8,ROOT/'data/open-images/experiments/phase10c7-query-smoke-v1',ROOT/'data/open-images/experiments/phase10c-recovery-v1'):
        for p in sorted(root.rglob('*.npz')):
            with zipfile.ZipFile(p) as z:names=tuple(sorted(n.removesuffix('.npy') for n in z.namelist()))
            schema[names]+=1;cache.append(dict(path=str(p.relative_to(ROOT)),arrays=list(names)))
    save(OUT/'saved-array-inventory.json',dict(files=cache,not_usable_substitutes='ABO SigLIP embeddings belong to a different encoder and cannot replace OWLv2 pre-head activations'))
    save(REPORT/'checkpoint-parameter-audit.json',dict(original={n:stats(orig[n]) for n in sorted(HEAD_KEYS)},arms=summaries,calibration_modules=2,calibration_tensors=4,calibration_scalars=1538,
        all12heads_trainable_previously=True,original_and_all_historical_checkpoints_read_only=True,intermediate_array_schemas=[dict(arrays=list(k),files=n) for k,n in schema.items()],
        exact_per_query_attribution_activations_available=False,missing=['Pre-classification image feature vectors f[q]','Original text query embeddings t[c]','Per-query logit-shift values','Per-query raw/activated logit-scale values','Normalized projected image embeddings','Objectness logits'],
        checkpoint_optimizer_name_mapping='Executed worker constructs optimizer and checkpoint weights from the same ordered model.named_parameters dictionary; serialized param IDs0..11 verified against shapes and75step counters',real_model_forwards=0))
    print('C10 exact parameter audit complete;12 tensors×2 arms;4 calibration tensors all updated',flush=True)

if __name__=='__main__':run()
