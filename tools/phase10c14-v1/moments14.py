"""Approximate aggregate gradients from saved Adam moments,not new inference."""
import math
import torch
from common14 import *


def recover_clipped(current,previous,beta1):
    if not 0<=beta1<1:raise ValueError('Invalid Adam beta1')
    return (current.double()-beta1*previous.double())/(1-beta1)


def run():
    torch.set_num_threads(2);original=original_heads();result={}
    for name,root in [('c11',C11_RUN),('c13',RUN13)]:
        previous=torch.load(next((root/'checkpoints').glob('update-000-*.pt')),map_location='cpu',weights_only=True);records=[]
        for i in range(1,76):
            current=torch.load(next((root/'checkpoints').glob('update-'+str(i).zfill(3)+'-*.pt')),map_location='cpu',weights_only=True);update=read(root/('update-'+str(i).zfill(3)+'.json'));group=current['optimizer']['param_groups'][0];names=group['param_names'];beta=group['betas'][0]
            if set(names)!=TRAIN_KEYS or current['updates']!=i:raise ValueError('Named optimizer scope/cursor changed')
            full=math.sqrt(sum(v*v for v in update['named_preclip_gradients'].values() if v is not None));coef=min(1,1/(full+1e-6));recovered={};errors=[];data={};reference={}
            for n in PROJECTION_KEYS:
                j=names.index(n);m=current['optimizer']['state'][j]['exp_avg'];old=previous['optimizer']['state'].get(j,{}).get('exp_avg',torch.zeros_like(m));g=recover_clipped(m,old,beta)/coef;recovered[n]=g;measured=update['named_preclip_gradients'][n];errors.append(abs(float(g.norm())-measured)/max(measured,1e-20))
                displacement=previous['weights'][n].double()-original[n].double();reference[n]=displacement;reg=displacement*current['projection_preservation']['lambda_value'] if name=='c13' else torch.zeros_like(g);data[n]=g-reg
            gd=math.sqrt(sum(float(g.square().sum()) for g in data.values()));rd=math.sqrt(sum(float(g.square().sum()) for g in reference.values()));dot=sum(float((data[n]*reference[n]).sum()) for n in PROJECTION_KEYS)
            records.append(dict(update=i,recovered_projection_data_gradient_norm=gd,recovered_gradient_cosine_with_pretrained_displacement=dot/(gd*rd) if gd*rd else None,max_relative_recovered_vs_recorded_norm_error=max(errors),recorded_box_head_gradient_norm=math.sqrt(sum(v*v for n,v in update['named_preclip_gradients'].items() if n.startswith('box_head.')))))
            previous=current
        result[name]=dict(records=records,max_relative_norm_error=max(r['max_relative_recovered_vs_recorded_norm_error'] for r in records),method='g_clipped approximately equals (Adam_exp_avg_t-beta1*exp_avg_previous)/(1-beta1);undo clip factor from recorded preclip norms;subtract analytic L2-SP gradient forC13',limitations=['FP32 Adam moment rounding and FP32 clip implementation prevent exact signed gradient recovery','Aggregate across an optimizer group only;cannot separate per-image or positive/native/source contributions','AdamW weight decay is decoupled and is not a loss gradient','No activation tensors or model forward inferred from moments'])
    save(REPORT14/'aggregate-gradient-recovery.json',dict(models=result,model_forwards=0,optimizer_updates=0,approximate_not_original_saved_vectors=True))
    print('Aggregate Adam-gradient CPU recovery complete;no per-image decomposition invented',flush=True)


if __name__=='__main__':run()
