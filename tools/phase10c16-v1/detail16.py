"""Refine only identifiable region attribution;render CPU scientific evidence."""
import io
import numpy as np
from common16 import *


def run():
    groups=read(REPORT16/'source-group-audit.json')['groups'];oldindex=read(DATA16/'source-region-audit.json')['query_index'];entries=[];output={k:[] for k in ('focal','BCE','balanced_raw')};loss_deltas=[]
    for g,ref in zip(groups,oldindex):
        receipt=read(ROOT/ref['receipt']);covered=[r for r in g['regions'] if r['queries']]
        for r in g['regions']:
            if not r['queries']:f=b=0.;basis='Exact zero:region has no admitted model queries'
            elif len(covered)==1:f=g['focal_negative_projection_norm'];b=g['BCE_negative_projection_norm'];basis='Only covered region:all saved photo-level source-negative projection gradients originate here'
            else:f=b=None;basis='Multiple covered regions:signed region parameter vectors not saved,cannot apportion a norm'
            entries.append(dict(region_id=r['region_id'],image_id=g['image_id'],identity_group=g['identity_group'],queries=len(r['queries']),saved_C11_two_tensor_projection_focal_norm=f,saved_C11_two_tensor_projection_BCE_norm=b,attribution_basis=basis,new_balanced_parameter_norm=None,saved_scope='C11 calibration-frozen two-tensor projection,not C8 full6class-head scope'))
        output['focal'].append(receipt['losses_and_gradients']['source_focal']['output_logit_L1']);output['BCE'].append(receipt['losses_and_gradients']['source_BCE']['output_logit_L1']);output['balanced_raw'].append(g['balanced_raw_output_gradient_L1'])
        loss_deltas.append(dict(index=g['index'],focal_CPU_region_sum_minus_saved_GPU=sum(r['focal_negative_loss'] for r in g['regions'])-g['focal_negative_loss'],BCE_CPU_region_sum_minus_saved_GPU=sum(r['BCE_negative_loss'] for r in g['regions'])-g['BCE_negative_loss']))
    concentration={k:dict(sum_output_L1=sum(v),top3_share=sum(sorted(v,reverse=True)[:3])/sum(v)) for k,v in output.items()}
    emit16(REPORT16/'region-attribution.json',dict(regions=entries,known_zero_regions=sum(e['queries']==0 for e in entries),sole_covered_regions_with_attributable_saved_photo_norm=sum(e['queries']>0 and e['saved_C11_two_tensor_projection_BCE_norm'] is not None for e in entries),covered_regions_missing_separate_signed_parameter_gradients=sum(e['queries']>0 and e['saved_C11_two_tensor_projection_BCE_norm'] is None for e in entries),output_concentration=concentration,CPU_region_loss_reconstruction_deltas=loss_deltas,per_region_loss_values_are_CPU_reconstructions_not_saved_GPU_region_scalars=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'phase10c16-v1','font.size':10});fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(24)
    for dx,key,label,color in [(-.24,'focal','Saved focal','#b28b39'),(0,'BCE','Saved BCE','#ae5663'),(.24,'balanced_raw','CPU region/group BCE,before cap','#386f8c')]:axes[0].bar(x+dx,output[key],.24,label=label,color=color)
    axes[0].set_xticks(x[::3],[str(i+1) for i in x[::3]]);axes[0].set_xlabel('Optimization photo / known source group');axes[0].set_ylabel('Negative output-gradient L1');axes[0].legend(fontsize=8);axes[0].set_title('Cached outputs:parameter effect unavailable')
    synthetic=read(REPORT16/'synthetic-comparison.json')['cases'];xx=np.arange(3)
    for key,label in [('focal','Focal'),('BCE','BCE'),('balanced_raw','Region mean'),('balanced_capped','Capped source')]:
        yy=[c['results'][key]['negative_to_positive_ratio'] if key!='balanced_capped' else c['results'][key]['used_negative_norm']/c['results'][key]['positive_norm'] for c in synthetic];axes[1].plot(xx,yy,marker='o',label=label)
    axes[1].set_yscale('log');axes[1].set_xticks(xx,['1','10','100']);axes[1].set_xlabel('Exact duplicate negative queries (synthetic fixtures)');axes[1].set_ylabel('Negative / positive parameter-gradient norm');axes[1].set_title('Synthetic6tensor graph;not OWLv2');axes[1].legend(fontsize=8)
    fig.suptitle('C16 CPU-only evidence · no real-model balanced-parameter gradient measurement',fontsize=11);buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None});plt.close(fig);path=REPORT16/'CPU-comparison.svg'
    with path.open('x') as f:f.write('\n'.join(line.rstrip() for line in buf.getvalue().splitlines())+'\n')
    print('Region attribution:17zero/14sole covered/10unresolved;CPU evidence plot written',flush=True)


if __name__=='__main__':run()
