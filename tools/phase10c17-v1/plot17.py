"""CPU scientific plot of measured vectors,no model or additional VJP."""
import io
import numpy as np
from common17 import *


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(REPORT17/'results.json');groups=read(REPORT17/'accumulation-group-summary.json')['groups'];sources=read(REPORT17/'source-group-summary.json')['groups'];plt.rcParams.update({'svg.hashsalt':'C17','font.size':10});fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained');x=np.arange(1,16)
    ax[0].plot(x,[g['metrics']['positive']['norm'] for g in groups],color='#687b84',label='Positive')
    for key,label,color in [('A_focal','Focal','#bb8d36'),('B_BCE','BCE','#b75663'),('C_post_cap','Balanced pre=post','#346c87')]:ax[0].plot(x,[max(g['metrics']['negatives'][key]['norm'],1e-9) for g in groups],marker='.',label=label,color=color)
    ax[0].set_yscale('log');ax[0].set_xlabel('Actual C8 accumulation group');ax[0].set_ylabel('Six-tensor class-head gradient L2');ax[0].set_title('Fixed original pretrained weights');ax[0].legend(fontsize=8)
    for key,label in [('B_BCE','BCE'),('C_post_cap','Balanced')]:ax[1].plot(x,[100*g['metrics']['negatives'][key]['negative_positive_ratio'] for g in groups],marker='.',label=label)
    ax[1].axhline(25,color='#af4e59',ls='--',label='25% cap');ax[1].set_yscale('symlog',linthresh=.001);ax[1].set_ylabel('Negative / positive norm (%)');ax[1].set_xlabel('Accumulation group');ax[1].set_title('0/15 cap activations;0 violations');ax[1].legend(fontsize=8)
    active=sorted([s['negatives']['C_post_cap']['norm'] for s in sources if s['negatives']['C_post_cap']['norm']>0],reverse=True);ax[2].bar(np.arange(1,len(active)+1),active,color='#346c87');ax[2].set_xlabel('19 contributing source groups,ranked');ax[2].set_ylabel('Actual source gradient norm');ax[2].set_title('Top3 share:51.74% (BCE69.66%)')
    fig.suptitle('C17 zero-update smoke ·59photos/295VJPs ·no precision/recall evaluation',fontsize=11);stream=io.StringIO();fig.savefig(stream,format='svg',metadata={'Date':None});plt.close(fig);atomic17(REPORT17/'gradient-comparison.svg',('\n'.join(line.rstrip() for line in stream.getvalue().splitlines())+'\n').encode());print('Measured-gradient plot written;no model calls',flush=True)


if __name__=='__main__':run()
