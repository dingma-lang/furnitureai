"""Confidence-band/output-gradient evidence and static scientific plot,CPU only."""
import numpy as np
from common15r import *


def run():
    index=read(RUNR/'combined-index.json')['records'];bands=[dict(label=label,queries=0,focal_output_L1=0.,BCE_output_L1=0.) for label in ('p<0.01','0.01<=p<0.05','0.05<=p<0.15','p>=0.15')]
    for row in index:
        with np.load(ROOT/row['raw_file'],allow_pickle=False) as z:
            added=z['source_added'][:,1].astype(bool);p=1/(1+np.exp(-z['logits'][added,1].astype(float)));f=np.abs(z['output_grad_source_focal'][added,1].astype(float));b=np.abs(z['output_grad_source_BCE'][added,1].astype(float))
        choices=np.select([p<.01,p<.05,p<.15],[0,1,2],default=3)
        for i,band in enumerate(bands):
            q=choices==i;band['queries']+=int(q.sum());band['focal_output_L1']+=float(f[q].sum());band['BCE_output_L1']+=float(b[q].sum())
    groups=read(REPORTR/'source-group-summary.json')['records'];total=sum(g['BCE_projection_L2'] for g in groups);largest=sorted(groups,key=lambda g:-g['BCE_projection_L2'])[:3]
    emit_r(REPORTR/'negative-gradient-bands.json',dict(bands=bands,top3_groups_share_of_sum_per_photo_BCE_projection_norms=sum(g['BCE_projection_L2'] for g in largest)/total,top3=[dict(index=g['index'],identity_group=g['identity_group'],BCE_projection_L2=g['BCE_projection_L2'],source_terms=g['source_terms']) for g in largest],warning='Shares of per-photo norms are descriptive,not a decomposition of a signed cohort gradient'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'phase10c15r-v1','font.size':10});fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained');x=np.arange(24)
    for offset,key,color,label in [(-.24,'positive_projection_L2','#386d8c','Positive'),(0,'focal_projection_L2','#bc8537','Source focal'),(.24,'BCE_projection_L2','#b44c58','Source BCE')]:ax[0].bar(x+offset,[max(g[key],1e-8) for g in groups],.24,color=color,label=label)
    ax[0].set_yscale('log');ax[0].set_title('Actual projection-gradient norms');ax[0].set_ylabel('Per-photo L2 (zeros plotted at floor)');ax[0].set_xlabel('Photo / known source group');ax[0].set_xticks(x[::3],[str(i+1) for i in x[::3]]);ax[0].legend(fontsize=8)
    ax[1].bar(x,[g['class_A_vs_B_cosine'] for g in groups],color=['#b44c58' if g['class_A_vs_B_cosine']<0 else '#386d8c' for g in groups]);ax[1].axhline(0,color='black',lw=.8);ax[1].set_ylim(-1.05,1.05);ax[1].set_title('Classification-gradient direction');ax[1].set_ylabel('Cosine A vs B;negative = reversal');ax[1].set_xlabel('Photo / known source group');ax[1].set_xticks(x[::3],[str(i+1) for i in x[::3]])
    xx=np.arange(4);ax[2].bar(xx-.18,[b['focal_output_L1'] for b in bands],.35,label='Focal',color='#bc8537');ax[2].bar(xx+.18,[b['BCE_output_L1'] for b in bands],.35,label='BCE',color='#b44c58');ax[2].set_xticks(xx,[b['label'] for b in bands],rotation=30,ha='right');ax[2].set_ylabel('Sum of GT-normalized output-gradient L1');ax[2].set_title('Same281 source-supported negatives');ax[2].legend()
    fig.suptitle('C15R · four reused + twenty recovered · fixed C11 weights · zero updates · no accuracy evaluation',fontsize=11)
    import io
    f=io.StringIO();fig.savefig(f,format='svg',metadata={'Date':None});plt.close(fig);atomic_r(REPORTR/'gradient-comparison.svg',('\n'.join(line.rstrip() for line in f.getvalue().splitlines())+'\n').encode());print('Confidence-band evidence and static plot written;no model calls',flush=True)


if __name__=='__main__':run()
