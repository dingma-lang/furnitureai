"""Standalone scientific SVG from saved CPU results;no inference or training."""
from common5 import *
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only plot')
    os.environ['MPLCONFIGDIR']=str(RUN/'CPU-analysis/matplotlib-config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    r=read(REPORT/'results.json');curves=read(REPORT/'precision-recall.json')
    colors=dict(original='#252525',C8='#c87920',C11='#248da7',E5='#b62e58')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for n in ('original','C8','C11','E5'):
        p=curves[n];m=r['comparison'][n]['headline']
        axs[0,0].plot([x['recall'] for x in p],[x['precision'] for x in p],label=n,color=colors[n],ls='--' if n=='E5' else '-',alpha=.9)
        axs[0,0].scatter([m['recall']],[m['precision']],color=colors[n],s=35)
    axs[0,0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Exposed partial-label panel; dots at confidence 0.15');axs[0,0].legend(fontsize=8)
    ep=r['training_epochs'];x=[e['epoch'] for e in ep]
    for key in ('positive','classification','box','weighted_KD'):
        axs[0,1].plot(x,[e['losses'][key] for e in ep],marker='o',label=key)
    axs[0,1].set(xlabel='Epoch',ylabel='Mean per-image loss',yscale='log',title='Fixed objective; weighted KD remains small');axs[0,1].legend(fontsize=8)
    ix=np.arange(7);width=.28
    for offset,n in ((-1,'C8'),(0,'C11'),(1,'E5')):
        axs[1,0].bar(ix+offset*width,[r['logit_drift'][n]['per_class_mean'][c] for c in CLASSES],width,color=colors[n],label=n)
    axs[1,0].set_xticks(ix,CLASSES,rotation=25);axs[1,0].set(ylabel='Mean logit change vs original',title='Same 68,400 query locations');axs[1,0].legend(fontsize=8)
    for q,label in (('0.5','Median'),('1','Maximum')):
        axs[1,1].plot(x,[e['weighted_KD_positive_ratio'][q] for e in ep],marker='o',label=label)
    axs[1,1].axhline(.5,color='#777',ls=':',label='Recovery bound (not enforced)')
    axs[1,1].set(xlabel='Epoch',ylabel='Weighted KD / positive gradient norm',yscale='log',title='Active training presentations');axs[1,1].legend(fontsize=8)
    fig.suptitle('Phase 10E.5 — selective teacher distillation; weak internal evaluation',fontsize=12)
    out=io.BytesIO();fig.savefig(out,format='svg');atomic(REPORT/'comparison-and-training.svg',out.getvalue());plt.close(fig)
    print('Saved scientific SVG from existing results',flush=True)
if __name__=='__main__':run()
