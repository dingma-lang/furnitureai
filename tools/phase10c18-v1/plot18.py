"""Static, measured training and PR comparison; no models or inference."""
from common18 import *


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(REPORT18/'results.json');t=read(REPORT18/'training-curves.json');pr=read(REPORT18/'precision-recall.json')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for k,points in pr.items():
        axes[0,0].plot([p['recall'] for p in points],[p['precision'] for p in points],label=k)
        h=r['comparison'][k]['headline'];axes[0,0].scatter([h['recall']],[h['precision']])
    axes[0,0].set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='Exposed partial-label panel; dots at0.15');axes[0,0].legend(fontsize=8)
    epochs=[e['epoch'] for e in t['epochs']]
    for name in ('positive_loss','box_loss','source_loss'):axes[0,1].plot(epochs,[e['mean_loss'][name] for e in t['epochs']],marker='o',label=name)
    axes[0,1].set(title='Per-photo loss means; descriptive',xlabel='Epoch');axes[0,1].legend(fontsize=8)
    for name in ('positive','source_pre','source_post'):axes[1,0].semilogy(epochs,[max(e['mean_parameter_gradient'][name],1e-12) for e in t['epochs']],marker='o',label=name)
    axes[1,0].set(title='Mean accumulated class-parameter gradient norms',xlabel='Epoch');axes[1,0].legend(fontsize=8)
    p=t['probes'];axes[1,1].plot([e['epoch'] for e in p],[e['terms_ge_015'] for e in p],marker='o',label='Source Chair scores >=0.15');axes[1,1].plot([e['epoch'] for e in p],[e['terms_ge_05'] for e in p],marker='o',label='>=0.50');axes[1,1].set(title='Fixed24-photo probes; admitted queries may change',xlabel='Epoch',ylabel='Query terms');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('C18 single controlled arm — no independent accuracy established')
    b=io.StringIO();fig.savefig(b,format='svg');plt.close(fig);atomic18(REPORT18/'diagnostic-curves.svg', ('\n'.join(line.rstrip() for line in b.getvalue().splitlines())+'\n').encode())


if __name__=='__main__':run()
