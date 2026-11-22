"""Static scientific figures from saved artifacts; no model or image loading."""
from common20 import *


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(REPORT20/'results.json');pr=read(REPORT20/'precision-recall.json');training=read(REPORT20/'training-curves.json')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for key in ('original','C8_B_E5','C18_E1','C20_E5'):
        p=pr[key];m=r['comparison'][key]['headline'];axes[0,0].plot([v['recall'] for v in p],[v['precision'] for v in p],label=key);axes[0,0].scatter([m['recall']],[m['precision']],s=25)
    axes[0,0].set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title='Exposed partial-label panel; dots at0.15');axes[0,0].legend(fontsize=8)
    updates=[read((RUN18 if i<=28 else RUN20)/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)]
    for name in ('positive','source_pre','source_post'):axes[0,1].semilogy(range(1,76),[max(u['metrics'][name]['norm'],1e-12) for u in updates],label=name)
    axes[0,1].axvline(28.5,color='gray',linestyle='--');axes[0,1].set(title='Measured accumulated class-gradient norms',xlabel='Update');axes[0,1].legend(fontsize=8)
    axes[1,0].plot(range(1,76),[u['metrics']['cap']['scale'] for u in updates]);axes[1,0].axvline(28.5,color='gray',linestyle='--');axes[1,0].set(title='Detached cap factor;13active groups',xlabel='Update',ylabel='Factor',ylim=(0,1.05))
    probes=training['probes'];axes[1,1].plot([p['epoch'] for p in probes],[p['terms_ge_015'] for p in probes],marker='o',label='>=0.15');axes[1,1].plot([p['epoch'] for p in probes],[p['terms_ge_05'] for p in probes],marker='o',label='>=0.50');axes[1,1].set(title='Trusted-negative terms; fixed24photo probes',xlabel='Epoch',ylabel='Terms');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('Recovered source-balanced experiment — no independent accuracy claim')
    b=io.StringIO();fig.savefig(b,format='svg');plt.close(fig);atomic20(REPORT20/'diagnostic-curves.svg', ('\n'.join(s.rstrip() for s in b.getvalue().splitlines())+'\n').encode())


if __name__=='__main__':run()
