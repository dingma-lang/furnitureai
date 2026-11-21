"""Small aggregate reports/plots from immutable CPU analysis artifacts."""
import csv
from common9 import *

def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def run():
    summary=read(REPORT/'prediction-summary.json');rows=[]
    for model,value in summary['models'].items():
        for threshold,cats in value['thresholds'].items():
            for category,c in cats.items():rows.append(dict(model=model,threshold=threshold,category=category,**{k:v for k,v in c.items() if k!='scored_FP_area_bins'}))
    write_csv(REPORT/'threshold-counts.csv',rows);write_csv(REPORT/'photograph-counts.csv',summary['per_photo'])
    observations=read(OUT/'assistant-observations.json')['records'];write_csv(REPORT/'sample-findings.csv',[{k:r[k] for k in ('sample_id','image_id','query_index','scope','taxonomy_primary','evidence_tier','observation','human_reviewed','training_enabled')} for r in observations])
    protocol=read(ROOT/'reports/phase10c8-v2/independent-evaluation-proposal.json')
    protocol['C9_clarification']='Explicitly annotate the expanded Chair concept including seating/bar stools. Native Chair exhaustiveness alone is insufficient. Preserve uncertain objects as ignored; audit object-level completeness without detector-output-driven source selection.'
    protocol['C9_no_acceptance_images_accessed']=True
    save(REPORT/'independent-evaluation-protocol.json',protocol)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors={'original':'#344d63','control':'#c18139','corrected':'#21847c'}
    for offset,model in enumerate(('original','control','corrected')):
        vals=summary['models'][model]['thresholds'];axes[0,0].bar(np.arange(3)+(offset-1)*.24,[vals[str(t)]['chair']['scored_fp'] for t in THRESHOLDS],.24,label=model,color=colors[model])
        axes[0,1].plot(THRESHOLDS,[vals[str(t)]['chair']['recall'] for t in THRESHOLDS],marker='o',label=model,color=colors[model])
        area=vals['0.15']['chair']['scored_FP_area_bins'];labels=['<1%','1-5%','5-20%','>=20%'];axes[1,0].bar(np.arange(4)+(offset-1)*.24,[area.get(x,0) for x in labels],.24,label=model,color=colors[model])
    axes[0,0].set(xticks=range(3),xticklabels=['≥.15','≥.50','≥.80'],ylabel='Native-scored Chair FP',title='Unmatched unknowns excluded; four FP-complete photos')
    axes[0,1].set(xlabel='Score threshold',ylabel='Recall on39 active Chair instances',ylim=(0,1.05),title='Threshold suppression loses Chairs')
    axes[1,0].set(xticks=range(4),xticklabels=['<1%','1–5%','5–20%','≥20%'],ylabel='Native-scored Chair FP',title='Predicted box area / photograph area at.15')
    epochs=[0,1,3,5];probes=read(ROOT/'reports/phase10c8-v2/comparison.json')['probes']
    for model,arm in (('control','A-control'),('corrected','B-corrected')):
        ys=[next(p['mean_added_loss_float64'] for p in probes if p['arm']==arm and p['epoch']==e) for e in epochs];axes[1,1].semilogy(epochs,ys,marker='o',label=model,color=colors[model])
    axes[1,1].set(xlabel='Epoch',ylabel='Mean additional negative focal loss /24photo probe',title='Real signal growth; no added training negatives scored≥.50')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Phase10C.9 — saved-query internal diagnostics, not independent accuracy');fig.tight_layout();fig.savefig(REPORT/'diagnostic-summary.svg');plt.close(fig)
    save(REPORT/'analysis-provenance.json',dict(C8_binding_sha256=sha(C8/'binding.json'),C8_metric_results_sha256=sha(C8/'metric-results.json'),
        diagnostic_native_context_sha256=sha(OUT/'development-native-context.json'),prediction_record_sha256=sha(OUT/'prediction-records.json'),
        source_supported_hard_negative_manifest_sha256=sha(OUT/'hard-negative-proposals.json'),assistant_observations_sha256=sha(OUT/'assistant-observations.json'),
        current_dashboard_data_sha256=sha(OUT/'dashboard-live.json'),all_new_tools={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.is_file()},
        image_or_model_downloads=0,model_forwards=0,optimizer_updates=0,training_labels_activated=0,human_confirmations_fabricated=0))

if __name__=='__main__':run()
