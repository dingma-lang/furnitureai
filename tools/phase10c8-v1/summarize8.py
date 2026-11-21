"""Read completed artifacts only: paired diagnostics, immutable checkpoint audit, plots."""
import csv,statistics
from common8 import *

def compact(m):
    return {k:v for k,v in m.items() if k!='pr_curve'}

def envelope(m,recall):
    eligible=[p for p in m['pr_curve'] if p['recall'] is not None and p['recall']>=recall]
    return max(eligible,key=lambda p:(p['precision'],-p['recall'])) if eligible else None

def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    import torch,numpy as np
    binding=read(RUN/'binding.json');prepared=verify_binding(binding);train,dev=subsets(prepared)
    queue=read(RUN/'queue-complete.json')
    if queue['optimizer_updates']!=150:raise ValueError('Not a complete approved A/B')
    entries=read(RUN/'metric-results.json')['entries'];by={(e['arm'],e['epoch']):e for e in entries};original=by[(ARMS[0],0)];reference_recall=original['metrics']['recall']
    curves=[];metrics=[];category_rows=[];photo_rows=[]
    for e in entries:
        m=e['metrics'];key=e['arm']+'-E'+str(e['epoch']);curve=m['pr_curve']
        curves.append(dict(model=key,points=curve))
        metrics.append(dict(model=key,updates=e['updates'],**{k:m[k] for k in ('precision','recall','ap50','ap75','tp','fp','fn')},precision_at_original_recall=envelope(m,reference_recall)))
        for c,v in m['categories'].items():category_rows.append(dict(model=key,category=c,**v))
    for row in dev:
        out=dict(image_id=row['image_id'],source_group=row['identity_group'])
        for arm,epoch,key in ((ARMS[0],0,'original'),(ARMS[0],5,'A'),(ARMS[1],5,'B')):
            pr=next(p for p in by[(arm,epoch)]['per_photo_Chair'] if p['image_id']==row['image_id']);out[key+'_Chair_tp']=pr['Chair_tp'];out[key+'_Chair_fp']=pr['Chair_fp']
        out['B_minus_A_Chair_fp']=out['B_Chair_fp']-out['A_Chair_fp'];photo_rows.append(out)
    probes=[];updates=[];checkpoints=[];completions=[]
    for arm in ARMS:
        folder=RUN/arm;completion=read(folder/'complete.json');completions.append(completion)
        if completion['updates']!=75 or completion['optimizer_hook_calls']!=75 or completion['training_backwards']!=295:raise ValueError('Unexpected update/gradient count')
        for epoch in PROBE_EPOCHS:
            p=folder/'probes'/('epoch-'+str(epoch));records=[read(p/(str(i).zfill(2)+'.json')) for i in range(1,25)];summary=read(p/'complete.json')
            if any(r['protected_violations'] for r in records):raise ValueError('Protected overlap violation')
            scores=[s for r in records for s in r['added_Chair_scores']]
            probes.append(dict(arm=arm,epoch=epoch,**{k:v for k,v in summary.items() if k not in ('arm','epoch')},
                mean_classification_loss=statistics.mean(r['classification_loss'] for r in records),mean_box_loss=statistics.mean(r['box_loss'] for r in records),
                mean_positive_classification=statistics.mean(r['positive_classification'] for r in records),mean_legacy_negative_classification=statistics.mean(r['legacy_negative_classification'] for r in records),
                mean_added_loss_float32=statistics.mean(r['added_Chair_negative_loss'] for r in records),
                ratio_of_mean_added_to_used_class_gradient=summary['mean_added_gradient_norm']/summary['mean_used_class_gradient_norm'] if summary['mean_used_class_gradient_norm'] else None,
                photos_with_added_queries=sum(r['added_active_Chair_terms']>0 for r in records),
                photos_with_nonzero_added_gradient=sum(r['additional_negative_gradient_norm']>0 for r in records),
                added_Chair_at_or_above_015=sum(s>=.15 for s in scores),added_Chair_quantiles=[float(np.quantile(scores,q)) if scores else None for q in (0,.5,.9,1)],
                gradient_probe_weights_unchanged=True,additional_gradient_applied_during_training=(arm==ARMS[1])))
        for i in range(1,76):
            u=read(folder/('update-'+str(i).zfill(3)+'.json'));rows=u['images'];rec=u['receipt'];p=RUN/rec['file']
            if u['updates']!=i or sha(p)!=rec['sha256']:raise ValueError('Update/checkpoint receipt mismatch')
            state=torch.load(p,map_location='cpu',weights_only=True)
            if state['updates']!=i or state['binding_sha256']!=sha(RUN/'binding.json') or state['frozen_sha256']!=completion['frozen_sha256'] or any(int(s['step'])!=i for s in state['optimizer']['state'].values()):raise ValueError('Snapshot cursor/optimizer mismatch')
            if not all(torch.isfinite(t).all() for t in state['weights'].values()):raise ValueError('Nonfinite checkpoint')
            if any(r['image_id'] not in {r['image_id'] for r in train} or r['protected_violations'] for r in rows):raise ValueError('Source/protected mask violation')
            checkpoints.append(rec);updates.append(dict(arm=arm,update=i,epoch=u['epoch'],mean_loss=statistics.mean(r['total_loss'] for r in rows),
                mean_classification_loss=statistics.mean(r['classification_loss'] for r in rows),mean_box_loss=statistics.mean(r['box_loss'] for r in rows),
                mean_positive_classification=statistics.mean(r['positive_classification'] for r in rows),mean_legacy_negative_classification=statistics.mean(r['legacy_negative_classification'] for r in rows),
                mean_added_negative_loss_float64=statistics.mean(r['added_Chair_negative_loss_float64'] for r in rows),
                additional_Chair_terms=sum(r['added_active_Chair_terms'] for r in rows),class_head_gradient_norm=u['class_head_gradient_norm'],total_head_gradient_norm=u['total_head_gradient_norm'],source_order=u['source_order']))
    for a,b in zip([u for u in updates if u['arm']==ARMS[0]],[u for u in updates if u['arm']==ARMS[1]]):
        if a['source_order']!=b['source_order']:raise ValueError('A/B source sampling differs')
    if len({c['initial_heads_sha256'] for c in completions})!=1 or len({c['frozen_sha256'] for c in completions})!=1:raise ValueError('Initial or frozen states differ')
    # Historical protocol/manifest agreement is a gate, never assumed from matching counts.
    historical=read(ROOT/'data/open-images/experiments/phase10b-heads-v1/internal-reference.json')
    manifest_path=ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'
    if historical['binding']['dataset_sha256']!=sha(manifest_path):raise ValueError('Historical evaluation manifest mismatch')
    recovered=read(ROOT/'data/open-images/experiments/phase10c-recovery-v1/comparison.json')['models']
    historical_original=recovered['original']['fixed']
    exact_original_match={k:original['metrics'][k]==historical_original[k] for k in ('tp','fp','fn','precision','recall','ap50','ap75')}
    old=[]
    for name,v in recovered.items():old.append(dict(model=name,**v['fixed'],categories=v['categories'],precision_at_original_recall=v.get('matched_original_recall'),source='immutable historical CPU results, same exact manifest and metric definitions; training seeds/control differ'))
    A=by[(ARMS[0],5)]['metrics'];B=by[(ARMS[1],5)]['metrics'];delta={k:B[k]-A[k] for k in ('precision','recall','ap50','ap75','tp','fp','fn')}
    def change(a,b):return dict(A=a,B=b,absolute_B_minus_A=b-a,relative_B_minus_A=(b-a)/a if a else None)
    effects=dict(aggregate=delta,Chair_FP=change(A['categories']['chair']['fp'],B['categories']['chair']['fp']),Chair_recall=change(A['categories']['chair']['recall'],B['categories']['chair']['recall']),
        original_recall=reference_recall,matched_recall_A=envelope(A,reference_recall),matched_recall_B=envelope(B,reference_recall),
        identical_initial_probe_arrays=True,identical_training_order=True,sole_intentional_difference='additional source-region Chair-negative mask')
    final_states=[torch.load(RUN/c['final_checkpoint']['file'],map_location='cpu',weights_only=True)['weights'] for c in completions]
    differences={}
    for name in final_states[0]:
        a=final_states[0][name].double();b=final_states[1][name].double();d=b-a
        differences[name]=dict(different_elements=int((d!=0).sum()),elements=d.numel(),delta_L2=float(d.norm()),delta_max_abs=float(d.abs().max()),relative_L2=float(d.norm()/a.norm()) if a.norm() else None)
    effects['final_parameter_differences']=differences
    effects['category_effects']={c:{k:change(A['categories'][c][k],B['categories'][c][k]) for k in ('tp','fp','fn','precision','recall','ap50','ap75','duplicate_fp','mean_matched_iou') if A['categories'][c][k] is not None and B['categories'][c][k] is not None} for c in CLASSES}
    output=dict(scope='19 exposed internal-development photos only; not independent seven-class accuracy',metrics=metrics,historical=old,historical_manifest_sha256=sha(manifest_path),fresh_original_exact_historical_match=exact_original_match,
        effects=effects,probes=probes,resources=completions,queue=queue,checkpoints_byte_and_optimizer_verified=len(checkpoints),
        all_updates_have_immutable_recovery=True,source_positive_boxes=182,training_photos=59,training_groups=56,development_photos=19,
        production_replaced=False,Phase10D_started=False,independent_acceptance_scored=False)
    save(REPORT/'comparison.json',output)
    write_csv(REPORT/'category-metrics.csv',category_rows);write_csv(REPORT/'per-photo-chair.csv',photo_rows)
    write_csv(REPORT/'training-curves.csv',[{k:v for k,v in u.items() if k!='source_order'} for u in updates]);write_csv(REPORT/'negative-signal.csv',[{k:v for k,v in p.items() if not isinstance(v,(list,dict))} for p in probes])
    save(RUN/'checkpoint-registry.json',dict(checkpoints=checkpoints,final=[c['final_checkpoint'] for c in completions]))
    save(RUN/'analysis-provenance.json',dict(analysis_sha256=sha(Path(__file__)),binding_sha256=sha(RUN/'binding.json'),metric_results_sha256=sha(RUN/'metric-results.json'),checkpoints_verified=len(checkpoints)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for arm in ARMS:
        us=[u for u in updates if u['arm']==arm];ps=[p for p in probes if p['arm']==arm];label='A legacy' if arm==ARMS[0] else 'B corrected'
        axes[0,0].plot([u['update'] for u in us],[u['mean_classification_loss'] for u in us],label=label)
        axes[0,1].semilogy([p['updates'] for p in ps],[max(p['mean_added_loss_float64'],1e-14) for p in ps],marker='o',label=label)
        axes[1,0].semilogy([p['updates'] for p in ps],[max(p['ratio_of_mean_added_to_used_class_gradient'],1e-14) for p in ps],marker='o',label=label)
        model=by[(arm,5)]['metrics'];axes[1,1].plot([p['recall'] for p in model['pr_curve']],[p['precision'] for p in model['pr_curve']],label=label)
        axes[1,1].scatter([model['recall']],[model['precision']],s=30)
    base=original['metrics'];axes[1,1].plot([p['recall'] for p in base['pr_curve']],[p['precision'] for p in base['pr_curve']],label='Original',color='black')
    titles=['Classification loss / optimizer group','Added negative focal loss (24-photo mean)','Ratio of mean added / classification gradient norms','Internal PR curves; dots at confidence 0.15']
    for ax,title in zip(axes.flat,titles):ax.set_title(title);ax.grid(alpha=.25);ax.legend()
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Optimizer updates')
    axes[1,1].set_xlabel('Recall');axes[1,1].set_ylabel('Precision');axes[1,1].set_xlim(0,1);axes[1,1].set_ylim(0,1.02)
    fig.suptitle('Phase 10C.8 — weak/partial-label internal diagnostics');fig.tight_layout();fig.savefig(REPORT/'diagnostic-curves.svg');plt.close(fig)
    print('PAIRED COMPARISON COMPLETE',effects,flush=True)

if __name__=='__main__':run()
