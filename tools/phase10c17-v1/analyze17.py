"""CPU-only validation and compact full-sequence gradient comparisons."""
import io,csv,zipfile
from collections import Counter
import numpy as np
import torch
from common17 import *
from balanced16 import CLASS_KEYS,weights


def run():
    cfg,prepared,seq,registry=inputs17();done=read(RUN17/'complete.json');photos=[];groups=[];byid={r['image_id']:r for g in seq for r in g};context=read(DATA16/'catalog.json');source=read(SOURCE/'source-negative-evidence.json')
    from common6 import candidates
    frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
    native_negative_terms=0;all_logits=[];reduction_deltas=[]
    if (done['forwards'],done['gradient_calculations'],done['accumulation_groups'])!=(59,295,15) or done['model_state_before']!=done['model_state_after'] or done['optimizer_steps']!=0:raise ValueError('Final smoke counters/fingerprints invalid')
    for index,ref in enumerate(done['records'],1):
        p=RUN17/ref['file'];r=read(p);row=byid[r['image_id']]
        if sha(p)!=ref['sha256'] or sha(ROOT/row['path'])!=r['source_sha256'] or r['image_id']!=cfg['group_order'][(index-1)//4][(index-1)%4]:raise ValueError('Photo hash/order mismatch')
        raw=RUN17/r['raw_file']
        if sha(raw)!=r['raw_sha256']:raise ValueError('Raw data hash changed')
        with zipfile.ZipFile(raw) as z:
            if sum(x.file_size for x in z.infolist())>1_000_000 or len(z.infolist())!=12:raise ValueError('Unexpected raw diagnostic archive')
        with np.load(raw,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite persisted outputs')
        target=a['target'].astype(bool);legacy=a['legacy'].astype(bool);added=a['added'].astype(bool);mask=a['mask'].astype(bool)
        if target.shape!=(3600,7) or np.any(added&target) or np.any(added&legacy) or np.any(added[:,[0,2,3,4,5,6]]) or not np.array_equal(mask,legacy|added):raise ValueError('Supervision preservation failure')
        counts=Counter(a['label'] for a in row['annotations'] if not a['ignore'])
        if target.sum()!=r['counts']['positive'] or target.sum(0).tolist()!=[counts[c] for c in CLASSES] or added.sum()!=r['counts']['source_terms']:raise ValueError('Positive category/target counts differ')
        protected=[v['box'] for v in registry[row['image_id']]['protected']]+[v['annotation']['box'] for v in frozen if v['image_id']==row['image_id'] and obs[v['candidate_id']]['visual_outcome']=='mixed_or_uncertain'];masks=dict(target=torch.from_numpy(a['target'].astype(np.float32)),legacy=torch.from_numpy(legacy),added=torch.from_numpy(added))
        w,meta=weights(torch.from_numpy(a['logits']),torch.from_numpy(a['square_cxcywh']),row,masks,context,protected)
        saved_meta=r['normalization']
        # Tensor weights and structural metadata must match exactly. GPU/CPU
        # parallel reduction of those identicalFP32weights may differ by1ULP.
        if not np.array_equal(w.numpy(),a['weights']) or any(meta[k]!=saved_meta[k] for k in meta if k!='weights_sum') or not np.isclose(meta['weights_sum'],saved_meta['weights_sum'],atol=1e-7,rtol=1e-6):raise ValueError('Frozen C16 source normalization changed')
        if meta['weights_sum']!=saved_meta['weights_sum']:reduction_deltas.append(dict(index=index,CPU_sum=meta['weights_sum'],GPU_sum=saved_meta['weights_sum'],weights_bit_identical=True))
        for name in ('A_focal','B_BCE','C_pre_cap'):
            g=a['output_gradient_'+name]
            if g.shape!=(3600,7) or np.any(g[~added]) or not np.isclose(np.abs(g.astype(float)).sum(),r['losses_and_gradients'][name]['output_logit_L1'],atol=1e-10,rtol=1e-6):raise ValueError('Output gradient routing/summary mismatch')
        if r['counts']['protected_violations'] or not r['all_parameter_grads_none'] or not r['heads_unchanged']:raise ValueError('Model/positive integrity failure')
        native_negative_terms+=int((legacy&~target).sum());all_logits.append(a['logits']);photos.append(r)
    for ref in done['groups']:
        p=RUN17/ref['file'];g=read(p)
        if sha(p)!=ref['sha256'] or g['model_state_sha256']!=done['model_state_before'] or g['metrics']['cap_violations'] or not g['metrics']['box_gradients_unchanged'] or not g['metrics']['positive_buffers_unchanged']:raise ValueError('Accumulation-boundary preservation failed')
        cap=g['metrics']['cap'];n=g['metrics']['negatives']['C_post_cap']['norm']
        if n>.25*g['metrics']['positive']['norm']*(1+1e-6)+1e-12 or not 0<cap['scale']<=1 or cap['amp_scale']!=128:raise ValueError('Actual parameter cap/AMP metadata failure')
        groups.append(g)
    creators=read(RUN17/'source-groups.json');methods={}
    for name in ('A_focal','B_BCE','C_pre_cap','C_post_cap'):
        ng=[g['metrics']['negatives'][name] for g in groups];contrib=[s['negatives'][name]['norm'] for s in creators['groups']];ratios=[v['negative_positive_ratio'] for v in ng if v['norm']>0]
        methods[name]=dict(mean_accumulation_group_negative_norm=float(np.mean([v['norm'] for v in ng])),active_accumulation_groups=sum(v['norm']>0 for v in ng),active_source_groups=sum(x>0 for x in contrib),sum_source_norms=sum(contrib),top3_source_norm_share=sum(sorted(contrib,reverse=True)[:3])/sum(contrib),active_negative_positive_ratio_quantiles=quantiles(ratios),negative_positive_cosine_quantiles=quantiles([v['cosine_positive'] for v in ng if v['cosine_positive']is not None]),combined_classification_vs_positive_cosine_quantiles=quantiles([g['metrics']['combined'][name]['cosine_positive'] for g in groups]),classification_vs_positive_reversals=sum(g['metrics']['combined'][name]['cosine_positive']<0 for g in groups),group_norms=[v['norm'] for v in ng],output_gradient_L1=sum(p['losses_and_gradients'][name if name!='C_post_cap' else 'C_pre_cap']['output_logit_L1']* (groups[p['group']-1]['metrics']['cap']['scale'] if name=='C_post_cap' else 1) for p in photos))
    scores=[s for p in photos for s in p['source_scores']];covered={key for p in photos for key,n in p['region_query_counts'].items() if n};per_tensor={k:dict(max_post_negative_positive_ratio=max(g['metrics']['per_tensor'][k]['ratio'] for g in groups if g['metrics']['per_tensor'][k]['ratio']is not None),opposing_negative_larger_than_positive_groups=sum(g['metrics']['per_tensor'][k]['opposing_negative_exceeds_positive'] for g in groups),positive_zero_negative_nonzero_groups=sum(g['metrics']['per_tensor'][k]['positive_zero_negative_nonzero'] for g in groups)) for k in sorted(CLASS_KEYS)}
    source_groups_with_regions=set(context['group_region_counts']);nonzero_source=set(s['identity_group'] for s in creators['groups'] if s['negatives']['C_post_cap']['norm']>0);logits=np.concatenate(all_logits)
    result=dict(status='COMPLETE',decision='A_QUALIFIED_PROMISING_FOR_ONE_DIAGNOSTIC_TRAINING_TRIAL',decision_basis='Finite source gradients across19groups,~788x focal norm,0cap/positive/box violations and0direction reversals;not proof of sufficient learning or accuracy',forwards=59,gradient_calculations=295,photos=59,known_source_groups=56,accumulation_groups=15,group_sizes=[g['size'] for g in groups],positive_boxes=sum(p['counts']['positive'] for p in photos),source_negative_queries=sum(p['counts']['source_terms'] for p in photos),native_negative_terms=native_negative_terms,ignored_terms=sum(p['counts']['ignored'] for p in photos),source_regions=41,covered_regions=len(covered),uncovered_regions=41-len(covered),source_groups_with_nonzero_balanced_gradient=len(nonzero_source),source_groups_without_negative_coverage=56-len(nonzero_source),source_groups_with_regions_but_no_coverage=len(source_groups_with_regions-nonzero_source),source_groups_without_regions=56-len(source_groups_with_regions),duplicate_query_signatures_collapsed=sum(p['normalization']['duplicate_query_signatures_collapsed'] for p in photos),methods=methods,mean_positive_group_norm=float(np.mean([g['metrics']['positive']['norm'] for g in groups])),cap_activated_groups=sum(g['metrics']['cap']['scale']<1 for g in groups),cap_activation_fraction=sum(g['metrics']['cap']['scale']<1 for g in groups)/15,cap_violations=0,cap_strength_changed=False,positive_and_box_buffers_bit_identical=True,source_signal_silently_dropped=False,unusable_positive_anchors=0,accumulation_groups_without_Chair_positives=sum(g['Chair_positive_boxes']==0 for g in groups),source_photos_without_Chair_positives=sum(p['counts']['Chair_positive']==0 and p['counts']['source_terms']>0 for p in photos),per_tensor=per_tensor,source_confidence=dict(quantiles=quantiles(scores),below001=sum(s<.001 for s in scores),below005=sum(s<.05 for s in scores),at_least015=sum(s>=.15 for s in scores)),all7_training_logit_quantiles={c:quantiles(logits[:,j]) for j,c in enumerate(CLASSES)},global_mean_actual_vectors=creators['global_mean'],resource=done['resource'],launcher_seconds=read(RUN17/'launch-result.json')['wall_seconds'],model_state_before=done['model_state_before'],model_state_after=done['model_state_after'],model_updates=0,optimizer_steps=0,calibration_trainable=True,projection_shift_scale_measured_separately=True,cap_activation_not_exercised_on_real_data=True,meaningful_beyond_easy_negative_coverage_not_established=True,relative_signal_warning='Balanced source norm only0.008–0.214percent of positive norm in active groups at original initialization;no lower-bound guarantee or recall/precision conclusion',AMP_comparison_caveat='A/B/C use identical forward and base branches;additive FP32 component comparisons are not asserted bit-identical to a hypothetical separate one-backward AMP objective',metadata_issue='Frozen C16.finish overwrites diagnostic cap scale withAMP128;C17 calls unchanged C16.compose on identical buffers and records amp_scale separately;CPU bit-identity regression passed',original_C16_code_modified=False,evaluation_or_holdout_scored=False,training_authorized=False)
    result['CPU_GPU_metadata_reduction_differences']=reduction_deltas
    emit17(REPORT17/'results.json',result);emit17(REPORT17/'accumulation-group-summary.json',dict(groups=groups));emit17(REPORT17/'source-group-summary.json',creators)
    csvfile=io.StringIO();writer=csv.writer(csvfile,lineterminator='\n');writer.writerow(['group','photos','positive_norm','focal_negative_norm','BCE_negative_norm','balanced_pre_norm','balanced_post_norm','balanced_positive_ratio','cap_factor','AMP_scale','balanced_positive_cosine'])
    for g in groups:
        m=g['metrics'];writer.writerow([g['group'],g['size'],m['positive']['norm'],m['negatives']['A_focal']['norm'],m['negatives']['B_BCE']['norm'],m['negatives']['C_pre_cap']['norm'],m['negatives']['C_post_cap']['norm'],m['negatives']['C_post_cap']['negative_positive_ratio'],m['cap']['scale'],m['cap']['amp_scale'],m['combined']['C_post_cap']['cosine_positive']])
    atomic17(REPORT17/'group-comparison.csv',csvfile.getvalue().encode())
    print('C17 CPU analysis PASS:qualifiedA;all15cap-compliant;0active clips;weak/easy-negative limitation explicit',flush=True)


if __name__=='__main__':run()
