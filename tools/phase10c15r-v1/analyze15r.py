"""Combine four immutable prior receipts with20recovered;no model inference."""
import csv,io,zipfile
from collections import Counter
import numpy as np
from common15r import *


def validated_records():
    cfg,prepared,rows,reg=inputs_r();prior,reused,_=verify_prior();done=read(RUNR/'complete.json')
    if done['new_forwards']!=20 or done['cumulative_forwards']!=25 or done['new_gradient_calculations']!=140 or done['model_state_before']!=done['model_state_after'] or done['model_state_before']!=cfg['initial_model_sha256']:raise ValueError('Recovery count/model integrity mismatch')
    _,train,dev,_,_=training_context();byid={r['image_id']:r for r in train}
    from common6 import candidates
    from loss15 import controlled_losses
    import torch
    frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');records=[];index=[]
    for i,ident in enumerate(cfg['photo_ids'],1):
        folder=PRIOR_RUN if i<=4 else RUNR;p=folder/(str(i).zfill(2)+'.json');r=read(p);row=byid[ident];require_training_row(row,train,dev)
        expected=cfg['reused_records'][i-1] if i<=4 else done['records'][i-5]
        if sha(p)!=expected['sha256'] or r['image_id']!=ident or r['identity_group']!=row['identity_group'] or r['source_sha256']!=row['sha256'] or sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Receipt/photo identity changed')
        raw=folder/r['raw_file']
        if sha(raw)!=r['raw_sha256']:raise ValueError('Raw diagnostic hash mismatch')
        with zipfile.ZipFile(raw) as z:
            if len(z.infolist())!=10 or sum(x.file_size for x in z.infolist())>1_000_000:raise ValueError('Unsafe compact query archive')
        with np.load(raw,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        cached=load_npz(C11_RUN/'probes/epoch-5'/(str(i).zfill(2)+'.npz'))
        if any(not np.isfinite(v).all() for v in a.values()) or any(not np.array_equal(a[k],cached[k]) for k in ('logits','square_cxcywh')):raise ValueError('Malformed/nonidentical model output')
        e=[e for e in bundle['records'] if e['image_id']==ident];protected=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        losses,masks,stats=controlled_losses(torch.from_numpy(a['logits']).requires_grad_(),torch.from_numpy(a['square_cxcywh']).requires_grad_(),row,e,protected)
        for key,saved_key in [('target','target'),('mask','mask'),('added','source_added')]:
            if not np.array_equal(masks[key].detach().numpy(),a[saved_key]):raise ValueError('Reused/recovered mask differs from unchanged evidence')
        if stats['protected_violations'] or r['counts']['protected_violations'] or not r['parameter_grads_all_none']:raise ValueError('Protected mask or gradient accumulation violation')
        for name in ('positive','source_focal','source_BCE'):
            g=a['output_grad_'+name]
            if g.shape!=(3600,7) or g.dtype!=np.float32 or (g[~a['mask'].astype(bool)]!=0).any() or (name.startswith('source') and (g[~a['source_added'].astype(bool)]!=0).any()):raise ValueError('Invalid saved output gradient')
            if not np.isclose(np.abs(g.astype(float)).sum(),r['losses_and_gradients'][name]['output_logit_L1'],atol=1e-10,rtol=1e-6):raise ValueError('Derivative summary inconsistent with saved arrays')
        if i>4 and (r['model_state_sha256']!=cfg['initial_model_sha256'] or r['photo_index']!=i or r['actual_forward']!=i-4):raise ValueError('Recovery model fingerprint/order drift')
        if int(a['source_added'].sum())!=r['counts']['source_negative'] or int(a['mask'].sum())+r['counts']['ignored']!=3600*7:raise ValueError('Supervision count mismatch')
        for name,loss in losses.items():
            if not np.isclose(float(loss.detach()),r['losses_and_gradients'][name]['loss'],atol=1e-5,rtol=1e-4):raise ValueError('CPU replay differs from persisted GPU loss')
        records.append(dict(r,photo_index=i,result_origin='prior_reused' if i<=4 else 'recovered'))
        index.append(dict(index=i,image_id=ident,identity_group=row['identity_group'],origin=records[-1]['result_origin'],receipt=str(p.relative_to(ROOT)),sha256=sha(p),raw_file=str(raw.relative_to(ROOT)),raw_sha256=sha(raw)))
    return records,index,cfg,done


def run():
    records,index,cfg,done=validated_records();components={};names=['positive','native_negative','source_focal','source_BCE','class_A','class_B','box']
    for name in names:
        values=[r['losses_and_gradients'][name] for r in records]
        components[name]=dict(mean_per_photo_loss=float(np.mean([v['loss'] for v in values])),sum_per_photo_output_logit_L1=sum(v['output_logit_L1'] for v in values),mean_per_photo_projection_L2=float(np.mean([v['projection_L2'] for v in values])),projection_L2_quantiles=quantiles([v['projection_L2'] for v in values]),mean_per_photo_box_head_L2=float(np.mean([v['box_head_L2'] for v in values])),max_individual_output_logit_gradient=max(v['max_abs_output_logit_gradient'] for v in values),max_combined_projection_element=max(v['projection_max_abs'] for v in values))
    pergroup=[]
    for r in records:
        d=r['losses_and_gradients'];p=d['positive']['projection_L2'];f=d['source_focal']['projection_L2'];b=d['source_BCE']['projection_L2']
        pergroup.append(dict(index=r['photo_index'],image_id=r['image_id'],identity_group=r['identity_group'],origin=r['result_origin'],positive_terms=r['counts']['positive'],source_terms=r['counts']['source_negative'],source_score_quantiles=quantiles(r['source_negative_scores']),positive_projection_L2=p,focal_projection_L2=f,BCE_projection_L2=b,BCE_to_positive_projection_ratio=b/p if p else None,BCE_to_focal_projection_ratio=b/f if f else None,positive_vs_BCE_cosine=r['projection_cosines']['positive_vs_source_BCE'],class_A_vs_B_cosine=r['projection_cosines']['classification_A_vs_B'],focal_vs_BCE_cosine=r['projection_cosines']['source_focal_vs_BCE']))
    scores=[s for r in records for s in r['source_negative_scores']];covered={k for r in records for k,v in r['source_region_query_counts'].items() if v};active=[r for r in pergroup if r['source_terms']>0];ratios=[r['BCE_to_positive_projection_ratio'] for r in active];opposed=[r['positive_vs_BCE_cosine'] for r in active]
    groupstats=dict(known_groups=len({r['identity_group'] for r in records}),groups_with_source_queries=len(active),groups_without_source_queries=24-len(active),BCE_nonzero_projection_groups=sum(r['BCE_projection_L2']>0 for r in pergroup),focal_nonzero_projection_groups=sum(r['focal_projection_L2']>0 for r in pergroup),BCE_exceeds_positive_projection_groups=sum(r['BCE_to_positive_projection_ratio']>1 for r in active),BCE_over10x_positive_projection_groups=sum(r['BCE_to_positive_projection_ratio']>10 for r in active),classification_direction_reversed_groups=sum(r['class_A_vs_B_cosine']<0 for r in pergroup),BCE_opposes_positive_groups=sum(x<0 for x in opposed),BCE_positive_ratio_quantiles_covered=quantiles(ratios),BCE_positive_cosine_quantiles_covered=quantiles(opposed),classification_A_B_cosine_quantiles=quantiles([r['class_A_vs_B_cosine'] for r in pergroup]))
    if len(records)!=24 or len(scores)!=281 or len(covered)!=24:raise ValueError('Full frozen C11 source coverage changed')
    summary=dict(status='COMPLETE',decision='B_EXCESSIVE_AT_UNCHANGED_COEFFICIENTS',interpretation='BCE restores a strong source-negative signal,but uneven magnitude and opposition reverse classification directions across many groups;this is suppressive risk,not observed recall regression',new_forwards=20,cumulative_forwards=25,reused_photos=4,recovered_photos=20,photos=24,new_gradient_calculations=140,cumulative_gradient_calculations=168,optimizer_steps=0,model_updates=0,counts={k:sum(r['counts'][k] for r in records) for k in records[0]['counts']},existing_regions=41,covered_regions=len(covered),uncovered_regions=41-len(covered),components=components,source_group_variability=groupstats,source_negative_confidence=dict(quantiles=quantiles(scores),below001=sum(s<.01 for s in scores),below005=sum(s<.05 for s in scores),above015=sum(s>=.15 for s in scores),above05=sum(s>=.5 for s in scores)),output_negative_BCE_to_focal=components['source_BCE']['sum_per_photo_output_logit_L1']/components['source_focal']['sum_per_photo_output_logit_L1'],mean_projection_BCE_to_focal=components['source_BCE']['mean_per_photo_projection_L2']/components['source_focal']['mean_per_photo_projection_L2'],resource=done['resource'],recovery_model_hash_before=done['model_state_before'],recovery_model_hash_after=done['model_state_after'],all20_recovery_full_fingerprints_unchanged=True,original_first_attempt_missing_measurements=dict(peak_VRAM=None,terminal_live_model_hash=None),full24_signed_projection_gradient_vector_unavailable=True,recovery20_only_mean_parameter_gradients=done['recovery20_only_mean_parameter_gradients'],recovery20_only_projection_cosines=done['recovery20_only_projection_cosines'],aggregation='Output-gradientL1 summed across per-GT-normalized photos;projection norm mean across24photos/known groups,not norm of full24mean gradient;20-only mean vectors explicitly separate',box_caveat='Four reused box-gradient summaries were calculated before the FP32 ordering repair;their loss values matched original,but exact GPU box-gradient bit identity was not established. Classification/source gradients are independent and unchanged.',precision_recall_or_AP_measured=False,training_authorized=False,new_negative_labels=0,automatic_retry=False)
    emit_r(RUNR/'combined-index.json',dict(records=index,original_first_attempt_unmodified=True));emit_r(REPORTR/'results.json',summary);emit_r(REPORTR/'source-group-summary.json',dict(records=pergroup,known_groups_not_proof_of_independent_physical_rooms=True))
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(['photo_index','origin','positive_terms','source_negative_terms','positive_projection_L2','focal_negative_projection_L2','BCE_negative_projection_L2','BCE_to_positive_ratio','positive_vs_BCE_cosine','classification_A_vs_B_cosine'])
    for r in pergroup:writer.writerow([r[k] for k in ['index','origin','positive_terms','source_terms','positive_projection_L2','focal_projection_L2','BCE_projection_L2','BCE_to_positive_projection_ratio','positive_vs_BCE_cosine','class_A_vs_B_cosine']])
    atomic_r(REPORTR/'source-group-comparison.csv',stream.getvalue().encode());print('COMBINED24 CPU AUDIT PASS;BCE excessive at current coefficients;no training',flush=True)


if __name__=='__main__':run()
