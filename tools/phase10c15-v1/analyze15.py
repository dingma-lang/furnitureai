"""CPU-only partial evidence and failure audit;never launches a model/retry."""
import csv,io,zipfile
import numpy as np
from common15 import *


def run():
    failure=read(RUN15/'failure.json');launch=read(RUN15/'launch-result.json');cfg=read(RUN15/'authorized-config.json')
    if failure['forwards']!=5 or failure['gradient_calculations']!=28 or launch['exit_code']!=1 or (RUN15/'complete.json').exists():raise ValueError('Expected stopped one-attempt partial run,not full completion')
    archive=read(RUN15/'executed-source-v1.json')
    for relative,expected in archive['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Executed source archive changed')
        original=HERE15/(ROOT/relative).name
        if cfg['files'][str(original.relative_to(ROOT))]!=expected:raise ValueError('Executed code archive not bound to actual approval')
    _,train,dev,_,reg=training_context();byid={r['image_id']:r for r in train};records=[]
    for i,ident in enumerate(cfg['photo_ids'][:4],1):
        path=RUN15/(str(i).zfill(2)+'.json');r=read(path);row=byid[ident];require_training_row(row,train,dev)
        if r['image_id']!=ident or r['source_sha256']!=row['sha256'] or sha(ROOT/row['path'])!=row['sha256'] or r['identity_group']!=row['identity_group'] or not r['parameter_grads_all_none']:raise ValueError('Partial receipt/source identity or zero-accumulation mismatch')
        raw=RUN15/r['raw_file']
        if sha(raw)!=r['raw_sha256']:raise ValueError('Saved raw query/gradient hash mismatch')
        with zipfile.ZipFile(raw) as z:
            if len(z.infolist())!=10 or sum(x.file_size for x in z.infolist())>1_000_000:raise ValueError('Unexpected compact diagnostic archive')
        with np.load(raw,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        cached=load_npz(C11_RUN/'probes/epoch-5'/(str(i).zfill(2)+'.npz'))
        if any(not np.isfinite(v).all() for v in a.values()) or not np.array_equal(a['logits'],cached['logits']) or not np.array_equal(a['square_cxcywh'],cached['square_cxcywh']):raise ValueError('Same-output/cache integrity mismatch')
        mask=a['mask'].astype(bool);added=a['source_added'].astype(bool);target=a['target'].astype(bool)
        if (added&target).any() or added[:,[0,2,3,4,5,6]].any() or (added&~mask).any():raise ValueError('Invalid source target expansion')
        for name in ['positive','source_focal','source_BCE']:
            g=a['output_grad_'+name]
            if g.shape!=(3600,7) or (g[~mask]!=0).any() or (name.startswith('source') and (g[~added]!=0).any()):raise ValueError('Unknown/protected output received source gradient')
            if not np.isclose(np.abs(g.astype(float)).sum(),r['losses_and_gradients'][name]['output_logit_L1'],atol=1e-10,rtol=1e-6):raise ValueError('Gradient summary mass inconsistent with saved arrays')
        if int(added.sum())!=r['counts']['source_negative']:raise ValueError('Mask count mismatch')
        # Validate original current source evidence and protection from saved outputs.
        from loss15 import controlled_losses
        from common6 import candidates
        import torch
        frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};e=[x for x in read(SOURCE/'source-negative-evidence.json')['records'] if x['image_id']==ident];protected=[p['box'] for p in reg[ident]['protected']]+[p['annotation']['box'] for p in frozen if p['image_id']==ident and obs[p['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
        _,mm,ss=controlled_losses(torch.from_numpy(a['logits']).requires_grad_(),torch.from_numpy(a['square_cxcywh']).requires_grad_(),row,e,protected)
        if not np.array_equal(mm['added'].numpy(),added) or not np.array_equal(mm['mask'].numpy(),mask) or ss['protected_violations']:raise ValueError('Saved source/protected mask differs from current validated evidence')
        records.append(r)
    aggregate={}
    for component in ['positive','native_negative','source_focal','source_BCE','class_A','class_B','box']:
        values=[r['losses_and_gradients'][component] for r in records]
        aggregate[component]=dict(mean_per_photo_loss=float(np.mean([v['loss'] for v in values])),sum_per_photo_output_logit_L1=sum(v['output_logit_L1'] for v in values),mean_per_photo_projection_L2=float(np.mean([v['projection_L2'] for v in values])),mean_per_photo_box_head_L2=float(np.mean([v['box_head_L2'] for v in values])),max_individual_output_logit_gradient=max(v['max_abs_output_logit_gradient'] for v in values),max_combined_source_projection_element=max(v['projection_max_abs'] for v in values))
    scores=[s for r in records for s in r['source_negative_scores']];covered={key for r in records for key,value in r['source_region_query_counts'].items() if value};cos=[r['projection_cosines']['positive_vs_source_BCE'] for r in records]
    summary=dict(status='STOPPED_PARTIAL',decision='D_INCONCLUSIVE',reason='Diagnostic implementation failed before completing24photos;terminal model-state hash and peak-memory telemetry were not persisted',actual_forwards=5,complete_photos=4,complete_source_groups=4,gradient_calculations=28,optimizer_steps=0,optimizer_constructed=False,model_updates=0,worker_seconds=failure['seconds'],launcher_wall_seconds=launch['wall_seconds'],peak_allocated_bytes=None,peak_observed_device_bytes=None,peak_memory_status='Unavailable:initial failure handler did not persist memory samples;no value reconstructed or estimated',terminal_live_model_hash_status='Unavailable after process exited;all4 saved receipts have model .grad None,no optimizer existed;checkpoint/source hashes independently verified',counts={k:sum(r['counts'][k] for r in records) for k in records[0]['counts']},covered_regions=len(covered),source_negative_score_quantiles=quantiles(scores),aggregate=aggregate,projection_positive_vs_BCE_cosines=cos,all_partial_BCE_projection_gradients_nonzero=all(r['losses_and_gradients']['source_BCE']['projection_L2']>0 for r in records),source_focal_zero_projection_photos=sum(r['losses_and_gradients']['source_focal']['projection_L2']==0 for r in records),aggregation='Output L1 is sum of per-photo normalized derivatives;parameter norms are mean per-photo norms,NOT norm of cohort-mean vector;partial cohort accumulator was not persisted',whole_cohort_effect_unmeasured=True,automatic_retry=False,training_ready=False,precision_recall_measured=False,new_negative_labels=0,new_GPU_after_failure=0,root_cause=dict(confirmed='New diagnostic reconstructed 5*L1_sum/nGT instead of original5*(L1_sum/nGT). FP32 operation order was changed and strict GPU equality guard stopped on fifth photo.',other_discovered_problem='Failure telemetry omitted peak VRAM and terminal live-model hash;future handler repaired without execution',CPU_fix='Normalize L1 first,then multiply by5,exactly as original;retain strict equality check',original_training_implementation_changed=False,executed_code_preserved='Private executed-source-v1/ archive;old config hashes remain bound to executed version',repaired_code_run_on_GPU=False))
    emit(REPORT15/'results.json',summary)
    public=[]
    for r in records:
        a=r['losses_and_gradients'];public.append(dict(image_id=r['image_id'],identity_group=r['identity_group'],counts=r['counts'],source_score_quantiles=quantiles(r['source_negative_scores']),losses_and_gradients=a,projection_cosines=r['projection_cosines'],C11_cached_outputs_bit_identical=True,parameter_grads_all_none=True,raw_sha256=r['raw_sha256']))
    emit(REPORT15/'per-image-summary.json',dict(records=public,scope='Four completed optimization-training photos only;not full cohort or detection evaluation'))
    recovery=dict(approved=False,GPU_authorized=False,training_authorized=False,Phase10D_authorized=False,action='Separately authorize inference/autograd-only recovery of missing20 gradient summaries with repaired FP32 ordering and telemetry',photos=[dict(index=i+1,image_id=ident,identity_group=byid[ident]['identity_group'],source_sha256=byid[ident]['sha256']) for i,ident in enumerate(cfg['photo_ids']) if i>=4],reused_completed_indices=[1,2,3,4],new_forwards=20,repeat_of_previously_forwarded_photo_indices=[5],total_forwards_including_failed_attempt=25,requires_explicit_revision_of_prior24_total_forward_limit=True,max_seconds=180,max_allocated_VRAM_bytes=MAX_VRAM,max_new_output_bytes=MAX_BYTES,optimizer_updates=0,initialization=cfg['head'],base_sha256=cfg['original_checkpoint_sha256'],same_seed=6106,preprocessing='Same pinned processor,prompts,FP16 forwards,FP32 loss andscale128',loss_change='Only repair FP32 normalization order in diagnostic box reconstruction;Focal/BCE masks and coefficients unchanged',destination='data/open-images/experiments/phase10c15-negative-gradient-recovery-v1/',destination_created=False,required_checks=['All24 saved-C11 CPU loss regressions pass','Reuse four completed receipts only after source/model/raw hashes verified','New same-output prediction equality checks remain strict','Persist per-photo peak VRAM and terminal model/buffer hash on success/failure','One launch only;stop on failure;no automatic training'],future_training_config_prepared=False,why_no_training_config='Full gradient diagnostic incomplete;insufficient basis to classify overall result A')
    emit(REPORT15/'recovery-proposal.json',recovery)
    print('C15 CPU PARTIAL AUDIT PASS:4photos,11source terms,5GPUforwards,28gradient calls;D_INCONCLUSIVE;no retry',flush=True)


if __name__=='__main__':run()
