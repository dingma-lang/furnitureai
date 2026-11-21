"""One matched head-only arm. CPU metrics never run inside this GPU process."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time,resource,threading,traceback
from common8 import *

def run(arm):
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));binding=read(RUN/'binding.json');prepared=verify_binding(binding);train,dev=subsets(prepared)
    if arm not in ARMS:raise PermissionError('Only the two approved arms')
    folder=RUN/arm;folder.mkdir(exist_ok=False);save(folder/'started.json',dict(arm=arm,optimizer_updates=0));settings=binding['configs'][arm];corrected=arm==ARMS[1]
    queue=read(RUN/'queue-started.json');deadline=queue['monotonic_start']+3600;updates=0;forwards=0;train_backwards=0;probe_grad_calls=0;step_hook=[0];started=time.monotonic();stop=threading.Event();violations=[];mem_samples=[]
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from expanded import configure_heads,HEAD_KEYS,HEAD_PARAMETERS
        from masks6 import validate_bundle,authorized
        from loss8 import objective,gradient_norm
        from checkpoint8 import save_snapshot
        from expanded import from_square,cxcywh_to_xyxy
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
        torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction((MAX_VRAM-512*2**20)/total,0);torch.cuda.reset_peak_memory_stats(0)
        def monitor():
            while not stop.wait(.05):
                try:
                    free,total_memory=torch.cuda.mem_get_info(0);used=total_memory-free;mem_samples.append(used)
                    if used>MAX_VRAM:violations.append('Whole-device VRAM above6.5GiB');return
                except Exception as e:violations.append('Memory monitor failed: '+str(e));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        def guard():
            if time.monotonic()>=deadline:raise RuntimeError('One-hour queue runtime bound')
            if violations:raise RuntimeError('; '.join(violations))
            if max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM:raise RuntimeError('Torch VRAM bound')
            if (RUN/'STOP').exists():raise RuntimeError('Graceful stop requested')
            for p in (SOURCE/'source-negative-evidence.json',ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json',ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json'):
                if sha(p)!=binding['files'][str(p.relative_to(ROOT))]:raise ValueError('Supervision/partition integrity changed during execution')
            if sum(p.stat().st_blocks*512 for p in RUN.rglob('*') if p.is_file())>4*2**30:raise RuntimeError('Conservative4GiB output bound')
        guard();model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False);configure_heads(model)
        heads={n:p for n,p in model.named_parameters() if p.requires_grad};class_heads={n:p for n,p in heads.items() if n.startswith('class_head.')}
        if set(heads)!=HEAD_KEYS or sum(p.numel() for p in heads.values())!=HEAD_PARAMETERS:raise ValueError('Gradient parameter scope changed')
        frozen=parameter_hash(model,True);initial_heads=parameter_hash(model,False)
        if arm==ARMS[1]:
            control=read(RUN/ARMS[0]/'complete.json')
            if control['updates']!=75 or control['initial_heads_sha256']!=initial_heads or control['frozen_sha256']!=frozen:raise ValueError('A/B initialization or control completion mismatch')
        optimizer=torch.optim.AdamW(list(heads.values()),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2000)
        optimizer.register_step_post_hook(lambda *_:step_hook.__setitem__(0,step_hook[0]+1))
        bundle=read(SOURCE/'source-negative-evidence.json');frozen_candidates=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen_candidates};validate_bundle(bundle,prepared['rows'],frozen_candidates,observations)
        if corrected:authorized(settings['source_configuration'],bundle,prepared)
        selected={r['image_id']:[e for e in bundle['records'] if e['image_id']==r['image_id']] for r in train};reg={p['image_id']:p for p in registry()['photos'] if p['role']=='optimization'}
        protected={r['image_id']:[p['box'] for p in reg[r['image_id']]['protected']]+[p['annotation']['box'] for p in frozen_candidates if p['image_id']==r['image_id'] and observations[p['candidate_id']]['visual_outcome']=='mixed_or_uncertain'] for r in train}
        probe_ids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']];byid={r['image_id']:r for r in train}
        def scope():
            if set(n for n,p in model.named_parameters() if p.requires_grad)!=HEAD_KEYS or {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in heads.values()}:raise ValueError('Trainable/optimizer scope drift')
            if any(p.grad is not None for n,p in model.named_parameters() if n not in HEAD_KEYS):raise ValueError('Frozen module gradient')
        def forward(row,gradient):
            nonlocal forwards
            part='optimization' if gradient else 'inner_development';p=image_path(row,part,prepared)
            with Image.open(p) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image frame changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            guard();forwards+=1
            if gradient:
                with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
            else:
                with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
            del batch;guard();return out
        def evaluate_raw(epoch):
            model.eval();outfolder=folder/'evaluation'/('epoch-'+str(epoch));outfolder.mkdir(parents=True,exist_ok=False);records=[];times=[]
            for index,row in enumerate(dev):
                t=time.monotonic();out=forward(row,False);torch.cuda.synchronize(0);score,label=out.logits[0].float().sigmoid().max(-1);box=from_square(cxcywh_to_xyxy(out.pred_boxes[0].float()),row['width'],row['height']).clamp(0,1)
                arrays=dict(logits=out.logits[0].float().cpu().numpy(),square_cxcywh=out.pred_boxes[0].float().cpu().numpy(),scores=score.cpu().numpy(),labels=label.cpu().numpy(),boxes=box.cpu().numpy());p=outfolder/(str(index+1).zfill(2)+'.npz');save_npz(p,arrays)
                receipt=dict(index=index+1,image_id=row['image_id'],source_sha256=row['sha256'],file=p.name,sha256=sha(p));save(outfolder/(str(index+1).zfill(2)+'.receipt.json'),receipt);records.append(receipt);times.append(time.monotonic()-t);del out,arrays,score,label,box
            save(outfolder/'raw-complete.json',dict(arm=arm,epoch=epoch,updates=updates,records=records,rows_sha256=digest(dev),FP32=True,seconds_per_image=sum(times)/19));configure_heads(model)
        def probe(epoch):
            nonlocal probe_grad_calls
            configure_heads(model);optimizer.zero_grad(set_to_none=True);initial=parameter_hash(model,False);outfolder=folder/'probes'/('epoch-'+str(epoch));outfolder.mkdir(parents=True,exist_ok=False);records=[]
            for index,ident in enumerate(probe_ids):
                row=byid[ident];out=forward(row,True);loss,parts,stats=objective(out.logits[0],out.pred_boxes[0],row,selected[ident],protected[ident],corrected);scale=scaler.get_scale()
                added_grad=torch.autograd.grad(parts['added']*scale,tuple(class_heads.values()),retain_graph=True);used_grad=torch.autograd.grad(parts['classification']*scale,tuple(class_heads.values()));probe_grad_calls+=2
                if any(not torch.isfinite(g).all() for g in added_grad+used_grad):raise ValueError('Nonfinite diagnostic gradient; stop arm')
                stats.update(image_id=ident,epoch=epoch,updates=updates,grad_scaler_scale=scale,
                    additional_negative_gradient_norm=gradient_norm(added_grad,scale),used_classification_gradient_norm=gradient_norm(used_grad,scale),
                    additional_gradient_applied_in_arm=corrected,additional_nonzero_gradient_elements=sum(int((g!=0).sum()) for g in added_grad),
                    class_head_additional_gradient_norms={n:gradient_norm([g],scale) for n,g in zip(class_heads,added_grad)},
                    class_head_used_gradient_norms={n:gradient_norm([g],scale) for n,g in zip(class_heads,used_grad)},probe_does_not_update_weights=True)
                arrays=dict(logits=out.logits[0].detach().float().cpu().numpy(),square_cxcywh=out.pred_boxes[0].detach().float().cpu().numpy());raw=outfolder/(str(index+1).zfill(2)+'.npz');save_npz(raw,arrays);stats['query_file_sha256']=sha(raw);stats['query_array_fingerprint']=hashlib.sha256(arrays['logits'].tobytes()+arrays['square_cxcywh'].tobytes()).hexdigest()
                if epoch==0 and corrected:
                    old=read(RUN/ARMS[0]/'probes/epoch-0'/(str(index+1).zfill(2)+'.json'))
                    if old['query_array_fingerprint']!=stats['query_array_fingerprint'] or old['additional_negative_gradient_norm']!=stats['additional_negative_gradient_norm']:raise ValueError('Initial A/B model-query/gradient conditions differ')
                save(outfolder/(str(index+1).zfill(2)+'.json'),stats);records.append(stats);del out,loss,parts,added_grad,used_grad,arrays
            if parameter_hash(model,False)!=initial or any(p.grad is not None for p in heads.values()):raise ValueError('Diagnostic probe modified weights or accumulated training gradients')
            save(outfolder/'complete.json',dict(arm=arm,epoch=epoch,updates=updates,photos=24,added_terms=sum(r['added_active_Chair_terms'] for r in records),regions_covered=len({k for r in records for k,n in r['region_query_counts'].items() if n}),
                mean_added_loss_float64=sum(r['added_Chair_negative_loss_float64'] for r in records)/24,mean_added_gradient_norm=sum(r['additional_negative_gradient_norm'] for r in records)/24,
                mean_used_class_gradient_norm=sum(r['used_classification_gradient_norm'] for r in records)/24,weights_unchanged=True))
            print('PROBE',arm,'epoch',epoch,'terms',sum(r['added_active_Chair_terms'] for r in records),'negativeLoss',sum(r['added_Chair_negative_loss_float64'] for r in records)/24,flush=True)
        def snapshot(epoch,next_group):
            payload=dict(binding_sha256=sha(RUN/'binding.json'),arm=arm,updates=updates,epoch=epoch,next_group=next_group,frozen_sha256=frozen,
                weights={n:p.detach().cpu().clone() for n,p in heads.items()},optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            return save_snapshot(folder,payload)
        save(folder/'initialization.json',dict(initial_heads_sha256=initial_heads,frozen_sha256=frozen,trainable_keys=sorted(heads),parameters=HEAD_PARAMETERS,seed=6106))
        probe(0)
        if arm==ARMS[0]:evaluate_raw(0)
        initial_receipt=snapshot(0,0);last_receipt=initial_receipt
        for epoch in range(5):
            groups=grouped_epoch(train,epoch)
            for group_index,group in enumerate(groups):
                guard();configure_heads(model);optimizer.zero_grad(set_to_none=True);values=[]
                for row in group:
                    out=forward(row,True);loss,parts,stats=objective(out.logits[0],out.pred_boxes[0],row,selected[row['image_id']],protected[row['image_id']],corrected)
                    scaler.scale(loss/len(group)).backward();train_backwards+=1;stats.update(image_id=row['image_id'],epoch=epoch+1,next_update=updates+1);values.append(stats);del out,loss,parts
                scaler.unscale_(optimizer);scope()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads.values()):raise ValueError('Missing/nonfinite training gradient; optimizer not called')
                cls_norm=gradient_norm([p.grad for p in class_heads.values()]);all_norm=gradient_norm([p.grad for p in heads.values()]);torch.nn.utils.clip_grad_norm_(list(heads.values()),1.,error_if_nonfinite=True)
                if updates>=75:raise RuntimeError('75update arm cap')
                completed_other=sum(read(RUN/a/'complete.json')['updates'] for a in ARMS if a!=arm and (RUN/a/'complete.json').exists())
                if completed_other+updates>=150:raise RuntimeError('150update queue cap')
                guard();save(folder/('update-'+str(updates+1).zfill(3)+'.intent.json'),dict(optimizer_update_reserved=updates+1,epoch=epoch+1,image_ids=[r['image_id'] for r in group]));scaler.step(optimizer)
                if step_hook[0]!=updates+1:raise RuntimeError('Optimizer skipped/duplicated; no replay')
                scaler.update();updates+=1
                if any(not torch.isfinite(p).all() for p in heads.values()):raise ValueError('Nonfinite updated parameters')
                if any(int(v['step'])!=updates for v in optimizer.state.values()):raise ValueError('Optimizer state counter mismatch')
                last_receipt=snapshot(epoch,group_index+1)
                save(folder/('update-'+str(updates).zfill(3)+'.json'),dict(arm=arm,updates=updates,epoch=epoch+1,images=values,class_head_gradient_norm=cls_norm,total_head_gradient_norm=all_norm,
                    receipt=last_receipt,scaler_scale=scaler.get_scale(),source_order=[r['image_id'] for r in group]));print('UPDATE',arm,updates,'/75','epoch',epoch+1,'classGrad',round(cls_norm,6),flush=True)
            optimizer.zero_grad(set_to_none=True)
            if parameter_hash(model,True)!=frozen:raise ValueError('Frozen parameters changed')
            if epoch+1 in PROBE_EPOCHS:probe(epoch+1)
            if epoch+1 in EVAL_EPOCHS:evaluate_raw(epoch+1)
        scope()
        if updates!=75 or step_hook[0]!=75 or parameter_hash(model,True)!=frozen:raise ValueError('Completed arm update/frozen scope mismatch')
        for n,h in read(ROOT/'reports/phase10a-v1/input-contract.json')['model_files'].items():
            if sha(BASE/n)!=h:raise ValueError('Original weights/processor changed')
        stop.set();thread.join(timeout=2);guard()
        save(folder/'complete.json',dict(status='PASS',arm=arm,updates=updates,epochs=5,initial_heads_sha256=initial_heads,final_heads_sha256=parameter_hash(model,False),frozen_sha256=frozen,
            final_checkpoint=last_receipt,initial_checkpoint=initial_receipt,model_forwards=forwards,training_backwards=train_backwards,diagnostic_autograd_calls=probe_grad_calls,
            seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),peak_device_used_bytes=max(mem_samples,default=0),
            evaluation_predictions_saved_before_metrics=True,nonfinite_losses_or_gradients=0,optimizer_hook_calls=step_hook[0]))
        print('ARM COMPLETE',arm,updates,'updates;',round(time.monotonic()-started,2),'seconds',flush=True)
    except BaseException as exc:
        save(folder/'failure.json',dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),updates_committed_or_applied=updates,optimizer_hook_calls=step_hook[0],model_forwards=forwards,seconds=time.monotonic()-started,automatic_retry=False));raise
    finally:stop.set()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);run(p.parse_args().arm)
