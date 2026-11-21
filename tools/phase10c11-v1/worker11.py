"""Phase10C.11 authorized execution; prepared algorithm plus detached diagnostics.

Imports and CUDA initialization occur only after a separately authorized config
and approval record. Historical workers are never patched or monkeypatched.
"""
import time,threading,traceback,uuid,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase10c10-v1'))
from common10 import *
from config10 import require_training_approval

def run():
    cfg=read(FUTURE/'authorized-config.json');approval=read(FUTURE/'approval-record.json');require_training_approval(cfg,approval)
    prepared,train,dev=verify_inputs();started=time.monotonic();deadline=read(FUTURE/'queue-started.json')['monotonic_start']+600
    updates=forwards=0;backwards=0;hook=[0];violations=[];stop=threading.Event();samples=[]
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from common8 import grouped_epoch,image_path,save_npz,PROMPTS
        from expanded import from_square,cxcywh_to_xyxy
        from loss8 import gradient_norm
        from observe11 import objective,verify_initial_outputs11
        from freeze10 import FrozenScope,head_weights,make_payload,restore_candidate
        from masks6 import validate_bundle,authorized
        from common6 import candidates,registry
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback or retry')
        torch.cuda.set_device(0);max_vram=cfg['limits']['max_vram_bytes'];torch.cuda.set_per_process_memory_fraction((max_vram-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def monitor():
            while not stop.wait(.05):
                try:
                    free,total=torch.cuda.mem_get_info(0);samples.append(total-free)
                    if total-free>max_vram:violations.append('Whole-device VRAM boundary');return
                except Exception as e:violations.append('VRAM monitor failed: '+str(e));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        def guard():
            if time.monotonic()>=deadline or violations or (FUTURE/'STOP').exists():raise RuntimeError('Runtime/memory/STOP boundary: '+str(violations))
            if max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>max_vram:raise RuntimeError('Allocator VRAM boundary')
            if sum(p.stat().st_blocks*512 for p in FUTURE.rglob('*') if p.is_file())>cfg['limits']['initial_output_storage_guard_bytes']:raise RuntimeError('4GiB output guard')
            for p in (SOURCE/'source-negative-evidence.json',ROOT/cfg['data']['manifest']):
                if sha(p)!=read(OUT/'input-contract.json')['files'][str(p.relative_to(ROOT))]:raise ValueError('Supervision source changed')
        guard();model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        scope=FrozenScope(model,original_heads());optimizer=scope.optimizer();scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2000);optimizer.register_step_post_hook(lambda *_:hook.__setitem__(0,hook[0]+1))
        bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};validate_bundle(bundle,prepared['rows'],frozen,observations);authorized(cfg['source_configuration'],bundle,prepared)
        reg={r['image_id']:r for r in registry()['photos'] if r['role']=='optimization'};selected={r['image_id']:[e for e in bundle['records'] if e['image_id']==r['image_id']] for r in train}
        protected={r['image_id']:[p['box'] for p in reg[r['image_id']]['protected']]+[p['annotation']['box'] for p in frozen if p['image_id']==r['image_id'] and observations[p['candidate_id']]['visual_outcome']=='mixed_or_uncertain'] for r in train}
        probe_ids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']];byid={r['image_id']:r for r in train};class_names=[n for n in scope.names if n.startswith('class_head.')]
        def forward(row,part):
            nonlocal forwards
            if forwards>=467:raise RuntimeError('Forward budget boundary')
            path=image_path(row,part,prepared)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source image registration changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            guard();forwards+=1
            if part=='optimization':
                with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
            else:
                with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
            del batch;guard();return out
        def probe(epoch):
            scope.train_mode();optimizer.zero_grad(set_to_none=True);before=named_sha(model,TRAIN_KEYS);folder=FUTURE/'probes'/('epoch-'+str(epoch));folder.mkdir(parents=True,exist_ok=False);records=[]
            for index,ident in enumerate(probe_ids,1):
                row=byid[ident];out=forward(row,'optimization');loss,parts,stats=objective(out.logits[0],out.pred_boxes[0],row,selected[ident],protected[ident],True);scale=scaler.get_scale()
                extra=torch.autograd.grad(parts['added']*scale,tuple(scope.params[n] for n in class_names),retain_graph=True);used=torch.autograd.grad(parts['classification']*scale,tuple(scope.params[n] for n in class_names))
                if any(not torch.isfinite(g).all() for g in extra+used):raise ValueError('Nonfinite probe gradient')
                stats.update(image_id=ident,grad_scaler_scale=scale,class_head_additional_gradient_norms={n:(gradient_norm([extra[class_names.index(n)]],scale) if n in class_names else None) for n in sorted(HEAD_KEYS) if n.startswith('class_head.')},class_head_used_gradient_norms={n:(gradient_norm([used[class_names.index(n)]],scale) if n in class_names else None) for n in sorted(HEAD_KEYS) if n.startswith('class_head.')},calibration_gradients_none=True,weights_unchanged=True)
                save(folder/(str(index).zfill(2)+'.json'),stats);save_npz(folder/(str(index).zfill(2)+'.npz'),dict(logits=out.logits[0].detach().float().cpu().numpy(),square_cxcywh=out.pred_boxes[0].detach().float().cpu().numpy()));records.append(stats);del out,loss,parts,extra,used
            if before!=named_sha(model,TRAIN_KEYS) or any(scope.params[n].grad is not None for n in TRAIN_KEYS):raise ValueError('Probe changed weights/accumulated gradients')
            scope.check(optimizer);save(folder/'complete.json',dict(epoch=epoch,updates=updates,records=24,added_terms=sum(r['added_active_Chair_terms'] for r in records),calibration_gradients_none=True))
        def evaluate_raw(epoch):
            scope.eval_mode();folder=FUTURE/'evaluation'/('epoch-'+str(epoch));folder.mkdir(parents=True,exist_ok=False);records=[]
            feature_records=[]
            for index,row in enumerate(dev,1):
                out=forward(row,'inner_development');score,label=out.logits[0].float().sigmoid().max(-1);box=from_square(cxcywh_to_xyxy(out.pred_boxes[0].float()),row['width'],row['height']).clamp(0,1)
                p=folder/(str(index).zfill(2)+'.npz');save_npz(p,dict(logits=out.logits[0].float().cpu().numpy(),square_cxcywh=out.pred_boxes[0].float().cpu().numpy(),scores=score.cpu().numpy(),labels=label.cpu().numpy(),boxes=box.cpu().numpy()))
                records.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],file=p.name,sha256=sha(p)))
                if epoch==0:
                    import numpy as np
                    fdir=FUTURE/'prehead-cache';fdir.mkdir(exist_ok=True);fp=fdir/(str(index).zfill(2)+'.npz');tmp=fp.with_suffix('.pending')
                    arrays=dict(image_features=out.image_embeds[0].reshape(3600,768).float().cpu().numpy(),text_features=out.text_embeds[0].float().cpu().numpy())
                    with tmp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
                    if tmp.stat().st_size>12000000:raise ValueError('Pre-head cache size bound')
                    os.link(tmp,fp);tmp.unlink();feature_records.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],file=fp.name,sha256=sha(fp),shape=[3600,768],preprocessing_binding=cfg['reference']['binding_sha256']))
                del out,score,label,box
            save(folder/'raw-complete.json',dict(epoch=epoch,updates=updates,records=records,rows_sha256=digest(dev),FP32=True))
            if epoch==0:save(FUTURE/'prehead-cache/manifest.json',dict(records=feature_records,backbone_sha256=scope.backbone_sha256,source='Actual approved initial FP32 model outputs only',synthetic=False))
            scope.train_mode()
        def snapshot(epoch,next_group):
            p=FUTURE/'checkpoints'/('update-'+str(updates).zfill(3)+'-'+uuid.uuid4().hex[:12]+'.pt');p.parent.mkdir(exist_ok=True)
            payload=make_payload(scope,optimizer,updates,sha(FUTURE/'authorized-config.json'),epoch=epoch,next_group=next_group,scaler=scaler.state_dict(),rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            tmp=p.with_suffix('.pending')
            with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.link(tmp,p);tmp.unlink();state=torch.load(p,map_location='cpu',weights_only=True)
            if state['updates']!=updates or any(not torch.equal(state['weights'][n],payload['weights'][n]) for n in HEAD_KEYS):raise ValueError('Snapshot integrity failed')
            rec=dict(file=str(p.relative_to(FUTURE)),sha256=sha(p),updates=updates);save(p.with_suffix('.receipt.json'),rec);return rec
        save(FUTURE/'initialization.json',dict(original_backbone_sha256=scope.backbone_sha256,frozen_sha256=scope.frozen_sha256,calibration_hashes={n:tensor_sha(t) for n,t in scope.calibration.items()},trainable_keys=scope.names))
        probe(0);evaluate_raw(0);verify_initial_outputs11();last=snapshot(0,0)
        for epoch in range(5):
            for group_index,group in enumerate(grouped_epoch(train,epoch)):
                guard();scope.train_mode();scope.check(optimizer);optimizer.zero_grad(set_to_none=True);values=[]
                for row in group:
                    out=forward(row,'optimization');loss,parts,stats=objective(out.logits[0],out.pred_boxes[0],row,selected[row['image_id']],protected[row['image_id']],True);scaler.scale(loss/len(group)).backward();backwards+=1;values.append(dict(stats,image_id=row['image_id']));del out,loss,parts
                scaler.unscale_(optimizer);norms=scope.gradients(optimizer);torch.nn.utils.clip_grad_norm_([scope.params[n] for n in scope.names],1.,error_if_nonfinite=True)
                if updates>=75:raise RuntimeError('75-update boundary')
                guard();save(FUTURE/('update-'+str(updates+1).zfill(3)+'.intent.json'),dict(update_reserved=updates+1,source_order=[r['image_id'] for r in group]));scaler.step(optimizer)
                if hook[0]!=updates+1:raise RuntimeError('Skipped/duplicated optimizer update; no replay')
                scaler.update();updates+=1;scope.check(optimizer)
                if any(not torch.isfinite(scope.params[n]).all() for n in TRAIN_KEYS):raise ValueError('Nonfinite updated weights')
                last=snapshot(epoch,group_index+1);save(FUTURE/('update-'+str(updates).zfill(3)+'.json'),dict(updates=updates,epoch=epoch+1,named_preclip_gradients=norms,images=values,receipt=last,source_order=[r['image_id'] for r in group],scaler_scale=scaler.get_scale()));print('UPDATE',updates,'/75',flush=True)
            optimizer.zero_grad(set_to_none=True);scope.check(optimizer,hash_frozen=True)
            if epoch+1 in (1,3,5):probe(epoch+1);evaluate_raw(epoch+1)
        if updates!=75 or hook[0]!=75 or forwards!=467:raise ValueError('Final execution count mismatch')
        scope.check(optimizer,hash_frozen=True);verify_inputs();stop.set();thread.join(timeout=2);guard()
        save(FUTURE/'complete.json',dict(updates=updates,epochs=5,forwards=forwards,training_backwards=backwards,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),peak_device_used_bytes=max(samples,default=0),calibration_unchanged=True,frozen_sha256=scope.frozen_sha256,original_backbone_sha256=scope.backbone_sha256,final_checkpoint=last,another_experiment_authorized=False))
    except BaseException as exc:
        save(FUTURE/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),updates_committed_or_applied=updates,optimizer_hook_calls=hook[0],forwards=forwards,automatic_retry=False));raise
    finally:stop.set()

if __name__=='__main__':run()
