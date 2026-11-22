"""One authorized FP32 box-only arm. No metrics, downloads or automatic retries."""
import random
import resource
import signal
import traceback
from commonD import *


def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cfg,prepared,train,dev,_=inputs()
    emit(RUN/'executed-once.json',dict(binding_sha256=sha(RUN/'binding.json'),retry=False))
    begin=read(RUN/'launch-once.json')['monotonic_start']
    state=dict(updates=0,optimizer_hook_calls=0,forwards=0,actual_forwards=0,phase='initializing',epoch=0,group=None,current_photo=None,training_backwards=0)
    requested=threading.Event();monitor_stop=threading.Event();violations=[]
    model=None;scope=None;opt=None;torch=None;thread=None;last=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:requested.set())

    def receipt(kind):
        result=dict(state,kind=kind,seconds=time.monotonic()-begin,allocated_output_bytes=allocated())
        if torch is not None and torch.cuda.is_initialized():
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return result

    def telemetry(kind):
        d=receipt(kind);emit(RUN/'telemetry'/('event-'+uuid.uuid4().hex+'.json'),d);return d

    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
        from scopeD import BoxScopeD,BOX_KEYS
        from box21 import frozen_fingerprint
        from common8 import image_path,PROMPTS,grouped_epoch
        from expanded import from_square,cxcywh_to_xyxy
        from telemetry19 import tensor_hash
        torch.set_num_threads(2);random.seed(6106);np.random.seed(6106);torch.manual_seed(6106)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no fallback/retry')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction((LIMITS['allocated_VRAM_bytes']-256*2**20)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats(0)

        def guard():
            budget_check(state,time.monotonic()-begin,torch.cuda.max_memory_allocated(0),allocated())
            if time.monotonic()-begin>=585 or requested.is_set() or violations or (RUN/'STOP').exists():
                raise RuntimeError('Graceful stop/deadline/resource boundary: '+str(violations))

        def monitor():
            while not monitor_stop.wait(2):
                try:
                    if torch.cuda.max_memory_allocated(0)>LIMITS['allocated_VRAM_bytes']:violations.append('Allocated VRAM cap')
                    telemetry('heartbeat')
                except BaseException as exc:violations.append('Telemetry failure: '+str(exc));return

        thread=threading.Thread(target=monitor,daemon=True);thread.start();guard()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
        persisted=model_state_matches_base(model)
        model=model.to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        scope=BoxScopeD(model);opt=scope.optimizer();scope.check(opt,check_hash=True)
        if scope.expected_frozen!=cfg['frozen_sha256'] or scope.derived_hashes!=cfg['unregistered_tensor_sha256']:raise ValueError('Actual GPU initial frozen state disagrees with CPU preflight')
        params=dict(model.named_parameters());heads={n:p for n,p in params.items() if n in BOX_KEYS}
        if any(p.dtype!=torch.float32 for p in params.values()):raise ValueError('FP32 only')
        def stepped(*_):state['optimizer_hook_calls']+=1
        opt.register_step_post_hook(stepped)
        def counted_forward(*_):
            state['actual_forwards']+=1
            if state['actual_forwards']!=state['forwards'] or state['actual_forwards']>322:
                raise RuntimeError('Unexpected detector forward without exact durable intent')
        model.register_forward_pre_hook(counted_forward)

        def mode(training):
            scope.check(opt)
            model.eval()
            if training:train_mode(scope,opt)
            scope.check(opt)

        def forward(row,optimization,training=False):
            guard();state['current_photo']=row['image_id']
            path=image_path(row,'optimization' if optimization else 'inner_development',prepared)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Original-image registration changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if list(batch['pixel_values'].shape)!=[1,3,960,960]:raise ValueError('Approved square preprocessing changed')
            consume_forward(state)
            emit(RUN/'forward-intents'/(str(state['forwards']).zfill(3)+'.json'),dict(state,source_sha256=row['sha256'],optimization=optimization,gradient_enabled=training,seconds=time.monotonic()-begin))
            moved={k:v.to('cuda') for k,v in batch.items()}
            torch.cuda.synchronize(0);start=time.monotonic()
            if training:out=model(**moved)
            else:
                with torch.inference_mode():out=model(**moved)
            torch.cuda.synchronize(0);latency=time.monotonic()-start
            if out.logits.shape!=(1,3600,7) or out.pred_boxes.shape!=(1,3600,4) or not torch.isfinite(out.logits).all() or not torch.isfinite(out.pred_boxes).all():raise ValueError('Actual prediction shape/nonfinite violation')
            if out.logits.requires_grad or (training and not out.pred_boxes.requires_grad):raise ValueError('Classification/box autograd isolation failed')
            guard();return out,latency

        def diagnostics(epoch):
            scope.check(opt,check_hash=True)
            values={}
            for n,p in heads.items():
                q=p.detach().cpu().double()
                values[n]=dict(sha256=tensor_hash(p),L2=float(q.norm()),minimum=float(q.min()),maximum=float(q.max()),mean=float(q.mean()))
            d=dict(epoch=epoch,updates=state['updates'],frozen_sha256=frozen_fingerprint(model),unregistered_tensor_sha256=scope.derived_hashes,box_parameters=values,resource=receipt('parameter_diagnostics'))
            emit(RUN/('parameters-epoch-'+str(epoch)+'.json'),d);return d

        def snapshot(epoch):
            nonlocal last
            guard();scope.check(opt,check_hash=True)
            payload=scope.checkpoint_payload(state['updates'],sha(RUN/'binding.json'))
            payload.update(optimizer=opt.state_dict(),epoch=epoch,next_group=0,seed=6106,rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),remaining_sequence_requires_separate_recovery_authorization=True)
            if any(int(v['step'])!=state['updates'] for v in opt.state.values()):raise ValueError('Optimizer state update count mismatch')
            stream=io.BytesIO();torch.save(payload,stream)
            if len(stream.getvalue())>16000000:raise RuntimeError('Bounded six-tensor snapshot exceeded16MB')
            path=RUN/'checkpoints'/('update-'+str(state['updates']).zfill(3)+'-'+uuid.uuid4().hex[:8]+'.pt')
            atomic(path,stream.getvalue());saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['binding_sha256']!=payload['binding_sha256'] or saved['frozen_sha256']!=cfg['frozen_sha256'] or set(saved['weights'])!=BOX_KEYS or any(not torch.equal(saved['weights'][k],payload['weights'][k]) for k in BOX_KEYS):raise ValueError('Saved checkpoint reload integrity failure')
            last=dict(file=str(path.relative_to(RUN)),sha256=sha(path),bytes=path.stat().st_size,updates=state['updates'],epoch=epoch,frozen_sha256=payload['frozen_sha256'])
            emit(path.with_suffix('.receipt.json'),last);guard();return last

        emit(RUN/'initialization.json',dict(original_checkpoint_sha256=BASE_SHA,persisted_tensors_exact_original=persisted,frozen_sha256=scope.expected_frozen,unregistered_tensor_sha256=scope.derived_hashes,trainable_keys=sorted(BOX_KEYS),trainable_parameters=1184260,optimizer_parameters=6,seed=6106,precision='FP32',device=torch.cuda.get_device_name(0),torch_version=torch.__version__,new_negatives_active=False,classification_loss=0))
        byid={r['image_id']:r for r in train};mode(False);diagnostics(0);snapshot(0)
        smoke=[]
        for i,ident in enumerate(cfg['smoke_ids'],1):
            state['phase']='zero-update-smoke';row=byid[ident];out,latency=forward(row,True)
            loss,parts=scope.loss(out.logits[0],out.pred_boxes[0],row)
            path=RUN/'smoke'/(str(i).zfill(2)+'.npz')
            arrays(path,dict(logits=out.logits[0].cpu().numpy(),square_cxcywh=out.pred_boxes[0].cpu().numpy()))
            smoke.append(dict(image_id=ident,source_sha256=row['sha256'],file=str(path.relative_to(RUN)),sha256=sha(path),latency_seconds=latency,loss=float(loss),parts=parts));del out,loss
        scope.check(opt,check_hash=True)
        if state['updates'] or opt.state or any(p.grad is not None for p in model.parameters()):raise ValueError('Zero-update smoke changed optimizer/gradients')
        emit(RUN/'smoke-pass.json',dict(status='PASS',forwards=4,updates=0,records=smoke,frozen_sha256=scope.expected_frozen,resource=receipt('smoke_pass')))
        print('SMOKE PASS4forwards;zero updates;classification frozen',flush=True)

        for epoch in range(5):
            state.update(phase='training',epoch=epoch+1);epoch_rows=[];epoch_gradients=[]
            groups=grouped_epoch(train,epoch)
            for group_index,group in enumerate(groups):
                state['group']=group_index;guard();mode(True);opt.zero_grad(set_to_none=True)
                if [r['image_id'] for r in group]!=cfg['orders'][epoch][group_index]:raise ValueError('Exact approved order changed')
                observations=[]
                for row in group:
                    out,latency=forward(row,True,training=True)
                    loss,parts=scope.loss(out.logits[0],out.pred_boxes[0],row)
                    (loss/len(group)).backward();state['training_backwards']+=1
                    observations.append(dict(image_id=row['image_id'],source_group=row['identity_group'],source_sha256=row['sha256'],loss=float(loss.detach()),parts=parts,latency_seconds=latency))
                    del loss,out
                scope.check(opt,require_gradients=True)
                grad={n:float(p.grad.detach().double().norm()) for n,p in heads.items()}
                total=float(torch.nn.utils.clip_grad_norm_(list(heads.values()),1.,error_if_nonfinite=True))
                if state['updates']>=75:raise RuntimeError('75 update cap before optimizer')
                guard();next_update=state['updates']+1
                emit(RUN/'updates'/('update-'+str(next_update).zfill(3)+'.intent.json'),dict(update=next_update,epoch=epoch+1,source_order=[r['image_id'] for r in group],binding_sha256=sha(RUN/'binding.json')))
                opt.step()
                if state['optimizer_hook_calls']!=next_update:raise RuntimeError('Skipped/duplicated optimizer step;no retry')
                state['updates']=next_update
                if any(not torch.isfinite(p).all() for p in heads.values()) or any(int(v['step'])!=next_update for v in opt.state.values()):raise ValueError('Nonfinite/update cursor violation')
                emit(RUN/'updates'/('update-'+str(next_update).zfill(3)+'.json'),dict(update=next_update,epoch=epoch+1,source_order=[r['image_id'] for r in group],images=observations,box_gradient_norm=total,per_parameter_gradient_norms=grad,classification_gradient_present=False,resource=receipt('update_committed')))
                epoch_rows.extend(observations);epoch_gradients.append(total)
                opt.zero_grad(set_to_none=True);scope.check(opt)
                print('UPDATE',next_update,'/75 epoch',epoch+1,'elapsed',round(time.monotonic()-begin,2),flush=True)
            diagnostics(epoch+1);cp=snapshot(epoch+1)
            emit(RUN/('epoch-'+str(epoch+1)+'.json'),dict(epoch=epoch+1,updates=state['updates'],photo_presentations=len(epoch_rows),matched_positive_targets=sum(r['parts']['matched'] for r in epoch_rows),mean_box_loss=float(np.mean([r['loss'] for r in epoch_rows])),mean_L1=float(np.mean([r['parts']['l1'] for r in epoch_rows])),mean_GIoU=float(np.mean([r['parts']['giou'] for r in epoch_rows])),mean_preclip_box_gradient_norm=float(np.mean(epoch_gradients)),frozen_sha256=scope.expected_frozen,checkpoint=cp,resource=receipt('epoch_complete')))

        state['phase']='frozen-query-logit-checks';mode(False);checks=[]
        for i,ref in enumerate(smoke,1):
            row=byid[ref['image_id']];out,latency=forward(row,True)
            with np.load(RUN/ref['file'],allow_pickle=False) as prior:
                z=out.logits[0].cpu().numpy();box=out.pred_boxes[0].cpu().numpy()
                exact=np.array_equal(z,prior['logits']);maximum=float(np.abs(z-prior['logits']).max());boxchange=float(np.abs(box-prior['square_cxcywh']).max())
            path=RUN/'logit-checks'/(str(i).zfill(2)+'.npz');arrays(path,dict(logits=z,square_cxcywh=box))
            result=dict(image_id=row['image_id'],classification_logits_bit_identical=bool(exact),maximum_absolute_logit_change=maximum,maximum_absolute_box_change=boxchange,sha256=sha(path),file=str(path.relative_to(RUN)),latency_seconds=latency)
            emit(path.with_suffix('.json'),result);checks.append(result);del out,z,box
            if not exact:raise ValueError('Original aligned FP32 classification logits changed')
        emit(RUN/'logit-checks/complete.json',dict(records=checks,all_original_logits_bit_identical=True,matching_precision='Same FP32 before/after,not FP16teacher cache'))
        state['phase']='final-internal-diagnostic';records=[]
        for i,row in enumerate(dev,1):
            out,latency=forward(row,False)
            z=out.logits[0];b=out.pred_boxes[0];scores,labels=z.sigmoid().max(-1)
            boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).clamp(0,1)
            values=dict(logits=z.cpu().numpy(),square_cxcywh=b.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),boxes=boxes.cpu().numpy())
            path=RUN/'evaluation'/(str(i).zfill(2)+'.npz');arrays(path,values)
            ref=dict(index=i,image_id=row['image_id'],source_sha256=row['sha256'],file=path.name,sha256=sha(path),latency_seconds=latency)
            emit(path.with_suffix('.receipt.json'),ref);records.append(ref);del out,z,b,scores,labels,boxes,values
        emit(RUN/'evaluation/raw-complete.json',dict(epoch=5,updates=75,rows_sha256=digest(dev),records=records,FP32=True,predictions_atomic_before_metrics=True))
        scope.check(opt,check_hash=True);guard()
        if state['updates']!=75 or state['forwards']!=322 or state['actual_forwards']!=322 or state['training_backwards']!=295 or state['optimizer_hook_calls']!=75 or sha(BASE/'model.safetensors')!=BASE_SHA:
            raise ValueError('Final counts/original preservation failed')
        state['phase']='complete';monitor_stop.set();thread.join(timeout=2)
        emit(RUN/'complete.json',dict(status='PASS',**receipt('complete'),epochs=5,final_checkpoint=last,frozen_sha256=scope.expected_frozen,classification_unchanged=True,original_file_unchanged=True,independent_or_reserved_scored=False,another_run=False,predictions_saved_before_metrics=True))
        print('PHASE10D COMPLETE;75updates;322forwards;CPU metrics only next',flush=True)
    except BaseException as exc:
        monitor_stop.set()
        failure=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),resource=receipt('failure'),automatic_retry=False,last_checkpoint=last,optimizer_state_changed=state['optimizer_hook_calls']>0,frozen_hash_current=None)
        if scope is not None and time.monotonic()-begin<590:
            try:failure['frozen_hash_current']=scope.expected_frozen if frozen_fingerprint(model)==scope.expected_frozen else 'CHANGED'
            except BaseException as e:failure['frozen_hash_unavailable']=str(e)
        try:emit(RUN/'failure.json',failure)
        except BaseException as e:exc.add_note('Failure telemetry persistence: '+str(e))
        raise
    finally:
        monitor_stop.set()
        if thread is not None:thread.join(timeout=2)


if __name__=='__main__':run()
