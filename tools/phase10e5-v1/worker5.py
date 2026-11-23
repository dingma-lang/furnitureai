"""One original-initialized arm. Pinned E4 precision/objective,extended telemetry."""
import random,resource,signal,traceback
from common5 import *

def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(128*2**20,128*2**20))
    cfg,context,index=inputs();launch=read(RUN/'launch-once.json')
    if launch['nonce']!=os.environ.get('FURNITURE_E5_NONCE'):raise PermissionError('Use single bounded E5 launcher')
    emit(RUN/'executed-once.json',dict(binding_sha256=sha(RUN/'binding.json'),automatic_retry=False))
    begin=launch['monotonic_start'];state=dict(forwards=0,actual_forwards=0,contractions=0,updates=0,epochs=0,VJPs=0,backwards=0,photo=None,phase='initializing')
    requested=threading.Event();finished=threading.Event();thread=None;errors=[];torch=None;scope=None;model=None;optimizer=None;last=None;old_grad=None;old_TF32=None
    def resources():
        value=dict(state,seconds=time.monotonic()-begin,allocated_output_bytes=allocated())
        if torch is not None and torch.cuda.is_initialized():value.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return value
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:requested.set())
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from scopeE import StudentScope,TRAIN_KEYS,HEAD_KEYS,original_heads,frozen_state
        from teacherE import load_teacher
        from common8 import image_path,grouped_epoch
        from common10 import named_sha
        from objectiveE import selective_mask
        sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
        from precision4 import combined,box_objective
        sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
        from stable3 import Capture
        sys.path.insert(0,str(ROOT/'tools/phase10e2-v1'))
        from measure2 import replay
        sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
        from commonD import model_state_matches_base
        from expanded import from_square,cxcywh_to_xyxy
        from observe5 import gradient_diagnostics,combined_gate
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;old_TF32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((LIMITS['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def guard():
            d=resources();budget(state,d['seconds'],d['peak_allocated_bytes'],d['allocated_output_bytes'])
            if requested.is_set() or (RUN/'STOP').exists() or errors or d['seconds']>=590:raise RuntimeError('Graceful E5 resource/STOP boundary: '+str(errors))
        def monitor():
            while not finished.wait(2):
                try:
                    d=resources();budget(state,d['seconds'],d.get('peak_allocated_bytes',0),d['allocated_output_bytes'])
                    emit(RUN/'telemetry'/('heartbeat-'+uuid.uuid4().hex[:8]+'.json'),d)
                except BaseException as e:errors.append(str(e));requested.set();return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        entries=context['entries'];train=[e['row'] for e in entries];dev=context['development'];byid={e['row']['image_id']:e for e in entries};refs={r['image_id']:r for r in index['records']};prepared=dict(rows=train+dev)
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True);model_state_matches_base(model)
        model=model.to('cuda');scope=StudentScope(model);processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        params=[scope.params[n] for n in scope.names];projection=[scope.params['class_head.dense0.'+s] for s in ('weight','bias')];box_parameters=[scope.params[n] for n in sorted(TRAIN_KEYS) if n.startswith('box_head.')]
        original=original_heads();initial_trainable=named_sha(model,TRAIN_KEYS);initial_frozen=scope.immutable_state
        emit(RUN/'initial-state.json',dict(trainable_sha256=initial_trainable,frozen_sha256=initial_frozen,trainable_keys=scope.names,original_sha256=BASE_SHA,parameter_count=sum(p.numel() for p in params)))
        def changes():
            out={}
            for name in sorted(TRAIN_KEYS):
                delta=(scope.params[name].detach().cpu()-original[name]).double();denominator=float(original[name].double().norm())
                out[name]=dict(delta_L2=float(delta.norm()),delta_maximum=float(delta.abs().max()),relative_delta_L2=float(delta.norm())/denominator if denominator else None)
            return out
        old_grad=torch.autograd.grad
        def grad(*args,**kwargs):
            guard();state['VJPs']+=1;result=old_grad(*args,**kwargs);guard();return result
        def count_detector(*_):
            state['actual_forwards']+=1
            if state['actual_forwards']!=state['forwards'] or state['actual_forwards']>373:raise RuntimeError('Unplanned model forward')
        model.register_forward_pre_hook(count_detector)
        def forward(row,optimization=True):
            guard();state['photo']=row['image_id']
            with Image.open(image_path(row,'optimization' if optimization else 'inner_development',prepared)) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source dimensions/orientation mismatch')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if optimization:
                observed={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
                if observed!=refs[row['image_id']]['reconstructed_processor_tensors']:raise ValueError('Preprocessing/prompt/query correspondence mismatch')
            if state['forwards']>=373:raise RuntimeError('Forward cap BEFORE call')
            state['forwards']+=1;emit(RUN/'telemetry'/f"forward-{state['forwards']:03}.json",dict(state,source_sha256=row['sha256']))
            if optimization:
                if state['contractions']>=354:raise RuntimeError('Contraction cap BEFORE call')
                state['contractions']+=1
                with Capture(model.class_head,enabled=True) as capture,torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
                z=out.logits[0].float();b=out.pred_boxes[0].float();snapshot=capture.snapshot
            else:
                with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
                z=out.logits[0].float();b=out.pred_boxes[0].float();snapshot=None
            if not torch.isfinite(z).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite forward output')
            guard();return z,b,snapshot
        state['phase']='preupdate-gate';gates=[];initial_terms=0
        for number,entry in enumerate(entries,1):
            row=entry['row'];teacher=load_teacher(refs[row['image_id']],row,index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda')
            z,b,s=forward(row);r=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),dict(logits=teacher.logits,square_cxcywh=teacher.boxes))
            loss,parts,stats=combined(model.class_head,s,b,tz,tb,entry,COEFFICIENT,enabled=True)
            mask,info=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected']);initial_terms+=info['terms']
            if row['image_id'] in cfg['gate_ids']:
                if not mask.any():raise ValueError('Safety representative lost nonzero coverage')
                before=named_sha(model,TRAIN_KEYS)
                measured=combined_gate(loss,parts,z,tz,mask,b,box_objective(z,b,row),params,projection,box_parameters,COEFFICIENT,grad)
                scope.check(hash_frozen=True)
                if named_sha(model,TRAIN_KEYS)!=before or any(p.grad is not None for p in model.parameters()) or optimizer is not None:raise ValueError('Safety gate mutated model/optimizer state')
                measured.update(image_id=row['image_id'],source_group=row['identity_group'],positive_classes=sorted({a['label'] for a in row['annotations'] if not a['ignore']}),replay=r,optimizer_constructed=False,model_updates=0)
                emit(RUN/'gate'/(str(len(gates)+1)+'.json'),measured);gates.append(measured)
            emit(RUN/'initial'/f'{number:02}.json',dict(image_id=row['image_id'],replay=r,terms=info['terms'],unit_KD_loss=float(parts['distillation'].detach())))
            del z,b,s,tz,tb,loss,parts,mask
        if initial_terms!=278 or len(gates)!=3 or named_sha(model,TRAIN_KEYS)!=initial_trainable:raise ValueError('Initial correspondence/safety completion failed')
        scope.check(hash_frozen=True);emit(RUN/'gate/complete.json',dict(status='PASS',records=gates,initial_terms=278,original_trainable_unchanged=True,frozen_unchanged=True,optimizer_updates=0,optimizer_constructed=False,resources=resources()))
        # No optimizer may exist until all representative combined-loss checks pass.
        optimizer=scope.optimizer();hook=[0];optimizer.register_step_post_hook(lambda *_:hook.__setitem__(0,hook[0]+1))
        state['phase']='training';binding=sha(RUN/'binding.json')
        for epoch in range(5):
            epoch_images=[];active_groups=set()
            for gi,group in enumerate(grouped_epoch(train,epoch)):
                guard();scope.train_mode();optimizer.zero_grad(set_to_none=True);values=[]
                if [r['image_id'] for r in group]!=context['orders'][epoch][gi]:raise ValueError('Approved data order changed')
                for row in group:
                    entry=byid[row['image_id']];teacher=load_teacher(refs[row['image_id']],row,index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda');z,b,s=forward(row)
                    loss,parts,stats=combined(model.class_head,s,b,tz,tb,entry,COEFFICIENT,enabled=True)
                    mask,info=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected'])
                    diagnostics=gradient_diagnostics(parts,z,tz,mask,projection,b,box_parameters,COEFFICIENT,grad)
                    if info['terms']:active_groups.add(row['identity_group'])
                    record=dict(image_id=row['image_id'],source_group=row['identity_group'],stats=stats,gradients=diagnostics,weighted_KD_loss=COEFFICIENT*float(parts['distillation'].detach()))
                    emit(RUN/'telemetry'/f"measurement-{epoch+1}-{len(epoch_images)+len(values)+1:02}.json",record)
                    state['backwards']+=1;(loss/len(group)).backward() # Pinned scale1;no GradScaler/manual128.
                    values.append(record);del z,b,s,tz,tb,loss,parts,mask
                norms=scope.gradients(optimizer);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if state['updates']>=75:raise RuntimeError('75update cap BEFORE step')
                emit(RUN/'updates'/f"{state['updates']+1:03}.intent.json",dict(state,epoch=epoch+1,source_order=[r['image_id'] for r in group]))
                guard();optimizer.step();state['updates']+=1
                if hook[0]!=state['updates']:raise ValueError('Optimizer step count mismatch')
                scope.check(optimizer)
                if any(not torch.isfinite(p).all() for p in params):raise ValueError('Nonfinite model update')
                emit(RUN/'updates'/f"{state['updates']:03}.json",dict(resources=resources(),epoch=epoch+1,images=values,preclip_gradient_norms=norms,parameter_changes=changes(),loss_scale=1.,coefficient=COEFFICIENT))
                epoch_images.extend(values)
            scope.check(optimizer,hash_frozen=True);state['epochs']=epoch+1
            payload=scope.payload(optimizer,state['updates'],binding,epoch=epoch+1,next_group=0,coefficient=COEFFICIENT,precision=cfg['precision'],rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            raw=io.BytesIO();torch.save(payload,raw);p=RUN/'checkpoints'/f'epoch-{epoch+1}.pt';atomic(p,raw.getvalue())
            check=torch.load(p,map_location='cpu',weights_only=True)
            if any(not torch.equal(check['weights'][n],payload['weights'][n]) for n in HEAD_KEYS):raise ValueError('Checkpoint round-trip integrity failure')
            last=dict(file=str(p.relative_to(RUN)),sha256=sha(p),updates=state['updates'])
            emit(RUN/'checkpoints'/f'epoch-{epoch+1}.json',dict(checkpoint=last,epoch=epoch+1,parameter_changes=changes(),active_KD_source_groups=sorted(active_groups),images=59,resources=resources()))
            print('E5 EPOCH',epoch+1,'updates',state['updates'],'KD groups',len(active_groups),'seconds',round(time.monotonic()-begin,2),flush=True)
            del payload,check,epoch_images
        state['phase']='evaluation';scope.eval_mode();optimizer.zero_grad(set_to_none=True);records=[]
        for number,row in enumerate(dev,1):
            z,b,_=forward(row,False);scores,labels=z.sigmoid().max(-1);boxes=from_square(cxcywh_to_xyxy(b),row['width'],row['height']).clamp(0,1)
            path=RUN/'predictions'/f'{number:02}.npz';arrays(path,dict(logits=z.cpu().numpy(),square_cxcywh=b.cpu().numpy(),scores=scores.cpu().numpy(),labels=labels.cpu().numpy(),boxes=boxes.cpu().numpy()))
            records.append(dict(image_id=row['image_id'],source_sha256=row['sha256'],file=path.name,sha256=sha(path)));del z,b,scores,labels,boxes
        emit(RUN/'predictions/raw-complete.json',dict(FP32=True,rows_sha256=context['development_rows_sha256'],records=records))
        scope.check(optimizer,hash_frozen=True);guard()
        if state['updates']!=75 or state['forwards']!=373 or state['actual_forwards']!=373 or state['contractions']!=354 or state['backwards']!=295 or sha(BASE/'model.safetensors')!=BASE_SHA:raise ValueError('Fixed endpoint/original preservation failure')
        state['phase']='complete';emit(RUN/'complete.json',dict(status='PASS',resources=resources(),final_checkpoint=last,initial_frozen_sha256=initial_frozen,final_frozen_sha256=frozen_state(model),
             trainable_sha256=named_sha(model,TRAIN_KEYS),parameter_changes=changes(),coefficient=COEFFICIENT,precision=cfg['precision'],predictions_saved_before_CPU_metrics=True,another_run=False,automatic_retry=False))
        print('E5 TRAINING COMPLETE;75 updates;CPU metrics only next',flush=True)
    except BaseException as exc:
        failure=dict(error=type(exc).__name__+': '+str(exc),stack=traceback.format_exc(),resources=resources(),last_checkpoint=last,automatic_retry=False,optimizer_constructed=optimizer is not None)
        if scope is not None and time.monotonic()-begin<593:
            try:failure.update(frozen_state=frozen_state(model),trainable_state=named_sha(model,TRAIN_KEYS))
            except BaseException as secondary:failure['fingerprint_error']=str(secondary)
        try:emit(RUN/'failure.json',failure)
        except BaseException as secondary:exc.add_note('Failure telemetry: '+str(secondary))
        raise
    finally:
        finished.set()
        if thread is not None:thread.join(timeout=2)
        if torch is not None and old_TF32 is not None:torch.backends.cuda.matmul.allow_tf32=old_TF32

if __name__=='__main__':run()
