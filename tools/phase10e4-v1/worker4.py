"""One bounded zero-update attempt. No training or optimizer construction."""
import random
import resource
import signal
import traceback
from common4 import *


def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE,(4*2**20,4*2**20))
    cfg,context,byid,index,refs=inputs()
    launch=read(RUN/'launch-once.json')
    if launch['nonce']!=os.environ.get('FURNITURE_E4_NONCE'):raise PermissionError('Use the single bounded launcher')
    emit(RUN/'executed-once.json',dict(binding_sha256=sha(RUN/'binding.json'),automatic_retry=False))
    begin=launch['monotonic_start']
    state=dict(forwards=0,completed_forwards=0,actual_detector_entries=0,contractions=0,
               completed_contractions=0,VJPs=0,completed_VJPs=0,optimizer_updates=0,
               phase='initializing',photo=None,source_group=None,component=None)
    requested=threading.Event();finished=threading.Event();thread=None;errors=[]
    model=None;scope=None;torch=None;real_grad=None;fingerprint=None
    previous_precision=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:requested.set())
    def resources():
        d=dict(state,seconds=time.monotonic()-begin,allocated_output_bytes=allocated())
        if torch is not None and torch.cuda.is_initialized():
            d.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return d
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from scopeE import StudentScope,TRAIN_KEYS,restore_candidate,original_heads,unregistered_tensors
        from teacherE import load_teacher
        from objectiveE import selective_mask,distill,logit_diagnostics
        sys.path.insert(0,str(ROOT/'tools/phase10e2-v1'))
        from measure2 import replay,summaries,cosine
        sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
        import telemetry3,components3
        from stable3 import Capture
        from telemetry3 import Trace,stats
        from precision4 import box_objective
        sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
        from commonD import model_state_matches_base
        from common8 import image_path,load_npz
        from telemetry19 import tensor_hash
        # Runtime-only publication routing. Scientific E3 code remains sealed.
        def bounded_json(path,value,cap=None):emit(path,value)
        telemetry3.atomic_json=bounded_json;components3.atomic_json=bounded_json
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        previous_precision=dict(TF32=torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction((LIMITS['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats(0)

        def guard(component=None):
            if component:state['component']=component
            budget(state,time.monotonic()-begin,torch.cuda.max_memory_allocated(0),allocated())
            if requested.is_set() or finished.is_set() or (RUN/'STOP').exists() or errors or time.monotonic()-begin>=230:
                raise RuntimeError('Graceful E4 boundary: '+str(errors))

        def monitor():
            while not finished.wait(2):
                try:
                    d=resources();budget(state,d['seconds'],d.get('peak_allocated_bytes',0),d['allocated_output_bytes'])
                    emit(RUN/'telemetry'/('heartbeat-'+uuid.uuid4().hex[:8]+'.json'),d)
                except BaseException as e:
                    errors.append(str(e));requested.set();return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
        model_state_matches_base(model);model=model.to('cuda');scope=StudentScope(model)
        processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        projection=[scope.params['class_head.dense0.'+s] for s in ('weight','bias')]
        box_parameters=[scope.params[n] for n in sorted(TRAIN_KEYS) if n.startswith('box_head.')]
        prepared=dict(rows=[e['row'] for e in context['entries']]);original=original_heads()

        def fingerprint():
            per={};h=hashlib.sha256()
            for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers()),('derived',unregistered_tensors(model).items())):
                for name,t in sorted(items):
                    key=kind+':'+name;per[key]=tensor_hash(t);h.update(key.encode());h.update(per[key].encode())
            return dict(sha256=h.hexdigest(),tensors=per)

        def count_detector(*_):
            state['actual_detector_entries']+=1
            if state['actual_detector_entries']!=state['forwards']:raise RuntimeError('Unintended model forward')
        model.register_forward_pre_hook(count_detector)
        real_grad=torch.autograd.grad
        def counted_grad(*args,**kwargs):
            guard()
            if state['VJPs']>=48:raise RuntimeError('VJP cap BEFORE call')
            state['VJPs']+=1
            emit(RUN/'VJP-intents'/f"{state['VJPs']:02}.json",dict(state,loss_scale=1.))
            result=real_grad(*args,**kwargs);state['completed_VJPs']+=1;guard();return result
        torch.autograd.grad=counted_grad

        class CountedCapture(Capture):
            def _finish(self,module,args,result):
                guard('head-verification')
                if state['contractions']>=25:raise RuntimeError('Contraction cap BEFORE call')
                state['contractions']+=1
                emit(RUN/'contraction-intents'/f"{state['contractions']:02}.json",dict(state))
                super()._finish(module,args,result)
                state['completed_contractions']+=1

        def forward(row,tag):
            guard('forward');state['photo']=row['image_id'];state['source_group']=row['identity_group']
            with Image.open(image_path(row,'optimization',prepared)) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Orientation/dimensions changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            actual={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
            if actual!=refs[row['image_id']]['reconstructed_processor_tensors']:raise ValueError('Pixel/token/query identity mismatch')
            if state['forwards']>=25:raise RuntimeError('Forward cap BEFORE call')
            state['forwards']+=1;emit(RUN/'forward-intents'/f"{state['forwards']:02}.json",dict(state,source_sha256=row['sha256']))
            trace=Trace(RUN/'telemetry'/('head-'+tag),enabled=True,context=dict(image_id=row['image_id'],source_group=row['identity_group'],device='cuda:0',forward_autocast='float16'))
            try:
                with CountedCapture(model.class_head,enabled=True,trace=trace) as capture,torch.autocast('cuda',dtype=torch.float16):
                    out=model(**{k:v.to('cuda') for k,v in batch.items()})
            finally:trace.close()
            z=out.logits[0].float();b=out.pred_boxes[0].float()
            torch.cuda.synchronize(0);state['completed_forwards']+=1
            if z.shape!=(3600,7) or b.shape!=(3600,4) or not torch.isfinite(z).all() or not torch.isfinite(b).all():raise ValueError('Invalid actual model outputs')
            path=RUN/'raw'/(tag+'.npz');arrays(path,dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()))
            guard();return z,b,capture.snapshot,dict(file=str(path.relative_to(RUN)),sha256=sha(path))

        state['phase']='original';scope.train_mode();beforeA=fingerprint();emit(RUN/'original-before.json',beforeA)
        for ident in cfg['original_photo_ids']:
            entry=byid[ident];teacher=load_teacher(refs[ident],entry['row'],index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda')
            z,b,snapshot,raw=forward(entry['row'],'original')
            comparison=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),dict(logits=teacher.logits,square_cxcywh=teacher.boxes))
            mask,info=selective_mask(z,b,tz,tb,entry['row'],entry['distill_regions'],entry['distill_protected'])
            kd=distill(z,tz,mask);delta=(z.detach()-tz)[mask];terminal=2*delta/int(mask.sum()) if mask.any() else delta
            emit(RUN/'original-correspondence.json',dict(raw=raw,replay=comparison,mask=info,
                 unit_KD_loss=float(kd.detach()),terminal_derivative=stats(terminal),
                 exact_zero_derivative=bool(not delta.any()),actual_original_VJPs=0,teacher_targets_modified=False))
            del z,b,snapshot,tz,tb,kd,mask
        scope.check(hash_frozen=True);afterA=fingerprint();emit(RUN/'original-after.json',afterA)
        if beforeA!=afterA:raise ValueError('Original in-memory model changed')
        checkpoint=ROOT/cfg['reference_checkpoint']
        if sha(checkpoint)!=cfg['reference_sha256']:raise ValueError('Adapted checkpoint changed')
        restore_candidate(scope,torch.load(checkpoint,map_location='cpu',weights_only=True));scope.check(hash_frozen=True)
        beforeB=fingerprint();emit(RUN/'adapted-before.json',beforeB);state['phase']='adapted'
        records=[];groups={};box_record=None
        for number,ident in enumerate(cfg['active_photo_ids'],1):
            entry=byid[ident];row=entry['row'];teacher=load_teacher(refs[ident],row,index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda')
            z,b,snapshot,raw=forward(row,f'adapted-{number:02}')
            expected=load_npz(ROOT/cfg['cachemap'][ident]['cache'])
            comparison=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),expected)
            folder=RUN/'measurements'/f'{number:02}'
            measured=components3.measure_components(model.class_head,snapshot,b,tz,tb,entry,box_parameters,folder,
                 guard=guard,context=dict(image_id=ident,source_group=row['identity_group'],source_sha256=row['sha256'],internal_partition='optimization'),enabled=True)
            gp=[g.cpu() for g in measured['positive']];gk=[g.cpu() for g in measured['distillation']]
            mask,info=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected'])
            if hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()!=cfg['cachemap'][ident]['mask_sha256']:
                raise ValueError('Live selective mask differs from frozen cache-derived eligibility')
            if ((measured['positive_output']!=0)&mask).any():raise ValueError('Positive/distillation output overlap')
            pn=summaries(gp)['L2'];kn=summaries(gk)['L2'];displacement=[scope.params['class_head.dense0.'+s].detach().cpu()-original['class_head.dense0.'+s] for s in ('weight','bias')]
            value=dict(image_id=ident,source_group=row['identity_group'],raw=raw,replay=comparison,terms=info['terms'],mask=info,
                       positive_projection=summaries(gp),unit_KD_projection=summaries(gk),unit_ratio=kn/pn if pn else None,
                       cosine=cosine(gp,gk),KD_cosine_past_projection_displacement=cosine(gk,displacement),
                       negative_KD_dot_output_drift=-float((measured['distillation_output'].double()*(z.detach()-tz).double()).sum()),
                       KD_output=stats(measured['distillation_output']),positive_output=stats(measured['positive_output']),
                       selected_student_scores=stats(z.detach()[mask].sigmoid()),selected_teacher_scores=stats(tz[mask].sigmoid()),
                       selected_student_scores_ge015=int((z.detach()[mask].sigmoid()>=.15).sum()),
                       diagnostics=logit_diagnostics(z.detach(),tz,mask),protected_overlap=0,
                       parameter_grads_none=all(p.grad is None for p in model.parameters()),loss_scale=1.)
            emit(folder/'summary.json',value);records.append(value)
            group=row['identity_group']
            if group not in groups:groups[group]=dict(photos=0,terms=0,positive=[torch.zeros_like(g) for g in gp],KD=[torch.zeros_like(g) for g in gk])
            data=groups[group];data['photos']+=1;data['terms']+=info['terms']
            for j in range(2):data['positive'][j].add_(gp[j]);data['KD'][j].add_(gk[j])
            if ident==cfg['box_probe_photo']:
                guard('box-VJP-intent');loss=box_objective(z,b,row)
                gradients=torch.autograd.grad(loss,[b,*box_parameters,*projection],retain_graph=True,allow_unused=True)
                if any(g is None or not torch.isfinite(g).all() for g in gradients[:7]) or any(g is not None for g in gradients[7:]):
                    raise ValueError('Native box VJP nonfinite/missing or reached classifier')
                box_record=dict(image_id=ident,loss=float(loss.detach()),box_output=stats(gradients[0]),
                                box_parameters=[stats(g) for g in gradients[1:7]],projection_gradients_none=True,
                                loss_scale=1.,backward='Native FP16 box-head adjoint;FP32 loss and master gradients;not the classification repair')
                emit(RUN/'box-probe.json',box_record);del loss,gradients
            scope.check()
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected accumulated parameter gradient')
            print('E4 PHOTO',number,'/',len(cfg['active_photo_ids']),'terms',info['terms'],'VJPs',state['VJPs'],flush=True)
            del measured,z,b,snapshot,tz,tb,mask,gp,gk
        scope.check(hash_frozen=True);afterB=fingerprint();emit(RUN/'adapted-after.json',afterB)
        if beforeB!=afterB or sha(BASE/'model.safetensors')!=BASE_SHA or sha(checkpoint)!=cfg['reference_sha256']:
            raise ValueError('Original/adapted checkpoint preservation failed')
        totalsP=[torch.zeros_like(projection[j],device='cpu') for j in range(2)];totalsK=[torch.zeros_like(t) for t in totalsP];source=[]
        for group,data in sorted(groups.items()):
            # Same source normalization as E2; one image per active source here.
            for j in range(2):data['positive'][j].div_(data['photos']);data['KD'][j].div_(data['photos']);totalsP[j].add_(data['positive'][j]);totalsK[j].add_(data['KD'][j])
            source.append(dict(source_group=group,photos=data['photos'],terms=data['terms'],positive=summaries(data['positive']),KD=summaries(data['KD']),cosine=cosine(data['positive'],data['KD'])))
        emit(RUN/'source-groups.json',dict(records=source,cosine_of_source_sums=cosine(totalsP,totalsK),positive_sum=summaries(totalsP),KD_sum=summaries(totalsK),normalization='Per-source mean,then sum;all15active groups have one photo. No loss-weight changes.'))
        if state['forwards']!=16 or state['contractions']!=16 or state['actual_detector_entries']!=16 or sum(r['terms'] for r in records)!=267:
            raise ValueError('Unexpected scope/term counters')
        guard();state['phase']='complete';torch.cuda.synchronize(0)
        emit(RUN/'complete.json',dict(status='PASS',**resources(),records=records,groups=len(groups),
             box_probe_complete=box_record is not None,model_weight_updates=0,optimizer_constructed=False,
             original_unchanged=True,adapted_unchanged=True,coefficient=None,training_authorized=False))
        print('E4 ZERO-UPDATE COMPLETE',state['forwards'],'forwards;',state['VJPs'],'VJPs',flush=True)
    except BaseException as exc:
        failure=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),**resources(),automatic_retry=False,model_fingerprint=None)
        if fingerprint is not None and time.monotonic()-begin<233:
            try:failure['model_fingerprint']=fingerprint()
            except BaseException as secondary:failure['fingerprint_error']=str(secondary)
        try:emit(RUN/'failure.json',failure)
        except BaseException as secondary:exc.add_note('Minimal failure persistence failed: '+str(secondary))
        raise
    finally:
        finished.set()
        if thread is not None:thread.join(timeout=2)
        if real_grad is not None:torch.autograd.grad=real_grad
        if previous_precision is not None:torch.backends.cuda.matmul.allow_tf32=previous_precision['TF32']


if __name__=='__main__':run()
