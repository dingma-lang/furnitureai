"""One 28-forward zero-update attempt;no optimizer construction or backward steps."""
import random,signal,traceback,resource
from common2 import *

def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cfg,context,byid,index,refs=inputs();emit(RUN/'executed-once.json',dict(binding_sha256=sha(RUN/'binding.json'),automatic_retry=False))
    begin=read(RUN/'launch-once.json')['monotonic_start'];state=dict(forwards=0,actual_forwards=0,gradient_calculations=0,analytic_zero_distill_derivatives=0,phase='initializing',photo=None)
    requested=threading.Event();stop=threading.Event();violations=[];torch=None;model=None;scope=None;thread=None;stage_before=None
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:requested.set())
    def telemetry(kind):
        d=dict(state,kind=kind,seconds=time.monotonic()-begin,allocated_output_bytes=allocated())
        if torch is not None and torch.cuda.is_initialized():d.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        emit(RUN/'telemetry'/('event-'+uuid.uuid4().hex+'.json'),d);return d
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from scopeE import StudentScope,TRAIN_KEYS,HEAD_KEYS,restore_candidate,frozen_state,original_heads,unregistered_tensors
        from objectiveE import objective,selective_mask,logit_diagnostics
        from teacherE import load_teacher
        from measure2 import replay,vjp,zero_derivative,summaries,cosine
        sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
        from commonD import model_state_matches_base
        from common8 import image_path,load_npz
        from telemetry19 import tensor_hash
        torch.set_num_threads(2);torch.manual_seed(6106);random.seed(6106);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no fallback/retry')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((LIMITS['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def guard():
            budget(state,time.monotonic()-begin,torch.cuda.max_memory_allocated(0),allocated())
            if time.monotonic()-begin>=172 or requested.is_set() or (RUN/'STOP').exists() or violations:raise RuntimeError('Graceful resource/STOP boundary: '+str(violations))
        def monitor():
            while not stop.wait(2):
                try:telemetry('heartbeat')
                except BaseException as e:violations.append('Failure telemetry: '+str(e));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True);model_state_matches_base(model);model=model.to('cuda')
        scope=StudentScope(model);processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        projection=[scope.params['class_head.dense0.'+s] for s in ('weight','bias')];box_parameters=[scope.params[n] for n in sorted(TRAIN_KEYS) if n.startswith('box_head.')]
        def fingerprint():
            h=hashlib.sha256();per={}
            for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers()),('derived',unregistered_tensors(model).items())):
                for n,t in sorted(items):per[kind+':'+n]=tensor_hash(t);h.update((kind+':'+n).encode());h.update(per[kind+':'+n].encode())
            return dict(sha256=h.hexdigest(),tensors=per)
        def counted(*_):
            state['actual_forwards']+=1
            if state['actual_forwards']!=state['forwards'] or state['actual_forwards']>28:raise RuntimeError('Unexpected detector forward')
        model.register_forward_pre_hook(counted)
        source_rows=[e['row'] for e in context['entries']];prepared=dict(rows=source_rows);original=original_heads();groups={};records=[];initial_records=[]
        def forward(row,folder,number):
            guard();state['photo']=row['image_id'];path=image_path(row,'optimization',prepared)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image orientation/frame mismatch')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            observed={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
            if observed!=refs[row['image_id']]['reconstructed_processor_tensors']:raise ValueError('Pixel/token preprocessing correspondence mismatch')
            if state['forwards']>=28:raise RuntimeError('Forward cap before call')
            state['forwards']+=1;emit(RUN/'forward-intents'/(str(state['forwards']).zfill(2)+'.json'),dict(state,source_sha256=row['sha256'],source_group=row['identity_group']))
            with torch.autocast('cuda',dtype=torch.float16):output=model(**{k:v.to('cuda') for k,v in batch.items()})
            z=output.logits[0].float();b=output.pred_boxes[0].float()
            if z.shape!=(3600,7) or b.shape!=(3600,4) or not torch.isfinite(z).all() or not torch.isfinite(b).all():raise ValueError('Actual output schema/nonfinite')
            # Durable outputs precede comparison/loss/VJP so failures cannot destroy replay evidence.
            p=RUN/folder/(str(number).zfill(2)+'.npz');arrays(p,dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()))
            guard();return z,b,dict(file=str(p.relative_to(RUN)),sha256=sha(p))
        def calculate(loss,z,b):
            guard()
            if state['gradient_calculations']>=48:raise RuntimeError('48VJP cap before calculation')
            state['gradient_calculations']+=1
            emit(RUN/'gradient-intents'/(str(state['gradient_calculations']).zfill(2)+'.json'),dict(state))
            result=vjp(loss,z,b,projection,box_parameters);guard();return result
        def measure(entry,z,b,tz,tb):
            row=entry['row'];total,parts,stats=objective(z,b,row,tz,tb,entry,1.)
            mask,mask_stats=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected'])
            pz,gp=calculate(parts['positive'],z,b)
            analytic=zero_derivative(z,tz,mask)
            if analytic:
                kz=torch.zeros_like(z);gk=[torch.zeros_like(p) for p in projection];state['analytic_zero_distill_derivatives']+=1;method='analytic exact zero from empty mask or identical selected logits;no VJP executed'
            else:kz,gk=calculate(parts['distillation'],z,b);method='actual scaled/unscaled model VJP'
            if (kz[~mask]!=0).any() or mask[:,[0,2,3,4,5,6]].any():raise ValueError('Distillation reached unknown/non-Chair output terms')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic accumulated parameter.grad')
            pn=summaries(gp)['L2'];kn=summaries(gk)['L2']
            delta=[scope.params['class_head.dense0.'+s].detach().cpu()-original['class_head.dense0.'+s] for s in ('weight','bias')]
            cp=[g.cpu() for g in gp];ck=[g.cpu() for g in gk]
            value=dict(losses=dict(positive=float(parts['positive'].detach()),box=stats['base']['box_loss'],classification=stats['base']['classification_loss'],unit_distillation=stats['distillation']),eligible_distillation_terms=mask_stats['terms'],protected_positive_terms=mask_stats['matched_positive_queries_excluded'],unknown_ignored_class_terms=3600*7-stats['base']['used_active_terms'],positive_distill_output_overlap=int(((pz!=0)&mask).sum()),KD_method=method,positive_projection=summaries(gp),unit_distill_projection=summaries(gk),positive_output=summaries([pz]),distill_output=summaries([kz]),distill_selected_output=summaries([kz[mask]]),projection_cosine=cosine(cp,ck),unit_distill_positive_ratio=kn/pn if pn else None,distill_gradient_cosine_observed_projection_drift=cosine(ck,delta),distill_negative_gradient_dot_logit_drift=-float((kz.double()*(z.detach()-tz).double()).sum()),box_output_gradient=None,box_parameter_gradients_none=True,frozen_parameter_grads_none=True,mask=mask_stats,diagnostics=logit_diagnostics(z.detach(),tz,mask))
            if value['positive_distill_output_overlap']:raise ValueError('Protected positive output targeted')
            return value,cp,ck,mask
        scope.train_mode();stage_before=fingerprint();emit(RUN/'state-A-before.json',stage_before)
        for i,ident in enumerate(cfg['photo_ids'][:4],1):
            state['phase']='original-A';entry=byid[ident];teacher=load_teacher(refs[ident],entry['row'],index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda');z,b,ref=forward(entry['row'],'original',i)
            replay_result=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),dict(logits=teacher.logits,square_cxcywh=teacher.boxes))
            measured,_,_,mask=measure(entry,z,b,tz,tb)
            result=dict(state='A-original',image_id=ident,source_group=entry['row']['identity_group'],replay=replay_result,measurement=measured,raw=ref);emit(RUN/'original'/(str(i).zfill(2)+'.json'),result);initial_records.append(result);del z,b,tz,tb,mask
        scope.check(hash_frozen=True);after=fingerprint();emit(RUN/'state-A-after.json',after)
        if after!=stage_before:raise ValueError('Original state changed during zero-update diagnostic')
        checkpoint=ROOT/cfg['reference_checkpoint']
        if sha(checkpoint)!=cfg['reference_sha256']:raise ValueError('C11 checkpoint identity mismatch')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);restore_candidate(scope,payload);scope.check(hash_frozen=True)
        stage_before=fingerprint();emit(RUN/'state-B-before.json',stage_before)
        for i,ident in enumerate(cfg['photo_ids'],1):
            state['phase']='adapted-B';entry=byid[ident];teacher=load_teacher(refs[ident],entry['row'],index['processor_contract_sha256']);tz,tb=teacher.tensors('cuda');z,b,ref=forward(entry['row'],'adapted',i)
            cached=load_npz(ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/probes/epoch-5'/f'{i:02}.npz')
            replay_result=replay(dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy()),cached)
            measured,gp,gk,mask=measure(entry,z,b,tz,tb)
            group=entry['row']['identity_group']
            if group not in groups:groups[group]=dict(photos=0,positive=[torch.zeros_like(g) for g in gp],distill=[torch.zeros_like(g) for g in gk],terms=0)
            groups[group]['photos']+=1;groups[group]['terms']+=measured['eligible_distillation_terms']
            for j in range(2):groups[group]['positive'][j].add_(gp[j]);groups[group]['distill'][j].add_(gk[j])
            result=dict(state='B-existing-C11',image_id=ident,source_group=group,replay=replay_result,measurement=measured,raw=ref);emit(RUN/'adapted'/(str(i).zfill(2)+'.json'),result);records.append(result);del z,b,tz,tb,mask,gp,gk
            scope.check();print('E2 PHOTO',i,'/24;forwards',state['forwards'],';VJPs',state['gradient_calculations'],flush=True)
        scope.check(hash_frozen=True);after=fingerprint();emit(RUN/'state-B-after.json',after)
        if after!=stage_before or any(p.grad is not None for p in model.parameters()) or sha(BASE/'model.safetensors')!=BASE_SHA or sha(checkpoint)!=cfg['reference_sha256']:raise ValueError('Reference/model preservation failed')
        totalp=[torch.zeros_like(projection[j],device='cpu') for j in range(2)];totalk=[torch.zeros_like(v) for v in totalp];source=[]
        for ident,data in sorted(groups.items()):
            for j in range(2):totalp[j].add_(data['positive'][j]);totalk[j].add_(data['distill'][j])
            source.append(dict(source_group=ident,photos=data['photos'],terms=data['terms'],positive=summaries(data['positive']),distillation=summaries(data['distill']),cosine=cosine(data['positive'],data['distill'])))
        emit(RUN/'source-groups.json',dict(records=source,positive_sum=summaries(totalp),distill_sum=summaries(totalk),cosine_of_sums=cosine(totalp,totalk),aggregation='Actual sums of measured per-photo vectors,not reconstructed from norms;24-photo diagnostic subset,not the full59-photo training gradient'))
        guard()
        if state['forwards']!=28 or state['actual_forwards']!=28 or state['gradient_calculations']>48:raise ValueError('Final counter mismatch')
        state['phase']='complete';stop.set();thread.join(timeout=2);resources=telemetry('complete')
        emit(RUN/'complete.json',dict(status='PASS',**resources,original_records=initial_records,adapted_records=records,optimizer_constructed=False,optimizer_steps=0,model_weight_updates=0,original_state_preserved=True,adapted_reference_preserved=True,coefficient_selected=None,training_authorized=False,another_run=False));print('E2 ZERO-UPDATE COMPLETE',state['forwards'],'forwards;',state['gradient_calculations'],'actual VJPs',flush=True)
    except BaseException as exc:
        stop.set();failure=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),state=dict(state),seconds=time.monotonic()-begin,automatic_retry=False,model_fingerprint=None)
        if torch is not None and torch.cuda.is_initialized():failure.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        if model is not None and time.monotonic()-begin<174:
            try:failure['model_fingerprint']=fingerprint()
            except BaseException as secondary:failure['fingerprint_error']=str(secondary)
        try:emit(RUN/'failure.json',failure)
        except BaseException as secondary:exc.add_note('Failure telemetry error: '+str(secondary))
        raise
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=2)

if __name__=='__main__':run()
