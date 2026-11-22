"""Exactly twenty additional forwards;the repaired loss is imported unchanged."""
import io,random,signal,threading,traceback
import numpy as np
from common15r import *
from gpu15 import state_fingerprint,vector_norm,cosine


def run():
    cfg,prepared,rows,reg=inputs_r();emit_r(RUNR/'executed-once.json',dict(configuration_sha256=sha(RUNR/'authorized-config.json'),no_retry=True))
    begin=read(RUNR/'launch-once.json')['monotonic_start'];state=dict(forwards=0,gradient_calculations=0,last_verified_model_sha256=None,last_verified_at_seconds=None,peak_observed_device_bytes=0,phase='initializing')
    stop=threading.Event();requested=threading.Event();violations=[];lock=threading.Lock();torch=None;model=None;before=None;thread=None;records=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:requested.set())
    def pulse(kind,terminal=False):
        with lock:
            rec=dict(state,kind=kind,elapsed_seconds=time.monotonic()-begin,cumulative_forwards=5+state['forwards'],optimizer_updates=0,terminal=terminal)
            if torch is not None and torch.cuda.is_initialized():
                try:rec.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                except Exception as exc:rec['memory_read_error']=str(exc)
            emit_r(RUNR/('telemetry-'+uuid.uuid4().hex+'.json'),rec)
            return rec
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from common8 import image_path,PROMPTS
        from freeze10 import FrozenScope,restore_candidate
        from loss15 import controlled_losses,gradients
        from masks6 import validate_bundle
        from common6 import candidates
        torch.set_num_threads(2);torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry/fallback')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((MAX_VRAM-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def guard():
            if time.monotonic()-begin>=175 or requested.is_set() or violations or (RUNR/'STOP').exists():raise RuntimeError('Runtime/STOP/VRAM guard: '+str(violations))
            if state['forwards']>20 or 5+state['forwards']>25 or max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM or allocated_r()>MAX_BYTES-2*2**20:raise RuntimeError('Forward/VRAM/output guard')
        def monitor():
            while not stop.wait(.5):
                try:
                    free,total=torch.cuda.mem_get_info(0);state['peak_observed_device_bytes']=max(state['peak_observed_device_bytes'],total-free)
                    if total-free>MAX_VRAM:violations.append('Observed whole-device VRAM limit')
                    pulse('heartbeat')
                except Exception as exc:violations.append('Memory/telemetry monitor failed: '+str(exc));return
        pulse('GPU_initialized');thread=threading.Thread(target=monitor,daemon=True);thread.start();guard()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        scope=FrozenScope(model,original_heads());payload=torch.load(ROOT/cfg['head']['path'],map_location='cpu',weights_only=True);restore_candidate(scope,payload,expected_binding=cfg['head_binding_sha256']);scope.train_mode()
        if any(not torch.equal(scope.params[n].detach().cpu(),payload['weights'][n]) for n in HEAD_KEYS):raise ValueError('C11 initialization mismatch')
        del payload;before=state_fingerprint(model)
        if before['sha256']!=cfg['initial_model_sha256']:raise ValueError('Recovery initialization differs from original attempt')
        state.update(last_verified_model_sha256=before['sha256'],last_verified_at_seconds=time.monotonic()-begin)
        emit_r(RUNR/'initialization.json',dict(model_state=before,trainable_keys=scope.names,calibration_keys=sorted(CALIBRATION_KEYS),frozen_sha256=scope.frozen_sha256,original_backbone_sha256=scope.backbone_sha256,head=cfg['head'],optimizer_constructed=False));pulse('initial_model_verified')
        bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};validate_bundle(bundle,prepared['rows'],frozen,obs)
        params=[scope.params[n] for n in scope.names];proj=[scope.names.index(n) for n in PROJECTION_KEYS];names=['positive','native_negative','source_focal','source_BCE','class_A','class_B','box'];cohort={name:[torch.zeros_like(p,device='cpu',dtype=torch.float64) for p in params] for name in names}
        def count(*_):
            guard()
            if state['forwards']>=20:raise RuntimeError('Twenty-forward limit before execution')
            state['forwards']+=1
        hook=model.register_forward_pre_hook(count)
        emit_r(RUNR/'resource-start.json',dict(device=torch.cuda.get_device_name(0),device_bytes=torch.cuda.get_device_properties(0).total_memory,limits=cfg['limits'],gradient_scale=128.,precision='Pinned FP16 forward/FP32 loss;same graph both variants',memory_sampling_seconds=.5))
        for index,row in zip(cfg['photo_indices'],rows):
            guard();scope.check();state['phase']='photo-'+str(index);ident=row['image_id'];selected=[e for e in bundle['records'] if e['image_id']==ident];protected=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and obs[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
            emit_r(RUNR/(str(index).zfill(2)+'.forward-intent.json'),dict(photo_index=index,image_id=ident,source_sha256=row['sha256'],next_recovery_forward=state['forwards']+1,cumulative_next_forward=state['forwards']+6));path=image_path(row,'optimization',prepared)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image dimensions/orientation changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
            del batch;z=out.logits[0].float();b=out.pred_boxes[0].float();guard()
            if z.shape!=(3600,7) or b.shape!=(3600,4):raise ValueError('Actual query geometry changed')
            cached=load_npz(C11_RUN/'probes/epoch-5'/(str(index).zfill(2)+'.npz'))
            if not all(np.array_equal(t.detach().cpu().numpy(),cached[k]) for t,k in [(z,'logits'),(b,'square_cxcywh')]):raise ValueError('Actual forward not bit-identical to pinned C11 cache')
            losses,masks,stats=controlled_losses(z,b,row,selected,protected);summary={};components={};arrays=dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy(),target=masks['target'].cpu().numpy().astype(np.uint8),mask=masks['mask'].cpu().numpy().astype(np.uint8),source_added=masks['added'].cpu().numpy().astype(np.uint8),matched=masks['matched'].cpu().numpy().astype(np.int16),matched_annotation=masks['matched_annotation'].cpu().numpy().astype(np.int16))
            for name in names:
                guard();state['gradient_calculations']+=1;g=gradients(losses[name],z,b,params,retain_graph=name!='box',scale=128.);gp=list(g[2:]);projection=[gp[j] for j in proj];gz,gb=g[:2]
                if name!='box' and (gb is not None or any(gp[j] is not None for j,n in enumerate(scope.names) if n.startswith('box_head.'))):raise ValueError('Classification crosses detached box mask')
                if name=='box' and any(x is not None for x in projection):raise ValueError('Box gradient reaches projection')
                if gz is not None and ((gz[~masks['mask']]!=0).any() or (name.startswith('source') and ((gz[~masks['added']]!=0).any() or (gz[masks['target'].bool()]!=0).any()))):raise ValueError('Ignored/positive/protected output received negative gradient')
                summary[name]=dict(loss=float(losses[name].detach()),output_logit_L1=float(gz.double().abs().sum()) if gz is not None else 0.,output_logit_L2=vector_norm([gz]),max_abs_output_logit_gradient=float(gz.abs().max()) if gz is not None else 0.,projection_L2=vector_norm(projection),projection_max_abs=max((float(x.abs().max()) for x in projection if x is not None),default=0.),box_head_L2=vector_norm([x for n,x in zip(scope.names,gp) if n.startswith('box_head.')]),box_output_L2=vector_norm([gb]),parameter_norms={n:(float(x.double().norm()) if x is not None else None) for n,x in zip(scope.names,gp)})
                for dest,value in zip(cohort[name],gp):
                    if value is not None:dest.add_(value.cpu().double()/20)
                if name in ('positive','source_focal','source_BCE'):arrays['output_grad_'+name]=gz.cpu().numpy()
                components[name]=projection;del g,gp,projection,gz,gb
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected .grad accumulation')
            scope.check();fingerprint=state_fingerprint(model)
            if fingerprint!=before:raise ValueError('Parameters or buffers changed during diagnostic')
            state.update(last_verified_model_sha256=fingerprint['sha256'],last_verified_at_seconds=time.monotonic()-begin)
            cv=dict(positive_vs_source_focal=cosine(components['positive'],components['source_focal']),positive_vs_source_BCE=cosine(components['positive'],components['source_BCE']),source_focal_vs_BCE=cosine(components['source_focal'],components['source_BCE']),classification_A_vs_B=cosine(components['class_A'],components['class_B']))
            counts=dict(positive=int((masks['target'].bool()&masks['mask']).sum()),negative=int((~masks['target'].bool()&masks['mask']).sum()),source_negative=int(masks['added'].sum()),ignored=int((~masks['mask']).sum()),protected_violations=stats['protected_violations']);prob=z.detach()[masks['added'][:,1],1].double().sigmoid().cpu().tolist();telemetry=pulse('photo_complete')
            rec=dict(photo_index=index,image_id=ident,identity_group=row['identity_group'],source_sha256=row['sha256'],dimensions=[row['width'],row['height']],actual_forward=state['forwards'],cumulative_forward=5+state['forwards'],gradient_calculations=7,C11_cached_outputs_bit_identical=True,counts=counts,losses_and_gradients=summary,projection_cosines=cv,source_negative_scores=prob,source_region_query_counts=stats['region_query_counts'],frozen_calibration_gradients_none=True,parameter_grads_all_none=True,optimizer_updates=0,model_state_sha256=fingerprint['sha256'],parameters_and_buffers_unchanged=True,telemetry=telemetry)
            buf=io.BytesIO();np.savez_compressed(buf,**arrays);fp=RUNR/(str(index).zfill(2)+'.npz');atomic_r(fp,buf.getvalue());rec.update(raw_file=fp.name,raw_sha256=sha(fp));jp=RUNR/(str(index).zfill(2)+'.json');emit_r(jp,rec);records.append(dict(index=index,image_id=ident,file=jp.name,sha256=sha(jp)))
            del out,z,b,losses,masks,summary,components,arrays,buf,prob,rec,fingerprint;guard();print('RECOVERED PHOTO',index,'/24;new forwards',state['forwards'],'/20;gradient calls',state['gradient_calculations'],flush=True)
        hook.remove();torch.cuda.synchronize();scope.check(hash_frozen=True);after=state_fingerprint(model)
        if after!=before or state['forwards']!=20 or state['gradient_calculations']!=140:raise ValueError('Final zero-update/count/fingerprint verification failed')
        state.update(last_verified_model_sha256=after['sha256'],last_verified_at_seconds=time.monotonic()-begin,phase='verified_complete');guard();stop.set();thread.join(timeout=1)
        cs={n:dict(projection_L2=vector_norm([v[j] for j in proj]),box_head_L2=vector_norm([x for key,x in zip(scope.names,v) if key.startswith('box_head.')])) for n,v in cohort.items()};cc={n:cosine([cohort[a][j] for j in proj],[cohort[b][j] for j in proj]) for n,a,b in [('positive_vs_source_focal','positive','source_focal'),('positive_vs_source_BCE','positive','source_BCE'),('source_focal_vs_BCE','source_focal','source_BCE'),('classification_A_vs_B','class_A','class_B')]}
        resource=pulse('complete',terminal=True);emit_r(RUNR/'complete.json',dict(status='PASS',new_forwards=20,cumulative_forwards=25,new_gradient_calculations=140,cumulative_gradient_calculations=168,reused_photos=4,new_photos=20,optimizer_constructed=False,optimizer_steps=0,model_updates=0,records=records,model_state_before=before['sha256'],model_state_after=after['sha256'],parameters_and_buffers_unchanged=True,resource=resource,recovery20_only_mean_parameter_gradients=cs,recovery20_only_projection_cosines=cc,whole24_signed_gradient_vectors_not_available=True,automatic_retry=False,training_authorized=False));print('C15R SINGLE RECOVERY COMPLETE;NO MODEL UPDATES',flush=True)
    except BaseException as exc:
        error=str(exc);trace=traceback.format_exc();terminal={}
        try:
            if model is not None and before is not None:
                after=state_fingerprint(model);state.update(last_verified_model_sha256=after['sha256'],last_verified_at_seconds=time.monotonic()-begin);terminal=dict(model_state_before=before['sha256'],model_state_after=after['sha256'],parameters_and_buffers_unchanged=after==before)
        except BaseException as secondary:terminal['terminal_fingerprint_error']=str(secondary)
        try:terminal['resource']=pulse('failure',terminal=True)
        except BaseException as secondary:terminal['telemetry_error']=str(secondary)
        emit_r(RUNR/'failure.json',dict(error=error,traceback=trace,**state,terminal=terminal,elapsed_seconds=time.monotonic()-begin,optimizer_updates=0,no_automatic_retry=True));raise
    finally:stop.set()


if __name__=='__main__':run()
