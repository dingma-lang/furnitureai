"""Single same-forward-graph gradient diagnostic. No optimizer is constructed."""
import io,random,threading,traceback
import numpy as np
from common15 import *


def state_fingerprint(model):
    import hashlib
    h=hashlib.sha256();records={}
    from common10 import tensor_sha
    # Include nonpersistent buffers too; state_dict alone omits them.
    for name,t in list(model.named_parameters())+list(model.named_buffers()):
        value=tensor_sha(t);records[name]=value;h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(list(t.shape)).encode());h.update(value.encode())
    return dict(sha256=h.hexdigest(),tensors=records)


def vector_norm(values):
    import torch
    return sum(float(v.double().square().sum()) for v in values if v is not None)**.5


def cosine(a,b):
    denom=vector_norm(a)*vector_norm(b)
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)/denom if denom else None


def run():
    cfg,prepared,rows,reg=inputs15();emit(RUN15/'executed-once.json',dict(config_sha256=sha(RUN15/'authorized-config.json'),no_retry=True))
    start=time.monotonic();forwards=calculations=0;violations=[];stop=threading.Event();samples=[];torch=None;model=None;before=None;records=[];thread=None
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
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no fallback or retry')
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((MAX_VRAM-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0);torch.cuda.reset_peak_memory_stats(0)
        def monitor():
            while not stop.wait(.025):
                try:
                    free,total=torch.cuda.mem_get_info(0);samples.append(total-free)
                    if total-free>MAX_VRAM:violations.append('Observed whole-device VRAM limit');return
                except Exception as exc:violations.append('GPU memory monitor: '+str(exc));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        def guard():
            if time.monotonic()-start>=180 or violations or (RUN15/'STOP').exists():raise RuntimeError('Time/VRAM/STOP limit: '+str(violations))
            if forwards>24 or max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM or allocated()>MAX_BYTES:raise RuntimeError('Forward/VRAM/storage limit')
        guard();model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).to('cuda')
        processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        scope=FrozenScope(model,original_heads());payload=torch.load(ROOT/cfg['head']['path'],map_location='cpu',weights_only=True)
        restore_candidate(scope,payload,expected_binding=cfg['head_binding_sha256']);scope.train_mode()
        if any(not torch.equal(scope.params[n].detach().cpu(),payload['weights'][n]) for n in HEAD_KEYS):raise ValueError('C11 head did not load exactly')
        del payload
        before=state_fingerprint(model);emit(RUN15/'initialization.json',dict(model_state=before,trainable_keys=sorted(TRAIN_KEYS),frozen_calibration=sorted(CALIBRATION_KEYS),frozen_parameter_sha256=scope.frozen_sha256,backbone_sha256=scope.backbone_sha256,original_checkpoint_sha256=cfg['original_checkpoint_sha256'],head=cfg['head'],optimizer_constructed=False))
        bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};validate_bundle(bundle,prepared['rows'],frozen,observations)
        params=[scope.params[n] for n in scope.names];proj=[scope.names.index(n) for n in PROJECTION_KEYS];names=['positive','native_negative','source_focal','source_BCE','class_A','class_B','box']
        cohort={name:[torch.zeros_like(p,device='cpu',dtype=torch.float64) for p in params] for name in names}
        def count_forward(*_):
            nonlocal forwards
            guard()
            if forwards>=24:raise RuntimeError('24forward limit before call')
            forwards+=1
        hook=model.register_forward_pre_hook(count_forward)
        emit(RUN15/'resource-start.json',dict(monotonic_start=start,device=torch.cuda.get_device_name(0),total_device_bytes=torch.cuda.get_device_properties(0).total_memory,limits=cfg['limits'],gradient_scale=cfg['gradient_scale'],precision='Same historical FP16 forward,FP32 loss;scaled autograd.grad/128;no optimizer or parameter.grad accumulation'))
        for index,row in enumerate(rows,1):
            guard();scope.check();ident=row['image_id'];selected=[e for e in bundle['records'] if e['image_id']==ident]
            protected=[p['box'] for p in reg[ident]['protected']]+[r['annotation']['box'] for r in frozen if r['image_id']==ident and observations[r['candidate_id']]['visual_outcome']=='mixed_or_uncertain']
            path=image_path(row,'optimization',prepared)
            emit(RUN15/(str(index).zfill(2)+'.forward-intent.json'),dict(image_id=ident,source_sha256=row['sha256'],next_forward=index))
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Image dimensions/orientation changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            with torch.autocast('cuda',dtype=torch.float16):out=model(**{k:v.to('cuda') for k,v in batch.items()})
            del batch;z=out.logits[0].float();b=out.pred_boxes[0].float()
            if z.shape!=(3600,7) or b.shape!=(3600,4):raise ValueError('Unexpected actual query shape')
            cached=load_npz(C11_RUN/'probes/epoch-5'/(str(index).zfill(2)+'.npz'))
            identical=all(np.array_equal(current.detach().cpu().numpy(),cached[key]) for current,key in [(z,'logits'),(b,'square_cxcywh')])
            if not identical:raise ValueError('Actual model output differs from exact C11 saved state;stop,not retry')
            losses,masks,stats=controlled_losses(z,b,row,selected,protected);summary={};components={};arrays=dict(logits=z.detach().cpu().numpy(),square_cxcywh=b.detach().cpu().numpy(),target=masks['target'].cpu().numpy().astype(np.uint8),mask=masks['mask'].cpu().numpy().astype(np.uint8),source_added=masks['added'].cpu().numpy().astype(np.uint8),matched=masks['matched'].cpu().numpy().astype(np.int16),matched_annotation=masks['matched_annotation'].cpu().numpy().astype(np.int16))
            for name in names:
                guard();calculations+=1;g=gradients(losses[name],z,b,params,retain_graph=name!='box',scale=cfg['gradient_scale']);gp=list(g[2:]);projection=[gp[i] for i in proj]
                if name!='box' and (g[1] is not None or any(gp[j] is not None for j,n in enumerate(scope.names) if n.startswith('box_head.'))):raise ValueError('Classification crossed detached box/mask boundary')
                if name=='box' and any(x is not None for x in projection):raise ValueError('Regression unexpectedly reaches classification projection')
                gz=g[0];gb=g[1]
                if gz is not None:
                    if (gz[~masks['mask']]!=0).any():raise ValueError('Ignored output term receives gradient')
                    if name.startswith('source') and ((gz[~masks['added']]!=0).any() or (gz[masks['target'].bool()]!=0).any()):raise ValueError('Source-negative gradient touches unapproved targets')
                summary[name]=dict(loss=float(losses[name].detach()),output_logit_L1=float(gz.double().abs().sum()) if gz is not None else 0.,output_logit_L2=vector_norm([gz]),max_abs_output_logit_gradient=float(gz.abs().max()) if gz is not None else 0.,projection_L2=vector_norm(projection),projection_max_abs=max((float(x.abs().max()) for x in projection if x is not None),default=0.),box_head_L2=vector_norm([x for n,x in zip(scope.names,gp) if n.startswith('box_head.')]),box_output_L2=vector_norm([gb]),parameter_norms={n:(float(x.double().norm()) if x is not None else None) for n,x in zip(scope.names,gp)})
                for dest,value in zip(cohort[name],gp):
                    if value is not None:dest.add_(value.cpu().double()/24)
                if name in ('positive','source_focal','source_BCE'):arrays['output_grad_'+name]=gz.cpu().numpy()
                components[name]=projection
                del g,gz,gb,gp,projection
            if any(p.grad is not None for p in model.parameters()):raise ValueError('autograd.grad accumulated model gradients')
            scope.check();counts=dict(positive=int((masks['target'].bool()&masks['mask']).sum()),negative=int((~masks['target'].bool()&masks['mask']).sum()),source_negative=int(masks['added'].sum()),ignored=int((~masks['mask']).sum()),protected_violations=stats['protected_violations'])
            cosine_values=dict(positive_vs_source_focal=cosine(components['positive'],components['source_focal']),positive_vs_source_BCE=cosine(components['positive'],components['source_BCE']),source_focal_vs_BCE=cosine(components['source_focal'],components['source_BCE']),classification_A_vs_B=cosine(components['class_A'],components['class_B']))
            negative=masks['added'][:,1];prob=z.detach()[negative,1].double().sigmoid();negative_scores=prob.cpu().tolist()
            rec=dict(image_id=ident,identity_group=row['identity_group'],source_sha256=row['sha256'],dimensions=[row['width'],row['height']],actual_forward=index,gradient_calculations=len(names),C11_cached_outputs_bit_identical=identical,counts=counts,losses_and_gradients=summary,projection_cosines=cosine_values,source_negative_scores=negative_scores,source_region_query_counts=stats['region_query_counts'],frozen_calibration_gradients_none=True,parameter_grads_all_none=True,optimizer_updates=0,elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),peak_observed_device_bytes=max(samples,default=0))
            # Persist compact raw evidence before the separate CPU analysis.
            stream=io.BytesIO();np.savez_compressed(stream,**arrays);npz=RUN15/(str(index).zfill(2)+'.npz');atomic_bytes(npz,stream.getvalue());rec.update(raw_file=npz.name,raw_sha256=sha(npz));emit(RUN15/(str(index).zfill(2)+'.json'),rec);records.append(dict(image_id=ident,file=str(index).zfill(2)+'.json',sha256=sha(RUN15/(str(index).zfill(2)+'.json'))))
            del out,z,b,losses,masks,summary,components,arrays,stream,negative,prob,rec;guard();print('GRADIENT PHOTO',index,'/24;autograd calls',calculations,flush=True)
        hook.remove();torch.cuda.synchronize();scope.check(hash_frozen=True);after=state_fingerprint(model)
        if after!=before or forwards!=24 or calculations!=168 or any(p.grad is not None for p in model.parameters()):raise ValueError('Zero-update/model fingerprint/count integrity failure')
        guard();stop.set();thread.join(timeout=1)
        cohort_summary={name:dict(projection_L2=vector_norm([values[j] for j in proj]),box_head_L2=vector_norm([x for n,x in zip(scope.names,values) if n.startswith('box_head.')]),parameter_norms={n:vector_norm([x]) for n,x in zip(scope.names,values)}) for name,values in cohort.items()}
        cohort_cosines={name:cosine([cohort[a][j] for j in proj],[cohort[b][j] for j in proj]) for name,a,b in [('positive_vs_source_focal','positive','source_focal'),('positive_vs_source_BCE','positive','source_BCE'),('source_focal_vs_BCE','source_focal','source_BCE'),('classification_A_vs_B','class_A','class_B')]}
        guard()
        emit(RUN15/'complete.json',dict(status='PASS',forwards=forwards,gradient_calculations=calculations,optimizer_constructed=False,optimizer_steps=0,model_updates=0,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),peak_observed_device_bytes=max(samples,default=0),model_state_before=before['sha256'],model_state_after=after['sha256'],all_parameter_and_buffer_hashes_unchanged=True,calibration_frozen=True,gradient_scale=128.,records=records,cohort_mean_parameter_gradients=cohort_summary,cohort_projection_cosines=cohort_cosines,cohort_aggregation='Arithmetic mean of24 per-photo GT-normalized gradient vectors at identical fixed C11 weights;not simulated optimizer updates',automatic_retry=False,follow_on_training_authorized=False))
        print('C15 ONE ZERO-UPDATE GPU SMOKE COMPLETE',flush=True)
    except BaseException as exc:
        telemetry={}
        try:
            if torch is not None and torch.cuda.is_initialized():telemetry.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),peak_observed_device_bytes=max(samples,default=0))
            if model is not None and before is not None:telemetry.update(model_state_after=state_fingerprint(model)['sha256'],model_state_before=before['sha256'],parameter_grads_all_none=all(p.grad is None for p in model.parameters()))
        except BaseException as secondary:telemetry['telemetry_error']=str(secondary)
        emit(RUN15/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),forwards=forwards,gradient_calculations=calculations,optimizer_updates=0,no_automatic_retry=True,seconds=time.monotonic()-start,telemetry=telemetry));raise
    finally:stop.set()


if __name__=='__main__':run()
