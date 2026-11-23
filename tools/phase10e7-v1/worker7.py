"""Exactly59inference-only forwards;immutable per-photo caches before CPU analysis."""
import resource,signal,traceback
from common7 import *

def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(128*2**20,128*2**20))
    cfg,c,train=inputs();launch=read(RUN/'launch-once.json')
    if launch['nonce']!=os.environ.get('FURNITURE_E7_NONCE'):raise PermissionError('Single bounded launcher required')
    emit(RUN/'executed-once.json',dict(binding_sha256=sha(RUN/'binding.json'),automatic_retry=False))
    state=dict(attempted_forwards=0,actual_forwards=0,completed_forwards=0,cached_photos=0,phase='CPU-loading',image_id=None)
    begin=launch['monotonic_start'];gpu_begin=None;torch=None;model=None;before=None;fingerprint=None;old_TF32=None
    stopped=threading.Event();finished=threading.Event();monitor_thread=None;errors=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.set())
    def resources():
        d=dict(state,wall_seconds=time.monotonic()-begin,GPU_runtime_seconds=time.monotonic()-gpu_begin if gpu_begin else 0,
               allocated_output_bytes=allocated(),optimizer_steps=0,backward_calls=0)
        if torch is not None and torch.cuda.is_initialized():
            d.update(allocated_VRAM_bytes=torch.cuda.memory_allocated(0),reserved_VRAM_bytes=torch.cuda.memory_reserved(0),
                 peak_allocated_VRAM_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_VRAM_bytes=torch.cuda.max_memory_reserved(0))
        return d
    def guard():
        d=resources();budget(state,d['wall_seconds'],d.get('peak_allocated_VRAM_bytes',0),d['allocated_output_bytes'])
        if stopped.is_set() or errors or (RUN/'STOP').exists() or d['wall_seconds']>=174:raise RuntimeError('E7graceful stop/resource boundary:'+str(errors))
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
        from commonD import model_state_matches_base
        from scopeD import unregistered_tensors
        from telemetry19 import model_fingerprint,tensor_hash
        from scopeE import StudentScope
        from common10 import named_sha,TRAIN_KEYS
        def fingerprint():
            return dict(registered=model_fingerprint(model),derived={n:tensor_hash(t) for n,t in unregistered_tensors(model).items()})
        torch.set_num_threads(2);torch.manual_seed(6106);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;old_TF32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        guard()
        model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
        original_tensors=model_state_matches_base(model)
        scope=StudentScope(model)
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        scope.restore(payload,binding=payload['binding_sha256'])
        done=read(CHECKPOINT.parent.parent/'complete.json')
        if named_sha(model,TRAIN_KEYS)!=done['trainable_sha256']:raise ValueError('Loaded model does not equal E5endpoint')
        del payload,scope
        model.requires_grad_(False);model.eval();before=fingerprint()
        if any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Inference model has gradients/train-mode enabled')
        emit(RUN/'model-before.json',dict(fingerprint=before,checkpoint_sha256=CHECKPOINT_SHA,original_tensors_verified=original_tensors,trainable_parameter_count=0))
        processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
        contract={r['image_id']:r for r in read(RUN/'preprocessing-manifest.json')['records']}
        gpu_begin=time.monotonic();state['phase']='CUDA-loading'
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable;no retry')
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction((LIMITS['VRAM_bytes']-512*2**20)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats(0)
        model=model.to('cuda');torch.cuda.synchronize();guard()
        if fingerprint()!=before:raise ValueError('Device transfer changed model state')
        def monitor():
            while not finished.wait(2):
                try:
                    guard();emit(RUN/'telemetry'/('heartbeat-'+uuid.uuid4().hex[:8]+'.json'),resources())
                except BaseException as e:errors.append(str(e));stopped.set();return
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        def count_actual(*_):
            state['actual_forwards']+=1
            if state['actual_forwards']!=state['attempted_forwards'] or state['actual_forwards']>59:raise RuntimeError('Unexpected extra detector forward')
        model.register_forward_pre_hook(count_actual)
        records=[];state['phase']='inference-only'
        for number,ident in enumerate(cfg['image_ids'],1):
            guard();row=require_train(ident,train);state['image_id']=ident
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Source changed immediately before forward')
            with Image.open(ROOT/row['path']) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source coordinate registration changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
            if tensors!=contract[ident]['processor_tensors']:raise ValueError('Preprocessing contract changed')
            consume_forward(state)
            emit(RUN/'telemetry'/f'forward-{number:02}.intent.json',dict(state,source_sha256=row['sha256'],source_group=row['identity_group']))
            guard();start=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                batch_gpu={k:v.to('cuda') for k,v in batch.items()};out=model(**batch_gpu)
            torch.cuda.synchronize();state['completed_forwards']+=1;forward_seconds=time.monotonic()-start
            logits=out.logits[0].float().cpu().numpy();square=out.pred_boxes[0].float().cpu().numpy()
            values=raw_values(logits,square,row['width'],row['height'])
            del out,batch_gpu,batch
            guard();p=RUN/'queries'/f'{number:02}.npz';arrays(p,values)
            with np.load(p,allow_pickle=False) as check:
                if set(check.files)!=set(values) or any(not np.array_equal(check[k],v) for k,v in values.items()):raise ValueError('Raw query atomic round-trip mismatch')
            receipt=dict(image_id=ident,source_sha256=row['sha256'],source_group=row['identity_group'],width=row['width'],height=row['height'],
                 historical_partition='train',internal_partition='optimization',file=str(p.relative_to(RUN)),sha256=sha(p),query_count=3600,class_count=7,
                 checkpoint_sha256=CHECKPOINT_SHA,binding_sha256=sha(RUN/'binding.json'),preprocessing_tensors=tensors,
                 query_layout=[60,60],query_identifier='row-major0..3599',coordinate_mapping=cfg['coordinate_mapping'],
                 precision=cfg['precision'],forward_seconds=forward_seconds,original_coordinates_clipped=False,
                 threshold_filtering_applied_to_raw_cache=False,negative_labels_created=0)
            emit(p.with_suffix('.json'),receipt);records.append(receipt);state['cached_photos']+=1
            emit(RUN/'telemetry'/f'forward-{number:02}.complete.json',resources())
            print('E7CACHED',number,'/59',ident,'seconds',round(time.monotonic()-begin,3),flush=True)
        state['phase']='post-inference-integrity';guard();after=fingerprint()
        if after!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Inference changed model state/gradient flags')
        if not state['attempted_forwards']==state['actual_forwards']==state['completed_forwards']==state['cached_photos']==59:raise ValueError('Incomplete59-forward run')
        if sha(CHECKPOINT)!=CHECKPOINT_SHA or sha(BASE/'model.safetensors')!=BASE_SHA:raise ValueError('Source weights changed')
        guard();emit(RUN/'model-after.json',dict(fingerprint=after,parameters_and_buffers_unchanged=True,derived_tensors_unchanged=True,gradients_present=False))
        emit(RUN/'raw-complete.json',dict(records=records,rows_sha256=cfg['rows_sha256'],query_count=59*3600,class_score_count=59*3600*7,
             parameters_and_buffers_unchanged=True,model_fingerprint=after,precision=cfg['precision'],resources=resources(),analysis_performed_on_GPU=False))
        state['phase']='complete';guard();emit(RUN/'complete.json',dict(status='PASS',resources=resources(),model_before=before,model_after=after,checkpoint_sha256=CHECKPOINT_SHA,
             automatic_retry=False,training=False,development_or_reserved_accessed=False))
        print('E7INFERENCE COMPLETE;59forwards;zero updates;raw caches immutable',flush=True)
    except BaseException as exc:
        failure=dict(error=type(exc).__name__+': '+str(exc),stack=traceback.format_exc(),resources=resources(),model_before=before,automatic_retry=False)
        if model is not None and callable(fingerprint) and time.monotonic()-begin<171:
            try:failure['model_after_failure']=fingerprint()
            except BaseException as e:failure['fingerprint_unavailable']=str(e)
        try:emit(RUN/'failure.json',failure)
        except BaseException as e:exc.add_note('Failure telemetry unavailable:'+str(e))
        raise
    finally:
        finished.set()
        if monitor_thread is not None:monitor_thread.join(timeout=1)
        if torch is not None and old_TF32 is not None:torch.backends.cuda.matmul.allow_tf32=old_TF32

if __name__=='__main__':run()
