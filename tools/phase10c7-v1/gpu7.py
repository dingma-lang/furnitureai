"""One process, at most24 original-model forwards; durable outputs before analysis."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,time,threading,resource,traceback,warnings
from common7 import *

def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));binding=inputs();rows=validate_binding(binding)
    save(RUN/'worker-started.json',dict(monotonic_start=time.monotonic(),pid=os.getpid(),training_authorized=False))
    if list(RUN.glob('forward-*.intent.json')):raise RuntimeError('Prior forward reservation exists; no GPU replay permitted')
    started=time.monotonic();calls=0;completed=[];stop=threading.Event();violations=[];monitor_values=[]
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        torch.set_num_threads(2);torch.manual_seed(binding['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback or automatic retry')
        torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory
        # Reserve0.5GiB for driver/context allocations beyond the Torch allocator.
        torch.cuda.set_per_process_memory_fraction((MAX_VRAM-512*2**20)/total,0)
        try:torch.cuda.reset_peak_memory_stats(0)
        except RuntimeError as e:warnings.warn('Optional peak reset unavailable; fresh-process peaks still recorded: '+str(e))
        def monitor():
            while not stop.wait(.05):
                try:
                    free,total_device=torch.cuda.mem_get_info(0);used=total_device-free;monitor_values.append(used)
                    if used>MAX_VRAM:violations.append('Device used VRAM exceeded6.5GiB');return
                except Exception as exc:violations.append('Memory monitor failed: '+str(exc));return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        def guard():
            if time.monotonic()-started>=MAX_SECONDS:raise RuntimeError('600-second runtime limit')
            if violations:raise RuntimeError('; '.join(violations))
            if max(torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0))>MAX_VRAM:raise RuntimeError('6.5GiB Torch VRAM limit')
            if (RUN/'STOP').exists():raise RuntimeError('Requested stop')
        guard();model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
        model.requires_grad_(False);model.eval();initial_state=state_hash(model);model.to('cuda');processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True)
        if not processor.image_processor.do_pad or not processor.image_processor.do_resize or processor.image_processor.size!={'height':960,'width':960}:raise ValueError('Unexpected image preprocessing')
        def reserve(_model,_args):
            nonlocal calls
            guard()
            if calls>=MAX_FORWARDS:raise RuntimeError('24-forward hard cap')
            row=rows[calls];save(RUN/('forward-'+str(calls+1).zfill(2)+'.intent.json'),dict(image_id=row['image_id'],image_sha256=row['sha256'],forward_reserved=True,seconds=time.monotonic()-started));calls+=1
        handle=model.register_forward_pre_hook(reserve)
        for index,row in enumerate(rows):
            guard();p=ROOT/row['path']
            if sha(p)!=row['sha256']:raise ValueError('Photograph bytes changed')
            with Image.open(p) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Original image frame/orientation changed')
                batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
            if tuple(batch['pixel_values'].shape)!=(1,3,960,960):raise ValueError('Unexpected processed dimensions')
            pixel_sha=hashlib.sha256(batch['pixel_values'].contiguous().numpy().tobytes()).hexdigest();begin=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                out=model(**{k:v.to('cuda') for k,v in batch.items()})
            torch.cuda.synchronize(0)
            logits=out.logits[0].float().cpu().numpy();square=out.pred_boxes[0].float().cpu().numpy();pixels,normalized=original_boxes(square,row['width'],row['height'])
            arrays=dict(logits=logits,square_cxcywh=square,original_xyxy_pixels=pixels,original_xyxy_normalized=normalized)
            path=RUN/('queries-'+str(index+1).zfill(2)+'.npz');write_arrays(path,arrays)
            receipt=dict(index=index+1,image_id=row['image_id'],source_sha256=row['sha256'],identity_group=row['identity_group'],raw_file=path.name,raw_sha256=sha(path),bytes=path.stat().st_size,queries=len(logits),forward_seconds=time.monotonic()-begin,
                pixel_values_sha256=pixel_sha,original_dimensions=[row['height'],row['width']],padding_bottom=max(row['width']-row['height'],0),padding_right=max(row['height']-row['width'],0),resize=[960,960],unclipped_coordinates=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            save(RUN/('queries-'+str(index+1).zfill(2)+'.receipt.json'),receipt);completed.append(receipt)
            print('DURABLE FORWARD',index+1,'/24',row['image_id'],'queries',len(logits),'peakMiB',round(receipt['peak_allocated_bytes']/2**20),flush=True)
            del out,batch,arrays,logits,square,pixels,normalized;guard()
        handle.remove()
        if calls!=24 or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Forward/no-gradient contract violated')
        model.to('cpu');final_state=state_hash(model)
        if initial_state!=final_state:raise ValueError('Model state changed without authorization')
        guard();stop.set();thread.join(timeout=2)
        if violations:raise RuntimeError('; '.join(violations))
        for name,h in read(ROOT/'reports/phase10a-v1/input-contract.json')['model_files'].items():
            if sha(BASE/name)!=h:raise ValueError('Original checkpoint/processor changed')
        save(RUN/'raw-complete.json',dict(status='PASS',forward_calls=calls,records=completed,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            peak_device_used_bytes=max(monitor_values,default=0),device=torch.cuda.get_device_name(0),dtype='fp16 autocast / fp32 weights',attention_implementation=model.config._attn_implementation,
            initial_model_state_sha256=initial_state,final_model_state_sha256=final_state,all_parameters_frozen=True,gradients_created=False,backward_calls=0,optimizer_steps=0,binding_sha256=sha(RUN/'binding.json')))
    except BaseException as e:
        save(RUN/'gpu-failure.json',dict(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),reserved_forwards=calls,completed_forwards=len(completed),seconds=time.monotonic()-started,automatic_retry=False))
        raise
    finally:stop.set()

if __name__=='__main__':run()
