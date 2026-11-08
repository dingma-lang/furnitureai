"""Lazy optional adapters. No downloader or package installation code."""
import os
from pathlib import Path
import math
import numpy as np
from PIL import Image

class BackendUnavailable(RuntimeError):
    pass


def local_model(config):
    path=Path(config.get('model_path','/nonexistent-m3-checkpoint')).expanduser()
    if not path.exists():raise BackendUnavailable('Local checkpoint missing; acquisition requires explicit approval. No download attempted.')
    if not config.get('approval_reference'):raise BackendUnavailable('Checkpoint authorization/provenance reference required')
    return path


def fit_pad(image,mask,long_edge):
    scale=min(1.,long_edge/max(image.size));size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    padded=(math.ceil(size[0]/8)*8,math.ceil(size[1]/8)*8)
    rgb=np.array(image.resize(size,Image.Resampling.LANCZOS));rgb=np.pad(rgb,((0,padded[1]-size[1]),(0,padded[0]-size[0]),(0,0)),mode='edge')
    m=Image.new('L',padded);m.paste(mask.resize(size,Image.Resampling.NEAREST),(0,0))
    return Image.fromarray(rgb),m,{'resized':size,'padded':padded,'original':image.size,'scale':scale}


def restore(image,mask,protected,config):
    backend=config.get('backend','classical')
    if backend=='classical':
        from photo_simulator.restoration import restore_background
        params=dict(config.get('classical',{}));params['mask_expand']=0;params['background_patches']=[]
        result,_,diagnostics=restore_background(np.array(image),np.array(mask)>0,np.array(protected)>0,params)
        raw=Image.fromarray(result);return raw,raw.copy(),{'classical':diagnostics,'gpu_peak_allocated_bytes':None,'gpu_peak_reserved_bytes':None}
    if backend not in ('lama','diffusion'):raise ValueError('Unknown restoration backend')
    path=local_model(config)
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[key]='1'
    try:import torch
    except ImportError as exc:raise BackendUnavailable('PyTorch unavailable in this isolated environment') from exc
    device=config.get('device','cpu')
    if device not in ('cpu','cuda'):raise ValueError('Device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():raise BackendUnavailable('Requested CUDA device unavailable')
    seed=config.get('seed',0);steps=config.get('steps',20);size=config.get('long_edge',512)
    if type(seed)!=int or not 0<=seed<2**32 or type(steps)!=int or not 1<=steps<=50 or type(size)!=int or not 64<=size<=1024:raise ValueError('Invalid bounded seed, steps or resolution')
    image_in,mask_in,mapping=fit_pad(image,mask,size)
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    if backend=='diffusion':
        try:from diffusers import AutoPipelineForInpainting
        except ImportError as exc:raise BackendUnavailable('diffusers unavailable; isolated dependency installation requires approval') from exc
        dtype=torch.float16 if device=='cuda' else torch.float32
        pipe=AutoPipelineForInpainting.from_pretrained(str(path),local_files_only=True,use_safetensors=True,variant='fp16',torch_dtype=dtype)
        if config.get('cpu_offload',False) and device=='cuda':pipe.enable_model_cpu_offload()
        else:pipe.to(device)
        pipe.enable_vae_slicing()
        guidance=float(config.get('guidance_scale',5));strength=float(config.get('strength',.99))
        if not math.isfinite(guidance) or not 0<=guidance<=15 or not math.isfinite(strength) or not 0<strength<=1:raise ValueError('Invalid guidance or strength')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        with torch.inference_mode():
            output=pipe(prompt=config.get('prompt','empty room background, no furniture'),negative_prompt=config.get('negative_prompt','sofa, furniture, people, objects'),image=image_in,mask_image=mask_in,width=image_in.width,height=image_in.height,num_inference_steps=steps,guidance_scale=guidance,strength=strength,generator=generator)
        if any(getattr(output,'nsfw_content_detected',None) or []):raise RuntimeError('Candidate flagged by model safety checker; not accepted')
        raw=output.images[0]
    else:
        # Explicitly authorized TorchScript exports only. Original Lightning .ckpt
        # needs an isolated legacy conversion step, not torch.load in production.
        if config.get('lama_format')!='torchscript-image-mask':raise BackendUnavailable('LaMa requires an authorized local TorchScript image/mask export; native Lightning checkpoint conversion is not installed')
        model=torch.jit.load(str(path),map_location=device).eval()
        tensor=torch.from_numpy(np.array(image_in).copy()).permute(2,0,1)[None].float().to(device)/255
        m=torch.from_numpy((np.array(mask_in)>0).astype(np.float32))[None,None].to(device)
        with torch.inference_mode():result=model(tensor,m)
        if not isinstance(result,torch.Tensor) or tuple(result.shape)!=(1,3,image_in.height,image_in.width) or not torch.isfinite(result).all() or result.min()<0 or result.max()>1:raise ValueError('LaMa export must return finite [1,3,H,W] RGB in [0,1]')
        raw=Image.fromarray((result[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
    if raw.size!=image_in.size:raise ValueError('Backend output resolution differs from padded input')
    meta={'mapping':mapping,'seed':seed,'steps':steps,'device':device,'gpu_peak_allocated_bytes':None,'gpu_peak_reserved_bytes':None,'determinism':'Seeded; bitwise reproducibility across hardware/software is not guaranteed'}
    if device=='cuda':
        torch.cuda.synchronize();meta.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    mapped=raw.crop((0,0,*mapping['resized'])).resize(image.size,Image.Resampling.LANCZOS)
    return raw,mapped,meta
