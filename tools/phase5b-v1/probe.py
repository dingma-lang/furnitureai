"""GPU forward/backward memory probes WITHOUT optimizers or parameter updates."""
import argparse
import gc
import json
import os
import time
from common import *

def run():
 import torch
 from PIL import Image,ImageOps
 from transformers import Owlv2ForObjectDetection,Owlv2Processor
 from adaptation import configure,detection_loss,trainable_hash
 if not torch.cuda.is_available():raise RuntimeError('GPU probe requires actual CUDA')
 out=OUT/'memory-probes-v1'
 if out.exists():raise FileExistsError('Fresh probe output required')
 out.mkdir();torch.cuda.set_per_process_memory_fraction(.80,0);processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
 row=sorted(load_rows('train'),key=lambda r:r['image_id'])[0]
 with Image.open(image_path(row,'train')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
 results=[]
 for mode,checkpointing in [('heads',False),('last1',True),('lora4',True)]:
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0);start=time.perf_counter();model=None;inputs=outputs=loss=None
  try:
   torch.manual_seed(20261105)
   model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True).to('cuda')
   names=configure(model,mode,checkpointing);before=trainable_hash(model)
   inputs=processor(text=[SETTINGS['prompts']],images=image,return_tensors='pt');inputs={k:v.to('cuda') for k,v in inputs.items()}
   with torch.autocast('cuda',dtype=torch.float16):outputs=model(**inputs)
   loss,parts=detection_loss(outputs.logits[0],outputs.pred_boxes[0],row);loss.backward();torch.cuda.synchronize(0)
   finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
   if not finite:raise ValueError('Missing/nonfinite trainable gradients')
   unchanged=trainable_hash(model)==before
   if not unchanged:raise ValueError('Parameter changed in no-update probe')
   result=dict(mode=mode,checkpointing=checkpointing,status='forward_backward_success_no_updates',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),trainable_modules=names,
       peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),seconds=time.perf_counter()-start,loss=float(loss.detach()),loss_parts=parts,trainable_hash_unchanged=unchanged,optimizer_created=False,optimizer_steps=0)
  except torch.cuda.OutOfMemoryError as exc:
   result=dict(mode=mode,status='cuda_oom_under_80_percent_process_cap',error=str(exc),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),optimizer_steps=0)
  results.append(result);save(out/(mode+'.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='trainable_modules'},indent=2),flush=True)
  model=inputs=outputs=loss=None;gc.collect();torch.cuda.empty_cache()
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original checkpoint changed')
 save(out/'summary.json',dict(device=torch.cuda.get_device_name(0),total_vram=torch.cuda.get_device_properties(0).total_memory,process_memory_fraction=.80,train_photo_id=row['image_id'],frozen_photos_used=0,optimizer_steps=0,results=results))

if __name__=='__main__':run()
