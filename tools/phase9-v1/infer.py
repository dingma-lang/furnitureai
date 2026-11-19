"""Offline sequential CUDA passes, frozen parameters and resumable immutable stages."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import gc
import hashlib
import math
import time
from core import *
from data import exposed_rows,final_rows_metadata,image
from fusion import views,map_box,internal_edge,nms

def runtime_hashes():return {n:sha(HERE/n) for n in ('core.py','data.py','fusion.py','infer.py')}
def cuda():
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: no automatic CPU inference fallback')
 torch.set_num_threads(4);torch.manual_seed(20261105);torch.use_deterministic_algorithms(True)
 torch.cuda.set_per_process_memory_fraction(.75,0)
 return torch
def load_detector(arm):
 import torch
 from transformers import Owlv2ForObjectDetection,Owlv2Processor
 start=time.monotonic();processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
 model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True)
 if arm=='adapted':
  registry=read(REGISTRY);p=Path(registry['checkpoint'])
  if sha(p)!=registry['sha256']:raise ValueError('Adapted checkpoint hash changed')
  values=torch.load(p,map_location='cpu',weights_only=True)['weights'];params=dict(model.named_parameters())
  expected={k for k in params if k.startswith(('class_head.','box_head.'))}
  if set(values)!=expected or len(expected)!=12:raise ValueError('Head delta key mismatch')
  with torch.no_grad():
   for name,value in values.items():
    if value.shape!=params[name].shape or not torch.isfinite(value).all():raise ValueError('Invalid head delta')
    params[name].copy_(value)
 elif arm!='original':raise ValueError('Unknown detector')
 model.requires_grad_(False);model.eval();model.to('cuda')
 if any(p.requires_grad for p in model.parameters()):raise ValueError('Inference model has trainable parameters')
 torch.cuda.synchronize();return model,processor,time.monotonic()-start

def pass_image(model,processor,im,spec):
 import torch
 from PIL import ImageOps
 w,h=im.size;x1,y1,x2,y2=spec['region'];bounds=(math.floor(x1*w),math.floor(y1*h),math.ceil(x2*w),math.ceil(y2*h))
 local=im.crop(bounds);region=[bounds[0]/w,bounds[1]/h,bounds[2]/w,bounds[3]/h]
 if spec['flip']:local=ImageOps.mirror(local)
 processor.image_processor.size={'height':spec['size'],'width':spec['size']}
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(0);start=time.monotonic()
 with torch.inference_mode():
  inputs=processor(text=[SETTINGS['prompts']],images=local,return_tensors='pt')
  pixel=inputs['pixel_values'];tensor_sha=hashlib.sha256(pixel.numpy().tobytes()).hexdigest();shape=list(pixel.shape)
  if shape[-2:]!=[spec['size'],spec['size']]:raise ValueError('Multi-scale processor did not change model tensor resolution')
  result=model(**{k:v.to('cuda') for k,v in inputs.items()},interpolate_pos_encoding=spec['size']!=960)
  post=processor.image_processor.post_process_object_detection(result,threshold=.001,target_sizes=[(local.height,local.width)])[0]
  raw=[]
  for b,s,c in zip(post['boxes'].float().cpu().tolist(),post['scores'].float().cpu().tolist(),post['labels'].cpu().tolist()):
   box=[b[0]/local.width,b[1]/local.height,b[2]/local.width,b[3]/local.height]
   mapped=map_box(box,region,spec['flip'])
   if mapped[2]<=mapped[0] or mapped[3]<=mapped[1]:continue
   raw.append(dict(box=mapped,label=CLASSES[c],score=s,view=spec['name'],internal_tile_edge=internal_edge(box,region)))
  output=nms(raw)
 torch.cuda.synchronize();seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated(0)
 if peak>6.5*2**30:raise RuntimeError('Unexpected VRAM budget use; stop instead of changing recipe')
 return dict(predictions=output,raw_candidate_count=len(raw),per_view_nms_iou=.5,seconds=seconds,peak_allocated_bytes=peak,
  pixel_values_sha256=tensor_sha,pixel_values_shape=shape,actual_region=region,crop_bounds=list(bounds),interpolate_pos_encoding=spec['size']!=960)

def capture(part='exposed',smoke=False):
 torch=cuda();rows=exposed_rows() if part=='exposed' else final_rows_metadata()
 if part=='final':
  selected=checked(OUT/'selection.json')
  if not selected or selected['code_hashes']!=code_hashes():raise PermissionError('Final requires locked selection')
  required=selected['required_views']
 else:required={a:[v['name'] for v in views()] for a in ('original','adapted')}
 if smoke:rows=rows[:1]
 started=time.monotonic();count=0;ledger=[]
 for arm,names in required.items():
  pending=[]
  for r in rows:
   for v in views():
    if v['name'] not in names:continue
    token=hashlib.sha256(r['image_id'].encode()).hexdigest()[:16];p=OUT/'passes'/part/arm/token/(v['name']+'.json')
    binding=dict(image_id=r['image_id'],image_sha256=r['sha256'],arm=arm,view=v,code=runtime_hashes(),source_model_sha256=read(OUT/'model-inventory.json')['weights'][arm]['sha256'])
    old=checked(p)
    if old:
     if old['binding']!=binding:raise ValueError('Resume binding mismatch')
    else:pending.append((r,v,p,binding))
  if not pending:continue
  model,processor,load_seconds=load_detector(arm);last=None;im=None
  try:
   for r,v,p,binding in pending:
    if time.monotonic()-started>1800:raise RuntimeError('30-minute invocation budget exhausted')
    if last!=r['image_id']:im=image(r);last=r['image_id']
    result=pass_image(model,processor,im,v);sealed(p,dict(binding=binding,**result));count+=1
    print(part,arm,count,r['image_id'],v['name'],round(result['seconds'],3),'s',round(result['peak_allocated_bytes']/2**30,3),'GiB',flush=True)
   ledger.append(dict(arm=arm,load_seconds=load_seconds,passes=len(pending)))
  except BaseException as e:
   save(OUT/('inference-failure-'+now().replace(':','-')+'.json'),dict(error=repr(e),part=part,arm=arm,completed_new_passes=count));raise
  finally:
   del model,processor;gc.collect();torch.cuda.empty_cache()
 if smoke:
  p=OUT/'smoke.json'
  if not p.exists():sealed(p,dict(passed=True,source_image=rows[0]['image_id'],completed_new_passes=count,loading=ledger,training_updates=0))
 else:
  p=OUT/(part+'-capture-complete.json')
  if not p.exists():sealed(p,dict(rows=len(rows),required_views=required,completed_new_passes_this_invocation=count,loading=ledger,seconds_this_invocation=time.monotonic()-started,training_updates=0))

def caches(part,arm,row):
 token=hashlib.sha256(row['image_id'].encode()).hexdigest()[:16];result={}
 for v in views():
  p=OUT/'passes'/part/arm/token/(v['name']+'.json')
  if p.exists():result[v['name']]=checked(p)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['exposed','final'],default='exposed');p.add_argument('--smoke',action='store_true');a=p.parse_args();capture(a.part,a.smoke)
