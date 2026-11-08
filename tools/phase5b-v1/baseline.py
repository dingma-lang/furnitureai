"""Execute original OWLv2 on locked eligible DEVELOPMENT photos only."""
import argparse
import gc
import html
import json
import os
from pathlib import Path
import sys
import time
from common import *
from metrics import nms,evaluate

def predict(model,processor,image):
 import torch
 with torch.inference_mode():
  inputs=processor(text=[SETTINGS['prompts']],images=image,return_tensors='pt');inputs={k:v.to(next(model.parameters()).device) for k,v in inputs.items()}
  outputs=model(**inputs)
  result=processor.image_processor.post_process_object_detection(outputs=outputs,threshold=SETTINGS['ap_score_floor'],target_sizes=[(image.height,image.width)])[0]
  raw=[]
  for b,s,c in zip(result['boxes'].float().cpu().tolist(),result['scores'].float().cpu().tolist(),result['labels'].cpu().tolist()):
   b=[max(0,min(v,lim)) for v,lim in zip(b,[image.width,image.height,image.width,image.height])]
   if b[2]<=b[0] or b[3]<=b[1]:continue
   raw.append(dict(box=[b[0]/image.width,b[1]/image.height,b[2]/image.width,b[3]/image.height],label=CLASSES[c],score=s))
 return raw,nms(raw,SETTINGS['nms_iou'],SETTINGS['max_detections'])

def gallery(rows,records,metrics,out):
 from PIL import Image,ImageDraw,ImageOps
 parts=['<!doctype html><meta charset="utf-8"><title>OWLv2 development benchmark</title><h1>Original OWLv2 — eligible development only</h1><p>Published native annotations versus detections at .15. Unknown classes unscored. No classifier disagreement or frozen-photo metrics.</p>']
 for row in rows:
  ident=row['image_id'];rec=records[ident]
  with Image.open(image_path(row,'development')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
  scale=800/image.width;h=round(image.height*scale);canvas=Image.new('RGB',(1600,h+50),'white');draw=ImageDraw.Draw(canvas)
  for col in (0,1):canvas.paste(image.resize((800,h)),(col*800,50));draw.text((col*800+8,8),'PUBLISHED BOXES' if col==0 else 'ORIGINAL OWLv2 @ .15',fill='black')
  for g in row['annotations']:
   b=g['box'];coords=[b[0]*800,b[1]*h+50,b[2]*800,b[3]*h+50];color='gray' if g['ignore'] else 'green';draw.rectangle(coords,outline=color,width=2);draw.text((coords[0],coords[1]),g['label'],fill='white',stroke_width=1,stroke_fill='black')
  for p in rec['predictions']:
   if p['score']<SETTINGS['operating_score']:continue
   b=p['box'];coords=[800+b[0]*800,b[1]*h+50,800+b[2]*800,b[3]*h+50];color='orange' if not row['coverage'][p['label']] else 'red';draw.rectangle(coords,outline=color,width=2);draw.text((coords[0],coords[1]),p['label']+f" {p['score']:.2f}",fill='white',stroke_width=1,stroke_fill='black')
  path=out/(ident+'.jpg');canvas.save(path,quality=90)
  events=[dict(category=c,**e) for c,es in metrics['diagnostics']['events'].items() for e in es if e['image_id']==ident]
  parts.append('<h2>'+ident+'</h2><img width="1400" src="'+ident+'.jpg"><pre>'+html.escape(json.dumps(dict(coverage=row['coverage_provenance'],events=events),indent=2))+'</pre>')
 with (out/'gallery.html').open('x') as f:f.write(''.join(parts))

def run(resume=False,device='cuda'):
 import torch
 import numpy as np
 from PIL import Image,ImageOps
 from transformers import Owlv2ForObjectDetection,Owlv2Processor
 rows=load_rows('development');out=OUT/'original-development-v1'
 binding=dict(dataset_sha256=sha(OUT/'dataset-lock.json'),checkpoint_sha256=OWL_SHA,model_files={p.name:sha(p) for p in OWL.iterdir() if p.is_file()},settings=SETTINGS,
  code={name:sha(HERE/name) for name in ('common.py','metrics.py','baseline.py')},device=device)
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original model changed')
 if out.exists():
  if not resume:raise FileExistsError('Explicit --resume required')
  if read(out/'binding.json')!=binding:raise ValueError('Resume binding mismatch')
 else:out.mkdir();save(out/'binding.json',binding)
 pending=[r for r in rows if not (out/(r['image_id']+'.json')).exists()]
 torch.manual_seed(SETTINGS['seed']);np.random.seed(SETTINGS['seed']);start=time.perf_counter();model=None;peak=0
 if pending:
  if device=='cuda':
   if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
   torch.cuda.set_per_process_memory_fraction(.80,0);torch.cuda.reset_peak_memory_stats(0)
  processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
  model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True).to(device).eval()
  for i,row in enumerate(pending):
   with Image.open(image_path(row,'development')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
   t=time.perf_counter();raw,filtered=predict(model,processor,image)
   if device=='cuda':torch.cuda.synchronize(0);peak=max(peak,torch.cuda.max_memory_allocated(0))
   record=dict(image_id=row['image_id'],image_sha256=row['sha256'],split='development',raw_candidates=raw,predictions=filtered,seconds=time.perf_counter()-t)
   save(out/(row['image_id']+'.json'),record);save(out/(row['image_id']+'.checksum.json'),dict(sha256=sha(out/(row['image_id']+'.json'))))
   print('DEVELOPMENT',i+1,'/',len(pending),row['image_id'],'AP candidates',len(filtered),'operating',sum(p['score']>=.15 for p in filtered),flush=True)
  save(out/'runtime.json',dict(seconds=time.perf_counter()-start,peak_allocated_bytes=peak,device=device,torch=torch.__version__,cuda=torch.version.cuda,frozen_images_loaded=0,official_phase2b_used=False))
 records={}
 for row in rows:
  path=out/(row['image_id']+'.json');r=read(path)
  if sha(path)!=read(out/(row['image_id']+'.checksum.json'))['sha256'] or r['image_sha256']!=row['sha256']:raise ValueError('Completed prediction corrupt')
  records[row['image_id']]=r
 result=evaluate(rows,{key:r['predictions'] for key,r in records.items()})
 if not (out/'metrics.json').exists():save(out/'metrics.json',result)
 elif read(out/'metrics.json')!=result:raise ValueError('Completed metrics differ')
 if not (out/'gallery.html').exists():gallery(rows,records,result,out)
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original checkpoint changed')
 if not (out/'complete.json').exists():save(out/'complete.json',dict(images=len(rows),metrics_sha256=sha(out/'metrics.json'),inference_checkpoint_sha256=OWL_SHA,frozen_evaluated=False))
 print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','categories')},indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=ap.parse_args();run(a.resume,a.device)
