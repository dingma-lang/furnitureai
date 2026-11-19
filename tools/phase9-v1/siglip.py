"""Frozen category-consistency check; never relabel, fit, or rescue unsuitable crops."""
import argparse
import gc
import hashlib
import math
import time
from core import *
from data import exposed_rows,final_rows_metadata,image
from infer import cuda,caches

def suitable(p,width,height):
 x1,y1,x2,y2=p['box'];w=(x2-x1)*width;h=(y2-y1)*height
 return min(w,h)>=32 and max(w/h,h/w)<=4 and min(x1,y1)> .005 and max(x2,y2)<.995

def should_abstain(prediction,verification):
 return verification.get('status')=='category_scored' and verification['category']!=prediction['label'] and verification['relative_softmax_margin']>=.10

def run(part='exposed'):
 torch=cuda();rows=exposed_rows() if part=='exposed' else final_rows_metadata()
 arms=('original','adapted') if part=='exposed' else [checked(OUT/'selection.json')['selected'].split('-',1)[0]]
 pending=[]
 for arm in arms:
  for r in rows:
   token=hashlib.sha256(r['image_id'].encode()).hexdigest()[:16];p=OUT/'siglip'/part/arm/(token+'.json')
   if not checked(p):pending.append((arm,r,p))
 if not pending:return
 from transformers import AutoModel,AutoImageProcessor
 model_manifest=read(SIGLIP/'checkpoint-manifest.json')
 for filename,digest in model_manifest['files'].items():
  if sha(SIGLIP/filename)!=digest:raise ValueError('Pinned encoder/preprocessor changed')
 head=torch.load(HEAD,map_location='cpu',weights_only=True)
 if head.get('feature_model_fingerprint')!=model_manifest['model_processor_fingerprint'] or tuple(head['classes'])!=CLASSES or set(head['state_dict'])!={'weight'}:raise ValueError('Classifier feature space/key mismatch')
 raw_weight=head['state_dict']['weight'].float()
 if not torch.isfinite(raw_weight).all() or torch.any(raw_weight.norm(dim=1)<1e-8):raise ValueError('Invalid prototype weights')
 weight=torch.nn.functional.normalize(raw_weight,dim=1)
 if weight.shape!=(7,768) or not torch.isfinite(weight).all():raise ValueError('Invalid classifier')
 processor=AutoImageProcessor.from_pretrained(str(SIGLIP),use_fast=False,local_files_only=True)
 start=time.monotonic();model=AutoModel.from_pretrained(str(SIGLIP),use_safetensors=True,local_files_only=True).to('cuda').eval().requires_grad_(False);torch.cuda.synchronize();load_seconds=time.monotonic()-start
 count=0;crop_count=0
 try:
  for arm,r,path in pending:
   if time.monotonic()-start>1800:raise RuntimeError('SigLIP invocation time cap')
   im=image(r);cache=caches(part,arm,r)['full960'];preds=cache['predictions'];checks=[];work=[]
   torch.cuda.reset_peak_memory_stats(0);torch.cuda.synchronize();clock=time.monotonic()
   for i,p in enumerate(preds):
    if not suitable(p,im.width,im.height):checks.append(dict(index=i,status='unsuitable_geometry_unresolved',category_not_forced=True));continue
    x1,y1,x2,y2=p['box'];bounds=[math.floor(x1*im.width),math.floor(y1*im.height),math.ceil(x2*im.width),math.ceil(y2*im.height)];crop=im.crop(bounds)
    work.append((i,crop,bounds));crop_count+=1
    if crop_count>10800:raise RuntimeError('Bounded verification crop cap')
   for pos in range(0,len(work),4):
    batch=work[pos:pos+4];inputs=processor(images=[v[1] for v in batch],return_tensors='pt')
    with torch.inference_mode():
     features=model.get_image_features(**{k:v.to('cuda') for k,v in inputs.items()}).float()
     if not torch.isfinite(features).all() or torch.any(features.norm(dim=-1)<1e-8):raise ValueError('Invalid encoder features')
     features=features/features.norm(dim=-1,keepdim=True)
     similarities=features.cpu()@weight.T;scores=(10*similarities).softmax(-1)
    for j,(index,crop,bounds) in enumerate(batch):
     values=scores[j].tolist();rank=sorted(range(7),key=lambda k:-values[k]);checks.append(dict(index=index,status='category_scored',category=CLASSES[rank[0]],relative_softmax_margin=values[rank[0]]-values[rank[1]],relative_softmax_scores=values,
      cosine_similarities=similarities[j].tolist(),crop_bounds=bounds,crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
      normalized_feature_sha256=hashlib.sha256(features[j].cpu().numpy().tobytes()).hexdigest(),calibrated_probability=False,original_label_preserved=True))
   torch.cuda.synchronize();seconds=time.monotonic()-clock;peak=torch.cuda.max_memory_allocated(0)
   if peak>6.5*2**30:raise RuntimeError('Unexpected SigLIP memory use; stop')
   checks.sort(key=lambda x:x['index']);sealed(path,dict(image_id=r['image_id'],arm=arm,checks=checks,seconds=seconds,peak_allocated_bytes=peak,
    classifier_sha256=PINS[str(HEAD.relative_to(ROOT))],processor_fingerprint=model_manifest['model_processor_fingerprint'],code_sha256=sha(__file__),source_predictions_sha256=hashlib.sha256(json.dumps(preds,sort_keys=True).encode()).hexdigest()))
   count+=1;print('SIGLIP',part,arm,count,'usable crops',len(work),'total',crop_count,flush=True)
 finally:
  del model;gc.collect();torch.cuda.empty_cache()
 p=OUT/(part+'-siglip-complete.json')
 if not p.exists():sealed(p,dict(images_this_invocation=count,crops_this_invocation=crop_count,model_load_seconds=load_seconds,no_model_parameters_updated=True,relative_scores_not_probabilities=True))

def record(part,arm,row):
 token=hashlib.sha256(row['image_id'].encode()).hexdigest()[:16]
 return checked(OUT/'siglip'/part/arm/(token+'.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['exposed','final'],default='exposed');args=p.parse_args();run(args.part)
