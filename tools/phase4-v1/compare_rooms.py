"""Offline isolated OWLv2 crop/classifier comparison. Never changes production labels/boxes."""
import argparse
import collections as C
import copy
import datetime as dt
import fcntl
import gc
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools/phase3-v1'))
from common import w,sha,read,save,atomic,CLASSES,PREP
from furniture_bench.data import load_image
from furniture_bench.geometry import crop_bounds
from furniture_bench.detection.owlv2 import Owlv2Detector
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

DEFAULT_OUT=ROOT/'reports/phase4-room-comparison-v1'
MANIFEST=ROOT/'data/detection-pilot/rooms.json'
MODEL=w.MODEL
OWL=ROOT/'models/owlv2-original-baseline-20261103'
SELECTED=ROOT/'data/abo/experiments/phase3-gpu-v1/final-selected/best.pt'
BASELINE=ROOT/'data/abo/experiments/phase3-gpu-v1/final-baseline/best.pt'
EXPECTED={'expanded':'51097ebf696a7765d023c8c02a9e4ea14965e0da04877cdd65be4416b9c13f0d','baseline':'2c61fe77229a67b1cdd2dc4cd3109fb0a9f3675ff5348519fddd8385dd9b6755'}
OWL_SHA='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
SETTINGS={'threshold':.15,'nms_iou':.5,'categories':list(CLASSES),'query_template':'a photo of a {category}',
 'max_rooms':3,'max_classified_crops_per_room':64,'embedding_batch_size':4,'oom_stage_attempts':2,
 'score_note':'s7 is uncalibrated seven-class softmax; cosine is similarity to a learned category prototype; neither is correctness probability',
 'no_ground_truth_metrics':True,'production_model_changed':False,'original_boxes_preserved':True,'downloads':False}


def stat(path):
 s=Path(path).stat();return [s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino]

def snapshot(output):
 inv=read(PREP/'eligibility-inventory.json')['images'];unopened={str((ROOT/r['local_path']).resolve()) for r in inv if r.get('proposed_split') in ('validation','test')}
 unopened|={str(p.resolve()) for part in ('validation','test') for p in (ROOT/f'data/abo/experiments/phase2a-v1/features-v1/{part}').glob('*.npy')}
 result={}
 for base in (ROOT/'models',ROOT/'reports',ROOT/'data',ROOT/'src',ROOT/'tools'):
  for p in base.rglob('*'):
   if not p.is_file() or p.is_relative_to(output) or p.is_relative_to(HERE) or 'evidence-completion-v1' in p.parts:continue
   result[str(p.relative_to(ROOT))]={'stat_only':stat(p)} if str(p.resolve()) in unopened else {'sha256':sha(p)}
 return result

def verify_preservation(out):
 source=read(out/'preservation-before.json');bad=[]
 for rel,expected in source.items():
  p=ROOT/rel
  if not p.is_file() or (stat(p)!=expected['stat_only'] if 'stat_only' in expected else sha(p)!=expected['sha256']):bad.append(rel)
 save(out/'preservation-after.json',{'checked':len(source),'mismatches':bad,'official_holdout_images_unopened':True})
 if bad:raise ValueError('Historical/production file changed: '+str(bad[:3]))
 return len(source)


def load_head(path,expected_sha,fingerprint):
 if sha(path)!=expected_sha:raise ValueError('Classifier SHA-256 mismatch')
 head=torch.load(path,map_location='cpu',weights_only=True)
 if tuple(head.get('classes',()))!=CLASSES or head.get('config',{}).get('kind')!='cosine':raise ValueError('Checkpoint category mapping/architecture differs')
 if head.get('feature_model_fingerprint')!=fingerprint:raise ValueError('Checkpoint embedding-space mismatch')
 if set(head.get('state_dict',{}))!={'weight'}:raise ValueError('Unexpected cosine head weights')
 weight=head['state_dict']['weight']
 if tuple(weight.shape)!=(7,768) or not torch.isfinite(weight).all() or torch.any(weight.norm(dim=1)<1e-8):raise ValueError('Invalid classifier weights')
 return {'weight':weight.detach().float(),'sha256':expected_sha,'checkpoint':str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),'config':head['config']}


def predict(shared_features,head):
 x=torch.as_tensor(shared_features,dtype=torch.float32)
 if x.ndim!=2 or x.shape[1]!=768 or not torch.isfinite(x).all() or not torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=1e-4):raise ValueError('Classification requires finite L2-normalized image features')
 with torch.inference_mode():
  similarities=torch.nn.functional.linear(x,torch.nn.functional.normalize(head['weight'],dim=1));logits=10*similarities;scores=logits.softmax(dim=1)
 rows=[]
 for similarity,logit,score in zip(similarities,logits,scores):
  order=score.argsort(descending=True);index=int(order[0])
  rows.append({'category':CLASSES[index],'cosine_similarity':float(similarity[index]),'uncalibrated_softmax7':float(score[index]),
   'softmax7_margin':float(score[index]-score[int(order[1])]),'all_cosine_similarities':similarity.tolist(),'all_logits':logit.tolist(),'all_softmax7':score.tolist(),'calibrated':False})
 return rows


def gpu_retry(operation,attempts=2):
 for attempt in range(attempts):
  try:return operation()
  except torch.cuda.OutOfMemoryError:
   torch.cuda.empty_cache()
   if attempt+1>=attempts:raise
 # Other CUDA/model/runtime failures propagate without fake outputs or CPU fallback.


def embed_shared(crops,model,processor,device,batch_size=4):
 vectors=[];proof=[];offset=0;oom=[]
 while offset<len(crops):
  group=crops[offset:offset+batch_size]
  try:
   inputs=processor(images=group,return_tensors='pt')
   pixels=inputs['pixel_values'];pixel_hashes=[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in pixels]
   with torch.inference_mode():values=model.get_image_features(**{k:v.to(device) for k,v in inputs.items()}).float().cpu().numpy()
   values=w.unit(values)
   for value,pixel_hash in zip(values,pixel_hashes):
    vectors.append(value);proof.append({'pixel_values_sha256':pixel_hash,'normalized_embedding_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'same_embedding_for_both_classifiers':True})
   offset+=len(group)
  except torch.cuda.OutOfMemoryError:
   inputs=None;values=None;torch.cuda.empty_cache()
   if batch_size==1:raise
   batch_size=max(1,batch_size//2);oom.append({'offset':offset,'reduced_batch':batch_size})
 return np.asarray(vectors,dtype=np.float32),proof,oom


def inputs_from_manifest(manifest):
 source=read(manifest);rows=source.get('images',[])
 if not 1<=len(rows)<=SETTINGS['max_rooms']:raise ValueError('Room experiment bounded to 1-3 authorized photographs')
 forbidden={r['sha256'] for r in read(PREP/'eligibility-inventory.json')['images'] if r.get('proposed_split') in ('validation','test')}
 forbidden|={r['sha256'] for r in read(ROOT/'data/abo/experiments/phase3-preparation-v1/frozen-development-protocol.json')['development_rows']}
 clean=[]
 for row in rows:
  path=(Path(manifest).parent/row['file']).resolve();rights=row.get('rights',{})
  if path.is_relative_to(ROOT/'data/abo') or rights.get('allowed_for_benchmark') is not True or not rights.get('source') or not rights.get('license'):raise ValueError('Only authorized non-ABO room photographs are allowed')
  actual=sha(path)
  if actual!=row.get('image_sha256') or actual in forbidden:raise ValueError('Room image hash changed or belongs to reserved evaluation')
  clean.append({'image_id':row['image_id'],'file':str(path),'sha256':actual,'rights':rights,'prior_annotations_used':False,
   'prior_annotation_status':row.get('annotation',{}).get('status','not_supplied')})
 if len({r['image_id'] for r in clean})!=len(clean):raise ValueError('Duplicate room IDs')
 return clean


def initialize(out,manifest,resume=False):
 if out.exists() and not resume:raise FileExistsError('Fresh report root required; explicit --resume reuses only matching stages')
 out.mkdir(exist_ok=True)
 rows=inputs_from_manifest(manifest);model_manifest=read(MODEL/'checkpoint-manifest.json');fingerprint=model_manifest['model_processor_fingerprint']
 if sha(SELECTED)!=EXPECTED['expanded'] or sha(BASELINE)!=EXPECTED['baseline'] or sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Pinned classifier/detector weights changed')
 for rel,h in model_manifest['files'].items():
  if sha(MODEL/rel)!=h:raise ValueError('Pinned SigLIP model/preprocessor changed')
 for arm,path in [('expanded',SELECTED),('baseline',BASELINE)]:load_head(path,EXPECTED[arm],fingerprint)
 binding={'rooms':rows,'manifest_sha256':sha(manifest),'settings':SETTINGS,'classifier_hashes':EXPECTED,'owl_sha256':OWL_SHA,
  'siglip_fingerprint':fingerprint,'siglip_preprocessor_sha256':sha(MODEL/'preprocessor_config.json'),'runner_sha256':sha(__file__)}
 lock=out/'input-lock.json'
 if lock.exists():
  if read(lock)!=binding:raise ValueError('Resume source/code/settings mismatch')
 else:
  save(out/'preservation-before.json',snapshot(out));save(lock,binding)
 return binding


def draw_comparison(image,rows,path):
 width=1000;scale=width/image.width;h=round(image.height*scale);legend=24+20*len(rows)
 canvas=Image.new('RGB',(2*width,h+110+legend),'#f7f9fc');draw=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('DejaVuSans.ttf',15);small=ImageFont.truetype('DejaVuSans.ttf',13)
 except OSError:font=small=ImageFont.load_default()
 for column,arm in enumerate(('baseline','expanded')):
  offset=column*width;canvas.paste(image.resize((width,h)),(offset,76))
  draw.text((offset+12,8),f'{arm.upper()} CLASSIFIER — same OWLv2 boxes / identical crop features',font=font,fill='#112233')
  draw.text((offset+12,31),'s7: uncalibrated closed-set score; not accuracy. No boxes added or moved.',font=small,fill='#334455')
  for n,row in enumerate(rows):
   prediction=row.get(arm);color='#ff3838' if row.get('models_disagree') else '#39d353'
   x1,y1,x2,y2=row['original_box'];box=[offset+x1*scale,76+y1*scale,offset+x2*scale,76+y2*scale]
   draw.rectangle(box,outline=color,width=3)
   label=f"{row['detection_id']} {prediction['category']}" if prediction else row['detection_id']+' unavailable'
   tx=max(offset,min(box[0],offset+width-200));ty=max(76,box[1]);draw.rectangle((tx,ty,tx+190,ty+19),fill='#101827');draw.text((tx+3,ty),label,font=small,fill='white')
   if prediction:
    text=f"{row['detection_id']} OWL:{row['detector_label']} → {prediction['category']}  s7={prediction['uncalibrated_softmax7']:.3f}  cos={prediction['cosine_similarity']:.3f}"
   else:text=f"{row['detection_id']} OWL:{row['detector_label']} — {row.get('classification_status','unavailable')}"
   draw.text((offset+12,90+h+n*20),text,font=small,fill='#a00000' if row.get('models_disagree') else '#112233')
 stream=io.BytesIO();canvas.save(stream,format='PNG');atomic(path,stream.getvalue())


def html_report(out,records,summary):
 blocks=[]
 for record in records:
  name=html.escape(record['image_id']);link=record.get('comparison_visual')
  section=f'<section><h2>{name}</h2><p>Status: {html.escape(record["status"])}</p>'
  if link:section+=f'<a href="{html.escape(link)}"><img src="{html.escape(link)}" alt="Identical boxes, previous and expanded classification"></a>'
  section+='<table><tr><th>Crop</th><th>OWLv2</th><th>Previous</th><th>Expanded</th><th>Diagnostic</th></tr>'
  for row in record.get('crops',[]):
   def cell(arm):
    p=row.get(arm)
    return html.escape(f"{p['category']} | s7 {p['uncalibrated_softmax7']:.3f} | cos {p['cosine_similarity']:.3f}") if p else 'Unavailable'
   section+=f'<tr><td><img class="crop" src="{html.escape(row["crop_file"])}"><br>{row["detection_id"]}</td><td>{row["detector_label"]} {row["detector_score"]:.3f}</td><td>{cell("baseline")}</td><td>{cell("expanded")}</td><td>{"Models disagree" if row.get("models_disagree") else "Models agree"}; not ground truth</td></tr>'
  blocks.append(section+'</table></section>')
 body='<!doctype html><meta charset="utf-8"><title>Phase 4 room crop comparison</title><style>body{font:16px system-ui;margin:2rem;background:#f7f9fc}img{max-width:100%;height:auto}.crop{max-width:200px;max-height:160px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:.6rem;text-align:left}section{margin:3rem 0}</style><h1>Experimental room crop classification</h1><p>Same OWLv2 detections and identical shared crop embeddings. Production unchanged. Scores are uncalibrated. Agreement is not accuracy. Missed objects cannot be recovered by crop classification. Existing assistant-provisional annotations were not used as ground truth.</p><p>'+html.escape(json.dumps(summary))+'</p>'+''.join(blocks)
 atomic(out/'comparison.html',body.encode())


def run(out,manifest,resume):
 binding=initialize(out,manifest,resume);fingerprint=binding['siglip_fingerprint'];started=time.monotonic()
 if (out/'completion.json').exists():verify_preservation(out);print('Existing comparison complete; no inference repeated');return
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; GPU inference is required, no CPU model fallback')
 torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.75,0);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats(0)
 detector=Owlv2Detector(str(OWL),device='cuda',threshold=.15,nms_iou=.5,categories=CLASSES,allow_download=False)
 records=[];errors=[];room_images={};detector_peak=0
 for number,row in enumerate(binding['rooms']):
  token=f'{number:02d}-'+hashlib.sha256(row['image_id'].encode()).hexdigest()[:12];folder=out/token;folder.mkdir(exist_ok=True)
  image=load_image(row['file']);room_images[row['image_id']]=image
  record={**row,'folder':token,'status':'pending','crops':[],'errors':[],'box_coordinate_system':'xyxy in EXIF-oriented source pixels'}
  detpath=folder/'detector.json'
  try:
   if detpath.exists():det=read(detpath)
   else:
    predictions,seconds=gpu_retry(lambda:detector.detect(image))
    det={'detections':copy.deepcopy(predictions),'raw_candidates':detector.last_candidates,'filter_report':detector.last_filter_report,'model':detector.metadata(),'inference_seconds':seconds,
     'input_sha256':row['sha256'],'stage_code_sha256':sha(__file__)};save(detpath,det)
   if det['input_sha256']!=row['sha256'] or det['stage_code_sha256']!=sha(__file__):raise ValueError('Cached detector stage mismatch')
   record['detections']=copy.deepcopy(det['detections']);record['detector_seconds']=det['inference_seconds'];record['detection_status']='complete'
   for i,detection in enumerate(det['detections']):
    bounds=crop_bounds(detection['box'],image.width,image.height);crop=image.crop(bounds);cropfile=folder/f'crop-{i:03d}.png'
    if not cropfile.exists():
     stream=io.BytesIO();crop.save(stream,format='PNG');atomic(cropfile,stream.getvalue())
    crop_record={'detection_id':f'd{i:03d}','original_box':list(detection['box']),'detector_label':detection['label'],'detector_score':detection['score'],
     'crop_box_integer_xyxy':list(bounds),'crop_file':str(cropfile.relative_to(out)),'crop_sha256':sha(cropfile),'classification_status':'pending' if i<64 else 'budget_skipped'}
    record['crops'].append(crop_record)
   record['status']='detected'
  except Exception as exc:
   record['detection_status']='error';record['status']='error';record['errors'].append({'stage':'detection_or_crop','exception':type(exc).__name__,'message':str(exc)})
  records.append(record);print(json.dumps({'room':row['image_id'],'detections':len(record.get('detections',[])),'stage':record['status']}),flush=True)
 detector_peak=torch.cuda.max_memory_allocated(0);detector.model=None;detector.processor=None;del detector;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
 heads={arm:load_head(path,EXPECTED[arm],fingerprint) for arm,path in [('baseline',BASELINE),('expanded',SELECTED)]}
 encoder=processor=None
 for record in records:
  if record['detection_status']=='error':continue
  folder=out/record['folder'];roompath=folder/'result.json'
  if roompath.exists():
   restored=read(roompath)
   if restored['sha256']!=record['sha256']:raise ValueError('Cached room result mismatch')
   record.update(restored);continue
  try:
   pending=[r for r in record['crops'] if r['classification_status']=='pending']
   if pending:
    featurepath=folder/'crop-features.npy';metapath=folder/'crop-features.json'
    if featurepath.exists() and metapath.exists():
     meta=read(metapath);features=np.load(featurepath,allow_pickle=False)
     if sha(featurepath)!=meta['feature_sha256'] or meta['crop_sha256s']!=[r['crop_sha256'] for r in pending]:raise ValueError('Cached crop feature hash mismatch')
    else:
     if encoder is None:
      from transformers import AutoModel,AutoImageProcessor
      encoder=gpu_retry(lambda:AutoModel.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cuda').eval().requires_grad_(False))
      processor=AutoImageProcessor.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,use_fast=False)
     images=[load_image(out/r['crop_file']) for r in pending];stage_start=time.monotonic();features,proof,oom=embed_shared(images,encoder,processor,'cuda',4)
     w.save_array(featurepath,features);meta={'feature_sha256':sha(featurepath),'crop_sha256s':[r['crop_sha256'] for r in pending],'proof':proof,'oom_events':oom,
      'model_fingerprint':fingerprint,'processor':processor.to_dict(),'embedding_normalization':'L2 float32 using same NumPy normalization as training','seconds':time.monotonic()-stage_start};save(metapath,meta)
    predictions={arm:predict(features,head) for arm,head in heads.items()}
    for i,r in enumerate(pending):
     r.update(baseline=predictions['baseline'][i],expanded=predictions['expanded'][i],shared_preprocessing=meta['proof'][i],classification_status='complete')
     r['models_disagree']=r['baseline']['category']!=r['expanded']['category'];r['expanded_differs_from_detector']=r['expanded']['category']!=r['detector_label']
   record['status']='complete';record['classification_status']='complete' if pending else 'no_detections';record['metrics']=None
   visual=folder/'side-by-side.png';draw_comparison(room_images[record['image_id']],record['crops'],visual);record['comparison_visual']=str(visual.relative_to(out));save(roompath,record)
  except Exception as exc:
   record['status']='partial';record['classification_status']='error';record['errors'].append({'stage':'crop_embedding_or_classifier','exception':type(exc).__name__,'message':str(exc)})
  for error in record['errors']:errors.append({'image_id':record['image_id'],**error})
 peak=max(detector_peak,torch.cuda.max_memory_allocated(0));encoder=None;processor=None;gc.collect();torch.cuda.empty_cache()
 allcrops=[c for r in records for c in r['crops']];classified=[r for r in allcrops if r['classification_status']=='complete']
 summary={'rooms':len(records),'detections':len(allcrops),'classified_crops':len(classified),'models_disagree':sum(r['models_disagree'] for r in classified),
  'expanded_differs_from_detector':sum(r['expanded_differs_from_detector'] for r in classified),'errors':sum(len(r['errors']) for r in records),'accuracy':None,
  'ground_truth_status':'no independently verified exhaustive boxes/crop labels; no accuracy claim','seconds_this_invocation':time.monotonic()-started,'peak_allocated_gpu_bytes':peak,
  'baseline_checkpoint':str(BASELINE.relative_to(ROOT)),'expanded_checkpoint':str(SELECTED.relative_to(ROOT)),'score_note':SETTINGS['score_note'],'official_holdout_scoring':False}
 save(out/'results.json',{'summary':summary,'images':records,'classifier_hashes':EXPECTED,'class_order':CLASSES,'settings':SETTINGS})
 html_report(out,records,summary);preserved=verify_preservation(out)
 if errors or any(r['status']!='complete' for r in records):
  errorpath=out/('errors-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');save(errorpath,{'images':records,'resume':'same command with --resume; successful stages retained'})
  print('Partial inference; explicit stage errors recorded. Use --resume after diagnosis.');return
 save(out/'completion.json',{'complete':True,'summary':summary,'historical_files_verified':preserved,'new_model_not_deployed':True})
 print(json.dumps(summary,indent=2))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=MANIFEST);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=args.output.resolve()
 if not out.is_relative_to(ROOT/'reports') or not out.name.startswith('phase4-room-comparison-'):raise ValueError('Isolated versioned phase4-room-comparison report root required')
 # Lock file lives beside the root to avoid accidentally creating an output before freshness checks.
 with (out.parent/(out.name+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(out,args.manifest.resolve(),args.resume)

if __name__=='__main__':main()
