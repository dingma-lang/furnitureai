#!/usr/bin/env python3
"""Offline, provisional weak-label GPU experiments. Verified Phase 2B is never invoked."""
import argparse
import collections as C
import csv
import datetime as dt
import fcntl
import gc
import gzip
import functools
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import tempfile
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image, ImageOps
import torch

PROJECT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUTPUT=PROJECT/'data/abo/experiments/weak-training-v1'
PREP=PROJECT/'data/abo/training-prep/phase1d-v1/proposal-v2'
META=PROJECT/'data/abo/metadata/phase1-v1'
OLD=PROJECT/'data/abo/experiments/auto-training-v1'
CACHE=PROJECT/'data/abo/experiments/phase2a-v1/features-v1'
MODEL=PROJECT/'models/siglip2-base-patch16-224-phase2a-v1'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
SEED=20261104
SCOPE='PROVISIONAL WEAK-LABEL DEVELOPMENT; not human-verified accuracy; adaptive internal model selection'
PROMPTS=[f'a photograph of a complete {c}' for c in CLASSES]+[
 'a close-up photograph of fabric or wood texture', 'a furniture dimension drawing or diagram',
 'a photograph of a furniture part or accessory', 'a collage of multiple furniture products',
 'a photograph of packaging', 'a blank image']
POLICY={'seed':SEED,'max_local_images':5000,'max_new_image_features':256,'max_new_storage_bytes':1024**3,
 'min_side':96,'min_gray_std':8,'min_laplacian_variance':8,'max_white_fraction':.99,
 'extra_view_margin':.01,'strong_contradiction_margin':.06,'batch_size':4,
 'max_queue_seconds':1800,'max_epochs':80,'patience':12,'folds':3,
 'human_review_required':False,'encoder_adaptation':False,'class_prompts':PROMPTS}
CONFIGS=[
 {'name':'linear-reference','kind':'linear','lr':.003,'wd':.001},
 {'name':'linear-lr1e2','kind':'linear','lr':.01,'wd':.001},
 {'name':'linear-smooth05','kind':'linear','lr':.003,'wd':.01,'smoothing':.05},
 {'name':'mlp64-d20','kind':'mlp','hidden':64,'dropout':.2,'lr':.001,'wd':.001},
 {'name':'mlp64-d40','kind':'mlp','hidden':64,'dropout':.4,'lr':.003,'wd':.01},
 {'name':'mlp128-d30','kind':'mlp','hidden':128,'dropout':.3,'lr':.001,'wd':.01},
 {'name':'cosine-head','kind':'cosine','lr':.003,'wd':.001},
 {'name':'nearest-centroid','kind':'centroid'}]
STOP=False


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def fingerprint(value):return hashlib.sha256(encoded(value)).hexdigest()
def atomic(path,body):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def save(path,value):atomic(path,encoded(value))
def save_tensor(path,value):
 stream=io.BytesIO();torch.save(value,stream);atomic(path,stream.getvalue())
def save_array(path,value):
 stream=io.BytesIO();np.save(stream,np.asarray(value,dtype=np.float32),allow_pickle=False);atomic(path,stream.getvalue())
def disk_guard(out):
 used=sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file())
 if used+16*1024**2>POLICY['max_new_storage_bytes']:raise RuntimeError('New experiment storage cap reached')
 return used

@functools.lru_cache(maxsize=1)
def helper():
 spec=importlib.util.spec_from_file_location('immutable_old_helpers',PROJECT/'tools/auto-training-v1/auto_train.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Groups:
 def __init__(self):self.parent={}
 def find(self,x):
  self.parent.setdefault(x,x)
  if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
  return self.parent[x]
 def union(self,items):
  items=list(items)
  if not items:return
  roots=[self.find(x) for x in items]; root=min(roots)
  for r in roots:self.parent[r]=root

def group_inventory(rows,identity):
 groups=Groups()
 # Group for leakage prevention; this does NOT merge physical product identities.
 for r in rows:groups.find(r['abo_product_id'])
 for group in identity['same_brand_model_candidates_not_verified_duplicates']:
  groups.union(group['product_ids'])
 for group in identity.get('duplicate_listing_pairs',[]):
  ids=group.get('product_ids',[]) if isinstance(group,dict) else group
  groups.union(ids)
 seen={}
 for r in rows:
  for field in ('group_id','family_group_id','human_audited_family_group_id','capture_cluster_id'):
   value=r.get(field)
   if value:
    key=(field,value)
    if key in seen:groups.union([seen[key],r['abo_product_id']])
    else:seen[key]=r['abo_product_id']
 return groups

def check_scope(rows,held,groups):
 reserved={groups.find(r['abo_product_id']) for r in held}
 for r in rows:
  if r.get('proposed_split') in ('validation','test') or groups.find(r['abo_product_id']) in reserved:raise ValueError('Reserved holdout identity/family leakage')
 for field in ('image_id','sha256','decoded_pixel_sha256','capture_cluster_id'):
  tokens={r[field] for r in held if r.get(field)}
  if any(r.get(field) in tokens for r in rows if r.get(field)):raise ValueError('Holdout image/capture leakage: '+field)

def image_path(row):
 rel=Path(row['local_path']);path=(PROJECT/rel).resolve();allowed=(PROJECT/'data/abo/image-pilots').resolve()
 if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(allowed) or sha(path)!=row['sha256']:raise ValueError('Unsafe/mismatched source image')
 return path

def quality(row):
 path=image_path(row)
 with Image.open(path) as image:image.verify()
 with Image.open(path) as image:
  im=ImageOps.exif_transpose(image).convert('RGB');array=np.asarray(im);gray=np.asarray(im.convert('L'),dtype=np.float32)
 lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
 measures={'width':im.width,'height':im.height,'gray_std':float(gray.std()),'laplacian_variance':float(lap.var()),'white_fraction':float(np.mean(np.all(array>245,axis=2)))}
 reasons=[]
 for bad,label in [(min(im.size)<POLICY['min_side'],'too_small'),(measures['gray_std']<POLICY['min_gray_std'],'blank_or_low_variation'),(measures['laplacian_variance']<POLICY['min_laplacian_variance'],'blur_risk'),(measures['white_fraction']>POLICY['max_white_fraction'],'nearly_blank')]:
  if bad:reasons.append(label)
 return measures,reasons

def preservation(out):
 paths=[]
 for root in (PROJECT/'data/abo',PROJECT/'models',PROJECT/'reports'):
  paths += [p for p in root.rglob('*') if p.is_file() and not p.is_relative_to(out) and 'evidence-completion-v1' not in p.parts]
 # Review application state is independently mutable; freeze evaluator/rules/review CSVs instead.
 return {str(p.relative_to(PROJECT)):sha(p) for p in paths}

def prepare(out, source_inventory=PREP/'eligibility-inventory.json'):
 target=out/'manifests-v1'
 if target.exists():raise FileExistsError('Manifest already frozen; use existing run or a fresh output root')
 source_inventory=Path(source_inventory).resolve()
 if not source_inventory.is_relative_to(PROJECT/'data/abo'):raise ValueError('Inventory must be a versioned local ABO artifact')
 inventory=read(source_inventory);allrows=inventory['images'];identity=read(META/'identity-review.json')
 original_reservation=read(PREP/'eligibility-inventory.json')['images']
 if len({r['image_id'] for r in allrows})!=len(allrows):raise ValueError('Duplicate inventory image IDs')
 requested={r['image_id'] for r in allrows}; official={}
 with gzip.open(META/'images/metadata/images.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   if r['image_id'] in requested:official[r['image_id']]=r['path']
 products=read(META/'furniture-metadata-audit.json')['products'];metadata={p['product_id']:p for p in products}
 groups=group_inventory(allrows+original_reservation,identity)
 held=[r for r in allrows+original_reservation if r.get('proposed_split') in ('validation','test')]
 reserved={groups.find(r['abo_product_id']) for r in held}
 shared={r['image_id'] for r in identity['shared_image_ids']}
 original=read(CACHE/'train/metadata.json');original_ids={r['image_id'] for r in original['rows']}
 accepted=[];excluded=[]
 for row in sorted(allrows,key=lambda r:r['image_id']):
  reason=[];meta=metadata.get(row['abo_product_id'])
  if row.get('proposed_split')!='train':reason.append('outside_existing_training_partition')
  if groups.find(row['abo_product_id']) in reserved:reason.append('reserved_holdout_identity_or_family')
  if row['image_id'] in shared:reason.append('shared_asset_identity_uncertain')
  expected='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/small/'+official.get(row['image_id'],'')
  if row['image_id'] not in official or row.get('source_url')!=expected:reason.append('unverified_official_image_provenance')
  if row['eligibility_status']=='rejected' or row.get('phase1d_block_reason') or row.get('rejection_reasons'):reason.append('prior_hard_exclusion')
  if not meta or not meta['metadata_screen_pass'] or meta['category']!=row['furniture_class']:reason.append('metadata_category_ambiguous')
  flags=row.get('review_reasons',[])
  if any('texture' in f or 'blur' in f or 'lifestyle' in f for f in flags):reason.append('unsuitable_or_mixed_scene_risk')
  measures=None
  if not reason:
   measures,q=quality(row);reason+=q
  if reason:excluded.append({'image_id':row['image_id'],'abo_product_id':row['abo_product_id'],'reasons':reason});continue
  accepted.append({**row,'leakage_group_id':groups.find(row['abo_product_id']),
   'weak_label':row['furniture_class'],'label_source':'official ABO product type/category/title, original metadata screen',
   'label_provenance':{'metadata_file':'data/abo/metadata/phase1-v1/furniture-metadata-audit.json','product_id':row['abo_product_id'],'raw_types':meta['raw_types'],'title':meta['name']},
   'quality_screen':measures,'original_cached_training_view':row['image_id'] in original_ids,
   'whole_object_status':'unverified automated screening only','capture_independence':'unverified; known capture clusters kept together',
   'human_verified':False})
 if len(accepted)>POLICY['max_local_images']:raise ValueError('Local scope cap exceeded')
 check_scope(accepted,held,groups)
 if sum(not r['original_cached_training_view'] for r in accepted)>POLICY['max_new_image_features']:raise ValueError('New embedding budget exceeded')
 target.mkdir()
 sources=[source_inventory,PREP/'eligibility-inventory.json',META/'images/metadata/images.csv.gz',META/'furniture-metadata-audit.json',META/'identity-review.json',CACHE/'train/metadata.json',CACHE/'train/features.npy',CACHE/'summary.json',MODEL/'checkpoint-manifest.json',PROJECT/'tools/auto-training-v1/auto_train.py',OLD/'leaderboard.csv',OLD/'final-train-only-head.pt',PROJECT/'data/abo/experiments/phase2a-v1/classifier-v1/head.pt']
 lock={str(p.relative_to(PROJECT)):sha(p) for p in sources}
 for name in ('pipeline.py',):lock[str((HERE/name).relative_to(PROJECT))]=sha(HERE/name)
 save(out/'preservation-before.json',preservation(out));save(target/'input-lock.json',lock)
 save(target/'policy.json',POLICY);save(target/'candidates.json',{'scope':SCOPE,'rows':accepted,'excluded':excluded})
 eligible_metadata=[p for p in products if p['metadata_screen_pass'] and p['qualified_link_count']>=2]
 local_products={r['abo_product_id'] for r in allrows}
 future=[{**p,'leakage_group_id':groups.find(p['product_id']),'status':'metadata_only_not_acquired_not_training','human_verified':False} for p in eligible_metadata if p['product_id'] not in local_products and groups.find(p['product_id']) not in reserved]
 save(target/'metadata-only-expansion.json',{'products':future,'source_license':'ABO CC BY 4.0; preserve attribution; no images acquired','by_class':dict(C.Counter(p['category'] for p in future))})
 save(target/'preparation-report.json',{'scope':SCOPE,'candidate_images':len(accepted),'candidate_products':len({r['abo_product_id'] for r in accepted}),'cached_views':sum(r['original_cached_training_view'] for r in accepted),'extra_views_to_embed':sum(not r['original_cached_training_view'] for r in accepted),'candidate_product_classes':dict(C.Counter({r['abo_product_id']:r['furniture_class'] for r in accepted}.values())),'screened_metadata_products':len(eligible_metadata),'metadata_only_nonreserved_candidates':len(future),'excluded_reasons':dict(C.Counter(x for r in excluded for x in r['reasons'])),'holdout_feature_arrays_or_decisions_read':False,'known_identity_grouping_is_not_physical_identity_merging':True})
 save(target/'frozen-artifacts.json',{p.name:sha(p) for p in target.iterdir() if p.is_file()})
 print(json.dumps(read(target/'preparation-report.json'),indent=2),flush=True)

def validate_inputs(out):
 for name,h in read(out/'manifests-v1/frozen-artifacts.json').items():
  if sha(out/'manifests-v1'/name)!=h:raise ValueError('Frozen manifest changed: '+name)
 for rel,h in read(out/'manifests-v1/input-lock.json').items():
  if sha(PROJECT/rel)!=h:raise ValueError('Pinned source changed: '+rel)
 if read(out/'manifests-v1/policy.json')!=POLICY:raise ValueError('Policy changed')
 disk_guard(out)

def gpu():
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
 torch.cuda.init();torch.cuda.set_per_process_memory_fraction(.75,0)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.cuda.reset_peak_memory_stats(0)
 return torch.device('cuda:0')

def unit(array):
 array=np.asarray(array,dtype=np.float32)
 if array.ndim!=2 or not np.isfinite(array).all():raise ValueError('Bad feature matrix')
 norms=np.linalg.norm(array,axis=1,keepdims=True)
 if np.any(norms<1e-8):raise ValueError('Zero features')
 return array/norms

def model_screen(row,similarities):
 label=CLASSES.index(row['weak_label']);order=np.argsort(-similarities);margin=float(similarities[order[0]]-similarities[order[1]])
 top=int(order[0]);match=top==label
 # Cached weak labels are retained unless a strong contradiction is found.
 # Previously pending extra views require agreement; this never upgrades human-review status.
 admit=(match or margin<POLICY['strong_contradiction_margin']) if row['original_cached_training_view'] else (match and margin>=POLICY['extra_view_margin'])
 return admit,{'top_prompt':PROMPTS[top],'metadata_label_agreement':match,'top_margin_cosine':margin,
  'metadata_label_cosine':float(similarities[label]),'confidence':'higher_weak_agreement' if match else 'uncertain_weak_label',
  'calibrated_probability':None,'confidence_is_not_accuracy':True}

def extract_screen(out):
 validate_inputs(out);dest=out/'features-v1'
 if (dest/'metadata.json').exists():return load_data(out)
 device=gpu();candidates=read(out/'manifests-v1/candidates.json')['rows']
 meta=read(CACHE/'train/metadata.json');summary=read(CACHE/'summary.json')
 if sha(CACHE/'train/metadata.json')!=summary['partitions']['train']['metadata_sha256'] or sha(CACHE/'train/features.npy')!=meta['feature_sha256']:raise ValueError('Cached training feature checksum mismatch')
 original=np.load(CACHE/'train/features.npy',allow_pickle=False)
 if original.shape!=(len(meta['rows']),meta['dimension']) or not np.allclose(np.linalg.norm(original,axis=1),1,atol=1e-4):raise ValueError('Invalid cached training vectors')
 lookup={r['image_id']:original[i] for i,r in enumerate(meta['rows'])}
 manifest=read(MODEL/'checkpoint-manifest.json')
 if meta['model_processor_fingerprint']!=manifest['model_processor_fingerprint']:raise ValueError('Embedding fingerprint mismatch')
 for name,h in manifest['files'].items():
  if sha(MODEL/name)!=h:raise ValueError('Frozen SigLIP checkpoint changed')
 from transformers import AutoModel,AutoImageProcessor,AutoTokenizer
 model=AutoModel.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval().requires_grad_(False)
 processor=AutoImageProcessor.from_pretrained(str(MODEL),local_files_only=True,use_fast=False)
 tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 dest.mkdir(exist_ok=True);shards=dest/'new-image-shards';shards.mkdir(exist_ok=True)
 textpath=dest/'text-prompts.npy'
 if textpath.exists():text=np.load(textpath,allow_pickle=False)
 else:
  tokens=tokenizer(PROMPTS,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
  with torch.inference_mode():text=unit(model.get_text_features(**{k:v.to(device) for k,v in tokens.items()}).float().cpu().numpy())
  save_array(textpath,text)
 for i,row in enumerate(candidates):
  if STOP:raise InterruptedError('Graceful stop during feature preparation; completed shards preserved')
  if row['image_id'] in lookup:continue
  path=shards/(hashlib.sha256(row['image_id'].encode()).hexdigest()+'.npy')
  if path.exists():value=np.load(path,allow_pickle=False)
  else:
   with Image.open(image_path(row)) as image:im=ImageOps.exif_transpose(image).convert('RGB')
   with torch.inference_mode():
    inputs=processor(images=im,return_tensors='pt');value=unit(model.get_image_features(**{k:v.to(device) for k,v in inputs.items()}).float().cpu().numpy())
   save_array(path,value)
  lookup[row['image_id']]=unit(value)[0]
 del model,processor,tokenizer;gc.collect();torch.cuda.empty_cache()
 rows=[];vectors=[];rejected=[]
 for row in candidates:
  vector=lookup[row['image_id']];admit,confidence=model_screen(row,vector@text.T)
  item={**row,'automated_model_screen':confidence,'embedding_model_fingerprint':meta['model_processor_fingerprint']}
  if admit:rows.append(item);vectors.append(vector)
  else:rejected.append({'image_id':row['image_id'],'reason':'model_label_or_negative_prompt_contradiction','screen':confidence})
 # Exact/known repeated captures are weighted once in each training epoch, not claimed independent.
 seen={};keep=[];values=[]
 for row,vector in zip(rows,vectors):
  token=(row['abo_product_id'],row.get('capture_cluster_id') or row.get('decoded_pixel_sha256') or row['sha256'])
  if token in seen:rejected.append({'image_id':row['image_id'],'reason':'redundant_known_capture','kept':seen[token]});continue
  seen[token]=row['image_id'];keep.append(row);values.append(vector)
 array=unit(values);folds=make_folds(keep)
 save_array(dest/'features.npy',array)
 save(dest/'metadata.json',{'scope':SCOPE,'rows':keep,'folds':folds,'feature_sha256':sha(dest/'features.npy'),'model_processor_fingerprint':meta['model_processor_fingerprint'],'policy_sha256':sha(out/'manifests-v1/policy.json'),'rejected':rejected,'new_image_features':len(list(shards.glob('*.npy'))),'cached_feature_vectors_reused':sum(r['original_cached_training_view'] for r in keep),'no_holdout_features_or_reviews_used':True})
 return array,keep,folds

def make_folds(rows):
 grouped=C.defaultdict(list)
 for i,r in enumerate(rows):grouped[r['leakage_group_id']].append(i)
 by_class=C.defaultdict(list)
 for group,indices in grouped.items():
  cats={rows[i]['weak_label'] for i in indices}
  if len(cats)!=1:raise ValueError('Conflicting labels in identity/family group')
  by_class[next(iter(cats))].append((group,indices))
 if set(by_class)!=set(CLASSES) or any(len(by_class[c])<POLICY['folds'] for c in CLASSES):raise ValueError('Insufficient groups for three all-category folds; do not weaken eligibility')
 folds=[[] for _ in range(POLICY['folds'])]
 for cls in CLASSES:
  for j,(group,indices) in enumerate(sorted(by_class[cls],key=lambda x:fingerprint([SEED,cls,x[0]]))):folds[j%len(folds)].extend(indices)
 for fold in folds:
  train=[i for i in range(len(rows)) if i not in set(fold)]
  for key in ('abo_product_id','leakage_group_id','sha256','decoded_pixel_sha256','capture_cluster_id'):
   a={rows[i][key] for i in train if rows[i].get(key)};b={rows[i][key] for i in fold if rows[i].get(key)}
   if a&b:raise ValueError('Internal leakage: '+key)
 return [sorted(f) for f in folds]

def load_data(out):
 validate_inputs(out);meta=read(out/'features-v1/metadata.json');path=out/'features-v1/features.npy'
 if sha(path)!=meta['feature_sha256']:raise ValueError('New feature cache checksum mismatch')
 array=np.load(path,allow_pickle=False);rows=meta['rows']
 if array.shape!=(len(rows),768) or not np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4):raise ValueError('Invalid new features')
 if make_folds(rows)!=meta['folds']:raise ValueError('Fold assignments changed')
 return array,rows,meta['folds']

class Cosine(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.weight=torch.nn.Parameter(torch.randn(len(CLASSES),dim)*.02)
 def forward(self,x):return 10*torch.nn.functional.linear(torch.nn.functional.normalize(x,dim=-1),torch.nn.functional.normalize(self.weight,dim=-1))

def network(cfg,dimension):
 if cfg['kind']=='cosine':return Cosine(dimension)
 return helper().network(cfg,dimension)

def metric(rows,indices,logits):
 mapped=[{**r,'furniture_class':r['weak_label']} for r in rows]
 return helper().score_logits(mapped,indices,logits)

def centroid(array,rows,indices):
 products=C.defaultdict(list)
 for i in indices:products[rows[i]['abo_product_id']].append(i)
 centers=[]
 for cls in CLASSES:
  means=[unit(array[pos].mean(axis=0,keepdims=True))[0] for pos in products.values() if rows[pos[0]]['weak_label']==cls]
  if not means:raise ValueError('Missing class support')
  centers.append(unit(np.mean(means,axis=0,keepdims=True))[0])
 return np.asarray(centers,dtype=np.float32)

def stopped(out,deadline):return STOP or (out/'STOP').exists() or time.monotonic()>deadline

def fit(out,array,rows,folds,cfg,fold,epochs,deadline,final=False,warm=False):
 device=gpu();directory=out/('final-warm-start' if warm else 'final-selected') if final else out/'trials'/cfg['name']/f'fold-{fold}'
 directory.mkdir(parents=True,exist_ok=True)
 signature=fingerprint({'cache':sha(out/'features-v1/metadata.json'),'cfg':cfg,'fold':fold,'epochs':epochs,'final':final,'warm':warm,'code':sha(__file__)})
 resultpath=directory/'result.json'
 if resultpath.exists():
  result=read(resultpath)
  if result['signature']!=signature:raise ValueError('Completed result signature mismatch')
  if sha(directory/'best.pt')!=result['checkpoint_sha256']:raise ValueError('Completed checkpoint changed')
  return result
 dev=[] if final else folds[fold];train=[i for i in range(len(rows)) if i not in set(dev)]
 torch.manual_seed(SEED+fold);torch.cuda.manual_seed_all(SEED+fold)
 if cfg['kind']=='centroid':
  centers=centroid(array,rows,train);metrics=metric(rows,dev,array[dev]@centers.T) if dev else None
  save_tensor(directory/'best.pt',{'centers':torch.from_numpy(centers),'config':cfg,'classes':CLASSES,'signature':signature})
  result={'signature':signature,'metrics':metrics,'best_epoch':0,'config':cfg,'scope':SCOPE,'checkpoint_sha256':sha(directory/'best.pt')};save(resultpath,result);return result
 model=network(cfg,array.shape[1]).to(device)
 if warm:
  checkpoint=torch.load(OLD/'final-train-only-head.pt',map_location='cpu',weights_only=True)
  if cfg['kind']!='linear' or checkpoint['classes']!=CLASSES:raise ValueError('Warm start architecture/classes mismatch')
  if checkpoint['model_processor_fingerprint']!=read(out/'features-v1/metadata.json')['model_processor_fingerprint']:raise ValueError('Warm-start feature mismatch')
  model.load_state_dict(checkpoint['state_dict'])
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
 x=torch.from_numpy(array).to(device);y=torch.tensor([CLASSES.index(r['weak_label']) for r in rows],device=device)
 byclass=C.defaultdict(set)
 for i in train:byclass[rows[i]['weak_label']].add(rows[i]['abo_product_id'])
 weights=np.asarray([1/math.sqrt(len(byclass[c])) for c in CLASSES]);weights/=weights.mean()
 lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=cfg.get('smoothing',0))
 checkpoint_path=directory/'progress.pt';start=0;curve=[];bestscore=-1;bestepoch=0;beststate=None;stale=0
 if checkpoint_path.exists():
  saved=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
  if saved['signature']!=signature:raise ValueError('Resume signature mismatch')
  model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
  start=saved['epoch'];curve=saved['curve'];bestscore=saved['bestscore'];bestepoch=saved['bestepoch'];beststate=saved['beststate'];stale=saved['stale']
 for epoch in range(start,epochs):
  if stopped(out,deadline):raise InterruptedError('Queue stopped at committed epoch; resume without deleting progress')
  if not final and stale>=POLICY['patience']:break
  torch.manual_seed(SEED+epoch*7919+fold);torch.cuda.manual_seed_all(SEED+epoch*7919+fold)
  chosen=helper().epoch_sample(rows,train,SEED+epoch*7919+fold);model.train();losses=[]
  for pos in range(0,len(chosen),32):
   ids=chosen[pos:pos+32];optimizer.zero_grad(set_to_none=True);loss=lossfn(model(x[ids]),y[ids])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward()
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
   optimizer.step();losses.append(float(loss.detach()))
  model.eval();measure=None
  if dev:
   with torch.inference_mode():logits=model(x[dev]).cpu().numpy()
   measure=metric(rows,dev,logits);score=measure['product']['macro_f1']
  else:score=epoch+1
  if score>bestscore:
   bestscore=score;bestepoch=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
  else:stale+=1
  curve.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'internal_dev':measure})
  save_tensor(checkpoint_path,{'signature':signature,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'epoch':epoch+1,'curve':curve,'bestscore':bestscore,'bestepoch':bestepoch,'beststate':beststate,'stale':stale})
  if (epoch+1)%10==0:print(json.dumps({'trial':cfg['name'],'fold':fold,'epoch':epoch+1,'best_product_macro_f1':bestscore if dev else None}),flush=True)
 if beststate is None:raise ValueError('No committed model state')
 model.load_state_dict(beststate)
 with torch.inference_mode():metrics=metric(rows,dev,model(x[dev]).cpu().numpy()) if dev else None
 save_tensor(directory/'best.pt',{'state_dict':beststate,'config':cfg,'classes':CLASSES,'dimension':array.shape[1],'signature':signature,'model_processor_fingerprint':read(out/'features-v1/metadata.json')['model_processor_fingerprint'],'best_epoch':bestepoch,'weak_labels_only':True})
 result={'signature':signature,'scope':SCOPE,'metrics':metrics,'best_epoch':bestepoch,'epochs_run':len(curve),'config':cfg,'checkpoint_sha256':sha(directory/'best.pt'),'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'warm_started_from':str(OLD/'final-train-only-head.pt') if warm else None,'dev_metric_for_final_checkpoint':None}
 save(directory/'curve.json',curve);save(resultpath,result);return result

def refresh(out):
 leaderboard=[];registry=[]
 for cfg in CONFIGS:
  paths=[out/'trials'/cfg['name']/f'fold-{i}'/'result.json' for i in range(POLICY['folds'])]
  if all(p.exists() for p in paths):
   results=[read(p) for p in paths];scores=[r['metrics']['product']['macro_f1'] for r in results]
   leaderboard.append({'experiment':cfg['name'],'product_macro_f1':statistics.mean(scores),'fold_std':statistics.stdev(scores),'product_balanced_accuracy':statistics.mean(r['metrics']['product']['balanced_accuracy'] for r in results),'folds':len(results),'scope':SCOPE})
 for path in out.rglob('best.pt'):
  registry.append({'path':str(path.relative_to(out)),'sha256':sha(path),'scope':SCOPE,'verified_holdout_evaluation':False})
 leaderboard.sort(key=lambda r:(-r['product_macro_f1'],r['experiment']))
 save(out/'leaderboard.json',leaderboard)
 stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=['experiment','product_macro_f1','fold_std','product_balanced_accuracy','folds','scope']);writer.writeheader();writer.writerows(leaderboard);atomic(out/'leaderboard.csv',stream.getvalue().encode())
 save(out/'checkpoint-registry.json',{'current':registry,'immutable_prior_checkpoints':[{'path':str(p.relative_to(PROJECT)),'sha256':sha(p),'used_for_internal_dev_initialization':False} for p in [OLD/'final-train-only-head.pt',PROJECT/'data/abo/experiments/phase2a-v1/classifier-v1/head.pt']]})
 return leaderboard

def smoke(out):
 array,rows,folds=extract_screen(out);deadline=time.monotonic()+300
 # Distinct immutable smoke config/directory; it cannot be mistaken for a full fold.
 cfg={**CONFIGS[0],'name':'smoke-three-epochs'}
 result=fit(out,array,rows,folds,cfg,0,3,deadline)
 save(out/'smoke-gate.json',{'passed':True,'code_sha256':sha(__file__),'cache_metadata_sha256':sha(out/'features-v1/metadata.json'),'result':result})
 print('GPU smoke passed',flush=True)

def batch(out):
 validate_inputs(out);gate=read(out/'smoke-gate.json')
 if not gate['passed'] or gate['code_sha256']!=sha(__file__) or gate['cache_metadata_sha256']!=sha(out/'features-v1/metadata.json'):raise ValueError('Smoke gate stale')
 array,rows,folds=load_data(out);deadline=time.monotonic()+POLICY['max_queue_seconds']
 save(out/'queue-config.json',{'configs':CONFIGS,'policy':POLICY,'scope':SCOPE,'runtime_budget_is_per_invocation':True})
 for cfg in CONFIGS:
  for fold in range(len(folds)):
   fit(out,array,rows,folds,cfg,fold,POLICY['max_epochs'],deadline);refresh(out);disk_guard(out)
 leaderboard=refresh(out);winner=next(c for c in CONFIGS if c['name']==leaderboard[0]['experiment'])
 results=[read(out/'trials'/winner['name']/f'fold-{i}'/'result.json') for i in range(len(folds))]
 epochs=max(1,int(statistics.median(r['best_epoch'] for r in results)))
 fit(out,array,rows,folds,winner,0,epochs,deadline,final=True)
 # A distinct training-only continuation reuses the prior trained classifier; it is NOT assigned CV scores.
 fit(out,array,rows,folds,CONFIGS[0],0,20,deadline,final=True,warm=True)
 refresh(out)
 save(out/'completion.json',{'status':'complete','winner':winner,'selection':leaderboard[0],'final_epochs':epochs,'warm_start_companion_has_no_dev_score':True,'scope':SCOPE,'holdout_metrics_computed':False,'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gpu':torch.cuda.get_device_name(0),'allocated_storage_bytes':disk_guard(out)})
 print('Bounded queue complete',flush=True)

def verify_preservation(out):
 snapshot=read(out/'preservation-before.json');bad=[p for p,h in snapshot.items() if sha(PROJECT/p)!=h]
 save(out/'preservation-after.json',{'checked':len(snapshot),'mismatches':bad})
 if bad:raise ValueError('Historical artifact changed: '+str(bad[:5]))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','smoke','run','status','verify']);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--source-inventory',type=Path,default=PREP/'eligibility-inventory.json')
 args=parser.parse_args();out=args.output.resolve()
 allowed=(PROJECT/'data/abo/experiments').resolve()
 if not out.is_relative_to(allowed) or out==allowed or out.is_symlink():raise ValueError('Output must be an isolated experiment child')
 if args.action=='status':
  print(json.dumps({'complete':read(out/'completion.json') if (out/'completion.json').exists() else False,'leaderboard':read(out/'leaderboard.json') if (out/'leaderboard.json').exists() else []},indent=2));return
 out.mkdir(exist_ok=True)
 with (out/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def request_stop(*_):
   global STOP;STOP=True
  signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
  try:
   if args.action=='prepare':prepare(out,args.source_inventory)
   elif args.action=='smoke':smoke(out)
   elif args.action=='run':batch(out)
   elif args.action=='verify':verify_preservation(out)
  except InterruptedError as exc:
   save(out/'status.json',{'status':'paused','reason':str(exc),'resume':'run again with the same code and output; remove your STOP file first'});print(str(exc),flush=True);return
  if (out/'preservation-before.json').exists():verify_preservation(out)

if __name__=='__main__':main()
