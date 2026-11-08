"""Bounded GPU heads; training-only model selection; frozen panel is comparison-only."""
import argparse
import collections as C
import csv
import datetime as dt
import fcntl
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
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase3-v1'))
from common import w,sha,read,save,atomic,fingerprint,PREP,META,WEAK,OUT as PREPARATION,CLASSES
import numpy as np
from PIL import Image,ImageOps
import torch

OUT=ROOT/'data/abo/experiments/phase3-gpu-v1'
IMAGES=ROOT/'data/abo/image-pilots/phase3-expansion-v1'
SEED=20261105
SCOPE='Provisional weak-label internal training development; not human-verified accuracy'
CONFIGS=[
 {'name':'cosine-reference','kind':'cosine','lr':.003,'wd':.001,'balance':'sqrt'},
 {'name':'cosine-lr1e3','kind':'cosine','lr':.001,'wd':.01,'balance':'sqrt'},
 {'name':'cosine-unweighted','kind':'cosine','lr':.003,'wd':.001,'balance':'none'},
 {'name':'linear-reference','kind':'linear','lr':.003,'wd':.001,'balance':'sqrt'},
 {'name':'linear-inverse','kind':'linear','lr':.01,'wd':.01,'balance':'inverse'},
 {'name':'mlp64-d20','kind':'mlp','hidden':64,'dropout':.2,'lr':.001,'wd':.001,'balance':'sqrt'},
 {'name':'mlp64-d40','kind':'mlp','hidden':64,'dropout':.4,'lr':.003,'wd':.01,'balance':'none'},
 {'name':'mlp128-d30','kind':'mlp','hidden':128,'dropout':.3,'lr':.001,'wd':.01,'balance':'sqrt'}]
POLICY={'seed':SEED,'expanded_internal_folds':3,'baseline_internal_folds':2,'epochs':80,'patience':12,
 'batch_size':32,'oom_min_batch':1,'cuda_fraction':.75,'max_gpu_fits':30,'max_epochs_total':2323,
 'queue_wall_seconds_per_invocation':1800,'storage_bytes':2*1024**3,'baseline_recipe':CONFIGS[0],
 'frozen_panel_for_selection':False,'frozen_panel_comparisons':1,'no_official_holdout_scoring':True,
 'all_heads_from_scratch':True,'encoder_adaptation':False,'feature_validation_atol':1e-5}
STOP=False


def stable_stat(path):
 s=Path(path).stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'inode':s.st_ino}

def preservation(out):
 inventory=read(PREP/'eligibility-inventory.json')['images']
 unopened={str((ROOT/r['local_path']).resolve()) for r in inventory if r.get('proposed_split') in ('validation','test')}
 unopened|={str(p.resolve()) for part in ('validation','test') for p in (ROOT/f'data/abo/experiments/phase2a-v1/features-v1/{part}').glob('*.npy')}
 unopened.add(str((PREPARATION/'frozen-development-features.npy').resolve()))
 result={}
 for base in (ROOT/'data/abo',ROOT/'models',ROOT/'reports'):
  for path in base.rglob('*'):
   if not path.is_file() or path.is_relative_to(out) or 'evidence-completion-v1' in path.parts:continue
   rel=str(path.relative_to(ROOT))
   result[rel]={'mode':'stat_only_never_opened','stat':stable_stat(path)} if str(path.resolve()) in unopened else {'mode':'sha256','sha256':sha(path)}
 return result

def preserve_check(out):
 snapshot=read(out/'preservation-before.json');bad=[];counts=C.Counter()
 for rel,expected in snapshot.items():
  path=ROOT/rel;counts[expected['mode']]+=1
  if not path.is_file() or (stable_stat(path)!=expected['stat'] if expected['mode']=='stat_only_never_opened' else sha(path)!=expected['sha256']):bad.append(rel)
 save(out/'preservation-after.json',{'files_checked':len(snapshot),'methods':dict(counts),'mismatches':bad,'official_holdout_images_never_opened':True})
 if bad:raise ValueError('Historical files changed: '+str(bad[:3]))
 return len(snapshot)


def canonical(path):
 with Image.open(path) as image:
  rgb=ImageOps.exif_transpose(image).convert('RGB');return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def guard_allowlist(rows,audit,risks):
 expected={r['image_id']:r for r in audit if r['status']=='provisional_accepted'}
 if len(expected)!=350 or len({r['abo_product_id'] for r in rows})!=196:raise ValueError('Expected final 350/196 allowlist only')
 if len(rows)!=len(expected) or len({r['image_id'] for r in rows})!=len(rows) or {r['image_id'] for r in rows}!=set(expected):raise ValueError('Allowlist contamination or duplicate IDs')
 for r in rows:
  if r!=expected[r['image_id']] or r['status']!='provisional_accepted' or r['product_identity_risk'] or r['product_id'] in risks:raise ValueError('Rejected/quarantined/unresolved image or product admitted')


def isolate(rows,reserved):
 for key in ('image_id','abo_product_id','leakage_group_id','sha256','decoded_pixel_sha256','capture_cluster_id'):
  known={r[key] for r in reserved if r.get(key)}
  if any(r.get(key) in known for r in rows if r.get(key)):raise ValueError('Reserved identity/image overlap: '+key)
 if any(r.get('proposed_split') in ('validation','test') for r in rows):raise ValueError('Official partition in training')
 labels=C.defaultdict(set)
 for r in rows:
  if r['weak_label'] not in CLASSES:raise ValueError('Out-of-taxonomy label')
  labels[r['abo_product_id']].add(r['weak_label'])
 if any(len(x)>1 for x in labels.values()):raise ValueError('Contradictory labels within product')


def group_folds(rows,n):
 # Unite known families and exact image/capture tokens solely as leakage constraints.
 parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 seen={}
 for i,r in enumerate(rows):
  for key in ('abo_product_id','leakage_group_id','sha256','canonical_pixels','capture_cluster_id'):
   token=r.get(key)
   if token:
    entry=(key,token)
    if entry in seen:
     a,b=find(i),find(seen[entry]);parent[max(a,b)]=min(a,b)
    else:seen[entry]=i
 groups=C.defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 byclass=C.defaultdict(list)
 for indices in groups.values():
  cats={rows[i]['weak_label'] for i in indices}
  if len(cats)!=1:raise ValueError('Conflicting known identity/capture group labels')
  byclass[next(iter(cats))].append(indices)
 if set(byclass)!=set(CLASSES) or min(map(len,byclass.values()))<n:raise ValueError('Insufficient known groups for all-category internal folds')
 folds=[[] for _ in range(n)]
 for cls in CLASSES:
  ordered=sorted(byclass[cls],key=lambda ids:fingerprint([SEED,cls,sorted(rows[i]['image_id'] for i in ids)]))
  for j,indices in enumerate(ordered):folds[j%n].extend(indices)
 for fold in folds:
  train=set(range(len(rows)))-set(fold)
  for indices in groups.values():
   if set(indices)&set(fold) and set(indices)&train:raise ValueError('Internal group leakage')
 return [sorted(f) for f in folds],len(groups)


def readiness(out):
 if (out/'input-lock.json').exists():validate(out);return
 if (out/'manifests').exists():raise FileExistsError('Partial readiness exists; inspect, do not replace')
 (out/'manifests').mkdir()
 protocol=read(PREPARATION/'frozen-development-protocol.json');base_meta=read(ROOT/'data/abo/experiments/phase3-baseline-v1/features-v1/metadata.json')
 source=ROOT/'data/abo/experiments/phase3-baseline-v1/features-v1/features.npy'
 if sha(source)!=base_meta['feature_sha256'] or base_meta['fit_count']!=182:raise ValueError('Baseline cache mismatch')
 baseline=base_meta['rows'][:182]
 if len({r['abo_product_id'] for r in baseline})!=85:raise ValueError('Baseline product count mismatch')
 audit=read(IMAGES/'audit-v3/screening.json')['images'];new=read(IMAGES/'audit-v3/provisional-candidates.json')['images'];risks=read(IMAGES/'audit-v3/product-identity-quarantine.json')['products']
 guard_allowlist(new,audit,risks)
 inventory=read(PREP/'eligibility-inventory.json')['images'];identity=read(META/'identity-review.json');groups=w.group_inventory(inventory,identity)
 reserved=[r for r in inventory if r.get('proposed_split') in ('validation','test')]+protocol['development_rows']
 reserved=[{**r,'leakage_group_id':groups.find(r['abo_product_id'])} for r in reserved]
 normal=[]
 for origin,items in [('baseline_fit',baseline),('new_allowed',new)]:
  for r in items:
   row={**r,'abo_product_id':r['abo_product_id'],'leakage_group_id':groups.find(r['abo_product_id']),'origin':origin,'proposed_split':'train','human_verified':False}
   path=w.image_path(row);row['canonical_pixels']=canonical(path);normal.append(row)
 isolate(normal,reserved)
 if len({r['image_id'] for r in normal})!=532:raise ValueError('Training image count mismatch')
 folds,ng=group_folds(normal,3);basefolds,bg=group_folds(normal[:182],2)
 newids={r['image_id'] for r in normal[182:]};allpair=read(IMAGES/'audit-v1/duplicate-and-overlap-pairs.json')
 if any((p['left_image_id'] in newids or p['right_image_id'] in newids) and p['left_product_id']!=p['right_product_id'] for p in allpair):raise ValueError('Accepted new product has unresolved recorded cross-product overlap')
 sources=[source,ROOT/'data/abo/experiments/phase3-baseline-v1/features-v1/metadata.json',PREPARATION/'frozen-development-protocol.json',PREP/'eligibility-inventory.json',META/'identity-review.json',
  IMAGES/'audit-v3/screening.json',IMAGES/'audit-v3/provisional-candidates.json',IMAGES/'audit-v3/product-identity-quarantine.json',IMAGES/'audit-v1/duplicate-and-overlap-pairs.json',IMAGES/'audit-v1/summary.json',
  ROOT/'tools/phase3-acquisition-v1/screen.py',ROOT/'tools/weak-training-v1/pipeline.py',ROOT/'tools/auto-training-v1/auto_train.py',w.MODEL/'checkpoint-manifest.json']
 sources+=list(HERE.glob('*.py'))
 for rel in read(w.MODEL/'checkpoint-manifest.json')['files']:sources.append(w.MODEL/rel)
 for r in normal[182:]:
  cache=IMAGES/'screening-cache-v1'/(r['sha256']+'.npy')
  if cache.exists():sources.append(cache)
 screening=read(IMAGES/'audit-v1/summary.json')
 if sha(ROOT/'tools/phase3-acquisition-v1/screen.py')!=screening['screening_code_sha256'] or screening['model_fingerprint']!=protocol['model_processor_fingerprint']:raise ValueError('Feature screening provenance mismatch')
 save(out/'preservation-before.json',preservation(out))
 dataset={'rows':normal,'expanded_folds':folds,'baseline_folds':basefolds,'baseline_images':182,'images':532,'products':281,'known_group_count':ng,'baseline_known_group_count':bg,
  'physical_identity_independence':'unverified; counts are known leakage components, not proven physical identities','frozen_panel_excluded_from_optimization':True,
  'classes':{c:{'images':sum(r['weak_label']==c for r in normal),'products':len({r['abo_product_id'] for r in normal if r['weak_label']==c}),'new_images':sum(r['weak_label']==c and r['origin']=='new_allowed' for r in normal)} for c in CLASSES}}
 save(out/'manifests/dataset.json',dataset);save(out/'manifests/policy.json',POLICY);save(out/'manifests/configurations.json',CONFIGS)
 save(out/'input-lock.json',{'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},'frozen_panel_stat_only':stable_stat(PREPARATION/'frozen-development-features.npy'),
  'frozen_panel_sha256_from_prior_preparation':protocol['development_feature_sha256'],
  'artifacts':{str(p.relative_to(out)):sha(p) for p in (out/'manifests').glob('*.json')}})
 print(json.dumps({'ready_images':532,'products':281,'known_groups':ng,'classes':dataset['classes'],'frozen_panel_not_loaded':True}),flush=True)


def validate(out):
 lock=read(out/'input-lock.json')
 for rel,h in lock['sources'].items():
  if sha(ROOT/rel)!=h:raise ValueError('Pinned input changed: '+rel)
 for rel,h in lock['artifacts'].items():
  if sha(out/rel)!=h:raise ValueError('Frozen manifest changed: '+rel)
 if stable_stat(PREPARATION/'frozen-development-features.npy')!=lock['frozen_panel_stat_only']:raise ValueError('Frozen panel changed')
 if read(out/'manifests/policy.json')!=POLICY or read(out/'manifests/configurations.json')!=CONFIGS:raise ValueError('Queue policy changed')


def gpu():return w.gpu()
def features(out):
 validate(out);manifest=read(out/'manifests/dataset.json');target=out/'features-v1'
 if (target/'metadata.json').exists():return load_features(out)
 device=gpu();started=time.monotonic();target.mkdir(exist_ok=True)
 original=np.load(ROOT/'data/abo/experiments/phase3-baseline-v1/features-v1/features.npy',mmap_mode='r',allow_pickle=False)
 baseline=np.asarray(original[:182]).copy();del original
 from transformers import AutoModel,AutoImageProcessor
 model=AutoModel.from_pretrained(str(w.MODEL),local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().requires_grad_(False).to(device)
 processor=AutoImageProcessor.from_pretrained(str(w.MODEL),local_files_only=True,use_fast=False)
 shards=target/'validated-shards';shards.mkdir(exist_ok=True);new=[];evidence=[]
 for n,row in enumerate(manifest['rows'][182:],1):
  if STOP or (out/'STOP').exists():raise InterruptedError('Feature validation stopped; committed shards retained')
  output=shards/(row['sha256']+'.npy');proof=output.with_suffix('.json')
  if output.exists() and proof.exists():
   record=read(proof)
   if record['feature_sha256']!=sha(output) or record['image_sha256']!=row['sha256']:raise ValueError('Invalid resumed feature shard')
   vector=np.load(output,allow_pickle=False)
  else:
   with Image.open(w.image_path(row)) as image:im=ImageOps.exif_transpose(image).convert('RGB')
   with torch.inference_mode():
    x=processor(images=im,return_tensors='pt');computed=w.unit(model.get_image_features(**{k:v.to(device) for k,v in x.items()}).float().cpu().numpy())
   prior=IMAGES/'screening-cache-v1'/(row['sha256']+'.npy');record={'image_sha256':row['sha256'],'model_fingerprint':read(w.MODEL/'checkpoint-manifest.json')['model_processor_fingerprint']}
   vector=computed
   if prior.exists():
    cached=np.load(prior,allow_pickle=False)
    if cached.shape==computed.shape and np.isfinite(cached).all() and np.allclose(cached,computed,atol=POLICY['feature_validation_atol'],rtol=0):
     vector=cached;record.update(reused=True,prior_sha256=sha(prior),max_difference=float(np.max(np.abs(cached-computed))))
    else:record.update(reused=False,reason='cached_vector_did_not_match_fresh_GPU_validation')
   else:record.update(reused=False,reason='no_provenance_checked_cached_embedding')
   w.save_array(output,vector);record['feature_sha256']=sha(output);save(proof,record)
  new.append(vector[0]);evidence.append(record)
  if n%50==0:print(json.dumps({'gpu_validated_new_embeddings':n,'total':350}),flush=True)
 matrix=np.concatenate([baseline,np.asarray(new,dtype=np.float32)])
 if matrix.shape!=(532,768) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4):raise ValueError('Feature matrix validation failed')
 torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated(0);del model,processor;torch.cuda.empty_cache()
 w.save_array(target/'fit-only.npy',matrix)
 save(target/'metadata.json',{'rows_sha256':sha(out/'manifests/dataset.json'),'feature_sha256':sha(target/'fit-only.npy'),'shape':list(matrix.shape),'reused_original_vectors':182,
  'new_cache_vectors_reused_after_GPU_validation':sum(r['reused'] for r in evidence),'new_vectors_extracted_or_rebuilt':sum(not r['reused'] for r in evidence),
  'model_processor_fingerprint':read(w.MODEL/'checkpoint-manifest.json')['model_processor_fingerprint'],'per_image_provenance':evidence,
  'peak_allocated_gpu_bytes':peak,'seconds':time.monotonic()-started,'frozen_panel_vectors_not_in_this_cache':True})
 return load_features(out)


def load_features(out):
 validate(out);meta=read(out/'features-v1/metadata.json');path=out/'features-v1/fit-only.npy'
 if sha(path)!=meta['feature_sha256'] or meta['rows_sha256']!=sha(out/'manifests/dataset.json'):raise ValueError('Feature fingerprint mismatch')
 array=np.load(path,allow_pickle=False)
 if array.shape!=(532,768) or not np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4):raise ValueError('Invalid training-only feature matrix')
 return array,read(out/'manifests/dataset.json')


def balance_weights(rows,indices,strategy):
 counts=C.Counter({rows[i]['abo_product_id']:rows[i]['weak_label'] for i in indices}.values())
 if set(counts)!=set(CLASSES):raise ValueError('Internal training fold missing a class')
 values=np.array([1 if strategy=='none' else 1/(counts[c]**(.5 if strategy=='sqrt' else 1)) for c in CLASSES]);return values/values.mean(),dict(counts)

def metrics(rows,indices,logits):return w.metric(rows,indices,logits)

def fit(out,relative,array,rows,train_ids,dev_ids,cfg,epochs,deadline):
 device=gpu();directory=out/relative;directory.mkdir(parents=True,exist_ok=True)
 signature=fingerprint({'code':sha(__file__),'cache':sha(out/'features-v1/metadata.json'),'cfg':cfg,'train':train_ids,'dev':dev_ids,'epochs':epochs,'seed':SEED,'policy':POLICY})
 if (directory/'result.json').exists():
  result=read(directory/'result.json')
  if result['signature']!=signature or sha(directory/'best.pt')!=result['checkpoint_sha256']:raise ValueError('Completed trial incompatible/changed')
  return result
 if set(train_ids)&set(dev_ids):raise ValueError('Internal index overlap')
 if {rows[i]['leakage_group_id'] for i in train_ids}&{rows[i]['leakage_group_id'] for i in dev_ids}:raise ValueError('Internal family leakage')
 seed=SEED+int(fingerprint([relative,cfg])[:8],16)%100000;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=w.network(cfg,array.shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
 weights,counts=balance_weights(rows,train_ids,cfg['balance']);criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
 # Only FIT/internal-development vectors enter this process; never frozen comparison vectors.
 x=torch.from_numpy(array).to(device);y=torch.tensor([CLASSES.index(r['weak_label']) for r in rows],device=device)
 state={'epoch':0,'curve':[],'best_score':-1.,'best_epoch':0,'stale':0,'best_state':None,'batch_size':POLICY['batch_size']}
 progress=directory/'progress.pt'
 if progress.exists():
  loaded=torch.load(progress,map_location='cpu',weights_only=True)
  if loaded['signature']!=signature:raise ValueError('Resume checkpoint signature mismatch')
  state=loaded['state'];model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
 started=time.monotonic();oom_events=[]
 for epoch in range(state['epoch'],epochs):
  if STOP or (out/'STOP').exists() or time.monotonic()>deadline:raise InterruptedError('Stopped at committed epoch; resume same command')
  if dev_ids and state['stale']>=POLICY['patience']:break
  chosen=w.helper().epoch_sample(rows,train_ids,seed+epoch*7919)
  while True:
   # Retry an OOM epoch from the prior committed state, never from partial gradient updates.
   rollback={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   rollback_opt=__import__('copy').deepcopy(optimizer.state_dict())
   torch.manual_seed(seed+epoch*7919);torch.cuda.manual_seed_all(seed+epoch*7919);model.train();losses=[]
   try:
    for pos in range(0,len(chosen),state['batch_size']):
     ids=chosen[pos:pos+state['batch_size']];optimizer.zero_grad(set_to_none=True);loss=criterion(model(x[ids]),y[ids])
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
     loss.backward()
     if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Invalid gradients')
     optimizer.step();losses.append(float(loss.detach()))
    break
   except torch.cuda.OutOfMemoryError:
    model.load_state_dict(rollback);optimizer.load_state_dict(rollback_opt);optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if state['batch_size']<=1:raise
    state['batch_size']=max(1,state['batch_size']//2);oom_events.append({'epoch':epoch+1,'reduced_batch_size':state['batch_size']})
    save(directory/f'oom-event-{len(oom_events)}.json',oom_events[-1])
  model.eval();measure=None
  if dev_ids:
   with torch.inference_mode():logits=model(x[dev_ids]).cpu().numpy()
   measure=metrics(rows,dev_ids,logits);score=measure['product']['macro_f1']
  else:score=epoch+1
  if score>state['best_score']:
   state.update(best_score=score,best_epoch=epoch+1,best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},stale=0)
  else:state['stale']+=1
  state['epoch']=epoch+1;state['curve'].append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'internal_dev':measure})
  w.save_tensor(progress,{'signature':signature,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'state':state})
  if (epoch+1)%10==0:print(json.dumps({'trial':relative,'epoch':epoch+1,'best_internal_f1':state['best_score'] if dev_ids else None}),flush=True)
 model.load_state_dict(state['best_state'])
 with torch.inference_mode():measure=metrics(rows,dev_ids,model(x[dev_ids]).cpu().numpy()) if dev_ids else None
 w.save_tensor(directory/'best.pt',{'state_dict':state['best_state'],'classes':CLASSES,'config':cfg,'signature':signature,'epoch':state['best_epoch'],
  'feature_model_fingerprint':read(out/'features-v1/metadata.json')['model_processor_fingerprint'],'train_product_ids':sorted({rows[i]['abo_product_id'] for i in train_ids}),
  'no_frozen_panel_training_or_tuning':True,'weak_labels_only':True})
 result={'signature':signature,'checkpoint_sha256':sha(directory/'best.pt'),'config':cfg,'metrics':measure,'best_epoch':state['best_epoch'],'epochs':state['epoch'],
  'training_products':counts,'training_images':len(train_ids),'internal_dev_images':len(dev_ids),'seconds_this_invocation':time.monotonic()-started,
  'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(0),'oom_events':oom_events,'scope':SCOPE}
 save(directory/'curve.json',state['curve']);save(directory/'result.json',result);return result


def fold_indices(total,dev):return sorted(set(range(total))-set(dev))
def ranking(results):return sorted(results,key=lambda r:(-r['mean_macro_f1'],-r['mean_balanced_accuracy'],r['name']))

def leaderboard(out):
 results=[]
 for cfg in CONFIGS:
  files=[out/'trials'/cfg['name']/f'fold-{i}/result.json' for i in range(3)]
  if all(p.exists() for p in files):
   r=[read(p) for p in files];f1=[x['metrics']['product']['macro_f1'] for x in r]
   results.append({'name':cfg['name'],'config':cfg,'mean_macro_f1':statistics.mean(f1),'sd_macro_f1':statistics.stdev(f1),
    'mean_balanced_accuracy':statistics.mean(x['metrics']['product']['balanced_accuracy'] for x in r),
    'fold_macro_f1':f1,'fold_balanced_accuracy':[x['metrics']['product']['balanced_accuracy'] for x in r], 'fold_epochs':[x['best_epoch'] for x in r]})
 results=ranking(results);save(out/'leaderboard.json',results)
 stream=io.StringIO();fields=['name','mean_macro_f1','sd_macro_f1','mean_balanced_accuracy','fold_macro_f1','fold_balanced_accuracy','fold_epochs'];writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results);atomic(out/'leaderboard.csv',stream.getvalue().encode())
 return results


def smoke(out):
 array,data=features(out);fold=data['expanded_folds'][0];deadline=time.monotonic()+300
 result=fit(out,'smoke',array,data['rows'],fold_indices(len(array),fold),fold,CONFIGS[0],3,deadline)
 save(out/'smoke-gate.json',{'passed':True,'code_sha256':sha(__file__),'data_sha256':sha(out/'features-v1/metadata.json'),'result':result})
 preserve_check(out);print('GPU smoke passed',flush=True)


def queue(out):
 validate(out);gate=read(out/'smoke-gate.json')
 if not gate['passed'] or gate['code_sha256']!=sha(__file__) or gate['data_sha256']!=sha(out/'features-v1/metadata.json'):raise ValueError('Smoke gate stale')
 array,data=load_features(out);rows=data['rows'];deadline=time.monotonic()+POLICY['queue_wall_seconds_per_invocation'];started=time.monotonic()
 save(out/'queue-policy.json',{'configs':CONFIGS,'policy':POLICY,'max_fits':30,'stages':'1 smoke, 24 expanded CV, 2 baseline CV, 2 or 3 final refits; one comparison only after freeze'})
 for cfg in CONFIGS:
  for i,dev in enumerate(data['expanded_folds']):
   fit(out,f'trials/{cfg["name"]}/fold-{i}',array,rows,fold_indices(len(rows),dev),dev,cfg,80,deadline);leaderboard(out)
 for i,dev in enumerate(data['baseline_folds']):fit(out,f'baseline-cv/fold-{i}',array[:182],rows[:182],fold_indices(182,dev),dev,CONFIGS[0],80,deadline)
 validate(out);preserve_check(out);board=leaderboard(out)
 if len(board)!=8:raise ValueError('Incomplete candidate selection')
 selected=board[0];cosine=next(r for r in board if r['name']=='cosine-reference')
 base=[read(out/f'baseline-cv/fold-{i}/result.json') for i in range(2)]
 freeze={'winner':selected,'baseline_recipe':CONFIGS[0],'baseline_epochs':int(statistics.median(r['best_epoch'] for r in base)),
  'selected_epochs':int(statistics.median(selected['fold_epochs'])),'expanded_cosine_epochs':int(statistics.median(cosine['fold_epochs'])),
  'selection_source':'training-only internal group folds','frozen_panel_accessed_for_selection':False,'policy_sha256':sha(out/'manifests/policy.json'),'features_sha256':sha(out/'features-v1/fit-only.npy')}
 freeze_path=out/'selection-frozen.json'
 if freeze_path.exists():
  if read(freeze_path)!=freeze:raise ValueError('Frozen selection changed')
 else:save(freeze_path,freeze)
 fit(out,'final-baseline',array[:182],rows[:182],list(range(182)),[],CONFIGS[0],max(1,freeze['baseline_epochs']),deadline)
 fit(out,'final-selected',array,rows,list(range(len(rows))),[],selected['config'],max(1,freeze['selected_epochs']),deadline)
 if selected['name']=='cosine-reference':control='final-selected'
 else:
  fit(out,'final-expanded-cosine',array,rows,list(range(len(rows))),[],CONFIGS[0],max(1,freeze['expanded_cosine_epochs']),deadline);control='final-expanded-cosine'
 registry={'baseline':{'path':'final-baseline/best.pt','sha256':sha(out/'final-baseline/best.pt')},'selected_expanded':{'path':'final-selected/best.pt','sha256':sha(out/'final-selected/best.pt')},
  'expanded_cosine_control':{'path':control+'/best.pt','sha256':sha(out/control/'best.pt')},'selection_sha256':sha(freeze_path)}
 registry_path=out/'comparison-checkpoints-frozen.json'
 if registry_path.exists():
  if read(registry_path)!=registry:raise ValueError('Comparison checkpoint registry changed')
 else:save(registry_path,registry)
 preserve_check(out)
 save(out/'queue-complete.json',{'complete':True,'gpu':torch.cuda.get_device_name(0),'seconds_this_invocation':time.monotonic()-started,'frozen_panel_not_yet_scored':not (out/'comparison/result.json').exists(),'checkpoint_registry_sha256':sha(registry_path)})
 print('Bounded training complete; candidate and comparison checkpoints frozen',flush=True)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','run','verify','status']);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();out=args.output.resolve()
 if not out.is_relative_to(ROOT/'data/abo/experiments') or not out.name.startswith('phase3-gpu-'):raise ValueError('Fresh isolated phase3-gpu output root required')
 if args.action=='status':
  print(json.dumps({'queue_complete':(out/'queue-complete.json').exists(),'comparison_complete':(out/'comparison/result.json').exists(),'leaderboard':read(out/'leaderboard.json') if (out/'leaderboard.json').exists() else []},indent=2));return
 out.mkdir(exist_ok=True)
 with (out/'training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def stop(*_):
   global STOP;STOP=True
  signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
  try:
   if args.action=='prepare':readiness(out)
   elif args.action=='smoke':smoke(out)
   elif args.action=='run':queue(out)
   elif args.action=='verify':validate(out);preserve_check(out)
  except InterruptedError as exc:save(out/'status.json',{'paused':True,'reason':str(exc),'resume':'same command after removing your STOP control file'});print(str(exc),flush=True)

if __name__=='__main__':main()
