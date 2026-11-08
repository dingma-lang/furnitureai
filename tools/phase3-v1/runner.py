"""Phase 3 train-only expansion with a fixed internal development panel. Dry run by default."""
import argparse
import collections as C
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from PIL import Image,ImageOps
from common import *


def check_isolation(fit,dev,official):
 for key in ('abo_product_id','leakage_group_id','sha256','decoded_pixel_sha256','capture_cluster_id'):
  a={r[key] for r in fit if r.get(key)};b={r[key] for r in dev+official if r.get(key)}
  if a&b:raise ValueError('FIT/evaluation isolation violation: '+key)
 if any(r.get('proposed_split') in ('validation','test') for r in fit+dev):raise ValueError('Official holdout admitted')
 labels=C.defaultdict(set)
 for r in fit+dev:labels[r['abo_product_id']].add(r['weak_label'])
 if any(len(v)>1 for v in labels.values()):raise ValueError('Contradictory product labels')


def pixel_signatures(path):
 with Image.open(path) as image:
  rgb=ImageOps.exif_transpose(image).convert('RGB')
  exact=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
  small=np.asarray(rgb.convert('L').resize((9,8)))
  bits=(small[:,1:]>small[:,:-1]).ravel();perceptual=sum(int(bit)<<i for i,bit in enumerate(bits))
 return exact,perceptual


def data_context(preparation):
 protocol=read(preparation/'frozen-development-protocol.json');inventory=read(PREP/'eligibility-inventory.json')['images'];identity=read(META/'identity-review.json')
 groups=w.group_inventory(inventory,identity)
 def grouped(rows):return [{**r,'leakage_group_id':groups.find(r['abo_product_id'])} for r in rows]
 fit=grouped(protocol['baseline_training_rows']);dev=grouped(protocol['development_rows']);official=grouped([r for r in inventory if r.get('proposed_split') in ('validation','test')])
 check_isolation(fit,dev,official)
 return protocol,fit,dev,official


def dry_run(preparation,mode):
 verify_plan(preparation);protocol,fit,dev,official=data_context(preparation);plan=read(preparation/'acquisition-plan.json')
 reserved={r['leakage_group_id'] for r in dev+official};local={r['abo_product_id'] for r in read(PREP/'eligibility-inventory.json')['images']}
 if any(p['family_group'] in reserved or p['product_id'] in local for p in plan['products']):raise ValueError('Expansion family/product isolation failure')
 missing=[o['image_id'] for o in plan['objects'] if not (ROOT/plan['destination']/o['destination']).is_file()]
 return {'dry_run':True,'network_requests':0,'training_performed':False,'official_holdout_scoring':False,'mode':mode,
 'baseline_fit_images':len(fit),'baseline_fit_products':len({r['abo_product_id'] for r in fit}),'frozen_dev_images':len(dev),'frozen_dev_products':len({r['abo_product_id'] for r in dev}),
 'candidate_new_products':plan['product_count'],'candidate_new_images':plan['image_count'],'new_images_missing':len(missing),
 'max_training_images_before_screen':len(fit)+(plan['image_count'] if mode=='expanded' else 0),
 'models':[c['name'] for c in protocol['configs']],'status':'ready_for_baseline_cache; expanded mode awaits approved acquisition' if missing else 'local inputs available; screening pending'}


def build_cache(preparation,out,mode):
 verify_plan(preparation);protocol,fit,dev,official=data_context(preparation);dest=out/'features-v1'
 if (dest/'metadata.json').exists():return load_cache(preparation,out,mode)
 original=read(WEAK/'features-v1/metadata.json');original_array=np.load(WEAK/'features-v1/features.npy',allow_pickle=False);lookup={r['image_id']:original_array[i] for i,r in enumerate(original['rows'])}
 fit_vectors=[lookup[r['image_id']] for r in fit];screen=[]
 if mode=='expanded':
  plan=read(preparation/'acquisition-plan.json');image_root=ROOT/plan['destination'];ledger=read(image_root/'transfer-ledger.json')
  if ledger['plan_binding']!=fingerprint(plan):raise ValueError('Acquisition provenance mismatch')
  for obj in plan['objects']:
   record=ledger['objects'].get(obj['image_id'],{})
   if record.get('status')!='downloaded' or sha(image_root/obj['destination'])!=record.get('sha256'):raise ValueError('Missing/changed acquired image')
  device=w.gpu()
  manifest=read(w.MODEL/'checkpoint-manifest.json')
  for rel,h in manifest['files'].items():
   if sha(w.MODEL/rel)!=h:raise ValueError('Frozen encoder checkpoint changed')
  from transformers import AutoModel,AutoImageProcessor
  model=AutoModel.from_pretrained(str(w.MODEL),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval().requires_grad_(False)
  processor=AutoImageProcessor.from_pretrained(str(w.MODEL),local_files_only=True,use_fast=False)
  text=np.load(WEAK/'features-v1/text-prompts.npy',allow_pickle=False)
  inventory=read(PREP/'eligibility-inventory.json')['images'];exact_seen={};near_seen=[]
  # Image-only comparisons against all existing local assets, including reserved examples; no scoring/labels.
  for r in inventory:
   exact,near=pixel_signatures(w.image_path(r));exact_seen[exact]=r['image_id'];near_seen.append((near,r['image_id']))
  cache=out/'new-feature-shards';cache.mkdir(parents=True,exist_ok=True)
  for obj in plan['objects']:
   if w.STOP or (out/'STOP').exists():raise InterruptedError('Feature screening stopped; committed vector shards retained')
   record=ledger['objects'][obj['image_id']];path=image_root/obj['destination'];exact,near=pixel_signatures(path)
   reasons=[]
   if exact in exact_seen:reasons.append('exact_pixel_duplicate:'+exact_seen[exact])
   close=next((i for n,i in near_seen if (n^near).bit_count()<=2),None)
   if close:reasons.append('perceptual_duplicate_suspect:'+close)
   row={'image_id':obj['image_id'],'abo_product_id':obj['product_id'],'leakage_group_id':obj['family_group'],'group_id':obj['family_group'],
    'family_group_id':obj['family_group'],'weak_label':obj['category'],'furniture_class':obj['category'],'proposed_split':'train',
    'local_path':str(path.relative_to(ROOT)),'source_url':obj['url'],'sha256':record['sha256'],'decoded_pixel_sha256':exact,'original_cached_training_view':False,
    'human_verified':False,'label_source':'ABO metadata with frozen automated screening','whole_object_status':'unverified','capture_independence':'unverified'}
   measures,quality_reasons=w.quality(row);reasons+=quality_reasons;row['quality_screen']=measures
   if reasons:screen.append({'image_id':obj['image_id'],'accepted':False,'reasons':reasons});continue
   featurepath=cache/(record['sha256']+'.npy')
   if featurepath.exists():vector=np.load(featurepath,allow_pickle=False)
   else:
    with Image.open(path) as image:im=ImageOps.exif_transpose(image).convert('RGB')
    with torch.inference_mode():
     inputs=processor(images=im,return_tensors='pt');vector=w.unit(model.get_image_features(**{k:v.to(device) for k,v in inputs.items()}).float().cpu().numpy())
    w.save_array(featurepath,vector)
   vector=w.unit(vector)[0];admit,confidence=w.model_screen(row,vector@text.T)
   row['automated_model_screen']=confidence
   screen.append({'image_id':obj['image_id'],'accepted':admit,'screen':confidence})
   if not admit:continue
   exact_seen[exact]=row['image_id'];near_seen.append((near,row['image_id']));fit.append(row);fit_vectors.append(vector)
  del model,processor;gc.collect();torch.cuda.empty_cache()
 check_isolation(fit,dev,official)
 rows=fit+dev;matrix=np.asarray(fit_vectors+[lookup[r['image_id']] for r in dev],dtype=np.float32);indices=list(range(len(fit),len(rows)))
 dest.mkdir(exist_ok=True);w.save_array(dest/'features.npy',matrix)
 save(dest/'metadata.json',{'mode':mode,'rows':rows,'folds':[indices],'fit_count':len(fit),'feature_sha256':sha(dest/'features.npy'),
  'protocol_sha256':sha(preparation/'frozen-development-protocol.json'),'model_processor_fingerprint':protocol['model_processor_fingerprint'],
  'code_sha256':sha(HERE/'runner.py'),'screening':screen,'frozen_development_unchanged':True})
 return load_cache(preparation,out,mode)


def load_cache(preparation,out,mode):
 verify_plan(preparation);meta=read(out/'features-v1/metadata.json');path=out/'features-v1/features.npy';protocol,base,dev,official=data_context(preparation)
 if meta['mode']!=mode or meta['protocol_sha256']!=sha(preparation/'frozen-development-protocol.json') or meta['code_sha256']!=sha(HERE/'runner.py') or sha(path)!=meta['feature_sha256']:raise ValueError('Incompatible or corrupted cache')
 rows=meta['rows'];fit=rows[:meta['fit_count']];actual_dev=rows[meta['fit_count']:]
 if actual_dev!=dev:raise ValueError('Frozen development examples/labels/order changed')
 check_isolation(fit,actual_dev,official);array=np.load(path,allow_pickle=False)
 if array.shape!=(len(rows),768) or not np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-4):raise ValueError('Bad feature array')
 # Compare exact frozen vectors, not only example IDs.
 fixed=np.load(preparation/'frozen-development-features.npy',allow_pickle=False)
 if not np.array_equal(array[meta['fit_count']:],fixed):raise ValueError('Frozen development vectors changed')
 return array,rows,meta['folds']


def uncertainty(rows,indices,logits,seed=SEED,repeats=1000):
 by=C.defaultdict(list)
 for pos,i in enumerate(indices):by[rows[i]['abo_product_id']].append(pos)
 records=[]
 for pid,positions in sorted(by.items()):
  r=rows[indices[positions[0]]];records.append((r['leakage_group_id'],CLASSES.index(r['weak_label']),int(logits[positions].mean(axis=0).argmax())))
 confusion=np.zeros((7,7),dtype=int)
 for _,t,pred in records:confusion[t,pred]+=1
 perclass={}
 for i,c in enumerate(CLASSES):
  tp=int(confusion[i,i]);support=int(confusion[i].sum());predicted=int(confusion[:,i].sum())
  perclass[c]={'products':support,'precision':tp/predicted if predicted else 0,'recall':tp/support if support else None}
 groups=sorted({g for g,_,_ in records});rng=np.random.default_rng(seed);samples=[];missing=0
 for _ in range(repeats):
  picked=rng.choice(groups,size=len(groups),replace=True);values=[(t,pred) for g in picked for group,t,pred in records if group==g]
  truth=[t for t,pred in values];pred=[pred for t,pred in values]
  if set(truth)!=set(range(7)):missing+=1;continue
  samples.append(w.helper().label_metrics(truth,pred)['macro_f1'])
 return {'scope':'weak-label internal development agreement only','per_category':perclass,'confusion_matrix':confusion.tolist(),
  'family_bootstrap_macro_f1_95_percentile_interval':list(map(float,np.quantile(samples,[.025,.975]))) if len(samples)>=100 else None,
  'bootstrap_valid':len(samples),'bootstrap_missing_category':missing,'bootstrap_repeats':repeats,'not_independent_accuracy_confidence_interval':True}


def run(preparation,out,mode):
 array,rows,folds=load_cache(preparation,out,mode);protocol=read(preparation/'frozen-development-protocol.json');deadline=time.monotonic()+protocol['queue_seconds_per_invocation']
 w.POLICY['max_new_storage_bytes']=2*1024**3
 def stop(*_):w.STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 results=[]
 for cfg in protocol['configs']:
  result=w.fit(out,array,rows,folds,cfg,0,protocol['max_epochs'],deadline);results.append(result)
  ranked=sorted(results,key=lambda r:(-r['metrics']['product']['macro_f1'],r['config']['name']))
  save(out/'leaderboard.json',[{'name':r['config']['name'],'weak_dev_product_macro_f1':r['metrics']['product']['macro_f1'],'scope':protocol['scope']} for r in ranked]);w.disk_guard(out)
 selected=ranked[0];cfg=selected['config'];path=out/'trials'/cfg['name']/'fold-0/best.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
 if cfg['kind']=='centroid':logits=array[folds[0]]@saved['centers'].numpy().T
 else:
  model=w.network(cfg,768);model.load_state_dict(saved['state_dict']);model.eval()
  with torch.inference_mode():logits=model(torch.from_numpy(array[folds[0]])).numpy()
 save(out/'comparison-report.json',{'scope':protocol['scope'],'mode':mode,'protocol_sha256':sha(preparation/'frozen-development-protocol.json'),'selected_recipe':cfg,
  'selection_metric':selected['metrics'],'uncertainty':uncertainty(rows,folds[0],logits),'no_official_holdout_scoring':True,'no_refit_on_frozen_development':True,
  'frozen_development_feature_sha256':sha(preparation/'frozen-development-features.npy')})
 save(out/'checkpoint-registry.json',[{'path':str(p.relative_to(out)),'sha256':sha(p),'scope':protocol['scope'],'selected':p==path} for p in out.rglob('best.pt')])
 save(out/'completion.json',{'complete':True,'mode':mode,'selected':str(path.relative_to(out)),'selected_sha256':sha(path)})


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preparation',type=Path,default=OUT);parser.add_argument('--mode',choices=['baseline','expanded'],default='expanded');parser.add_argument('--output',type=Path);parser.add_argument('--build-cache',action='store_true');parser.add_argument('--train',action='store_true');args=parser.parse_args()
 status=dry_run(args.preparation,args.mode)
 if not args.build_cache and not args.train:print(json.dumps(status,indent=2));return
 out=safe_output(args.output or ROOT/f'data/abo/experiments/phase3-{args.mode}-v1',ROOT/'data/abo/experiments')
 if out in (WEAK,OUT,ROOT/'data/abo/experiments/phase2a-v1',w.OLD):raise ValueError('Historical output forbidden')
 out.mkdir(exist_ok=True)
 import fcntl
 with (out/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.build_cache:build_cache(args.preparation,out,args.mode)
  if args.train:
   array,rows,folds=load_cache(args.preparation,out,args.mode)
   # Mandatory separate 3-epoch smoke gate on the same FIT/development partition.
   gate=out/'smoke-gate.json';binding={'code':sha(HERE/'runner.py'),'cache':sha(out/'features-v1/metadata.json')}
   if gate.exists():
    if read(gate)['binding']!=binding:raise ValueError('Stale smoke gate')
   else:
    result=w.fit(out,array,rows,folds,{**w.CONFIGS[0],'name':'smoke'},0,3,time.monotonic()+300);save(gate,{'passed':True,'binding':binding,'result':result})
   try:run(args.preparation,out,args.mode)
   except InterruptedError as exc:save(out/'status.json',{'paused':True,'reason':str(exc),'resume':'same command, after removing your STOP control file'})

if __name__=='__main__':main()
