"""User-approved Phase 5C: exactly one of two prepared head-only fits."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import importlib.metadata as metadata
import json
import random
import signal
import time
import uuid
from pathlib import Path
from common import *
from control import RUN_ROOT,FIT_IDS,ledger_init,check_resume,reserve,applied,frozen_hash,assert_scope,atomic
from checkpoints import Lease,commit_json,snapshot,recover
STOP=False

def stop(*_):
 global STOP
 STOP=True

def validate(fit_id):
 if fit_id not in FIT_IDS:raise ValueError('Exactly two approved configurations only')
 path=OUT/'configs'/(fit_id+'.json');config=read(path);pre=read(RUN_ROOT/'preflight.json')
 if sha(path)!=pre['config_sha256'][fit_id] or sha(OUT/'dataset-lock.json')!=config['dataset_lock_sha256']:raise ValueError('Prepared config/data changed')
 required=dict(mode='heads',epochs=10,accumulation=4,max_updates=70,fit_split='train',selection_split='development',batch_size=1,precision='fp16',seed=20261105,weight_decay=.01,max_seconds=1800,memory_fraction=.8)
 if any(config.get(k)!=v for k,v in required.items()):raise ValueError('Prepared bounds violated')
 if config['learning_rate']!={'heads-lr1e-5':1e-5,'heads-lr3e-5':3e-5}[fit_id]:raise ValueError('Unapproved LR')
 return config

def run(fit_id,resume=False,pause_after=None):
 config=validate(fit_id);out=RUN_ROOT/fit_id
 if not read(RUN_ROOT/'authorization.json')['detector_updates_authorized']:raise PermissionError('Missing user authorization')
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original OWLv2 changed')
 model_files={p.name:sha(p) for p in OWL.iterdir() if p.is_file()}
 if model_files!=read(OUT/'original-development-v1/binding.json')['model_files']:raise ValueError('Original processor/model changed')
 protocol_files={n:sha(HERE/n) for n in ('common.py','adaptation.py','baseline.py','metrics.py','control.py')}
 binding=dict(schema='phase5c-approved-heads-v1',config=config,config_sha256=sha(OUT/'configs'/(fit_id+'.json')),dataset_sha256=sha(OUT/'dataset-lock.json'),base_sha256=OWL_SHA,model_files=model_files,
     protocol_files=protocol_files,environment={n:metadata.version(n) for n in ('torch','transformers','numpy','scipy','Pillow')})
 if out.exists():
  if not resume:raise FileExistsError('Explicit --resume required')
  if read(out/'binding.json')!=binding:raise ValueError('Protocol/config/model binding changed')
  if (out/'complete.json').exists():print('SKIP COMPLETE',fit_id,flush=True);return
 else:out.mkdir();commit_json(out/'binding.json',binding)
 with Lease(out/'worker.lock'):
  commit_json(out/('execution-'+uuid.uuid4().hex+'.json'),dict(worker_sha256=sha(__file__),checkpoints_code_sha256=sha(HERE/'checkpoints.py'),resume=resume,pause_after=pause_after,unix_ns=time.time_ns()))
  import torch
  from PIL import Image,ImageOps
  from transformers import Owlv2ForObjectDetection,Owlv2Processor
  from adaptation import configure,detection_loss,trainable_state,load_delta
  from baseline import predict
  from metrics import evaluate
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
  torch.cuda.set_per_process_memory_fraction(.8,0);torch.cuda.reset_peak_memory_stats(0);torch.manual_seed(config['seed']);random.seed(config['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
  started=time.perf_counter()
  model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
  names=configure(model,'heads',False);original_frozen=frozen_hash(model)
  if (out/'scope.json').exists():
   if read(out/'scope.json')['frozen_parameter_sha256']!=original_frozen:raise ValueError('Original frozen parameters differ')
  else:commit_json(out/'scope.json',dict(trainable_names=names,trainable_count=1579526,frozen_parameter_sha256=original_frozen,frozen_modules=['owlv2','objectness_head','layer_norm']))
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=.01);assert_scope(model,optimizer)
  scaler=torch.amp.GradScaler('cuda',init_scale=1024.0);restored=recover(out,binding);step=epoch=start_group=0;best=None;elapsed_before=0.;epoch_losses=[];peak_before=0
  if restored:
   _,state,source,digest=restored;load_delta(model,state['weights']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);random.setstate(state['rng']['python']);torch.set_rng_state(state['rng']['cpu']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
   step=state['step'];epoch=state['epoch'];start_group=state['next_group'];best=state.get('best');elapsed_before=state['elapsed_seconds'];epoch_losses=state['epoch_losses'];peak_before=state.get('peak_allocated_bytes',0)
   commit_json(out/('recovery-'+uuid.uuid4().hex+'.json'),dict(step=step,epoch=epoch,next_group=start_group,source=source,sha256=digest,optimizer_restored=True,rng_restored=True))
  ledger_init();check_resume(fit_id,step)
  train_rows=load_rows('train');dev_rows=load_rows('development')
  if len(train_rows)!=27 or len(dev_rows)!=18:raise ValueError('Approved counts differ')
  for row in train_rows:image_path(row,'train')
  for row in dev_rows:image_path(row,'development')
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  actual_calls=[step];optimizer.register_step_post_hook(lambda *_:actual_calls.__setitem__(0,actual_calls[0]+1))
  def elapsed():return elapsed_before+time.perf_counter()-started
  def peak():return max(peak_before,torch.cuda.max_memory_allocated(0))
  def persist(ep,ng):
   return snapshot(out,dict(weights=trainable_state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
       step=step,epoch=ep,next_group=ng,best=best,elapsed_seconds=elapsed(),epoch_losses=epoch_losses,peak_allocated_bytes=peak(),optimizer_hook_calls=actual_calls[0]),binding)
  def assert_frozen():
   assert_scope(model,optimizer)
   if frozen_hash(model)!=original_frozen:raise ValueError('Frozen parameters changed')
  while epoch<10:
   order=list(train_rows);random.Random(config['seed']+epoch).shuffle(order);groups=[order[i:i+4] for i in range(0,27,4)]
   for index in range(start_group,len(groups)):
    if STOP or (out/'STOP').exists() or elapsed()>=1800:
     persist(epoch,index);assert_frozen();print('SAFE STOP',fit_id,step,flush=True);return
    model.train();model.owlv2.text_model.eval()
    group_rng=dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    for numeric_attempt in range(11):
     if numeric_attempt:
      random.setstate(group_rng['python']);torch.set_rng_state(group_rng['cpu']);torch.cuda.set_rng_state_all(group_rng['cuda'])
     optimizer.zero_grad(set_to_none=True);losses=[];components=[]
     for row in groups[index]:
      with Image.open(image_path(row,'train')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
      inputs=processor(text=[SETTINGS['prompts']],images=image,return_tensors='pt');inputs={k:v.to('cuda') for k,v in inputs.items()}
      with torch.autocast('cuda',dtype=torch.float16):outputs=model(**inputs)
      loss,parts=detection_loss(outputs.logits[0],outputs.pred_boxes[0],row)
      scaler.scale(loss/len(groups[index])).backward();losses.append(float(loss.detach()));components.append(parts)
     scaler.unscale_(optimizer);assert_scope(model,optimizer)
     named_grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
     missing=[n for n,g in named_grads.items() if g is None]
     if missing:raise ValueError('Missing head gradient: '+str(missing))
     bad=[n for n,g in named_grads.items() if not torch.isfinite(g).all()]
     if not bad:break
     scale=scaler.get_scale()
     event=dict(step_before=step,epoch=epoch+1,group=index,attempt=numeric_attempt,losses=losses,nonfinite_gradient_tensors=bad,
                old_scale=scale,new_scale=max(1.,scale/2),optimizer_called=False,action='same-batch RNG-preserving replay at lower FP16 loss scale')
     commit_json(out/('amp-overflow-'+uuid.uuid4().hex+'.json'),event)
     print('AMP OVERFLOW',fit_id,'before update',step+1,'finite losses',losses,'scale',scale,'->',max(1.,scale/2),'tensors',bad,flush=True)
     if scale<=1. or numeric_attempt==10:raise ValueError('Nonfinite gradients persist at minimum FP16 loss scale; optimizer not called')
     scaler.update(new_scale=scale/2)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    reserve(fit_id,step);scaler.step(optimizer)
    if actual_calls[0]!=step+1:raise RuntimeError('GradScaler did not apply exactly one optimizer call')
    scaler.update();step+=1;applied(fit_id,step)
    loss_record=dict(step=step,epoch=epoch+1,group=index,images=[r['image_id'] for r in groups[index]],samples=len(groups[index]),loss=sum(losses)/len(losses),components=components,grad_norm=float(norm),scaler_scale=scaler.get_scale(),numeric_replays=numeric_attempt)
    epoch_losses.append(loss_record);checkpoint,digest=persist(epoch,index+1)
    commit_json(out/f'update-{step:05d}.json',dict(**loss_record,checkpoint=checkpoint,sha256=digest,peak_allocated_bytes=peak()))
    print('UPDATE',fit_id,step,'epoch',epoch+1,'loss',loss_record['loss'],'peak',peak(),flush=True)
    if pause_after is not None and step>=pause_after:
     assert_frozen();commit_json(out/f'paused-integrity-step-{step:05d}.json',dict(step=step,checkpoint=checkpoint,sha256=digest,frozen_parameter_sha256=original_frozen,head_only=True,optimizer_hook_calls=actual_calls[0]));print('SMOKE PAUSE',fit_id,step,flush=True);return
   model.eval();predictions={}
   for row in dev_rows:
    with Image.open(image_path(row,'development')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
    raw,filtered=predict(model,processor,image);predictions[row['image_id']]=dict(image_id=row['image_id'],image_sha256=row['sha256'],predictions=filtered)
   metrics=evaluate(dev_rows,{k:v['predictions'] for k,v in predictions.items()});score=metrics['macro_ap50_supported']
   payload=dict(epoch=epoch+1,step=step,mean_training_loss=sum(r['loss']*r['samples'] for r in epoch_losses)/27,metrics=metrics,predictions=predictions)
   path=out/f'validation-epoch-{epoch+1:03d}.json'
   if path.exists():
    if read(path)!=payload:raise ValueError('Recovered validation differs')
   else:commit_json(path,payload)
   if best is None or score>best['score']:
    checkpoint,digest=persist(epoch,len(groups));best=dict(score=score,epoch=epoch+1,step=step,checkpoint=checkpoint,sha256=digest,selection='maximum development native macro AP50; earliest epoch wins ties')
   print('EPOCH',fit_id,epoch+1,'training_loss',payload['mean_training_loss'],'AP50',score,'AP75',metrics['macro_ap75_supported'],flush=True)
   epoch+=1;start_group=0;epoch_losses=[];persist(epoch,0);assert_frozen()
  check_resume(fit_id,step);assert_frozen()
  if step!=70 or actual_calls[0]!=70:raise ValueError('Unexpected update count')
  if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original checkpoint changed')
  commit_json(out/'complete.json',dict(fit_id=fit_id,epochs=epoch,optimizer_updates=step,optimizer_hook_calls=actual_calls[0],best=best,seconds=elapsed(),peak_allocated_bytes=peak(),frozen_parameter_sha256=original_frozen,
      frozen_parameters_unchanged=True,only_detector_heads_updated=True,base_checkpoint_sha256=OWL_SHA,frozen_photographs_used=0))
  print('COMPLETE',fit_id,step,'best',best,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fit-id',choices=FIT_IDS,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--pause-after-updates',type=int);a=ap.parse_args();run(a.fit_id,a.resume,a.pause_after_updates)
