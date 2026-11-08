"""Future bounded adaptation runner. Parameter updates require explicit approval flag.

This module is prepared and tested in Phase 5B; its update loop is NOT executed now.
"""
import argparse
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import random
import signal
import time
from pathlib import Path
from common import *
from checkpoints import commit_json,snapshot,recover,Lease
STOP=False

def stop_handler(*_):
 global STOP
 STOP=True

def validate_config(config):
 allowed=dict(schema='phase5b-detector-adaptation-v1',mode='heads',epochs=10,accumulation=4,max_updates=70,selection_split='development',fit_split='train',precision='fp16',batch_size=1)
 for k,v in allowed.items():
  if config.get(k)!=v:raise ValueError('Unapproved/unsupported training bound: '+k)
 if config.get('learning_rate') not in (1e-5,3e-5) or config.get('weight_decay')!=.01 or config.get('seed')!=SETTINGS['seed']:raise ValueError('Unexpected preset')
 if config.get('max_seconds')!=1800 or config.get('memory_fraction')!=.80:raise ValueError('Resource bounds changed')
 if config.get('dataset_lock_sha256')!=sha(OUT/'dataset-lock.json') or config.get('base_checkpoint_sha256')!=OWL_SHA:raise ValueError('Dataset/model binding differs')
 return config

def capture_rng(torch):return dict(python=random.getstate(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(torch,rng):random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])

def train(config_path,out,resume=False,allow_updates=False):
 if not allow_updates:raise PermissionError('Detector parameter updates require separate user approval and --allow-parameter-updates')
 config=validate_config(read(config_path));out=Path(out).resolve()
 model_files={p.name:sha(p) for p in OWL.iterdir() if p.is_file()}
 if model_files!=read(OUT/'original-development-v1/binding.json')['model_files']:raise ValueError('Pinned model/preprocessor files changed')
 if not out.is_relative_to(ROOT/'data/open-images/experiments') or out==OUT:raise ValueError('Fresh isolated training output under experiments required')
 import importlib.metadata as metadata
 environment={name:metadata.version(name) for name in ('torch','transformers','numpy','scipy','Pillow')}
 binding=dict(explicit_update_flag=True,environment=environment,model_files=model_files,config=config,dataset_sha256=sha(OUT/'dataset-lock.json'),model_sha256=OWL_SHA,code={n:sha(HERE/n) for n in ('train.py','common.py','adaptation.py','checkpoints.py','baseline.py','metrics.py')})
 if out.exists():
  if not resume:raise FileExistsError('Existing fit requires explicit --resume')
  if read(out/'binding.json')!=binding:raise ValueError('Resume configuration/code differs')
  if (out/'complete.json').exists():print('SKIP COMPLETED',out,flush=True);return
 else:out.mkdir(parents=True);commit_json(out/'binding.json',binding)
 with Lease(out/'worker.lock'):
  import torch
  from PIL import Image,ImageOps
  from transformers import Owlv2ForObjectDetection,Owlv2Processor
  from adaptation import configure,detection_loss,trainable_state,load_delta
  from baseline import predict
  from metrics import evaluate
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training')
  torch.cuda.set_per_process_memory_fraction(.80,0);torch.cuda.reset_peak_memory_stats(0);torch.manual_seed(config['seed']);random.seed(config['seed']);torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark=False
  if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original model changed')
  model=Owlv2ForObjectDetection.from_pretrained(str(OWL),local_files_only=True,use_safetensors=True).to('cuda');processor=Owlv2Processor.from_pretrained(str(OWL),local_files_only=True)
  configure(model,'heads',False);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=.01)
  scaler=torch.amp.GradScaler('cuda',init_scale=1024.0);state=recover(out,binding);step=epoch=start_group=0;best=None;elapsed_before=0.
  if state:
   _,state,_,_=state;load_delta(model,state['weights']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);restore_rng(torch,state['rng']);step=state['step'];epoch=state['epoch'];start_group=state['next_group'];best=state.get('best');elapsed_before=state.get('elapsed_seconds',0.)
  train_rows=load_rows('train');dev_rows=load_rows('development');started=time.perf_counter()
  if len(train_rows)!=27 or len(dev_rows)!=18:raise ValueError('Locked counts changed')
  signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
  def stopped():return STOP or (out/'STOP').exists() or elapsed_before+time.perf_counter()-started>=config['max_seconds']
  def persist(ep,next_group):
   value=dict(weights=trainable_state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=capture_rng(torch),step=step,epoch=ep,next_group=next_group,best=best,elapsed_seconds=elapsed_before+time.perf_counter()-started)
   return snapshot(out,value,binding)
  while epoch<config['epochs']:
   order=list(train_rows);random.Random(config['seed']+epoch).shuffle(order);groups=[order[i:i+4] for i in range(0,len(order),4)]
   for index in range(start_group,len(groups)):
    if stopped():
     persist(epoch,index);print('GRACEFUL STOP',step,flush=True);return
    if step>=config['max_updates']:raise RuntimeError('Optimizer update bound reached')
    model.train();model.owlv2.text_model.eval();optimizer.zero_grad(set_to_none=True);loss_values=[]
    for row in groups[index]:
     with Image.open(image_path(row,'train')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
     inputs=processor(text=[SETTINGS['prompts']],images=image,return_tensors='pt');inputs={k:v.to('cuda') for k,v in inputs.items()}
     with torch.autocast('cuda',dtype=torch.float16):outputs=model(**inputs)
     loss,parts=detection_loss(outputs.logits[0],outputs.pred_boxes[0],row)
     scaler.scale(loss/len(groups[index])).backward();loss_values.append(float(loss.detach()))
    scaler.unscale_(optimizer);grads=[p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in grads):raise ValueError('Training gradient failure; no optimizer step')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    scaler.step(optimizer);scaler.update();step+=1
    checkpoint,digest=persist(epoch,index+1)
    commit_json(out/f'update-{step:05d}.json',dict(step=step,epoch=epoch,loss=sum(loss_values)/len(loss_values),checkpoint=checkpoint,sha256=digest,peak_allocated_bytes=torch.cuda.max_memory_allocated(0)))
    print('UPDATE',step,'epoch',epoch,'loss',sum(loss_values)/len(loss_values),flush=True)
   # Development only, never frozen evaluation. All seven native metrics retained.
   model.eval();preds={}
   for row in dev_rows:
    with Image.open(image_path(row,'development')) as im:image=ImageOps.exif_transpose(im).convert('RGB')
    _,preds[row['image_id']]=predict(model,processor,image)
   metrics=evaluate(dev_rows,preds);score=metrics['macro_ap50_supported'];validation=out/f'validation-epoch-{epoch+1:03d}.json'
   payload=dict(step=step,epoch=epoch+1,metrics=metrics)
   if validation.exists():
    if read(validation)!=payload:raise ValueError('Recovered validation differs')
   else:commit_json(validation,payload)
   if best is None or score>best['score']:
    best=dict(score=score,epoch=epoch+1,step=step,selection='development native macro AP50; not verified real-world accuracy')
    checkpoint,digest=persist(epoch+1,0);best.update(checkpoint=checkpoint,sha256=digest)
   epoch+=1;start_group=0;persist(epoch,0)
   if stopped():print('GRACEFUL STOP AFTER VALIDATION',step,flush=True);return
  if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original model changed')
  commit_json(out/'complete.json',dict(updates=step,epochs=epoch,best=best,seconds=elapsed_before+time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),frozen_images_used=0,base_checkpoint_sha256=OWL_SHA))
  print('COMPLETE',out,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--allow-parameter-updates',action='store_true')
 a=ap.parse_args();train(a.config,a.out,a.resume,a.allow_parameter_updates)
