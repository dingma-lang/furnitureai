"""Expanded-data adapter and explicit approval gate. No parameter update entrypoint.

Prepares variable-size rows/loss/inference interfaces for a separately approved
head-only runner. Does not reuse the historical worker's fixed 27-image loop.
"""
from core import *
import random
import sys

ORIGINAL=ROOT/'models/owlv2-original-baseline-20261103'
ORIGINAL_SHA='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
REGISTRY=ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json'

def batches(rows,epoch,accumulation=4,seed=20261105):
 """Variable-size approved training groups; development cannot enter updates."""
 if not rows or accumulation<1 or any(r.get('split')!='train' for r in rows):raise ValueError('Training rows only')
 order=list(rows);random.Random(seed+epoch).shuffle(order)
 return [order[i:i+accumulation] for i in range(0,len(order),accumulation)]

def update_schedule(training_rows,epochs=10,max_updates=500):
 if not 1<=epochs<=10 or not 1<=max_updates<=500:raise ValueError('Future proposal budget exceeded')
 schedule=[]
 for epoch in range(epochs):
  for group in batches(training_rows,epoch):
   if len(schedule)>=max_updates:return schedule
   schedule.append(dict(epoch=epoch+1,images=[r['image_id'] for r in group],loss_normalizer=len(group)))
 return schedule

def checked_image(row,partition,locked_rows):
 if partition not in ('train','development') or row.get('split')!=partition or row not in locked_rows:raise ValueError('Not a locked training/development member')
 path=Path(row['absolute_file']).resolve()
 if not path.is_relative_to(DEST.resolve()) or not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Changed or unsafe expanded photograph')
 return path

def load_baseline_model(mode):
 """Future inference loader: immutable base plus optional original Phase5C delta."""
 spec=checkpoint_spec(mode)
 if mode=='expanded_heads':raise ValueError('Use a separately approved future expanded checkpoint registry')
 import torch
 from transformers import Owlv2ForObjectDetection,Owlv2Processor
 sys.path.insert(0,str(ROOT/'tools/phase5c-v1'))
 from adaptation import configure,load_delta
 model=Owlv2ForObjectDetection.from_pretrained(spec['base'],local_files_only=True,use_safetensors=True)
 if mode=='phase5c':
  configure(model,'heads',False);state=torch.load(spec['delta'],map_location='cpu',weights_only=True);load_delta(model,state['weights'])
 for p in model.parameters():p.requires_grad_(False)
 return model.eval(),Owlv2Processor.from_pretrained(spec['base'],local_files_only=True)

def checkpoint_spec(mode):
 if mode not in ('original','phase5c','expanded_heads'):raise ValueError('Unsupported model variant')
 if sha(ORIGINAL/'model.safetensors')!=ORIGINAL_SHA:raise ValueError('Original detector changed')
 result=dict(mode=mode,base=str(ORIGINAL),base_sha256=ORIGINAL_SHA)
 if mode=='phase5c':
  record=read(REGISTRY)
  if record['sha256']!=SELECTED_SHA or sha(record['checkpoint'])!=SELECTED_SHA:raise ValueError('Phase 5C head checkpoint changed')
  result.update(delta=record['checkpoint'],delta_sha256=SELECTED_SHA)
 return result

def plans():
 common=dict(seed=20261105,precision='fp16',batch_size=1,accumulation=4,epochs=10,max_optimizer_updates=500,
  learning_rate=1e-5,weight_decay=.01,gradient_clip=1.,memory_fraction=.8,architecture='class_head and box_head only',
  partial_label_loss_source='tools/phase5c-v1/adaptation.py',loss_source_sha256=sha(ROOT/'tools/phase5c-v1/adaptation.py'),
  evaluation_source='tools/phase5c-v1/metrics.py',evaluation_source_sha256=sha(ROOT/'tools/phase5c-v1/metrics.py'),
  preprocessing_source='tools/phase5c-v1/baseline.py',preprocessing_source_sha256=sha(ROOT/'tools/phase5c-v1/baseline.py'),
  confidence=.15,ap_floor=.001,nms_iou=.5,iou=[.5,.75],selection='native per-category reporting and macro AP50 over explicitly supported fresh development categories; unsupported categories NA, never zero or silently counted',
  separate_reporting=['nonambiguous-category macro AP50/AP75','Desk/Cabinetry source-native sensitivity','per-category FP/FN/duplicates/fragmentation'],
  scope='No original or reserved frozen development panel, official Phase2B holdout or Phase5 frozen evaluation; no selection on previous development predictions',
  training_approved=False,training_readiness='blocked pending acquisition, screening, supported-class coverage and separate final configuration approval')
 write_new(OUT/'training-proposal.json',dict(baselines=[checkpoint_spec('original'),checkpoint_spec('phase5c')],experiment=dict(common,initialization=checkpoint_spec('expanded_heads')),actual_optimizer_updates=0,
  expected_budget_rule='min(10 epochs, 500 optimizer updates); actual ceil(N_train/4)*epochs pinned after screened manifest exists',
  rationale='Start from immutable original; conservative LR avoids selecting the prior higher-LR localization regressions. Original and Phase5C baselines score exactly the same newly approved development rows. No model selection against historical development sets.',
  expected_vram='Prior same-resolution head-only run peaked at 1.93 GiB allocated; plan <=80% of 8 GiB with batch1; stop on OOM instead of widening scope',
  optimization_policy='Desk and Cabinetry positive and negative coverage masked until scope is resolved; native annotations retained for separate source-relative evaluation. All missing and descendant-only coverage remains unknown.'))

def prepare_dataset():
 """Future post-acquisition operation: fresh lock, no updates or inference."""
 screen=read(DEST/'screening-v1.json');m=read(MANIFEST);lookup={r['image_id']:r for r in m['images']};rows=[]
 for r in screen['images']:
  if r['status']!='provisional_automated_candidate':continue
  source=lookup[r['image_id']]
  if r['proposed_split']!=source['proposed_split']:raise ValueError('Partition reassigned')
  path=(DEST/r['file']).resolve()
  if not path.is_relative_to(DEST.resolve()) or sha(path)!=r['sha256']:raise ValueError('Unsafe/changed screened photo')
  # Gradient masks must never silently rewrite development scoring targets.
  runtime_annotations=r['annotations'] if r['proposed_split']=='train' else source['annotations']
  runtime_coverage=r['coverage'] if r['proposed_split']=='train' else source['coverage']
  rows.append(dict(r,annotations=runtime_annotations,coverage=runtime_coverage,evaluation_annotations=source['annotations'],evaluation_coverage=source['coverage'],split=r['proposed_split'],absolute_file=str(path),label_source='Open Images native human annotations; automated suitability provisional'))
 isolation(rows,set(read(OUT/'protected-identities.json')))
 if not any(r['split']=='train' for r in rows) or not any(r['split']=='development' for r in rows):raise ValueError('No usable train/development set')
 write_new(OUT/'screened-training-lock-v1.json',dict(rows=rows,screening_sha256=sha(DEST/'screening-v1.json'),acquisition_manifest_sha256=sha(MANIFEST),training_approved=False,source_native_annotations_unchanged=True,desk_cabinet_optimization_masked=True))

def training_gate(approval_path,config_path,lock_path):
 record=read(approval_path)
 if record.get('detector_training_approved') is not True or record.get('config_sha256')!=sha(config_path) or record.get('dataset_sha256')!=sha(lock_path):raise PermissionError('Separate final training approval required')
 config=read(config_path)
 if config.get('architecture')!='class_head and box_head only' or config.get('epochs',11)>10 or config.get('max_optimizer_updates',501)>500:raise ValueError('Unsupported proposed adaptation scope')
 return config

if __name__=='__main__':plans()
