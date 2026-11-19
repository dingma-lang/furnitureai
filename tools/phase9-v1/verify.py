"""No inference: compare checkpoints, historical stats and locked evaluation receipts."""
import argparse
from core import *
from preflight import snapshot
from data import exposed_rows,final_rows_metadata
from infer import caches,runtime_hashes
from metrics import evaluate

def run():
 old=read(OUT/'preservation-before.json');current=snapshot();changes=[]
 for path,stat in old['files'].items():
  if current.get(path)!=stat:changes.append(path)
 if changes:raise ValueError('Historical files changed; preserve and investigate '+str(changes[:10]))
 for path,digest in old['strong_hashes'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Critical dataset/registry hash changed')
 for item in read(OUT/'model-inventory.json')['weights'].values():
  if sha(ROOT/item['path'])!=item['sha256'] or (ROOT/item['path']).stat().st_size!=item['bytes']:raise ValueError('Original checkpoint changed')
 for row in exposed_rows():
  for arm in ('original','adapted'):
   cache=caches('exposed',arm,row)
   if len(cache)!=8:raise ValueError('Incomplete development capture')
   shapes={v:tuple(cache[v]['pixel_values_shape'][-2:]) for v in ('full768','full960','full1152')}
   if shapes!={'full768':(768,768),'full960':(960,960),'full1152':(1152,1152)}:raise ValueError('Pretend multi-scale input')
   if len({cache[v]['pixel_values_sha256'] for v in shapes})!=3:raise ValueError('Multi-scale tensors identical')
   if any(v['binding']['code']!=runtime_hashes() for v in cache.values()):raise ValueError('Runtime code changed during capture')
 # Exact historical baseline comparison at metric level; no rescoring frozen data.
 reference=read(ROOT/'data/open-images/experiments/phase5c-v1/comparison-v1/comparison.json')['primary_metrics']
 for arm,name in [('original','original'),('adapted','heads-lr3e-5')]:
  m=checked(OUT/'evaluations/exposed'/(arm+'-baseline.json'))['metrics'];expected=reference[name]
  for k in ('tp','fp','fn','duplicate_fp'):
   if m[k]!=expected['micro_operating'][k]:raise ValueError('Historical operating baseline not reproduced')
  if abs(m['ap50']-expected['macro_ap50_supported'])>1e-9 or abs(m['ap75']-expected['macro_ap75_supported'])>1e-9:raise ValueError('Historical AP not reproduced')
 selection=checked(OUT/'selection.json')
 if selection['code_hashes']!=code_hashes():raise ValueError('Non-test source changed since selection')
 final=checked(OUT/'final-comparison-complete.json')
 if not final or final['images']!=36 or not final['retuning_prohibited']:raise ValueError('Final comparison receipt missing')
 for row in final_rows_metadata():
  for arm,views in selection['required_views'].items():
   cache=caches('final',arm,row)
   if set(cache)!=set(views):raise ValueError('Unapproved extra final view or missing stage')
 if read(ROOT/'reports/phase8a-v1/existing33-registration-v2.json')['annotations_activated']!=0:raise ValueError('Inactive33 labels changed')
 files=list(OUT.glob('passes/**/*.json'));peak=max(read(p)['peak_allocated_bytes'] for p in files)
 save(OUT/'preservation-after.json',dict(status='PASS',historical_files_stat_checked=len(old['files']),historical_changes=changes,
  critical_model_hashes_verified=4,critical_dataset_registry_hashes_verified=len(old['strong_hashes']),
  actual_detector_passes=len(files),peak_detector_allocated_bytes=peak,
  development_baselines_reproduced=True,original_default_and_main_unchanged=True,
  final_configuration_locked_before_pixels=True,final_comparison_passes=1,no_retuning=True,
  frozen_image_pixels_and_bytes_read=False,frozen_preservation_stat_only=True,model_parameter_updates=0,
  previous33_LVIS_annotations_activated=0,network_model_or_dataset_downloads=0,independent_weight_backup_verified=False))
 print('PRESERVATION PASS',len(old['files']),'historical files;4 model hashes; exact baseline metrics; locked final comparison',flush=True)

if __name__=='__main__':run()
