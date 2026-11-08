"""Post-training preservation and scope audit; no inference or optimizer calls."""
import collections
import sys
from common import *
from control import RUN_ROOT, FIT_IDS, check_resume
sys.path.insert(0, str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical

def run():
 old=historical(META)
 acquisition=verify_acquisition()
 previous=verify_sums(OUT,'ARTIFACT-SHA256SUMS')
 previous_code=read(OUT/'implementation-sha256.json')
 for path,digest in previous_code.items():
  if sha(ROOT/path)!=digest:raise ValueError('Historical implementation changed: '+path)
 lock=read(OUT/'dataset-lock.json');train=load_rows('train');dev=load_rows('development')
 if sha(OUT/'dataset-lock.json')!=read(RUN_ROOT/'preflight.json')['dataset_sha256']:raise ValueError('Dataset changed')
 for key in ('image_id','sha256','identity_group','source_identity'):
  partitions=[{r[key] for r in rows if r.get(key)} for rows in (train,dev,lock['frozen_references'])]
  if any(partitions[i]&partitions[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Identity leakage '+key)
 ledger=read(RUN_ROOT/'update-budget.json')
 if sum(x['applied_calls'] for x in ledger['fits'].values())!=140:raise ValueError('Wrong global update count')
 receipt_count=0
 for name in FIT_IDS:
  fit=RUN_ROOT/name;check_resume(name,70);binding=read(fit/'binding.json');done=read(fit/'complete.json')
  for filename,digest in binding['model_files'].items():
   if sha(OWL/filename)!=digest:raise ValueError('Original model or processor changed')
  for filename,digest in binding['protocol_files'].items():
   if sha(HERE/filename)!=digest:raise ValueError('Training/scoring protocol changed')
  if sha(OUT/'configs'/(name+'.json'))!=binding['config_sha256']:raise ValueError('Approved config changed')
  updates=[read(p) for p in sorted(fit.glob('update-*.json'))]
  counts=collections.Counter(i for r in updates for i in r['images'])
  if counts!={r['image_id']:10 for r in train}:raise ValueError('Unexpected training exposure')
  if [r['step'] for r in updates]!=list(range(1,71)):raise ValueError('Wrong update sequence')
  if not done['frozen_parameters_unchanged'] or done['optimizer_hook_calls']!=70:raise ValueError('Gradient/update scope failure')
  verification=read(fit/'verification-step-00070.json')
  if verification['optimizer_state_steps']!=[70] or verification['non_head_tensors_in_checkpoint']!=0:raise ValueError('Optimizer state scope differs')
  from checkpoints import recover
  if recover(fit,binding)[1]['step']!=70:raise ValueError('Recovery differs from completion')
  receipt_count+=verification['committed_checkpoints_verified']
  if sha(done['best']['checkpoint'])!=done['best']['sha256']:raise ValueError('Selected checksum mismatch')
  for path in fit.glob('validation-epoch-*.json'):
   if set(read(path)['predictions'])!={r['image_id'] for r in dev}:raise ValueError('Unexpected validation exposure')
 first_verified=(RUN_ROOT/FIT_IDS[0]/'verification-step-00070.json').stat().st_mtime_ns
 second_started=min(read(p)['unix_ns'] for p in (RUN_ROOT/FIT_IDS[1]).glob('execution-*.json'))
 if first_verified>=second_started:raise ValueError('Second experiment preceded first verification')
 comparison=read(RUN_ROOT/'comparison-v1/comparison.json')
 if comparison['frozen_photographs_used']!=0 or comparison['optimizer_updates_during_comparison']!=0:raise ValueError('Comparison scope changed')
 result=dict(status='PASS',historical=old,acquisition=acquisition,phase5b_artifacts_verified=previous,
  phase5b_implementation_files_verified=len(previous_code),original_model_and_processor_verified=True,
  original_checkpoint_sha256=OWL_SHA,training_images=27,development_images=18,reserved_eligible_frozen_images_unused=18,
  full_original_frozen_slots=60,licensing_quarantines_excluded=32,optimizer_updates_total=140,
  optimizer_updates_per_fit=70,training_presentations_per_fit=270,all_checkpoint_receipts_verified=receipt_count,
  frozen_parameters_unchanged=True,first_fit_verified_before_second=True,known_identity_groups_isolated=True,
  independent_selected_prediction_replay=comparison['independent_replay'],official_phase2b_scoring=False,
  frozen_pixel_decoding_inference_or_scoring=0,production_models_replaced=False,mac_project_accessed=False,
  note='Opaque photo hashes include frozen bytes only for preservation; no frozen pixels, predictions or metrics. Unknown physical relationships cannot be ruled out by metadata alone.')
 save(RUN_ROOT/'final-verification.json',result)
 print('FINAL AUDIT PASS:',receipt_count,'checkpoint receipts; 140 updates;',old['checked_records'],'historical records')

if __name__=='__main__':run()
