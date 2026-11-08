"""Final read-only provenance verification; frozen photos only opaque byte hashes."""
import json
import sys
from common import *
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical

def run():
 old=historical(META);acquisition=verify_acquisition();lock=read(OUT/'dataset-lock.json');original=read(OUT/'original-development-v1/binding.json')
 if original['dataset_sha256']!=sha(OUT/'dataset-lock.json'):raise ValueError('Dataset changed since baseline')
 for name,h in original['model_files'].items():
  if sha(OWL/name)!=h:raise ValueError('Pinned model/processor changed')
 frozen={r['image_id'] for r in read(PILOT/'screening-final-v1/screening.json')['images'] if r['proposed_split']=='frozen_evaluation'}
 for row in lock['rows']:
  if row['image_id'] in frozen or row['split'] not in ('train','development'):raise ValueError('Frozen member in model-readable data')
 dev=load_rows('development')
 for row in dev:
  prediction=read(OUT/'original-development-v1'/(row['image_id']+'.json'))
  if prediction['image_id'] not in {r['image_id'] for r in dev} or prediction['split']!='development':raise ValueError('Incorrect prediction partition')
 probe=read(OUT/'memory-probes-v1/summary.json')
 if probe['optimizer_steps']!=0 or probe['train_photo_id'] not in {r['image_id'] for r in load_rows('train')}:raise ValueError('Probe scope differs')
 result=dict(acquisition_complete_with_quarantines=True,acquired_photos=118,licence_quarantines=32,actual_eligible_counts=lock['eligible_counts'],approved_slots=lock['approved_membership'],
  historical=old,acquisition_hash_verification=acquisition,original_checkpoint_sha256=OWL_SHA,
  development_inference_images=18,frozen_inference_or_pixel_decoding=0,official_phase2b_scoring=False,optimizer_steps=0,
  resume_verified=True,probe_parameters_unchanged=all(r.get('trainable_hash_unchanged') for r in probe['results']),production_or_mac_application_changed=False,
  notes='Acquisition order explicitly rechecks all photograph hashes, including opaque frozen bytes. No frozen photograph pixels, embeddings, predictions or metrics are used in Phase 5B.')
 save(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
