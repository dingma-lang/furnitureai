"""Pin experiment protocol and preservation records before any inference."""
import collections as C
import subprocess
from core import *
from data import exposed_rows,final_rows_metadata
from fusion import views

def snapshot():
 result={}
 for name in ('data','models','reports','src','tools','scripts','tests','tests_photo_simulator','tests_inpainting_eval'):
  for p in (ROOT/name).rglob('*'):
   if not p.is_file() or p.is_relative_to(OUT) or p.is_relative_to(HERE) or p.is_relative_to(PUBLIC) or '__pycache__' in p.parts:continue
   s=p.stat();result[str(p.relative_to(ROOT))]=[s.st_size,s.st_mtime_ns]
 return result

def run():
 starting=subprocess.check_output(['git','rev-parse','phase9-baseline-20261118^{commit}'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
 if starting!=remote:raise ValueError('Remote backup changed; inspect before proceeding')
 registry=read(REGISTRY);pins=dict(PINS);pins[str(Path(registry['checkpoint']).relative_to(ROOT))]=registry['sha256'];weights={}
 for path,expected in pins.items():
  p=ROOT/path;actual=sha(p)
  if actual!=expected:raise ValueError('Checkpoint changed '+path)
  label='original' if path.startswith('models/owlv2') else 'siglip' if path.startswith('models/siglip') else 'classifier' if path.startswith('data/abo') else 'adapted'
  weights[label]=dict(path=path,sha256=actual,bytes=p.stat().st_size,independent_backup_verified=False,
   backup_status='Not in Git; no independent off-workstation backup was verified. No large copies made.')
 save(OUT/'model-inventory.json',dict(weights=weights,code_backup_commit=starting,GitHub_main_verified=remote,all_model_weights_unchanged_before_inference=True))
 old=exposed_rows();fresh=final_rows_metadata();sealed(OUT/'fresh-metadata-lock.json',dict(rows=fresh,images=36,pixels_not_opened=True))
 observed=[]
 # Check known model-evaluation evidence, excluding screening/audit metadata.
 for base in (ROOT/'data/open-images/experiments',ROOT/'reports/phase4-room-comparison-v1',ROOT/'reports/m6-consumer-developer-integration/bounded-comparison'):
  for p in base.rglob('*.json'):
   if p.is_relative_to(OUT) or not any(s in p.name for s in ('prediction','validation-epoch','dataset-lock','comparison','input-lock')):continue
   if p.stat().st_size>30*2**20:continue
   text=p.read_text()
   hits=[r['image_id'] for r in fresh if '"'+r['image_id']+'"' in text]
   if hits:observed.append(dict(path=str(p.relative_to(ROOT)),IDs=hits))
 if observed:raise ValueError('Fresh development IDs appear in prior model-use evidence: '+str(observed))
 save(OUT/'freshness-audit.json',dict(prior_model_use_hits=observed,existing_detector_experiments=['phase5b-v1','phase5c-v1'],
  fresh_exposed_ID_and_hash_overlap=False,scope='No recorded use in inspected local model evidence; cannot prove absence of unrecorded external use',
  fresh_pixels_not_opened=True,frozen_photo_pixels_or_bytes_not_read=True))
 protocol=dict(starting_commit=starting,branch='experiment/phase9-multiview-detection',baseline_tag='phase9-baseline-20261118',
  goal='Higher native-source precision at comparable recall; not public-benchmark or independently verified accuracy',
  exposed_images=18,exposed_native_boxes=41,fresh_images=36,fresh_native_boxes=126,final_comparisons_max=1,
  per_image_detector_views=views(),models=['original','adapted'],max_development_detector_passes=288,
  method_recipes=['baseline','scale_nms','scale_agreement','scale_weighted','tiled_guarded','flip_agreement'],
  additional_arms=['original_adapted_agreement','original_siglip_consistency','adapted_siglip_consistency'],max_selection_arms=15,
  global_confidence=.15,ap_floor=.001,nms=.5,consensus_iou=.55,scale_consensus_minimum=2,
  siglip_contradiction_margin=.10,SigLIP_crops_max=10800,unknowns_or_bad_geometry_kept_advisory=True,
  selection_rule=dict(reference='adapted-baseline',minimum_precision_gain=.03,maximum_recall_loss_objects=1,maximum_ap75_loss=.02,
   no_TP_loss_categories=['sofa','chair','lamp'],ranking=['precision','ap75','f1','name'],no_qualifier='retain adapted baseline; no robust winner'),
  max_invocation_seconds=1800,allocator_fraction=.75,soft_peak_allocated_stop_GiB=6.5,new_storage_cap_bytes=536870912,
  original_default_unchanged=True,no_training_no_downloads_no_dependencies=True,
  partial_scope='Unknown classes and nonexhaustive unmatched proposals unscored, not automatic FPs; final positive-only subsets lower FP observability',
  genuine_views='No verified grouped same-room capture set identified; interface only pending independently verified identities/registration')
 sealed(OUT/'protocol.json',protocol);save(OUT/'preservation-before.json',dict(files=snapshot(),strong_hashes={p:sha(ROOT/p) for p in [str(OLD_LOCK.relative_to(ROOT)),str(READINESS.relative_to(ROOT)),str(REGISTRY.relative_to(ROOT))]},
  frozen_photo_preservation='stat only; no reads/decodes',backup_limitations='GitHub backs up tracked code only; ignored data/checkpoints have no independently verified backup'))
 print('PREFLIGHT PASS; pinned four checkpoint hashes;18 exposed/36 final metadata-only;15 recipe arms; zero frozen photo reads',flush=True)

if __name__=='__main__':run()
