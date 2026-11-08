"""Read-only preflight against all historical Phase 1–5 artifacts."""
import ast
import inspect
import sys
from common import *
from control import RUN_ROOT,FIT_IDS,ledger_init
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical

def source_function(path,name):
 s=path.read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);return ast.get_source_segment(s,node)

def run():
 if (RUN_ROOT/'preflight.json').exists():raise FileExistsError('Preflight already committed')
 prior=verify_sums(OUT,'ARTIFACT-SHA256SUMS');old=historical(META);acq=verify_acquisition()
 if sha(OUT/'dataset-lock.json')!='7d160c5419e791c62377d8758c3e3d585a0a700324dfcd01fecf0e44b8d73bf9':raise ValueError('Approved dataset lock changed')
 for name in ('metrics.py','baseline.py','common.py'):
  if sha(HERE/name)!=sha(ROOT/'tools/phase5b-v1'/name):raise ValueError('Primary protocol changed')
 if source_function(HERE/'adaptation.py','detection_loss')!=source_function(ROOT/'tools/phase5b-v1/adaptation.py','detection_loss'):raise ValueError('Prepared loss changed')
 lock=read(OUT/'dataset-lock.json');screen=read(PILOT/'screening-final-v1/screening.json');rights=read(PILOT/'licensing-audit.json');rights_by={r['image_id']:r for r in rights['records']};byid={r['image_id']:r for r in screen['images']}
 train=load_rows('train');dev=load_rows('development');reserved={r['image_id'] for r in lock['frozen_references']};quarantined={r['image_id'] for r in rights['records'] if r['status']!='verified_individual_current_source'}
 for rows,part in ((train,'train'),(dev,'development')):
  for r in rows:
   image_path(r,part)
   if r['image_id'] in reserved|quarantined or byid[r['image_id']]['eligibility_status']!='provisional_candidate':raise ValueError('Excluded image entered approved pool')
   if rights_by[r['image_id']]['status']!='verified_individual_current_source':raise ValueError('Missing photo rights')
 if (len(train),len(dev),len(reserved),len(quarantined))!=(27,18,18,32):raise ValueError('Counts changed')
 for key in ('image_id','sha256','identity_group','source_identity'):
  a={r[key] for r in train};b={r[key] for r in dev};c={r.get(key) for r in lock['frozen_references'] if r.get(key)}
  if a&b or a&c or b&c:raise ValueError('Known identity leakage')
 values=dict(source_phase5b_artifacts_verified=prior,historical=old,acquisition=acq,original_model_sha256=OWL_SHA,dataset_sha256=sha(OUT/'dataset-lock.json'),
   config_sha256={name:sha(OUT/'configs'/(name+'.json')) for name in FIT_IDS},training_images=27,development_images=18,reserved_images_unused=18,licensing_quarantines_excluded=32,
   primary_evaluation_unchanged=True,loss_unchanged=True,source_groups_disjoint=True,
   gradient_scope='class_head and box_head only; 1,579,526 parameters',unknown_class_gradients_masked=True,objectness_frozen=True,
   annotation_caveats=['Desk labels include booth/side furniture and differ from ordinary furniture semantics','Cabinetry may be an entire kitchen run while model detects individual doors/units','Published explicit class negatives remain source-native weak supervision, not independent manual verification; unknown/ignored class regions are masked and no global objectness background is trained'],
   separate_sensitivity='Report results excluding Desk, Cabinetry, or both after primary comparison; never use sensitivity metrics for selection or change primary labels/GT')
 save(RUN_ROOT/'preflight.json',values)
 save(RUN_ROOT/'authorization.json',dict(detector_updates_authorized=True,source='Explicit user approval: Phase 5C controlled OWLv2 head-only fine-tuning',fit_ids=list(FIT_IDS),max_calls_per_fit=70,max_calls_total=140,
   training_images=27,development_images=18,frozen_access_authorized=False,backbone_or_lora_authorized=False,downloads_authorized=False))
 ledger_init();print('PREFLIGHT PASS',len(train),len(dev),'historical checks',old['checked_records'],'phase5b files',prior,flush=True)

if __name__=='__main__':run()
