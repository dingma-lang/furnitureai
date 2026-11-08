"""Phase 5B isolated, offline development benchmark. No frozen-photo loading."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/open-images/experiments/phase5b-v1'
PILOT=ROOT/'data/open-images/image-pilots/phase5-v1'
META=ROOT/'data/open-images/metadata/phase5-v1'
OWL=ROOT/'models/owlv2-original-baseline-20261103'
OWL_SHA='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
MIDS={'sofa':'/m/02crq1','chair':'/m/01mzpv','table':'/m/04bcr3','bed':'/m/03ssj5','desk':'/m/01y9k5','cabinet':'/m/01s105','lamp':'/m/0dtln'}
SETTINGS=dict(prompts=['a photo of a '+c for c in CLASSES],ap_score_floor=.001,operating_score=.15,nms_iou=.5,max_detections=300,precision='float32',ap='101-point interpolated',iou_thresholds=[.5,.75],seed=20261105)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def source_code():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*.py'))}
def verify_sums(base,name):
 count=0
 for line in (base/name).read_text().splitlines():
  expected,rel=line.split('  ',1);p=(base/rel).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=expected:raise ValueError('Historical artifact changed '+str(p))
  count+=1
 return count

def verify_acquisition():
 result=dict(metadata=verify_sums(META,'ARTIFACT-SHA256SUMS'),photographs_and_evidence=verify_sums(PILOT,'ARTIFACT-SHA256SUMS'),photo_hashes=verify_sums(PILOT,'PHOTO-SHA256SUMS'))
 if sha(OWL/'model.safetensors')!=OWL_SHA:raise ValueError('Original OWLv2 changed')
 return result

def lock_dataset():
 if (OUT/'dataset-lock.json').exists():raise FileExistsError('Dataset already locked; use existing immutable lock')
 integrity=verify_acquisition();screen=read(PILOT/'screening-final-v1/screening.json');proposal=read(META/'pilot-proposal-v4/proposed-image-acquisition.json');lookup={r['image_id']:r for r in proposal['images']}
 owners={k:{} for k in ('sha256','identity_group','source_identity')};partition_counts={p:0 for p in ('train','development','frozen_evaluation')};eligible_counts={p:0 for p in partition_counts};rows=[];reserved=[]
 for r in screen['images']:
  orig=lookup[r['image_id']];part=r['proposed_split'];partition_counts[part]+=1
  if orig['proposed_split']!=part or orig['annotations']!=r['original_annotations']:raise ValueError('Membership/box discrepancy')
  for k in owners:
   value=r.get(k)
   if not value:continue
   if value in owners[k] and owners[k][value]!=part:raise ValueError('Known partition leakage '+k)
   owners[k][value]=part
  if r['eligibility_status']!='provisional_candidate':continue
  eligible_counts[part]+=1
  if part=='frozen_evaluation':
   reserved.append(dict(image_id=r['image_id'],proposed_split=part,sha256=r['sha256'],identity_group=r['identity_group'],file=r['file']));continue
  annotations=r['original_annotations'];coverage={};reasons={}
  for c,mid in MIDS.items():
   val=r['native_image_level_labels'].get(mid)
   has_box=any(g['label']==c for g in annotations)
   coverage[c]=(val==0 or (val==1 and has_box))
   reasons[c]='published_negative' if val==0 else ('published_positive_with_native_boxes' if val==1 and has_box else 'unannotated_or_positive_without_native_box')
   if has_box and val!=1:raise ValueError('Native box contradicts positive image label')
  rows.append(dict(image_id=r['image_id'],split=part,file=r['file'],sha256=r['sha256'],width=r['width'],height=r['height'],identity_group=r['identity_group'],source_identity=r['source_identity'],
       annotations=annotations,coverage=coverage,coverage_provenance=reasons,native_image_level_labels=r['native_image_level_labels'],licence_evidence=r['licence_evidence'],original_source_boxes=r['all_original_source_rows'],
       eligibility='provisional_published_native_annotations',human_verified=False,scope_note='Agreement with published native annotations; no human application-label verification'))
 lock=dict(schema='phase5b-development-only-v1',source_manifest_sha256=sha(META/'pilot-proposal-v4/proposed-image-acquisition.json'),source_screening_sha256=sha(PILOT/'screening-final-v1/screening.json'),approved_membership=partition_counts,eligible_counts=eligible_counts,
     rows=rows,frozen_references=reserved,settings=SETTINGS,mapping=MIDS,integrity=integrity,frozen_pixels_decoded=False,downloads=False,official_phase2b_used=False)
 save(OUT/'dataset-lock.json',lock);return lock

def load_rows(part):
 if part not in ('train','development'):raise ValueError('Frozen evaluation access prohibited in Phase 5B')
 lock=read(OUT/'dataset-lock.json')
 if lock['settings']!=SETTINGS:raise ValueError('Settings changed after lock')
 return [r for r in lock['rows'] if r['split']==part]

def image_path(row,part):
 if part not in ('train','development') or row['split']!=part:raise ValueError('Partition access denied')
 # Validate against locked whitelist, not a caller-provided tag or filename.
 locked={r['image_id']:r for r in load_rows(part)}
 if row!=locked.get(row['image_id']):raise ValueError('Not an exact locked member')
 p=(PILOT/row['file']).resolve()
 if not p.is_relative_to((PILOT/'photos').resolve()) or sha(p)!=row['sha256']:raise ValueError('Image path/hash mismatch')
 return p
