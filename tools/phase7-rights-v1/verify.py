"""Independent audit-byte, identity and historical preservation checks; no source requests."""
import importlib.util
import sys
from audit_core import *
from transport import allowed_request,provider

def seal(base):
 n=0
 for line in (base/'ARTIFACT-SHA256SUMS').read_text().splitlines():
  expected,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=expected:raise ValueError('Historical artifact seal changed '+str(path))
  n+=1
 for relative,expected in read(base/'implementation-sha256.json').items():
  if sha(ROOT/relative)!=expected:raise ValueError('Historical implementation changed '+relative)
 return n

def run():
 m=manifest();byid={r['coco_image_id']:r for r in m['images']};audit=read(OUT/'rights-audit.json');net=read(OUT/'network-ledger.json');summary=read(OUT/'summary.json');proposed=read(OUT/'proposed-image-acquisition-v1.json')
 if len(audit['records'])!=400 or {r['coco_image_id'] for r in audit['records']}!=set(byid):raise ValueError('Audit ID inventory changed')
 if any(r['provisional_partition']!=byid[r['coco_image_id']]['proposed_partition'] or r['evidence_attempts']>1 for r in audit['records']):raise ValueError('Assignment/attempt mismatch')
 if any(c['attempt']!=1 for c in net['candidates'].values()):raise ValueError('Attempt limit')
 actual=sum(r['actual_body_bytes'] for r in net['requests'])
 if actual!=net['actual_body_bytes'] or actual!=audit['actual_body_bytes'] or not actual<=net['conservative_charge']<=CAP:raise ValueError('Transfer ledger mismatch')
 blocked={};successful_documents=0
 for r in net['requests']:
  row=byid[r['coco_image_id']]
  if not allowed_request(row,r['url'],r['method']):raise ValueError('Unapproved request or photograph-body request')
  host=provider(r['url'])
  if host in blocked:raise ValueError('Request followed host rate-limit circuit opening')
  if r.get('status')==429:blocked[host]=r['checked_utc']
  if r.get('file'):
   p=OUT/r['file']
   if p.stat().st_size!=r['actual_body_bytes']:raise ValueError('Saved source response length mismatch')
   if r.get('sha256') and sha(p)!=r['sha256']:raise ValueError('Source evidence hash mismatch')
  successful_documents+=r['method']=='GET' and r.get('status')==200 and r.get('complete')
 decisions={r['coco_image_id']:r for r in read(OUT/'identity-audit.json')['records']}
 audit_by_id={r['coco_image_id']:r for r in audit['records']}
 for row in proposed['images']:
  iid=row['coco_image_id'];original=byid[iid];result=audit_by_id[iid]
  if not decisions[iid]['eligible_for_acquisition_proposal'] or result['licensing_status']!=STATUSES[0]:raise ValueError('Unverified/ineligible proposed photo')
  if row['original_lvis_furniture_annotations']!=original['original_lvis_furniture_annotations'] or row['source_image_scope']!=original['source_image_scope'] or row['proposed_partition']!=original['proposed_partition']:raise ValueError('Original annotation/split changed')
  proof=row['verified_photo_rights']
  if sha(ROOT/proof['rights_record'])!=proof['rights_record_sha256'] or sha(ROOT/proof['evidence_file'])!=proof['evidence_sha256']:raise ValueError('Proposed source evidence hash changed')
 if sum(r['remote_object']['bytes'] for r in proposed['images'])!=proposed['expected_photo_body_bytes']:raise ValueError('Expected transfer mismatch')
 registration=read(OUT/'existing-ten-photo-registration-assessment.json')
 if registration['photos']!=10 or registration['native_annotations']!=33 or registration['annotations_merged_into_training']!=0:raise ValueError('Registration scope changed')
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
 spec=importlib.util.spec_from_file_location('phase7_rights_preservation',ROOT/'tools/phase7-v1/preserve.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);historical=mod.verify()
 seals={'phase7_preparation':seal(ROOT/'reports/phase7-data-strategy-v1'),'phase7_lvis_metadata':seal(BASE)}
 space();commit(OUT/'integrity-verification.json',dict(status='PASS',candidates=400,original_core_boxes_masks=905,original_all_native_annotations=1080,
  attempted_candidates=summary['attempted_candidates'],successful_source_document_GETs=successful_documents,actual_body_bytes=actual,conservative_charge=net['conservative_charge'],
  max_attempts_per_candidate=1,no_requests_after_host429=True,rate_limited_hosts=list(blocked),photo_bodies_downloaded=0,proposed_eligible_images=proposed['image_count'],
  original_partitions_and_annotations_unchanged=True,historical=historical,seals=seals,allocated_bytes_at_verification=allocated(),
  frozen_image_pixels_decoded_or_scored=0,perceptual_duplicate_screening=False,model_training_inference_or_rendering=False,mac_project_accessed=False))
 print('RIGHTS AND HISTORICAL PRESERVATION PASS',flush=True)

if __name__=='__main__':run()
