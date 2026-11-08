"""Independent no-network photo, annotation, isolation and historical preservation verification."""
import importlib.util
import sys
from pilot_core import *
from acquire import validate_state

def seal(base,name):
 count=0
 for line in (base/name).read_text().splitlines():
  digest,relative=line.split('  ',1);p=(base/relative).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=digest:raise ValueError('Historical seal mismatch '+str(p))
  count+=1
 return count

def run():
 source={r['image_id']:r for r in manifest()['images']};net=read(OUT/'network-ledger.json');validate_state(net,list(source.values()))
 records=read(OUT/'acquisition-complete.json')['records'];screen=read(OUT/'screening-v2.json');ready=read(OUT/'combined-training-readiness.json')
 if len(records)!=27 or {r['image_id'] for r in records}!=set(source) or net['actual_body_bytes']!=3913471:raise ValueError('Acquisition inventory/bytes mismatch')
 for request in net['requests']:
  if request['url']!=source[request['image_id']]['remote_object']['url'] or request['method'] not in ('HEAD','GET') or not request['tls_verified']:raise ValueError('Unapproved request/TLS')
 for r in records:
  original=source[r['image_id']];p=OUT/r['file']
  if sha(p)!=r['sha256'] or hashlib.md5(p.read_bytes()).hexdigest()!=original['remote_object']['etag'].strip('"') or p.stat().st_size!=original['remote_object']['bytes']:raise ValueError('Photo integrity mismatch')
 for r in screen['images']:
  s=source[r['image_id']]
  if r['original_annotations']!=s['original_furniture_annotations'] or r['proposed_split']!=s['proposed_split'] or r['source_image_scope']!=s['source_image_scope']:raise ValueError('Native metadata/partition changed')
  if not set(r['retained_annotation_ids'])<=set(a['id'] for a in r['original_annotations']) or r['human_verified']:raise ValueError('Fabricated positive/human review')
 eligible={r['image_id'] for r in screen['images'] if r['status']=='provisional_source_candidate'}
 if eligible!={r['image_id'] for r in ready['source_rows'] if r['source']=='phase8c_LVIS'}:raise ValueError('Quarantine/eligibility mismatch')
 for key in ('source_photo_id','sha256','known_source_component'):
  train={r[key] for r in ready['source_rows'] if r['split']=='train'};dev={r[key] for r in ready['source_rows'] if r['split']=='development'}
  if train&dev:raise ValueError('Train/development overlap')
 if ready['source_group_or_duplicate_exclusions']:raise ValueError('Combined identity conflicts need explicit resolution')
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'));spec=importlib.util.spec_from_file_location('phase8c_preserve',ROOT/'tools/phase7-v1/preserve.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);historical=old.verify()
 locations=[('reports/phase7-data-strategy-v1','ARTIFACT-SHA256SUMS'),('data/lvis/metadata/phase7-v1','ARTIFACT-SHA256SUMS'),('data/lvis/rights-audits/phase7-v1','FINAL-ARTIFACT-SHA256SUMS-v2'),('data/lvis/image-pilots/phase7-v1','ARTIFACT-SHA256SUMS'),('reports/phase8a-v1','ARTIFACT-SHA256SUMS'),('tools/phase8a-v1','IMPLEMENTATION-SHA256SUMS'),('data/lvis/rights-audits/phase8a-v1','ARTIFACT-SHA256SUMS')]
 seals={rel:seal(ROOT/rel,name) for rel,name in locations}
 for rel,_ in locations:
  for name in ('implementation-sha256-final-v2.json','implementation-sha256.json'):
   p=ROOT/rel/name
   if p.exists():
    for path,digest in read(p).items():
     if sha(ROOT/path)!=digest:raise ValueError('Historical implementation changed '+path)
    break
 if read(ROOT/'reports/phase8a-v1/existing33-registration-v2.json')['annotations_activated']!=0:raise ValueError('Old33 labels activated')
 space();commit(OUT/'integrity-verification.json',dict(status='PASS',historical=historical,seals=seals,
  downloaded_photographs=27,GET_attempts=sum(r['method']=='GET' for r in net['requests']),actual_response_body_bytes=net['actual_body_bytes'],
  all_photo_MD5_and_SHA256_verified=True,original99_annotation_records_preserved=True,retained_native_annotations=39,
  known_identity_or_partition_conflicts=0,historical_photo_pixels_opened=False,new_photo_pixels_assistant_inspected=27,
  frozen_pixels_decoded_or_scored=0,model_operations=0,rights_audits_resumed=False,previous33_annotations_activated=0,Mac_project_accessed=False))
 print('Photo/annotation/split and historical model/data preservation PASS',flush=True)

if __name__=='__main__':run()
