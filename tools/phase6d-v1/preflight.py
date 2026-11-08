"""Reverify immutable source evidence in its original isolated parser namespace."""
import collections as C
import datetime as dt
import gzip
import io
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase6c-v1'))
from common import read,sha,commit,now,PIN as CANDIDATE_PIN
from evidence import classify

def run():
 rights=ROOT/'data/coco/rights-audits/phase6c-v1';path=rights/'proposed-image-acquisition-v1.json';out=ROOT/'data/coco/image-pilots/phase6d-v1'
 if out.exists():raise FileExistsError('Destination exists; preflight must not modify it')
 assert sha(path)=='bc6722b2b603e88e437961c2a8c7b8f6d5dfe16cd07141ee8b478d5ef8390725'
 m=read(path);identity={r['image_id']:r for r in read(rights/'identity-audit.json')['records']};records={r['image_id']:r for r in read(rights/'rights-audit-v3.json')['records']};proposed={r['image_id']:r for r in m['images']};base=read(ROOT/'data/coco/metadata/phase6b-v1/proposed-image-candidates-v1.json');base_rows={r['image_id']:r for r in base['images']}
 assert len(proposed)==72 and C.Counter(r['proposed_split'] for r in proposed.values())==dict(train=60,development=12)
 blocked={r['image_id'] for r in identity.values() if r['identity_flags']};assert len(blocked)==20 and not blocked&proposed.keys()
 bad={r['image_id'] for r in records.values() if r['licensing_status']!='verified_commercially_compatible'};assert len(bad)==308 and not bad&proposed.keys()
 groups={};ages=[]
 for r in proposed.values():
  source=records[r['image_id']];d=identity[r['image_id']]
  assert d['eligible_for_image_acquisition_proposal'] and not d['identity_flags'] and r['proposed_split']==base_rows[r['image_id']]['proposed_split']
  assert r['identity_group']==r['known_source_group']==d['known_source_group']
  if r['identity_group'] in groups:assert groups[r['identity_group']]==r['proposed_split']
  groups[r['identity_group']]=r['proposed_split']
  assert sha(ROOT/r['rights_record'])==r['rights_record_sha256'] and sha(ROOT/r['source_document'])==r['source_document_sha256']
  data=(ROOT/r['source_document']).read_bytes()
  if data[:2]==b'\x1f\x8b':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(4194305)
  assert len(data)<=4194304
  evidence=classify(data.decode('utf-8'),base_rows[r['image_id']],r['verified_source_page'])
  assert evidence['licensing_status']=='verified_commercially_compatible' and evidence['creator']==r['creator'] and evidence['license_url']==r['current_license_url'] and evidence['title']==r['source_title']
  age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(r['rights_checked_utc'])).total_seconds()/86400;assert 0<=age<=7;ages.append(age)
  assert r['original_furniture_annotations']==base_rows[r['image_id']]['original_furniture_annotations']
 assert len(groups)==66
 expected={'owlv2':('models/owlv2-original-baseline-20261103/model.safetensors','e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'),'siglip2':('models/siglip2-base-patch16-224-phase2a-v1/model.safetensors','612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b')}
 model_hashes={}
 for name,(rel,digest) in expected.items():assert sha(ROOT/rel)==digest;model_hashes[name]=digest
 checkpoint=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json');assert sha(checkpoint['checkpoint'])==checkpoint['sha256']=='d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6';model_hashes['phase5c']=checkpoint['sha256']
 out.mkdir(parents=True)
 commit(out/'authorization.json',dict(image_acquisition_approved=True,user_order='Explicit user approval: Phase6D COCO photograph acquisition and screening only',manifest_sha256=sha(path),recorded_utc=now(),image_ids=sorted(proposed),train=60,development=12,max_transfer=100000000,max_storage=536870912,per_image_max=3145728,max_attempts=2,model_operations_authorized=False))
 commit(out/'preflight.json',dict(status='PASS',rights_documents_reverified=72,max_rights_age_days=max(ages),original_overlap_exclusions_still_excluded=20,all_other_unverified_or_incompatible_excluded=308,known_source_groups_isolated=66,model_hashes=model_hashes,original_metadata_annotations_preserved=True))
 (out/'approved-manifest.json').write_bytes(path.read_bytes())
 print('PREFLIGHT PASS:72 rights documents;66 known groups;328 exclusions preserved; models unchanged',flush=True)

if __name__=='__main__':run()
