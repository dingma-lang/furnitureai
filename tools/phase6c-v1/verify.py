"""Read-only rights/proposal replay and preservation; no network or image decode."""
import collections as C
import gzip
import io
import sys
from common import *
from evidence import classify
from integrity import build_graph

def sums(base,name):
 n=0
 for line in (base/name).read_text().splitlines():
  expected,rel=line.split('  ',1);p=(base/rel).resolve()
  if not p.is_relative_to(base.resolve()) or sha(p)!=expected:raise ValueError('Historical artifact changed '+str(p))
  n+=1
 return n

def run():
 m=manifest();rows={r['image_id']:r for r in m['images']};records=read(OUT/'rights-audit-v3.json')['records'];byid={r['image_id']:r for r in records};plan=read(OUT/'proposed-image-acquisition-v1.json');state=read(OUT/'network-ledger.json')
 assert len(records)==len(byid)==400 and set(byid)==set(rows)
 assert C.Counter(r['proposed_split'] for r in records)==dict(train=320,development=80)
 assert all(r['licensing_status'] in STATUSES and r['proposed_split']==rows[r['image_id']]['proposed_split'] for r in records)
 assert state['actual_body_bytes']==sum(r['actual_bytes'] for r in state['requests'])==12536007
 assert state['conservative_charge']<=CAP
 docs=0;head_photos=0
 for q in state['requests']:
  row=rows[q['image_id']];photo=row['source_identity'].removeprefix('flickr-photo:')
  if q['photo_object']:
   assert q['method']=='HEAD' and q['url']==row['proposed_secure_image_url'] and q['actual_bytes']==0;head_photos+=1
  else:assert allowed_page(q['url'],photo)
  assert q['tls_certificate_chain_verified'] and q['complete']
  if q['method']=='GET':
   assert not q['photo_object'] and q['status']==200 and q['attempt']<=2
   assert sha(OUT/q['file'])==q['sha256'] and (OUT/q['file']).stat().st_size==q['actual_bytes']<=PAGE
   docs+=1
 assert head_photos==400 and docs==95
 retries=[q for q in state['requests'] if q['attempt']>1]
 assert len(retries)==1 and retries[0]['status']==429 and retries[0]['method']=='HEAD'
 earlier=next(q for q in state['requests'] if q['url']==retries[0]['url'] and q['method']=='HEAD' and q['attempt']==1)
 assert (dt.datetime.fromisoformat(retries[0]['checked_utc'])-dt.datetime.fromisoformat(earlier['checked_utc'])).total_seconds()>=120
 # The one persistent429 ends all subsequent source requests in the retry pass.
 assert not any(q['checked_utc']>retries[0]['checked_utc'] and not q['photo_object'] for q in state['requests'])
 replay=0
 for r in records:
  gets=[e for e in r['evidence_requests'] if e['method']=='GET' and e.get('file')]
  if not gets:continue
  e=gets[-1];q=next(q for q in state['requests'] if q.get('file')==e['file']);data=(OUT/e['file']).read_bytes()
  if q['headers'].get('Content-Encoding')=='gzip':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(DECODED+1)
  assert len(data)<=DECODED
  rec=classify(data.decode('utf-8'),rows[r['image_id']],e['url']);assert rec['licensing_status']==r['licensing_status']
  for key in ('creator','creator_url','license_url','title','source_url'):
   if key in rec:assert r.get(key)==rec[key]
  replay+=1
 assert replay==95
 uf,protected,source_scope,_=build_graph(records);groups=C.defaultdict(set)
 for r in records:groups[uf.find(r['source_identity'])].add(r['proposed_split'])
 kept=plan['images'];assert len(kept)==72
 assert C.Counter(r['proposed_split'] for r in kept)==dict(train=60,development=12)
 parts=C.defaultdict(set)
 for row in kept:
  raw=byid[row['image_id']];original=rows[row['image_id']];root=uf.find(row['source_identity'])
  assert raw['licensing_status']==STATUSES[0] and raw['image_object_status']=='available'
  assert root not in protected and len(groups[root])==1
  assert row['proposed_split']==original['proposed_split'] and row['original_furniture_annotations']==original['original_furniture_annotations']
  assert row['remote_object']==raw['image_object'] and row['image_sha256'] is None and not row['visual_quality_verified'] and not row['perceptual_duplicates_checked']
  assert sha(ROOT/row['rights_record'])==row['rights_record_sha256'] and sha(ROOT/row['source_document'])==row['source_document_sha256']
  assert row['title'] if 'title' in row else row['source_title']
  parts[row['proposed_split']].add(root)
 assert not parts['train']&parts['development']
 assert len(parts['train']|parts['development'])==66
 assert sum(r['remote_object']['bytes'] for r in kept)==10853122
 # No photographs or inference; old heldout paths are stat-only where required.
 sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
 from verify_metadata import historical
 old=historical(ROOT/'data/open-images/metadata/phase5-v1');checks={}
 locations=[('openimages_metadata',ROOT/'data/open-images/metadata/phase5-v1'),('phase5_photos',ROOT/'data/open-images/image-pilots/phase5-v1'),('phase5b',ROOT/'data/open-images/experiments/phase5b-v1'),('phase5c',ROOT/'data/open-images/experiments/phase5c-v1'),('phase6a_preparation',ROOT/'data/open-images/preparation/phase6a-v1'),('phase6a_photos',ROOT/'data/open-images/image-pilots/phase6a-v1'),('phase6b_metadata',BASE)]
 for label,base in locations:checks[label]=sums(base,'ARTIFACT-SHA256SUMS')
 for label,base in locations[2:]:
  impl=base/'implementation-sha256.json'
  if impl.exists():
   for path,digest in read(impl).items():assert sha(ROOT/path)==digest
 owl=sha(ROOT/'models/owlv2-original-baseline-20261103/model.safetensors');assert owl=='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
 sig=ROOT/'models/siglip2-base-patch16-224-phase2a-v1';sig_files=sums(sig,'SHA256SUMS');sig_hash=sha(sig/'model.safetensors');assert sig_hash=='612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'
 checkpoint=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json');assert sha(checkpoint['checkpoint'])==checkpoint['sha256']=='d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6'
 assert allocated()+allocated(HERE)+16*2**20<DISK
 commit(OUT/'verification.json',dict(status='PASS',candidate_source_ids_verified=400,original_partition_slots=dict(train=320,development=80),licence_documents_replayed=95,photo_object_heads=400,photo_body_gets=0,actual_body_bytes=12536007,
  capped_retry_after_cooldown=1,rate_limit_retry_persisted_and_stopped=True,known_source_groups_eligible=66,photo_proposal_images=72,proposal_partition_counts=dict(train=60,development=12),expected_photo_bytes=10853122,
  historical=old,old_artifact_ledgers_checked=checks,siglip_model_processor_files_verified=sig_files,model_hashes=dict(owlv2=owl,siglip2=sig_hash,phase5c=checkpoint['sha256']),
  original_COCO_annotations_unchanged=True,old_frozen_memberships_unchanged=True,perceptual_duplicate_or_visual_checks_performed=False,new_photograph_bodies_downloaded=0,model_inference_or_training=0,mac_project_accessed=False))
 print('VERIFY PASS:400 source IDs,95 document replays,72 eligible images; historical hashes intact',flush=True)

if __name__=='__main__':run()
