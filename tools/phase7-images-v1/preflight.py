"""Offline licence replay, exact annotation/provenance checks and preserved exclusions."""
import collections as C
import gzip
import io
import sys
from pilot_core import *

def run():
 if OUT.exists():raise FileExistsError('Destination exists; no overwrite')
 m=manifest();original=read(MANIFEST);parent={r['coco_image_id']:r for r in read(META/'proposed-photo-rights-audit-v1.json')['images']};decisions={r['coco_image_id']:r for r in read(RIGHTS/'identity-audit.json')['records']}
 ids={r['coco_image_id'] for r in m['images']};excluded={r['coco_image_id'] for r in decisions.values() if r['licensing_status']=='verified_commercially_compatible' and r['identity_flags']}
 if len(excluded)!=9 or excluded&ids or 154613 in ids:raise ValueError('Nine identity/frozen exclusions not preserved')
 pending={r['coco_image_id'] for r in read(RIGHTS/'unattempted-checkpoint-v1.json')['records']}
 if len(pending)!=296 or pending&ids:raise ValueError('Unattempted source subset overlaps acquisition')
 sys.path.insert(0,str(ROOT/'tools/phase7-rights-v1'))
 from source_evidence import classify
 evidence=[];groups={}
 for r in m['images']:
  iid=r['coco_image_id'];p=r['verified_photo_rights'];d=decisions[iid]
  if not d['eligible_for_acquisition_proposal'] or r['proposed_split']!=parent[iid]['proposed_partition'] or r['original_furniture_annotations']!=parent[iid]['original_lvis_furniture_annotations']:raise ValueError('Original annotation or assignment changed')
  if sha(ROOT/p['rights_record'])!=p['rights_record_sha256'] or sha(ROOT/p['evidence_file'])!=p['evidence_sha256']:raise ValueError('Recorded evidence changed')
  raw=(ROOT/p['evidence_file']).read_bytes()
  if raw[:2]==b'\x1f\x8b':
   with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:raw=f.read(4194305)
  if len(raw)>4194304:raise ValueError('Decoded source document bound')
  parsed=classify(raw.decode('utf-8'),parent[iid],p['evidence_url'])
  if parsed['licensing_status']!='verified_commercially_compatible' or parsed['creator']!=p['creator'] or parsed['license_url']!=p['license_url'] or parsed['title']!=p['title']:raise ValueError('Photo-specific grant contradiction')
  age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(p['evidence_checked_utc'])).total_seconds()/86400
  if not 0<=age<=7:raise ValueError('Saved evidence is stale; no live licensing retry permitted')
  if r['identity_group'] in groups and groups[r['identity_group']]!=r['proposed_split']:raise ValueError('Known-group leakage')
  groups[r['identity_group']]=r['proposed_split'];evidence.append(dict(coco_image_id=iid,creator=p['creator'],creator_url=p['creator_url'],photo_license=p['license_url'],source_url=p['source_url'],title=p['title'],evidence_checked_utc=p['evidence_checked_utc'],evidence_file=p['evidence_file'],evidence_sha256=p['evidence_sha256'],rights_record=p['rights_record'],rights_record_sha256=p['rights_record_sha256'],attribution=p['attribution'],age_days=age))
 expected=read(META/'acquisition-complete.json')['archives'];hashes={}
 for name,info in expected.items():
  digest=sha(META/name)
  if digest!=info['sha256']:raise ValueError('LVIS source archive modified')
  hashes[name]=digest
 OUT.mkdir(parents=True)
 approval=Path('/home/tristary/.codex/attachments/96ebda36-cd94-4b21-8a03-dfd289c25c99/Pasted text.txt')
 commit(OUT/'authorization.json',dict(image_acquisition_approved=True,manifest_sha256=PIN,user_approval_file=str(approval),user_approval_sha256=sha(approval),recorded_utc=now(),image_ids=sorted(ids),train=55,development=11,max_transfer=CAP,max_storage=DISK,per_image_max=MAX_IMAGE,max_attempts=2,rights_checks_resumed=False,model_operations_authorized=False))
 commit(OUT/'preflight.json',dict(status='PASS',rights_documents_reverified=66,licensing_evidence=evidence,nine_identity_quarantines_excluded=sorted(excluded),known_frozen_related_154613_excluded=True,unattempted296_not_resumed=True,known_source_groups=len(groups),source_archive_hashes=hashes,original_annotations_preserved=sum(len(r['original_furniture_annotations']) for r in m['images']),live_licensing_requests=0))
 (OUT/'approved-manifest.json').write_bytes(MANIFEST.read_bytes());space(8368471)
 print('PREFLIGHT PASS:66 saved same-photo grants;9 exclusions;296 rights checks untouched',flush=True)

if __name__=='__main__':run()
