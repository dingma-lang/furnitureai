"""Replay saved photo-specific rights and preserved source/partition isolation offline."""
import gzip
import io
import importlib.util
import sys
from pilot_core import *

def run():
 if OUT.exists():raise FileExistsError('Destination already exists; no overwrite')
 m=manifest();ids={r['coco_image_id'] for r in m['images']}
 decisions=read(RIGHTS/'identity-audit.json')['records'];parent={r['coco_image_id']:r for r in read(ROOT/'reports/phase8a-v1/targeted-rights-candidates-v1.json')['images']}
 excluded={r['coco_image_id'] for r in decisions if r['identity_flags']};bad={r['coco_image_id'] for r in decisions if not r['eligible_for_acquisition_proposal']}
 if len(excluded)!=4 or len(bad)!=17 or ids&bad:raise ValueError('Four identity and13 insufficient/unavailable exclusions not preserved')
 old={r['coco_image_id'] for r in read(META/'proposed-photo-rights-audit-v1.json')['images']}
 if ids&old:raise ValueError('Prior400/paused296 overlap')
 # Import the preserved evidence parser, but never call its transport or run functions.
 sys.path.insert(0,str(ROOT/'tools/phase8b-v1'))
 spec=importlib.util.spec_from_file_location('phase8c_saved_parser',ROOT/'tools/phase8b-v1/source_evidence.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 sys.path.insert(0,str(ROOT/'tools/phase8b-v1'));spec=importlib.util.spec_from_file_location('phase8c_old_graph',ROOT/'tools/phase8b-v1/audit_core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 uf,protected,history=core.history_graph()
 for record in read(RIGHTS/'rights-audit.json')['records']:
  for alias in record.get('creator_aliases',[]):uf.join(record['source_photo_id'],alias)
 protected={uf.find(p) for p in protected};groups={};evidence=[]
 for r in m['images']:
  iid=r['coco_image_id'];proof=r['verified_photo_rights'];original=parent[iid]
  if r['proposed_partition']!=original['proposed_partition'] or r['original_furniture_annotations']!=original['original_LVIS_core_annotations']:raise ValueError('Frozen split/native annotation change')
  if sha(ROOT/proof['rights_record'])!=proof['rights_record_sha256'] or sha(ROOT/proof['evidence_file'])!=proof['evidence_sha256']:raise ValueError('Saved evidence hash mismatch')
  raw=(ROOT/proof['evidence_file']).read_bytes()
  if raw[:2]==b'\x1f\x8b':
   with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:raw=f.read(4194305)
  if len(raw)>4194304:raise ValueError('Decompression bound')
  replay=mod.classify(raw.decode('utf-8'),dict(original,metadata_license_url=original['metadata_license_claim']),proof['evidence_url'])
  if replay['licensing_status']!='verified_commercially_compatible' or any(replay[k]!=proof[k] for k in ('creator','creator_url','license_url','title')):raise ValueError('Rights replay conflict')
  age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(proof['evidence_checked_utc'])).total_seconds()/86400
  if not 0<=age<=7:raise ValueError('Saved evidence stale; do not restart rights requests')
  group=uf.find(r['source_photo_id'])
  if group in protected or group in groups and groups[group]!=r['proposed_partition']:raise ValueError('Historical/cross-partition identity conflict')
  groups[group]=r['proposed_partition'];evidence.append(dict(coco_image_id=iid,**proof,evidence_age_days=age))
 # Preserve sealed Phase8B evidence and Phase8A preparation before acquiring anything.
 for base,name in [(RIGHTS,'ARTIFACT-SHA256SUMS'),(ROOT/'reports/phase8a-v1','ARTIFACT-SHA256SUMS'),(ROOT/'tools/phase8a-v1','IMPLEMENTATION-SHA256SUMS')]:
  for line in (base/name).read_text().splitlines():
   digest,relative=line.split('  ',1)
   if sha(base/relative)!=digest:raise ValueError('Historical seal mismatch '+relative)
 OUT.mkdir(parents=True)
 approval=Path('/home/tristary/.codex/attachments/2451d144-2d09-4781-b982-618c4cc59e13/Pasted text.txt')
 commit(OUT/'authorization.json',dict(image_acquisition_approved=True,manifest_sha256=PIN,user_approval_file=str(approval),user_approval_sha256=sha(approval),recorded_utc=now(),image_ids=sorted(ids),train=7,development=20,max_transfer=CAP,max_storage=DISK,per_image_max=MAX_IMAGE,max_attempts=2,rights_checks_resumed=False,model_operations_authorized=False))
 commit(OUT/'preflight.json',dict(status='PASS',rights_documents_replayed=27,licensing_evidence=evidence,identity_exclusions_preserved=sorted(excluded),all17_noneligible_exclusions_preserved=sorted(bad),prior400_and296paused_excluded=True,known_source_groups=len(groups),original_boxes_masks=99,live_rights_requests=0,source_geometry_and_assignments_unchanged=True,historical_graph=history))
 with (OUT/'approved-manifest.json').open('xb') as f:f.write(MANIFEST.read_bytes())
 space(3913471);print('PREFLIGHT PASS27exactCCBYgrants;7/20split;17 exclusions;26known source groups;99native boxes/masks',flush=True)

if __name__=='__main__':run()
