"""Offline source identity, geometry, historical grouping and cooldown checks."""
import sqlite3
from audit_core import *

def run():
 m=manifest();uf,protected,history=history_graph();rows=m['images']
 for r in rows:
  if uf.find(r['source_photo_id']) in protected:raise ValueError('Known historical identity overlap')
 if len({uf.find(r['source_photo_id']) for r in rows})!=44:raise ValueError('Candidates not source-isolated')
 db=sqlite3.connect(f'file:{BASE}/furniture-index.sqlite?mode=ro',uri=True)
 for r in rows:
  dims=db.execute("SELECT width,height FROM images WHERE split='train' AND id=?",(r['lvis_image_id'],)).fetchone()
  if dims!=(r['width'],r['height']) or r['coco_image_id']!=r['lvis_image_id']:raise ValueError('Source dimensions/identity mismatch')
 db.close()
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase8b_source_reader',ROOT/'tools/phase7-acquisition-v1/reader.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
 archive=BASE/'lvis_v1_train.json.zip'
 if sha(archive)!=m['source_archive_sha256']:raise ValueError('Source archive hash changed')
 wanted={a['id']:a for r in rows for a in r['original_lvis_furniture_annotations']};seen=set()
 for kind,a in reader.stream(archive):
  if kind=='annotations' and a['id'] in wanted:
   if a!=wanted[a['id']]:raise ValueError('Native annotation changed')
   seen.add(a['id'])
 if len(wanted)!=164 or seen!=set(wanted):raise ValueError('164 native annotation inventory changed')
 previous=[r for r in read(ROOT/'data/lvis/rights-audits/phase7-v1/network-ledger.json')['requests'] if r.get('status')==429]
 commit(OUT/'preflight.json',dict(status='PASS',manifest_sha256=PIN,exact_candidates=44,original_allocation={'train':8,'development':36},
  unchanged_source_boxes_masks=164,previous400_and_paused296_excluded=True,known_historical_source_overlaps=[],historical_graph=history,
  previous_rate_limit=previous[-1] if previous else None,provider_access_not_yet_checked=True,photograph_bodies=0))
 commit(OUT/'authorization.json',dict(phase='8B',rights_audit_approved=True,manifest_sha256=PIN,
  user_explicit_approval=True,max_cumulative_response_body_bytes=CAP,max_new_allocated_storage_bytes=DISK,
  single_new_approved_ID_access_check_authorized=True,old296_resume=False,photo_bodies=False,training=False,checked_utc=now()))
 print('PREFLIGHT PASS44 exactIDs8/36;164 unchanged boxes/masks; prior400/paused296excluded',flush=True)

if __name__=='__main__':run()
