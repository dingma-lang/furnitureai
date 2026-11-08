"""Read-only exact-ID/source-geometry/partition validation before any network request."""
import collections as C
import importlib.util
import sqlite3
import sys
from audit_core import *

def run():
 m=manifest();ids={r['coco_image_id'] for r in m['images']}
 frozen={r['lvis_image_id'] for r in read(BASE/'frozen-and-source-group-isolation.json')['frozen_source_group_hits']}
 oldrights=read(ROOT/'data/coco/rights-audits/phase6c-v1/license-attribution-ledger.json')['records']
 if ids&frozen or ids&{r['coco_image_id'] for r in oldrights}:raise ValueError('Historical/frozen/prior400 ID overlap')
 group_parts=C.defaultdict(set);lookup={r['coco_image_id']:r for r in m['images']};expected={a['id']:a for r in m['images'] for a in r['original_lvis_furniture_annotations']}
 if len(expected)!=1080:raise ValueError('Frozen original annotation inventory changed')
 core=sum(a['category_id'] in CORE for a in expected.values())
 if core!=905:raise ValueError('905 core boxes/masks not preserved')
 db=sqlite3.connect(f'file:{BASE}/furniture-index.sqlite?mode=ro',uri=True)
 for r in m['images']:
  im=db.execute("SELECT width,height FROM images WHERE split='train' AND id=?",(r['lvis_image_id'],)).fetchone()
  if im!=(r['width'],r['height']) or r['coco_image_id']!=r['lvis_image_id'] or not all(a['image_id']==r['lvis_image_id'] for a in r['original_lvis_furniture_annotations']):raise ValueError('Original LVIS identity or geometry mismatch')
  group_parts[r['known_source_group']].add(r['proposed_partition'])
 if any(len(p)>1 for p in group_parts.values()):raise ValueError('Known group crosses provisional partitions')
 db.close()
 # Read the sealed bounded streaming adapter in a dedicated module namespace.
 sys.path.insert(0,str(ROOT/'tools/phase7-acquisition-v1'))
 spec=importlib.util.spec_from_file_location('phase7_rights_source_reader',ROOT/'tools/phase7-acquisition-v1/reader.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 archive=BASE/'lvis_v1_train.json.zip';digest=sha(archive)
 if digest!='334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba':raise ValueError('Original LVIS train archive changed')
 seen=set()
 for kind,a in mod.stream(archive):
  if kind=='annotations' and a['id'] in expected:
   if a!=expected[a['id']]:raise ValueError('Frozen box/mask/source annotation modified')
   seen.add(a['id'])
 if seen!=set(expected):raise ValueError('Expected original source annotations absent')
 # Existing source graph covers all local Open Images metadata, old audits and frozen references.
 spec=importlib.util.spec_from_file_location('phase7_rights_known_source',ROOT/'tools/phase7-acquisition-v1/cross_reference.py')
 cross=importlib.util.module_from_spec(spec);spec.loader.exec_module(cross)
 uf,blocked,history,_=cross.identity_graph();overlaps=[]
 for r in m['images']:
  if uf.find(r['source_photo_id']) in blocked:overlaps.append(r['coco_image_id'])
 if overlaps:raise ValueError('Known historical source/creator overlap')
 old_net=read(ROOT/'data/coco/rights-audits/phase6c-v1/network-ledger.json');old429=[r for r in old_net['requests'] if r.get('status')==429]
 previous=old429[-1] if old429 else None
 # This new user approval authorizes one new-candidate attempt only; old429 IDs are excluded.
 commit(OUT/'preflight.json',dict(status='PASS',manifest_sha256=PIN,exact_candidates=400,original_allocation=dict(C.Counter(r['proposed_partition'] for r in m['images'])),
  core_boxes_masks_verified=905,all_native_annotations_verified=1080,train_archive_sha256=digest,original_geometry_unchanged=True,
  historical_graph=history,known_historical_source_overlaps=overlaps,five_frozen_related_ids_excluded=sorted(frozen),previous400_ids_excluded=True,
  previous_rate_limit=previous,current_source_requests=0,annotations_and_photograph_licensing_separate=True,
  physical_identity_verified=False,perceptual_or_visual_screening_performed=False))
 space();print('PREFLIGHT PASS:400 exact IDs,905 core boxes/masks,1080 unchanged native annotations',flush=True)

if __name__=='__main__':run()
