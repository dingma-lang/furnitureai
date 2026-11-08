"""Independent new-image hashes/geometry and historical byte-preservation audit."""
import importlib.util
import sys
from PIL import Image
from pilot_core import *
from signatures import signature

def seals(base,filename='ARTIFACT-SHA256SUMS'):
 n=0
 for line in (base/filename).read_text().splitlines():
  expected,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=expected:raise ValueError('Historical artifact changed '+str(path))
  n+=1
 return n

def run():
 source={r['coco_image_id']:r for r in manifest()['images']};records=read(OUT/'acquisition-complete.json')['records'];screen={r['coco_image_id']:r for r in read(OUT/'screening-v2.json')['images']};net=read(OUT/'network-ledger.json')
 if len(records)!=66 or {r['coco_image_id'] for r in records}!=set(source):raise ValueError('Approved IDs changed')
 gets=[r for r in net['requests'] if r['method']=='GET']
 if len(gets)!=66 or sum(r['actual_bytes'] for r in gets)!=8368471 or net['actual_body_bytes']!=8368471 or net['conservative_charge']!=8368471:raise ValueError('Transfer ledger mismatch')
 if any(r['attempt']!=1 or not r['complete'] for r in gets):raise ValueError('Unexpected retry/incomplete state')
 for r in records:
  row=source[r['coco_image_id']];p=OUT/r['file'];current=screen[r['coco_image_id']]
  if sha(p)!=r['sha256'] or p.stat().st_size!=row['remote_object']['bytes'] or r['proposed_split']!=row['proposed_split'] or current['original_annotations']!=row['original_furniture_annotations']:raise ValueError('Source/hash/assignment changed')
  with Image.open(p) as im:im.verify()
  with Image.open(p) as im:im.load();features=signature(im);dims=im.size
  if dims!=(row['width'],row['height']) or any(current[k]!=v for k,v in features.items()):raise ValueError('Decode/signature mismatch')
  proof=row['verified_photo_rights']
  if sha(ROOT/proof['rights_record'])!=proof['rights_record_sha256'] or sha(ROOT/proof['evidence_file'])!=proof['evidence_sha256']:raise ValueError('Rights evidence changed')
 if any(OUT.rglob('*.part')) or any(OUT.rglob('*.tmp')):raise ValueError('Incomplete image/ledger files')
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
 spec=importlib.util.spec_from_file_location('phase7_images_history',ROOT/'tools/phase7-v1/preserve.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);history=old.verify()
 checks=dict(phase7_preparation=seals(ROOT/'reports/phase7-data-strategy-v1'),LVIS_metadata=seals(META),LVIS_rights=seals(RIGHTS,'FINAL-ARTIFACT-SHA256SUMS-v2'))
 for base,filename in [(ROOT/'reports/phase7-data-strategy-v1','implementation-sha256.json'),(META,'implementation-sha256.json'),(RIGHTS,'implementation-sha256-final-v2.json')]:
  for relative,digest in read(base/filename).items():
   if sha(ROOT/relative)!=digest:raise ValueError('Historical implementation changed')
 space();commit(OUT/'integrity-verification.json',dict(status='PASS',approved_images=66,decoded_and_rehashed_images=66,actual_body_bytes=8368471,attempts=66,retries=0,
  source_annotations_verified=sum(len(r['original_furniture_annotations']) for r in source.values()),native_core_annotations_in_parent=108,
  historical=history,additional_historical_seals=checks,original_model_hashes=history['model_hashes'],
  original_source_geometry_and_partitions_unchanged=True,rights_licensing_retries=0,unattempted296_rights_not_resumed=True,
  old_image_pixels_decoded=0,official_holdout_scoring=0,model_training_inference_rendering=0,Mac_project_accessed=False,allocated_bytes_at_verification=allocated()+allocated(HERE)))
 print('NEW IMAGES AND HISTORICAL PRESERVATION PASS',flush=True)

if __name__=='__main__':run()
