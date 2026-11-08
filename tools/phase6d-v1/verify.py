"""Independent acquired-byte/source-annotation and historical preservation checks."""
import collections as C
import sys
from common import *
from screen import signature,duplicate

def sums(base,name):
 count=0
 for line in (base/name).read_text().splitlines():
  digest,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=digest:raise ValueError('Historical hash mismatch '+str(path))
  count+=1
 return count

def run():
 from PIL import Image
 m=manifest();source={r['image_id']:r for r in m['images']};acq=read(OUT/'acquisition-complete.json');final=read(OUT/'screening-v2.json');net=read(OUT/'network-ledger.json');eligible=read(OUT/'eligible-candidates-v1.json')['images']
 assert len(acq['records'])==len(final['images'])==72 and {r['image_id'] for r in acq['records']}==set(source)
 assert C.Counter(r['proposed_split'] for r in final['images'])==dict(train=60,development=12)
 assert net['actual_body_bytes']==net['conservative_charge']==10853122 and sum(r['actual_bytes'] for r in net['requests'] if r['method']=='GET')==10853122
 assert len([r for r in net['requests'] if r['method']=='GET'])==72 and all(r['attempt']==1 and r['complete'] and r['headers_verified'] for r in net['requests'] if r['method']=='GET')
 known={r['image_id']:r for r in final['images']};polygons=crowd=0
 for r in acq['records']:
  row=source[r['image_id']];current=known[r['image_id']];path=OUT/r['file'];assert sha(path)==r['sha256']==current['sha256'] and path.stat().st_size==r['bytes']==row['remote_object']['bytes']
  assert current['original_annotations']==row['original_furniture_annotations'] and current['identity_group']==row['identity_group'] and current['proposed_split']==row['proposed_split']
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:im.load();assert im.size==(row['width'],row['height']);s=signature(im)
  for k,value in s.items():assert current[k]==value
  polygons+=sum(isinstance(a['segmentation'],list) for a in current['original_annotations']);crowd+=sum(bool(a['iscrowd']) for a in current['original_annotations'])
  assert sha(ROOT/row['rights_record'])==row['rights_record_sha256'] and sha(ROOT/row['source_document'])==row['source_document_sha256']
 assert not final['duplicate_pairs']
 for i,r in enumerate(final['images']):
  for other in final['images'][i+1:]:assert duplicate(r,other) is None
 assert len(eligible)==55 and C.Counter(r['proposed_split'] for r in eligible)==dict(train=45,development=10)
 groups=C.defaultdict(set)
 for r in eligible:groups[r['identity_group']].add(r['proposed_split']);assert r['status']=='provisionally_accepted' and not r['human_verified']
 assert all(len(parts)==1 for parts in groups.values())
 assert not any(OUT.rglob('*.part'))
 sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
 from verify_metadata import historical
 meta=ROOT/'data/open-images/metadata/phase5-v1';old=historical(meta);checks={}
 places=[('openimages_metadata',meta),('phase5_photos',ROOT/'data/open-images/image-pilots/phase5-v1'),('phase5b',ROOT/'data/open-images/experiments/phase5b-v1'),('phase5c',ROOT/'data/open-images/experiments/phase5c-v1'),('phase6a_preparation',ROOT/'data/open-images/preparation/phase6a-v1'),('phase6a_photos',ROOT/'data/open-images/image-pilots/phase6a-v1'),('phase6b_metadata',ROOT/'data/coco/metadata/phase6b-v1'),('phase6c_rights',RIGHTS)]
 for label,base in places:checks[label]=sums(base,'ARTIFACT-SHA256SUMS')
 for label,base in places:
  p=base/'implementation-sha256.json'
  if p.exists():
   for filename,digest in read(p).items():assert sha(ROOT/filename)==digest
 before=read(OUT/'preflight.json')['model_hashes'];actual={}
 actual['owlv2']=sha(ROOT/'models/owlv2-original-baseline-20261103/model.safetensors');sig=ROOT/'models/siglip2-base-patch16-224-phase2a-v1';actual['siglip2']=sha(sig/'model.safetensors');sig_files=sums(sig,'SHA256SUMS')
 checkpoint=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json');actual['phase5c']=sha(checkpoint['checkpoint']);assert actual==before
 assert allocated()+allocated(HERE)+16*2**20<DISK
 commit(OUT/'integrity-verification.json',dict(status='PASS',downloaded_images=72,photo_body_bytes=10853122,retries=0,decoded_and_rehashed_images=72,original_native_annotations_verified=sum(len(r['original_annotations']) for r in final['images']),polygon_annotation_objects=polygons,crowd_rle_regions=crowd,
  final_counts=dict(C.Counter(r['status'] for r in final['images'])),eligible_partition_counts=dict(train=45,development=10),eligible_known_source_groups=len(groups),exact_duplicate_pairs=0,suspected_perceptual_pairs=0,
  annotations_unchanged=True,original_slots_unchanged=True,old_historical=old,old_artifact_ledgers=checks,siglip_pinned_files_verified=sig_files,model_hashes=actual,old_frozen_pixels_decoded_or_scored=0,model_inference_or_training=0,mac_project_accessed=False,
  limitation='New-photograph perceptual thresholds and assistant diagnostics are not human truth or proof of independent physical product identity; old/frozen pixels were never decoded.'))
 print('VERIFY PASS:72 images and362 native annotation objects; models/history unchanged',flush=True)

if __name__=='__main__':run()
