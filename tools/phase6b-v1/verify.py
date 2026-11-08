"""Independent source recount, candidate reproduction and historical preservation."""
import collections as C
import sys
from common import *
from streaming import entries
from audit import rank_select,group_split,historic_groups,flickr_id

def sums(base,name):
 count=0
 for line in (base/name).read_text().splitlines():
  expected,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=expected:raise ValueError('Historical hash mismatch '+str(path))
  count+=1
 return count

def run():
 archive=OUT/'annotations_trainval2017.zip';a=read(OUT/'metadata-audit.json');plan=read(OUT/'proposed-image-candidates-v1.json');ledger=read(OUT/'network-ledger.json')
 assert archive.stat().st_size==SIZE and sha(archive)==SHA
 assert read(OUT/'acquisition-complete.json')['local_sha256']==SHA
 assert ledger['archive_attempts']==1 and ledger['actual_body_bytes']==252925128 and ledger['conservative_charged_bytes']<=TRANSFER
 assert len(ledger['requests'])==4 and all(r['complete'] for r in ledger['requests'])
 assert sum(r['actual_bytes'] for r in ledger['requests'])==ledger['actual_body_bytes']
 assert [r['url'] for r in ledger['requests'] if r['kind']=='archive']==[URL]
 files=read(OUT/'extraction-complete.json')['members'];assert {r['file'].removeprefix('extracted/') for r in files}==set(MEMBERS)
 for r in files:assert sha(OUT/r['file'])==r['sha256'] and (OUT/r['file']).stat().st_size==r['bytes']
 assert sum(r['bytes'] for r in files)==489773314
 selected={r['coco_image_id']:r for r in plan['images']};source_selected=C.defaultdict(list);fields=set();counts={}
 for part in ('train2017','val2017'):
  image_ids=set();native={i:C.Counter() for i in MAP};unique={i:set() for i in MAP};all_annotations=0
  for key,row in entries(OUT/'extracted/annotations'/('instances_'+part+'.json')):
   if key=='images':image_ids.add(row['id']);fields.update(row)
   if key!='annotations':continue
   all_annotations+=1;c=row['category_id']
   if c not in MAP:continue
   native[c]['boxes']+=1;unique[c].add(row['image_id'])
   native[c]['crowd_boxes']+=int(row['iscrowd'])
   native[c]['polygons']+=isinstance(row['segmentation'],list)
   if part=='train2017' and row['image_id'] in selected:source_selected[row['image_id']].append(row)
  assert len(image_ids)==a['source_partitions'][part]['images'] and all_annotations==a['source_partitions'][part]['annotations']
  for c,(source,target) in MAP.items():
   expected=a['source_partitions'][part]['categories'][source]
   assert native[c]['boxes']==expected['boxes'] and len(unique[c])==expected['images']
   assert native[c]['crowd_boxes']==expected.get('crowd_boxes',0) and native[c]['polygons']==expected.get('segmentation_polygon',0)
  counts[part]=dict(images=len(image_ids),annotations=all_annotations)
 assert len(selected)==400 and all(source_selected[i]==r['original_furniture_annotations'] for i,r in selected.items())
 # Reproduce the proposal using the persisted, immutable metadata inventory.
 eligible=[];row_count=0
 with (OUT/'candidate-inventory.jsonl').open() as f:
  for line in f:
   r=json.loads(line);row_count+=1
   assert r['license_verified'] is False and r['image_sha256'] is None
   if r['metadata_eligible_for_rights_audit']:eligible.append(r)
 assert row_count==25282 and len(eligible)==3132
 repeated=group_split(rank_select(eligible))
 assert [(r['coco_image_id'],r['selection_anchor'],r['proposed_split']) for r in repeated]==[(r['coco_image_id'],r['selection_anchor'],r['proposed_split']) for r in plan['images']]
 source_lookup,protected,source_scope=historic_groups()
 owners={};partition_groups=C.defaultdict(set)
 for r in plan['images']:
  assert r['source_partition']=='train2017' and not r['license_verified'] and r['metadata_license_id']==4
  photo=flickr_id(r['original_flickr_asset_url']);expected=source_lookup.get(photo,'coco-flickr:'+photo)
  assert r['identity_group']==expected and expected not in protected and expected not in owners
  owners[expected]=r['image_id'];partition_groups[r['proposed_split']].add(expected)
 assert not partition_groups['train']&partition_groups['development']
 assert C.Counter(r['proposed_split'] for r in plan['images'])==dict(train=320,development=80)
 # Existing data/model bytes only; no photograph decoding or inference.
 sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
 from verify_metadata import historical
 meta=ROOT/'data/open-images/metadata/phase5-v1';old=historical(meta);checks={}
 for label,base in [('openimages_metadata',meta),('phase5_photos',ROOT/'data/open-images/image-pilots/phase5-v1'),('phase5b',ROOT/'data/open-images/experiments/phase5b-v1'),('phase5c',ROOT/'data/open-images/experiments/phase5c-v1'),('phase6a_preparation',ROOT/'data/open-images/preparation/phase6a-v1'),('phase6a_photos',ROOT/'data/open-images/image-pilots/phase6a-v1')]:checks[label]=sums(base,'ARTIFACT-SHA256SUMS')
 for base in (ROOT/'data/open-images/experiments/phase5b-v1',ROOT/'data/open-images/experiments/phase5c-v1',ROOT/'data/open-images/preparation/phase6a-v1',ROOT/'data/open-images/image-pilots/phase6a-v1'):
  for path,digest in read(base/'implementation-sha256.json').items():assert sha(ROOT/path)==digest
 owl=ROOT/'models/owlv2-original-baseline-20261103/model.safetensors';owl_hash=sha(owl);assert owl_hash=='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
 sig=ROOT/'models/siglip2-base-patch16-224-phase2a-v1';sig_files=sums(sig,'SHA256SUMS');sig_hash=sha(sig/'model.safetensors');assert sig_hash=='612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'
 reg=read(ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json');assert sha(reg['checkpoint'])==reg['sha256']=='d7c30f22eacd0483062b17f44776ff769f886a5a5f546fbf81e5f3d2b4e877a6'
 result=dict(status='PASS',archive_sha256=SHA,archive_attempts=1,actual_response_body_bytes=ledger['actual_body_bytes'],independently_recounted_source=counts,exact_selected_annotation_records_verified=sum(len(r) for r in source_selected.values()),
  inventory_rows=row_count,metadata_candidates=3132,known_groups_selected=400,draft_partition_counts=dict(train=320,development=80),candidate_selection_reproduced=True,
  creator_groups_independently_verified=False,individual_photograph_rights_verified=0,source_image_metadata_fields=sorted(fields),
  historical=old,old_artifact_checks=checks,siglip_files_verified=sig_files,model_hashes=dict(owlv2=owl_hash,siglip2=sig_hash,phase5c=reg['sha256']),
  new_photographs_or_weights_downloaded=0,photo_pixels_opened=0,model_inference_or_training=0,frozen_partitions_changed=False,mac_application_accessed=False,python=sys.executable)
 commit(OUT/'integrity-verification.json',result)
 commit(OUT/'acquisition-verified-commit.json',dict(archive_sha256=SHA,raw_completion_sha256=sha(OUT/'acquisition-complete.json'),ledger_sha256=sha(OUT/'network-ledger.json'),atomic_publication=True))
 commit(OUT/'extraction-verified-commit.json',dict(extraction_receipt_sha256=sha(OUT/'extraction-complete.json'),members=files,atomic_publication=True))
 print('VERIFY PASS: archive,1662 selected annotation records,400 reproducible groups; historical records',old['checked_records'],flush=True)

if __name__=='__main__':run()
