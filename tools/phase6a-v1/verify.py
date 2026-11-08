"""Read-only historical integrity, licensing, and proposal isolation audit."""
import sys
from core import *
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from verify_metadata import historical

def verify_sums(base,filename):
 count=0
 for line in (base/filename).read_text().splitlines():
  expected,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=expected:raise ValueError('Changed artifact '+str(path))
  count+=1
 return count

def run():
 preserved=historical(META);checked={}
 for name,base,filename in [('metadata',META,'ARTIFACT-SHA256SUMS'),('photos_and_evidence',OLD,'ARTIFACT-SHA256SUMS'),('photos',OLD,'PHOTO-SHA256SUMS'),('phase5b',ROOT/'data/open-images/experiments/phase5b-v1','ARTIFACT-SHA256SUMS'),('phase5c',ROOT/'data/open-images/experiments/phase5c-v1','ARTIFACT-SHA256SUMS')]:checked[name]=verify_sums(base,filename)
 for base in ['phase5b-v1','phase5c-v1']:
  for path,digest in read(ROOT/'data/open-images/experiments'/base/'implementation-sha256.json').items():
   if sha(ROOT/path)!=digest:raise ValueError('Historical implementation changed')
 m=read(MANIFEST);isolation(m['images'],set(read(OUT/'protected-identities.json')))
 if len(m['images'])>400 or len({r['identity_group'] for r in m['images']})!=len(m['images']):raise ValueError('Selection exceeds distinct group limit')
 for row in m['images']:
  if sha(ROOT/row['license_record'])!=row['license_record_sha256']:raise ValueError('Changed licensing evidence')
  lic=read(ROOT/row['license_record']);response=OUT/'rights-audit'/lic['response_file']
  if sha(response)!=lic['response_sha256']:raise ValueError('Changed source HTML')
  if not row['license_verified'] or license_url(lic['license_url'])!=row['license_url'] or photo_id(lic['source_url'])!=row['source_identity'].removeprefix('flickr-photo:'):raise ValueError('Individual rights identity mismatch')
  for g in row['annotations']:box(g['box'])
 for name,digest in read(OUT/'candidate-manifest.json')['metadata_sha256'].items():
  if sha(META/name)!=digest:raise ValueError('Changed source metadata')
 if DEST.exists():raise ValueError('Photograph acquisition unexpectedly started')
 result=dict(status='PASS',historical=preserved,checksum_records_verified=checked,selected_image_count=len(m['images']),selected_known_groups=len(m['images']),
  finalized_manifest_sha256=sha(MANIFEST),image_bodies_downloaded=0,detector_optimizer_updates=0,
  original_and_phase5c_checkpoint_hashes_verified=True,phase5c_selected_checkpoint_sha256=SELECTED_SHA,
  frozen_pixels_decoded_or_scored=0,official_phase2b_scoring=False,mac_application_accessed=False,production_model_replaced=False,
  preserved_partition_policy='All prior 150 Open Images memberships and complete known source groups excluded; entire source test table not used for expansion; original annotations and all frozen memberships unchanged.',
  identity_limitations='Source-photo/photographer/original MD5 groups isolated. Cross-author physical-object, image-byte and perceptual overlaps pending approved photographs; no new image hashes invented.')
 write_new(OUT/'verification.json',result);print('PRESERVATION PASS',checked,'historical records',preserved['checked_records'])

if __name__=='__main__':run()
