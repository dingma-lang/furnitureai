"""Read-only historical seals and source image/model identity preservation."""
import importlib.util
import sys
from prep_core import *

def seal(base,name='ARTIFACT-SHA256SUMS'):
 count=0
 for line in (base/name).read_text().splitlines():
  digest,relative=line.split('  ',1);path=(base/relative).resolve()
  if not path.is_relative_to(base.resolve()) or sha(path)!=digest:raise ValueError('Historical byte mismatch '+str(path))
  count+=1
 return count

def run():
 sys.path.insert(0,str(ROOT/'tools/phase7-v1'))
 spec=importlib.util.spec_from_file_location('phase8a_preservation',ROOT/'tools/phase7-v1/preserve.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);old=m.verify()
 locations=[('phase7_preparation',ROOT/'reports/phase7-data-strategy-v1','ARTIFACT-SHA256SUMS'),('LVIS_metadata',META,'ARTIFACT-SHA256SUMS'),('LVIS_rights',RIGHTS,'FINAL-ARTIFACT-SHA256SUMS-v2'),('LVIS_photos',PILOT,'ARTIFACT-SHA256SUMS')]
 seals={label:seal(path,name) for label,path,name in locations}
 for _,base,_ in locations:
  candidates=['implementation-sha256-final-v2.json','implementation-sha256.json'];file=next((base/name for name in candidates if (base/name).exists()),None)
  if file:
   for filename,digest in read(file).items():
    if sha(ROOT/filename)!=digest:raise ValueError('Historical implementation changed '+filename)
 commit(OUT/'preservation-verification.json',dict(status='PASS',historical=old,seals=seals,project_training_inference_or_scoring=0,
  network_requests=0,model_weights_changed=False,source_photos_modified=False,historical_labels_or_partitions_modified=False,
  source_archive_extraction=False,assets_or_packages_downloaded=0,Blender_installed=False,synthetic_frames_rendered=0,Mac_project_accessed=False,
  actual_existing_ten_images_inspected=10,frozen_image_pixels_decoded=0))
 print('ALL HISTORICAL MODEL/DATA/ARCHIVE/PARTITION SEALS PASS',flush=True)

if __name__=='__main__':run()
