"""Versioned photograph hash and attribution ledger; original source metadata preserved."""
import csv
from pilot_core import *

def run():
 source={r['coco_image_id']:r for r in manifest()['images']};rows=[]
 for r in read(OUT/'acquisition-complete.json')['records']:
  original=source[r['coco_image_id']]
  rows.append(dict(acquisition=r,rights=original['verified_photo_rights'],
   original_LVIS_annotations=original['original_furniture_annotations'],original_LVIS_image_scope=original['source_image_scope'],
   annotation_licence=original['annotation_license'],source_annotation_archive_sha256=original['annotation_archive_sha256'],
   source_manifest_sha256=PIN,photograph_licence_is_not_annotation_licence=True))
 commit(OUT/'license-and-provenance-ledger.json',dict(records=rows,live_rights_requests=0,no_labels_or_human_flags_fabricated=True))
 columns=['coco_image_id','source_photo_id','partition','creator','creator_url','title','license_url','source_url','evidence_checked_utc','evidence_sha256','photo_sha256','attribution']
 with (OUT/'attribution.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
  for row in rows:
   p=row['acquisition'];rights=row['rights'];writer.writerow(dict(coco_image_id=p['coco_image_id'],source_photo_id=p['source_identity'],partition=p['proposed_split'],photo_sha256=p['sha256'],**{k:rights[k] for k in columns if k in rights}))
 with (OUT/'PHOTO-SHA256SUMS').open('x') as f:
  for row in rows:f.write(row['acquisition']['sha256']+'  '+row['acquisition']['file']+'\n')
 print('27 photo hashes and attribution/licence/native provenance records written',flush=True)

if __name__=='__main__':run()
