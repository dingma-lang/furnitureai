"""Reassess previous ten-photo/33-annotation opportunity without activating or decoding."""
import sqlite3
from pilot_core import *

def run():
 prior=RIGHTS/'existing-ten-photo-registration-assessment.json';assessment=read(prior)
 db=sqlite3.connect(f'file:{META}/furniture-index.sqlite?mode=ro',uri=True);rows=[]
 for r in assessment['records']:
  if sha(ROOT/r['existing_photo_file'])!=r['existing_photo_sha256']:raise ValueError('Existing source photo SHA changed')
  dims=db.execute('SELECT width,height FROM images WHERE id=?',(r['lvis_image_id'],)).fetchone()
  if dims!=(r['existing_photo_width'],r['existing_photo_height']):raise ValueError('Original image-coordinate mismatch')
  rows.append(dict(r,reassessment_utc=now(),metadata_identity_and_coordinates_still_consistent=True,annotation_training_eligible=False,
    outstanding_checks=['Native Desk/Cabinet/Lamp semantic unit consistency','Actual source-mask/image visual registration','Native federated partial-annotation loss policy','Keep original development membership; no source annotation or split correction']))
 db.close()
 if len(rows)!=10 or sum(len(r['annotation_diagnostics']) for r in rows)!=33:raise ValueError('Existing overlay scope changed')
 commit(OUT/'existing-ten-registration-reassessment.json',dict(records=rows,source_assessment=str(prior.relative_to(ROOT)),source_assessment_sha256=sha(prior),
   photographs=10,annotations=33,existing_photo_hashes_verified=10,image_pixels_decoded=0,model_inference_or_scoring=0,annotations_activated_or_merged=0,
   status='Metadata/hash/coordinate consistency remains supported; semantic and visual alignment still unresolved'))
 print('TEN EXISTING PHOTOS:33 annotations remain inactive',flush=True)

if __name__=='__main__':run()
