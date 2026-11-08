"""Inspect only the ten authorised existing COCO photos; all33 LVIS labels stay inactive."""
import math
import sqlite3
from PIL import Image,ImageDraw
from prep_core import *

def run():
 prior=read(PILOT/'existing-ten-registration-reassessment.json');folder=OUT/'existing-overlay-gallery-v1';folder.mkdir()
 images=[];rows=[];colors={181:'magenta',361:'orange',626:'cyan'}
 db=sqlite3.connect(f'file:{META}/furniture-index.sqlite?mode=ro',uri=True)
 coco_rights={r['coco_image_id']:r for r in read(ROOT/'data/coco/image-pilots/phase6d-v1/license-and-provenance-ledger.json')['records']}
 for r in prior['records']:
  iid=r['coco_image_id'];path=ROOT/r['existing_photo_file'];digest=sha(path)
  if digest!=r['existing_photo_sha256']:raise ValueError('Existing image hash changed')
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:im.load();canvas=im.convert('RGB');dims=im.size;orientation=im.getexif().get(274,1)
  native=db.execute('SELECT width,height,url,neg,nex FROM images WHERE id=?',(iid,)).fetchone()
  if dims!=(native[0],native[1]) or dims!=(r['existing_photo_width'],r['existing_photo_height']) or orientation!=1:raise ValueError('Actual image coordinate/identity mismatch')
  draw=ImageDraw.Draw(canvas);details=[]
  for entry in r['annotation_diagnostics']:
   a=entry['native_annotation'];x,y,w,h=a['bbox'];color=colors[a['category_id']]
   draw.rectangle([x,y,x+w,y+h],outline=color,width=2);draw.text((x,y),str(a['id'])+':'+str(a['category_id']),fill=color)
   for polygon in a['segmentation']:
    pts=list(zip(polygon[::2],polygon[1::2]));draw.line(pts+[pts[0]],fill=color,width=1)
   details.append(dict(annotation=a,native_category_id=a['category_id'],bbox_polygon_max_delta_pixels=entry['bbox_polygon_extent_max_delta_pixels'],
     coordinate_metadata_consistent=True,visual_semantic_assessment='pending_assistant_inspection',independent_semantic_verification_complete=False,active=False,human_reviewed=False))
  name=f'{iid:012d}.jpg';canvas.save(folder/name,quality=94);images.append((r,canvas.copy()))
  rows.append(dict(coco_image_id=iid,original_partition=r['existing_partition'],local_photo_file=r['existing_photo_file'],sha256=digest,dimensions=list(dims),orientation=orientation,
   lvis_source_url=native[2],native_image_scope=dict(neg_category_ids=__import__('json').loads(native[3]),not_exhaustive_category_ids=__import__('json').loads(native[4])),
   photo_rights=coco_rights[iid],overlay_file=str((folder/name).relative_to(ROOT)),annotations=details,source_assessment_sha256=sha(PILOT/'existing-ten-registration-reassessment.json'),
   active=False,human_reviewed=False))
 if len(rows)!=10 or sum(len(r['annotations']) for r in rows)!=33:raise ValueError('33-label scope changed')
 for start in (0,5):
  batch=images[start:start+5];sheet=Image.new('RGB',(1800,800),'white');d=ImageDraw.Draw(sheet)
  for index,(r,im) in enumerate(batch):
   im.thumbnail((580,330));x=index%3*600;y=index//3*400;sheet.paste(im,(x,y));d.text((x+4,y+335),f"{r['coco_image_id']:012d} {r['existing_partition']} | {len(r['annotation_diagnostics'])} labels",fill='black')
  sheet.save(folder/f'sheet-{start//5+1:02d}.jpg',quality=94)
 with (folder/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>Inactive existing LVIS overlays</title><p>Assistant inspection only; source geometry unchanged; all33 annotations inactive. Magenta Cabinet181, orange Desk361, cyan Lamp626.</p>'+''.join(f'<p>{r["coco_image_id"]} {r["original_partition"]}</p><img style="max-width:100%" src="{r["coco_image_id"]:012d}.jpg">' for r in rows))
 db.close();commit(OUT/'existing33-registration-v1.json',dict(records=rows,actual_photos_decoded=10,source_photos_modified=0,annotations=33,annotations_activated=0,independent_human_verification=False,
  pixel_frame_policy='Native top-left xywh/polygon pixels, exact actual COCO dimensions, EXIF1, no source transform',model_inference_or_scoring=0))
 print('EXISTING33 actual image identities/dimensions checked; overlays prepared; all inactive',flush=True)

if __name__=='__main__':run()
