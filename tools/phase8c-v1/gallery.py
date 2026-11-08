"""Original/annotation pairs for every approved photograph; no visual approvals."""
import math
from PIL import Image,ImageDraw
from pilot_core import *

COLORS={982:'red',232:'blue',1050:'green',77:'purple',361:'orange',181:'magenta',626:'cyan'}

def run():
 folder=OUT/'diagnostic-gallery-v1';folder.mkdir();entries=[];sheets=[]
 rows=read(OUT/'screening-v1.json')['images']
 for r in rows:
  with Image.open(OUT/r['file']) as src:im=src.convert('RGB')
  draw=ImageDraw.Draw(im)
  for a in r['original_annotations']:
   color=COLORS.get(a['category_id'],'gray')
   for poly in a['segmentation']:
    pts=list(zip(poly[::2],poly[1::2]));draw.line(pts+[pts[0]],fill=color,width=2)
   x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
   draw.text((x+2,max(0,y-12)),str(a['id'])+':'+CLASSES.get(a['category_id'],str(a['category_id'])),fill=color,stroke_width=1,stroke_fill='black')
  filename=f"{r['coco_image_id']:012d}-overlay.jpg";im.save(folder/filename,quality=95)
  entries.append(dict(coco_image_id=r['coco_image_id'],original='../'+r['file'],original_sha256=r['sha256'],overlay=filename,partition=r['proposed_split']))
 for part in ('train','development'):
  subset=[r for r in rows if r['proposed_split']==part]
  for start in range(0,len(subset),3):
   batch=subset[start:start+3];canvas=Image.new('RGB',(1920,1040),'white');draw=ImageDraw.Draw(canvas)
   for i,r in enumerate(batch):
    for y,path in [(24,OUT/r['file']),(544,folder/f"{r['coco_image_id']:012d}-overlay.jpg")]:
     with Image.open(path) as src:im=src.convert('RGB');im.thumbnail((640,480));canvas.paste(im,(i*640,y))
    draw.text((i*640+4,4),str(r['coco_image_id'])+' '+part+' ORIGINAL',fill='black')
    draw.text((i*640+4,524),str(r['coco_image_id'])+' native overlays; not human verification',fill='black')
   filename=f'{part}-{start//3+1:02d}-pairs.jpg';canvas.save(folder/filename,quality=95);sheets.append(filename)
 commit(folder/'index.json',dict(images=entries,sheets=sheets,source_geometry_modified=False,human_reviewed=False))
 with (folder/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>Phase8C diagnostics</title><p>Original photographs and native box/mask overlays. Assistant diagnostics only; creator and CC BY credits in ../attribution.csv. No ground-truth accuracy established.</p>'+''.join('<h2>'+s+'</h2><img style="max-width:100%" src="'+s+'">' for s in sheets))
 print('GALLERY',len(entries),'images;',len(sheets),'paired sheets',flush=True)

if __name__=='__main__':run()
