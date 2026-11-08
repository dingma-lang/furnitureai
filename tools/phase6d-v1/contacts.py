"""Bounded native-box galleries; drawn only from the72 new approved photos."""
import math
from common import *

def run():
 from PIL import Image,ImageOps,ImageDraw
 rows=read(OUT/'screening-v1.json')['images'];folder=OUT/'contact-sheets-v1';folder.mkdir();links=[];colors={62:'blue',63:'red',65:'purple',67:'green'}
 for part in ('train','development'):
  group=[r for r in rows if r['proposed_split']==part]
  for start in range(0,len(group),15):
   batch=group[start:start+15];canvas=Image.new('RGB',(1500,325*math.ceil(len(batch)/4)+40),'white');draw=ImageDraw.Draw(canvas)
   draw.text((5,5),part+' | original COCO source boxes | assistant diagnostics, no human verification',fill='black')
   for i,r in enumerate(batch):
    with Image.open(OUT/r['file']) as source:im=ImageOps.exif_transpose(source).convert('RGB')
    scale=min(365/im.width,250/im.height);thumb=im.resize((round(im.width*scale),round(im.height*scale)));x=i%4*375;y=i//4*325+40;canvas.paste(thumb,(x,y))
    for a in r['original_annotations']:
     bx,by,w,h=a['bbox'];draw.rectangle((x+bx*scale,y+by*scale,x+(bx+w)*scale,y+(by+h)*scale),outline=colors[a['category_id']],width=2)
    draw.text((x+3,y+253),r['image_id'],fill='black');draw.text((x+3,y+269),r['status'][:48],fill='black');draw.text((x+3,y+285),','.join(CLASSES[c] for c in sorted({a['category_id'] for a in r['original_annotations']})),fill='black');draw.text((x+3,y+301),','.join(r['flags'])[:48],fill='black')
   name=part+f'-{start//15+1:03d}.jpg';canvas.save(folder/name,quality=91);links.append(name)
 with (folder/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>Phase6D diagnostic contacts</title><p>Original source annotations; no human confirmation. <a href="../attribution.csv">Creator/title/CC BY credits</a>. Changes: thumbnails and source-box overlays.</p>'+''.join(f'<h2>{n}</h2><img style="max-width:100%" src="{n}">' for n in links))
 print('CONTACTS',links,flush=True)

if __name__=='__main__':run()
