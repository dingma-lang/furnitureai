"""Bounded new-photo contact sheets and unchanged source polygon/box overlays."""
import math
from PIL import Image,ImageDraw
from pilot_core import *

COLORS={982:'red',232:'blue',1050:'green',77:'purple',361:'orange',181:'magenta',626:'cyan'}
def run():
 folder=OUT/'contact-sheets-v1';folder.mkdir();links=[]
 for part in ('train','development'):
  rows=[r for r in read(OUT/'screening-v1.json')['images'] if r['proposed_split']==part]
  for start in range(0,len(rows),15):
   batch=rows[start:start+15];canvas=Image.new('RGB',(1500,365*math.ceil(len(batch)/4)+35),'white');d=ImageDraw.Draw(canvas)
   d.text((5,5),part+' | native LVIS masks/boxes | assistant diagnostics, no human labels',fill='black')
   for i,r in enumerate(batch):
    with Image.open(OUT/r['file']) as image:im=image.convert('RGB')
    scale=min(365/im.width,280/im.height);thumb=im.resize((round(im.width*scale),round(im.height*scale)));x=i%4*375;y=i//4*365+35;canvas.paste(thumb,(x,y))
    for a in r['original_annotations']:
     color=COLORS.get(a['category_id'],'gray')
     for polygon in a['segmentation']:
      pts=[(x+xx*scale,y+yy*scale) for xx,yy in zip(polygon[::2],polygon[1::2])];d.line(pts+[pts[0]],fill=color,width=1)
     bx,by,bw,bh=a['bbox'];d.rectangle([x+bx*scale,y+by*scale,x+(bx+bw)*scale,y+(by+bh)*scale],outline=color,width=2)
    d.text((x+3,y+283),str(r['coco_image_id'])+' | '+r['status'][:30],fill='black')
    d.text((x+3,y+300),','.join(CLASSES.get(cid,str(cid)) for cid in r['native_categories']),fill='black')
    d.text((x+3,y+317),','.join(r['flags'])[:52],fill='black')
   name=part+f'-{start//15+1:03d}.jpg';canvas.save(folder/name,quality=92);links.append(name)
 with (folder/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>LVIS Phase7 source diagnostics</title><p>Only approved new images. Original source masks and boxes, thumbnail derivatives; no human verification. Creator/CC BY attribution in ../attribution.csv.</p>'+''.join('<h2>'+n+'</h2><img style="max-width:100%" src="'+n+'">' for n in links))
 print('CONTACTS',links,flush=True)

if __name__=='__main__':run()
