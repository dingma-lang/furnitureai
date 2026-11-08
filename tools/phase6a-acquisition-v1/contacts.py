"""Bounded contacts of the 41 newly approved photos; no historical pixels."""
import html
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase6a-v1'))
from core import *

def run():
 from PIL import Image,ImageOps,ImageDraw
 rows=read(DEST/'screening-v1.json')['images'];out=DEST/'contact-sheets-v1';out.mkdir()
 links=[];colors=dict(sofa='red',chair='blue',table='green',bed='purple',desk='orange',cabinet='cyan',lamp='magenta')
 for part in ('train','development'):
  group=[r for r in rows if r['proposed_split']==part]
  for start in range(0,len(group),16):
   batch=group[start:start+16];canvas=Image.new('RGB',(1600,320*math.ceil(len(batch)/4)+40),'white');draw=ImageDraw.Draw(canvas)
   draw.text((8,8),part+' | source-native boxes; provisional automated status; no human review flags',fill='black')
   for i,row in enumerate(batch):
    path=DEST/row['file']
    if sha(path)!=row['sha256']:raise ValueError('Photo changed')
    with Image.open(path) as src:im=ImageOps.exif_transpose(src).convert('RGB')
    scale=min(390/im.width,240/im.height);thumb=im.resize((round(im.width*scale),round(im.height*scale)));x=i%4*400;y=i//4*320+40;canvas.paste(thumb,(x,y))
    for g in row['original_annotations']:
     b=g['box'];draw.rectangle((x+b[0]*thumb.width,y+b[1]*thumb.height,x+b[2]*thumb.width,y+b[3]*thumb.height),outline=colors[g['label']],width=2)
    for off,text in [(244,row['image_id']),(258,row['status']),(272,','.join(row['flags'])),(288,'source: '+','.join(sorted({g['label'] for g in row['original_annotations']})))]:draw.text((x+3,y+off),text[:60],fill='black')
   name=part+f'-{start//16+1:03d}.jpg';canvas.save(out/name,quality=90);links.append(name)
 with (out/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>Phase6A approved photographs</title><h1>Source annotations; no human verification implied</h1>'+''.join(f'<h2>{html.escape(name)}</h2><img style="max-width:100%" src="{html.escape(name)}">' for name in links))
 print('CONTACTS',links)

if __name__=='__main__':run()
