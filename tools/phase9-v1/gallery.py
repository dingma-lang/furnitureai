"""Authorized exposed-development galleries only; unmodified native GT and local attribution."""
import html
from PIL import Image,ImageDraw
from core import *
from data import exposed_rows,image
from fusion import inter,area

def run():
 selected=checked(OUT/'selection.json')['selected'];rows=exposed_rows()
 methods=list(dict.fromkeys(['original-baseline','adapted-baseline','adapted-scale_nms','adapted-tiled_guarded','adapted-scale_weighted',selected]))
 if len(methods)<6:methods.append('crossmodel_agreement')
 methods=methods[:6];results={n:checked(OUT/'evaluations/exposed'/(n+'.json')) for n in methods}
 folder=OUT/'galleries/exposed';folder.mkdir(parents=True);pages=[];attribution=[]
 for r in rows:
  im=image(r);canvas=Image.new('RGB',(1800,990),'white');draw=ImageDraw.Draw(canvas)
  draw.text((8,5),'BLUE: published native GT | GREEN: source match | RED: source FP | ORANGE: unscored | mixed?: geometric flag only',fill='black')
  for k,name in enumerate(methods):
   x=k%3*600;y=k//3*480+25;copy=im.copy();copy.thumbnail((590,410));canvas.paste(copy,(x+5,y+45))
   draw.text((x+5,y),name,fill='black');draw.text((x+5,y+16),'confidence .15; native IoU .5; not verified real-world truth',fill='black')
   for g in r['annotations']:
    b=g['box'];xy=[x+5+b[0]*copy.width,y+45+b[1]*copy.height,x+5+b[2]*copy.width,y+45+b[3]*copy.height];draw.rectangle(xy,outline='blue' if not g['ignore'] else 'gray',width=2)
    draw.text((xy[0],xy[1]),'GT '+g['label'],fill='blue',stroke_width=1,stroke_fill='white')
   result=results[name];event={}
   for c,v in result['metrics']['categories'].items():
    for e in v['events']:
     if e['image_id']==r['image_id']:event[e['index']]=e
   for j,p in enumerate(result['predictions'][r['image_id']]):
    if p['score']<.15:continue
    e=event.get(j);color='green' if e and e['outcome']=='tp' else 'red' if e else 'orange';b=p['box'];xy=[x+5+b[0]*copy.width,y+45+b[1]*copy.height,x+5+b[2]*copy.width,y+45+b[3]*copy.height]
    draw.rectangle(xy,outline=color,width=2);mixed=sum(inter(b,g['box'])/area(g['box'])>=.7 for g in r['annotations'] if not g['ignore'])>=2
    text=p['label']+f" {p['score']:.2f}"+(' dup' if e and e.get('duplicate') else '')+(' mixed?' if mixed else '')
    draw.text((xy[0],max(y+32,xy[1]-11)),text,fill=color,stroke_width=1,stroke_fill='white')
   current=[e for e in event.values() if e['outcome']=='tp'];missed=sum(not g['ignore'] for g in r['annotations'])-len(current)
   draw.text((x+5,y+458),f'Source matched: {len(current)}; missed annotations: {missed}. Unlabelled objects unknown.',fill='black')
  token=hashlib.sha256(r['image_id'].encode()).hexdigest()[:16]+'.jpg';canvas.save(folder/token,quality=90)
  pages.append('<h2>'+html.escape(r['image_id'])+'</h2><img style="max-width:100%" src="'+token+'">')
  attribution.append(dict(image_id=r['image_id'],licence_evidence=r.get('licence_evidence'),original_sha256=r['sha256'],derivative=token))
 save(folder/'attribution.json',dict(records=attribution,source_photographs_not_reencoded=True,viewable_derivatives_not_for_Git=True))
 with (folder/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>Phase9 exposed-development comparison</title><h1>18 exposed photos — source-native comparison only</h1><p>No image is independently human verified. Native Cabinetry/Desk unit ambiguity remains. Local creator/source/licence evidence is preserved in attribution.json. Galleries and photographs are excluded from Git.</p>'+''.join(pages))
 print('GALLERY18 exposure-only frames; source attribution retained; no frozen or fresh images opened for gallery',flush=True)

if __name__=='__main__':run()
