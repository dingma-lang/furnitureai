"""Bounded evidence plates: unchanged photographs, saved prediction boxes only."""
from PIL import Image,ImageOps,ImageDraw,ImageFont
from common9 import *

def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def overlay(draw,b,size,color,width=2):
    draw.rectangle([round(v*s) for v,s in zip(b,[size[0],size[1]]*2)],outline=color,width=width)

def plate(records,rows,path,title):
    sheet=Image.new('RGB',(1800,990),'#f5f6f8');draw=ImageDraw.Draw(sheet);draw.text((12,8),title,font=font(20),fill='black')
    for i,p in enumerate(records):
        row=rows[p['image_id']];x=(i%3)*600;y=42+(i//3)*470
        draw.text((x+8,y),p['sample_id']+' '+p['image_id'],font=font(13),fill='black')
        draw.text((x+8,y+22),f"q{p['query_index']} Chair score={p['score']:.4f}  {p.get('status','UNVERIFIED TRAINING QUERY')}",font=font(15),fill='black')
        with Image.open(source_path(row)) as opened:
            if opened.size!=(row['width'],row['height']) or opened.getexif().get(274,1)!=1:raise ValueError('Orientation/dimensions changed')
            source=opened.convert('RGB');context=ImageOps.contain(source,(280,350));d=ImageDraw.Draw(context)
            for a in row['annotations']:overlay(d,a['box'],context.size,'#929292' if a['ignore'] else '#00a7df',2)
            overlay(d,p['box'],context.size,'#ef2222',3);sheet.paste(context,(x+7,y+57))
            b=p['box'];dx=max(.025,(b[2]-b[0])*.25);dy=max(.025,(b[3]-b[1])*.25);frame=[max(0,b[0]-dx),max(0,b[1]-dy),min(1,b[2]+dx),min(1,b[3]+dy)]
            bounds=[math.floor(frame[0]*source.width),math.floor(frame[1]*source.height),math.ceil(frame[2]*source.width),math.ceil(frame[3]*source.height)]
            crop=source.crop(bounds);crop=ImageOps.contain(crop,(300,350));cd=ImageDraw.Draw(crop)
            exact=[bounds[0]/source.width,bounds[1]/source.height,bounds[2]/source.width,bounds[3]/source.height]
            mapped=[(b[0]-exact[0])/(exact[2]-exact[0]),(b[1]-exact[1])/(exact[3]-exact[1]),(b[2]-exact[0])/(exact[2]-exact[0]),(b[3]-exact[1])/(exact[3]-exact[1])]
            overlay(cd,mapped,crop.size,'#ef2222',2);sheet.paste(crop,(x+294,y+57))
        draw.text((x+8,y+420),'Blue=active GT; grey=ignored; red=saved query',font=font(13),fill='#303f50')
        draw.text((x+8,y+441),'Visual judgment is not human verification.',font=font(13),fill='#303f50')
    sheet.save(path,quality=94)

def run():
    train,dev=subsets();rows={r['image_id']:r for r in train+dev};folder=OUT/'plates';folder.mkdir(exist_ok=False);samples=read(OUT/'review-sample.json')['records'];pages=[]
    for start in range(0,len(samples),6):
        path=folder/('development-'+str(start//6+1).zfill(2)+'.jpg');plate(samples[start:start+6],rows,path,'C9 development diagnostics ONLY — source-scored mismatch is not a confirmed visual error');pages.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),sample_ids=[p['sample_id'] for p in samples[start:start+6]]))
    unverified=read(OUT/'hard-negative-proposals.json')['unverified_queries'];selected=[]
    for p in sorted(unverified,key=lambda p:(-p['witness']['score'],p['image_id'],p['witness']['query_index'])):
        same=[q for q in selected if q['image_id']==p['image_id']]
        if len(same)>=2:continue
        m=metric_module()
        if any(m.iou(q['box'],p['witness']['box'])>.3 for q in same):continue
        selected.append(dict(p['witness'],image_id=p['image_id'],sample_id='T'+str(len(selected)+1).zfill(3),status='TRAINING UNVERIFIED - NOT A LABEL'))
        if len(selected)==12:break
    for start in range(0,len(selected),6):
        path=folder/('training-'+str(start//6+1).zfill(2)+'.jpg');plate(selected[start:start+6],rows,path,'C9 optimization-training queries — unverified, no training labels or approvals');pages.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),sample_ids=[p['sample_id'] for p in selected[start:start+6]]))
    save(OUT/'training-review-sample.json',dict(records=selected,training_enabled=False))
    save(OUT/'visual-plates.json',dict(pages=pages,development_sample=len(samples),training_unverified_sample=len(selected),source_images_modified=False,model_forwards=0))
    print('C9 plates',len(pages),'samples',len(samples)+len(selected),flush=True)

if __name__=='__main__':run()
