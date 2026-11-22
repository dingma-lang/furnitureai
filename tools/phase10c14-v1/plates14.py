"""Private contact sheets from exact optimization photos; source files untouched."""
from PIL import Image,ImageDraw,ImageFont
from common14 import *


def run():
    _,train,dev,photos,reg=training_context();byid={r['image_id']:r for r in train};folder=RAW14/'visual';folder.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    sets={'proposals':read(RAW14/'new-hard-negative-candidates.json')['candidates'],'context':read(RAW14/'training-inspection-sample.json')['records']};records=[];regions=read(SOURCE/'source-negative-evidence.json')['records']
    for name,items in sets.items():
        for start in range(0,len(items),6):
            plate=Image.new('RGB',(1500,1120),'white');draw=ImageDraw.Draw(plate)
            for slot,item in enumerate(items[start:start+6]):
                row=byid[item['image_id']];require_training_row(row,train,dev);path=ROOT/row['path']
                if sha(path)!=row['sha256']:raise ValueError('Source photo changed')
                with Image.open(path) as opened:
                    if opened.size!=(row['width'],row['height']) or opened.getexif().get(274,1)!=1:raise ValueError('Image registration/orientation differs')
                    im=opened.convert('RGB')
                full=im.copy();d=ImageDraw.Draw(full)
                for a in row['annotations']:
                    b=a['box'];d.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#159447' if not a['ignore'] else '#d03c25',width=2)
                for e in regions:
                    if e['image_id']==row['image_id']:
                        b=e['box'];d.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#168eb7',width=2)
                if name=='proposals':
                    pix=item['box_pixels'];poly=item['native_polygon'];d.line([tuple(x) for x in poly]+[tuple(poly[0])],fill='#c17700',width=2);ident=item['candidate_key'][:8];score=max(o['Chair_probability'] for o in item['query_observations'])
                else:
                    b=item['query_box'];pix=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height];ident=item['sample_id'];score=item['Chair_probability'];g=item['native_box'];d.rectangle([g[0]*im.width,g[1]*im.height,g[2]*im.width,g[3]*im.height],outline='#9c47bf',width=2)
                d.rectangle(pix,outline='#e600af',width=3);x=(slot%3)*500;y=(slot//3)*560
                draw.text((x+8,y+5),ident+' '+item['native_label']+' Chair='+format(score,'.4f'),font=font,fill='black');draw.text((x+8,y+26),row['image_id'],font=font,fill='black');full.thumbnail((480,275));plate.paste(full,(x+10,y+48))
                pad=15;crop=im.crop((max(0,int(pix[0])-pad),max(0,int(pix[1])-pad),min(im.width,int(pix[2])+pad+1),min(im.height,int(pix[3])+pad+1)));crop.thumbnail((480,195));plate.paste(crop,(x+10,y+335))
                draw.text((x+8,y+535),'Magenta=query; green=GT; blue=existing negative',font=font,fill='black')
                records.append(dict(kind=name,item_id=item.get('candidate_key',item.get('sample_id')),image_id=row['image_id'],plate=str((folder/(name+'-'+str(start//6+1)+'.jpg')).relative_to(ROOT))))
            plate.save(folder/(name+'-'+str(start//6+1)+'.jpg'),quality=92)
    save(RAW14/'visual-index.json',dict(records=records,all_optimization_only=True,source_images_unmodified=True))
    print('Training-only contact sheets created',len(list(folder.glob('*.jpg'))),flush=True)


if __name__=='__main__':run()
