"""Bounded query-local training-only inspection plates,not invented labels."""
from PIL import Image,ImageDraw,ImageFont
from common21 import *


def run():
    records=read(DATA21/'hard-negative-candidates.json')['records'];cfg,prepared,train,dev,reg,inventory,photos=context21();byid={r['image_id']:r for r in train};pages=[]
    if len(records)>36:raise ValueError('Initial targeted review bound36;do not inspect every image')
    for start in range(0,len(records),9):
        batch=records[start:start+9];plate=Image.new('RGB',(1500,1140),'#eeeeee');draw=ImageDraw.Draw(plate)
        for index,record in enumerate(batch):
            row=byid[record['image_id']];require_training_row(row,train,dev)
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Source photograph changed')
            with Image.open(ROOT/row['path']) as source:
                if source.size!=(row['width'],row['height']) or source.getexif().get(274,1)!=1:raise ValueError('Source coordinate frame mismatch')
                image=source.convert('RGB')
            x,y=(index%3)*500,(index//3)*380;b=record['box_pixels'];pad=90;crop=[max(0,b[0]-pad),max(0,b[1]-pad),min(image.width,b[2]+pad),min(image.height,b[3]+pad)];im=image.crop(crop);d=ImageDraw.Draw(im);d.rectangle([b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],outline='red',width=2);im.thumbnail((480,285));plate.paste(im,(x+10,y+75))
            text=f"{start+index+1}: {record['native_label']} / {record['candidate_id'][-8:]}\n{record['image_id']}\nChair max {max(o['Chair_score'] for o in record['query_observations']):.3f} | red = exact proposed query region"
            draw.multiline_text((x+10,y+8),text,fill='black',spacing=4)
        p=DATA21/'contact-sheets'/('query-local-'+str(start//9+1)+'.png');buf=io.BytesIO();plate.save(buf,format='PNG');atomic21(p,buf.getvalue());pages.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),candidate_ids=[r['candidate_id'] for r in batch]))
    emit21(DATA21/'contact-sheets/index.json',dict(pages=pages,source='Exact optimization photos;red rectangles only;no predictions recomputed',model_inference=0));print('Bounded targeted review plates',len(pages),flush=True)


if __name__=='__main__':run()
