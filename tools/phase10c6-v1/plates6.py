"""Private evidence plates: original frame, native crop and proposed inner crop."""
from PIL import Image,ImageDraw,ImageFont
from common6 import *

def run(expansion=False):
    records=candidates()[187:] if expansion else read(RAW/'candidates.json')['candidates'];folder=RAW/('inspection-expansion' if expansion else 'inspection');folder.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for start in range(0,len(records),6):
        sheet=Image.new('RGB',(1500,1300),'white');d=ImageDraw.Draw(sheet)
        for slot,r in enumerate(records[start:start+6]):
            x=slot%3*500;y=slot//3*650
            with Image.open(ROOT/r['path']) as opened:im=opened.convert('RGB')
            full=im.copy();draw=ImageDraw.Draw(full)
            for p in r['protected']:
                b=p['box'];draw.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#e53935',width=2)
            b=r['annotation']['box'];bb=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height];draw.rectangle(bb,outline='#285ad8',width=3)
            if r['proposal']:draw.rectangle(r['proposal']['box_pixels'],outline='#00bb66',width=2)
            d.text((x+8,y+5),r['candidate_id']+' '+r['native_label']+' ['+r['annotation']['dataset']+']',fill='black',font=font)
            d.text((x+8,y+28),'red protected · blue native · green proposed interior',fill='black',font=font)
            shown=full.copy();shown.thumbnail((480,280));sheet.paste(shown,(x+8,y+55))
            pad=8;crop=full.crop((max(0,int(bb[0])-pad),max(0,int(bb[1])-pad),min(im.width,int(bb[2])+pad+1),min(im.height,int(bb[3])+pad+1)))
            crop.thumbnail((295,255));sheet.paste(crop,(x+8,y+360))
            if r['proposal']:
                b=r['proposal']['box_pixels'];cut=im.crop(b);cut.thumbnail((170,190))
                if cut.width<150 and cut.height<150:cut=cut.resize((150,150),Image.Resampling.NEAREST)
                sheet.paste(cut,(x+318,y+380));d.text((x+318,y+352),'inner pixels',fill='black',font=font);d.text((x+318,y+575),str(b[2]-b[0])+'×'+str(b[3]-b[1])+'px',fill='black',font=font)
            else:d.text((x+312,y+380),'No certified\ninterior proposal',fill='#b02020',font=font)
            d.text((x+8,y+625),r['image_id'],fill='black',font=font)
        sheet.save(folder/('plate-'+str(start//6+(33 if expansion else 1)).zfill(2)+'.jpg'),quality=94)
    print('Inspection plates',len(list(folder.glob('*.jpg'))))

if __name__=='__main__':run('--expansion' in sys.argv)
