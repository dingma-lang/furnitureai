"""Freeze initial123 plus64 diverse existing native objects for visual inspection."""
from collections import Counter,defaultdict
from PIL import Image,ImageDraw,ImageFont
from common6 import *
from geometry6 import propose

def run():
    if RAW.exists() or REPORT.exists():raise FileExistsError('Fresh version required')
    RAW.mkdir(parents=True);REPORT.mkdir(parents=True);(RAW/'visual').mkdir();(RAW/'crops').mkdir()
    invpath=ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json';inv=read(invpath);reg=registry();rp={p['image_id']:p for p in reg['photos'] if p['role']=='optimization'}
    photos=[p for p in inv['photos'] if p['eligible_for_primary_AB']];items=[];seen=set()
    priority=('people','plants_flowers','furniture_parts')
    for p in sorted(photos,key=lambda p:p['image_id']):
        for a in sorted(p['annotations'],key=lambda a:a['annotation_key']):
            if a['metadata_candidate_eligible'] and a['mapped_auxiliary_category'] in priority:
                items.append(dict(photo=p,annotation=a,cohort='initial123'));seen.add(a['object_hypothesis_group'])
    if len(items)!=123:raise ValueError('Initial approved evidence inventory changed')
    extra_groups=('animals','wall_art','windows_doors','screens','kitchen_appliances','tableware_small_objects','clothing_bags','people_parts')
    initial_photos={i['photo']['image_id'] for i in items};extra=[]
    for group in extra_groups:
        options=[dict(photo=p,annotation=a,cohort='expanded-local') for p in photos for a in p['annotations'] if a['metadata_candidate_eligible'] and a['mapped_auxiliary_category']==group]
        # Prefer diverse photos/groups and sufficiently large objects; no swaps of
        # any training/holdout memberships or reliance on model confidence.
        used=set()
        for _ in range(8):
            options.sort(key=lambda i:(i['photo']['identity_group'] in used,i['photo']['image_id'] in initial_photos,i['annotation']['dataset']!='COCO',-((i['annotation']['box'][2]-i['annotation']['box'][0])*(i['annotation']['box'][3]-i['annotation']['box'][1])),i['annotation']['annotation_key']))
            selected=next((i for i in options if i['annotation']['object_hypothesis_group'] not in seen),None)
            if selected is None:break
            extra.append(selected);seen.add(selected['annotation']['object_hypothesis_group']);used.add(selected['photo']['identity_group']);initial_photos.add(selected['photo']['image_id']);options.remove(selected)
    items+=extra;records=[]
    for index,i in enumerate(items):
        p,a=i['photo'],i['annotation'];photo=rp[p['image_id']];path=ROOT/p['path']
        if sha(path)!=p['sha256']:raise ValueError('Source photo changed')
        with Image.open(path) as im:
            if im.size!=(p['width'],p['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source frame/orientation unsupported; no silent transform')
        region=propose(a['original_annotation'],p['width'],p['height'],photo['protected'])
        r=dict(candidate_id='C'+str(index+1).zfill(3),cohort=i['cohort'],image_id=p['image_id'],sha256=p['sha256'],path=p['path'],width=p['width'],height=p['height'],
            identity_group=p['identity_group'],category=a['mapped_auxiliary_category'],native_label=a['native_label'],annotation=a,rights=p['licence'],
            protected=photo['protected'],proposal=region,visual_inspected=False,human_reviewed=False,decision_level=None)
        records.append(r)
        if index%20==0:print('PREPARED',index+1,'/',len(items),flush=True)
    save(RAW/'candidates.json',dict(schema='phase10c6-frozen-native-candidates-v1',source_inventory_sha256=sha(invpath),registry_sha256=digest(reg),policy=POLICY,candidates=records,selection_seed=6106,deterministic_sort=True,proposals_automatically_approved=False))
    # Six candidates per plate, each with photograph context and native region.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for start in range(0,len(records),6):
        sheet=Image.new('RGB',(1500,1100),'white');d=ImageDraw.Draw(sheet)
        for slot,r in enumerate(records[start:start+6]):
            x=(slot%3)*500;y=(slot//3)*550
            with Image.open(ROOT/r['path']) as opened:im=opened.convert('RGB')
            full=im.copy();draw=ImageDraw.Draw(full)
            for p in r['protected']:
                b=p['box'];draw.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#dd3333',width=2)
            b=r['annotation']['box'];bb=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height];draw.rectangle(bb,outline='#285ad8',width=3)
            if r['proposal']:draw.rectangle(r['proposal']['box_pixels'],outline='#00ba66',width=3)
            full.thumbnail((480,245));sheet.paste(full,(x+10,y+50))
            pad=8;crop=im.crop((max(0,int(bb[0])-pad),max(0,int(bb[1])-pad),min(im.width,int(bb[2])+pad+1),min(im.height,int(bb[3])+pad+1)))
            crop.thumbnail((480,215));sheet.paste(crop,(x+10,y+305));crop.save(RAW/'crops'/f"{r['candidate_id']}.png")
            d.text((x+10,y+5),r['candidate_id']+' '+r['native_label']+' ['+r['annotation']['dataset']+']',fill='black',font=font)
            d.text((x+10,y+26),'green=polygon-interior proposal; red=protected; blue=source bbox',fill='black',font=font)
            d.text((x+10,y+526),'source '+r['image_id'],fill='black',font=font)
        sheet.save(RAW/'visual'/('plate-'+str(start//6+1).zfill(2)+'.jpg'),quality=92)
    print('FROZEN',len(records),'candidates;',len({r['image_id'] for r in records}),'photos;',sum(r['proposal'] is not None for r in records),'interior proposals;visual inspection still pending')

if __name__=='__main__':run()
