"""Bounded CPU contact sheets;no inferred or human labels are approved."""
from common7 import *

def run():
    from PIL import Image,ImageDraw,ImageFont
    _,_,train=inputs()
    obs=[json.loads(s) for s in (RUN/'analysis/high-confidence-Chair.jsonl').read_text().splitlines()]
    choices=sorted([r for r in obs if r['evidence_tier']=='B_automatic_unverified'],key=lambda r:(-r['student_Chair_score'],r['image_id'],r['query_index']))
    used={(r['image_id'],r['query_index']) for r in choices}
    extra=sorted([r for r in obs if r['student_Chair_score']>=.8 and r['reported_Chair_detection'] and not r['previously_observed_high_Chair_query'] and r['within_original_frame']],key=lambda r:(-r['student_Chair_score'],r['image_id'],r['query_index']))
    chosen_photos={r['image_id'] for r in choices}
    for r in extra:
        if (r['image_id'],r['query_index']) in used or r['image_id'] in chosen_photos:continue
        choices.append(r);used.add((r['image_id'],r['query_index']));chosen_photos.add(r['image_id'])
        if len(choices)>=24:break
    choices=choices[:24];pages=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for start in range(0,len(choices),8):
        page=Image.new('RGB',(1520,1360),'white');d=ImageDraw.Draw(page)
        subset=choices[start:start+8]
        for i,r in enumerate(subset):
            r['sample_id']='E7V'+str(start+i+1).zfill(2);row=require_train(r['image_id'],train);x=(i%2)*760;y=(i//2)*340
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Source photograph changed')
            with Image.open(ROOT/row['path']) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Registration changed')
                im=im.convert('RGB');overview=im.copy();od=ImageDraw.Draw(overview)
                for a in row['annotations']:
                    b=a['box'];od.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='yellow',width=3)
                b=r['box'];px=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height];od.rectangle(px,outline='red',width=4)
                overview.thumbnail((430,245));page.paste(overview,(x+5,y+85));crop=im.crop(tuple(px));crop.thumbnail((310,245));page.paste(crop,(x+440,y+85))
            lines=[f"{r['sample_id']} Chair={r['student_Chair_score']:.3f} original={r['original_same_query_Chair_score']:.3f} q={r['query_index']}",r['image_id']+' | native='+str(r['native_label']),
                   r['evidence_tier']+' | protected='+str(len(r['protected_overlap_keys'])),'Red=query;yellow=original positives/ignored;right=exact query crop']
            for j,line in enumerate(lines):d.text((x+5,y+3+19*j),line,font=font,fill='black')
        p=RUN/'analysis/contact-sheets'/f'queries-{start//8+1}.png';stream=io.BytesIO();page.save(stream,format='PNG');atomic(p,stream.getvalue())
        pages.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),sample_ids=[r['sample_id'] for r in subset]))
    emit(RUN/'analysis/inspection-samples.json',dict(records=choices,pages=pages,selection='All21B plus3new high-score protected-context crops;targeted,not prevalence sampling',human_reviews_created=0,labels_activated=0))
    print('Created',len(choices),'samples on',len(pages),'bounded contact sheets')

if __name__=='__main__':run()
