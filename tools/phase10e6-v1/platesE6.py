"""Bounded targeted inspection using existing optimization images only."""
from commonE6 import *
def run():
    from PIL import Image,ImageDraw,ImageFont
    c,train,_=context();rows=[json.loads(line) for line in (DATA/'high-confidence-observations.jsonl').read_text().splitlines()]
    choices=[];used=set();groups=set()
    contexts=['person_or_body_part','plant_or_decoration','appliance_or_screen','other_furniture','furniture_fragment_or_ambiguous_part','uncertain_or_unannotated_real_object','seating_or_ambiguous_seating','other_native_object']
    for target in ('chair','sofa','bed'):
        limit=16 if target=='chair' else 4
        for kind in contexts:
            opts=[r for r in rows if r['target']==target and r['source_context']==kind and r['within_original_frame'] and r['reported_target_detection'] and r['known_target_IoU']<.5]
            opts.sort(key=lambda r:(r['source_group'] in groups,-r['student_score'],r['candidate_id']))
            for r in opts:
                key=(r['image_id'],target,kind)
                if key in used:continue
                used.add(key);groups.add(r['source_group']);choices.append(dict(r,sample_id='V'+str(len(choices)+1).zfill(2)))
                if sum(v['target']==target and v['source_context']==kind for v in choices)>=2:break
            if sum(v['target']==target for v in choices)>=limit:break
    choices=choices[:24]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);pages=[]
    for start in range(0,len(choices),8):
        subset=choices[start:start+8];page=Image.new('RGB',(1520,1360),'white');d=ImageDraw.Draw(page)
        for i,r in enumerate(subset):
            row=require_train(r['image_id'],train)
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Source bytes changed')
            x=(i%2)*760;y=(i//2)*340
            with Image.open(ROOT/row['path']) as im:
                if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source registration changed')
                im=im.convert('RGB');overview=im.copy();od=ImageDraw.Draw(overview)
                for a in row['annotations']:
                    b=a['box'];od.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='yellow',width=3)
                b=r['box'];px=[b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height]
                od.rectangle(px,outline='red',width=4);overview.thumbnail((430,245));page.paste(overview,(x+5,y+85))
                crop=im.crop(tuple(px));crop.thumbnail((310,245));page.paste(crop,(x+440,y+85))
            title=f"{r['sample_id']} {r['target']}={r['student_score']:.3f} original={r['original_score_same_query']:.3f} q={r['query_index']}"
            lines=[title,r['image_id']+' | '+r['source_context'],f"native={r['native_label']} | protected overlaps={r['protected_overlap_count']}",'Red=query;yellow=original positives/ignored;right=exact query crop']
            for j,line in enumerate(lines):d.text((x+5,y+3+19*j),line,font=font,fill='black')
        path=DATA/'contact-sheets'/f'queries-{start//8+1}.png';out=io.BytesIO();page.save(out,format='PNG');atomic(path,out.getvalue())
        pages.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),sample_ids=[r['sample_id'] for r in subset]))
    emit(DATA/'inspection-samples.json',dict(records=choices,pages=pages,selection='Stratified high-score source contexts,optimization only;not a prevalence estimate or label acceptance',human_reviews_created=0))
    print('Prepared',len(choices),'targeted crops on',len(pages),'plates',flush=True)
if __name__=='__main__':run()
