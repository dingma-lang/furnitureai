"""Local photographic evidence charts from saved boxes; no image/model inference."""
import html
import json
import math
import os
from pathlib import Path
import sys
from PIL import Image,ImageDraw,ImageFont,ImageOps
from audit import OUT,ROOT,REC,read,save,sha,internal_rows,source_path,CLASSES

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(size):return ImageFont.truetype(FONT,size)
def rect(draw,box,size,color,width=2):
    w,h=size;draw.rectangle(tuple(int(x*s) for x,s in zip(box,(w,h,w,h))),outline=color,width=width)
def base(row,maxsize=(740,560)):
    with Image.open(source_path(row)) as im:return ImageOps.contain(im.convert('RGB'),maxsize)
def gt(im,row):
    d=ImageDraw.Draw(im)
    for a in row['annotations']:
        color='#a5a5a5' if a['ignore'] else ('#00bfff' if a['label']=='chair' else '#008a00')
        rect(d,a['box'],im.size,color,2)
        x,y=int(a['box'][0]*im.width),int(a['box'][1]*im.height)
        d.text((x+1,y+1),a['label']+(' [inactive]' if a['ignore'] else ''),font=font(12),fill=color,stroke_width=1,stroke_fill='black')

def run():
    if (OUT/'index.html').exists():raise FileExistsError('Visual report already exists')
    rows={r['image_id']:r for r in internal_rows()};a=read(OUT/'analysis.json');sample=read(OUT/'review-sample.json')['records']
    predictions=read(REC/'post-nms-predictions.json');lookup={}
    import csv
    with (OUT/'predictions-operating.csv').open() as f:
        for r in csv.DictReader(f):lookup[r['image_id'],int(r['index'])]=r['status']
    images=OUT/'images';images.mkdir(exist_ok=False)
    files={}
    for k,(ident,row) in enumerate(rows.items()):
        im=base(row,(1100,900));name=f'scene-{k:02d}.jpg';im.save(images/name,quality=90);files[ident]='images/'+name
    pages=[]
    for k,p in enumerate(a['photographs'][:4]):
        row=rows[p['image_id']];canvas=Image.new('RGB',(1800,670),'#f2f4f6');d=ImageDraw.Draw(canvas)
        left=base(row,(585,550));gt(left,row);canvas.paste(left,(0,75))
        d.text((10,12),'Available native GT',font=font(20),fill='black');d.text((10,40),'Blue=Chair; grey=inactive native label',font=font(14),fill='black')
        d.text((610,20),'Original / LR1e-5 predictions',font=font(19),fill='black')
        d.multiline_text((615,150),'Per-image boxes were not saved.\nOriginal: 19 Chair FPs overall.\nLR1e-5: 641 Chair FPs overall.\n\nNo boxes are reconstructed or invented.\nPhotograph-level original counts: unavailable.',font=font(17),fill='#444',spacing=9)
        right=base(row,(585,550));gt(right,row);rd=ImageDraw.Draw(right)
        ps=[(j,x) for j,x in enumerate(predictions[row['image_id']]) if x['label']=='chair' and x['score']>=.15 and lookup.get((row['image_id'],j))=='fp']
        ps.sort(key=lambda v:-v[1]['score'])
        for j,x in ps[:20]:
            rect(rd,x['box'],right.size,'#ed1515',2);rd.text((int(x['box'][0]*right.width),int(x['box'][1]*right.height)),f'{j}:{x["score"]:.2f}',font=font(11),fill='red',stroke_width=1,stroke_fill='white')
        canvas.paste(right,(1200,75));d.text((1210,12),'Recovered LR3e-5 E5',font=font(20),fill='black')
        d.text((1210,40),f'Top20 of {p["chair_FP"]} scored Chair FPs; red=unmatched',font=font(14),fill='black')
        d.text((10,630),row['image_id']+' | These are source-native mismatches; incomplete annotations may explain some.',font=font(17),fill='black')
        name=f'comparison-{k+1:02d}.jpg';canvas.save(OUT/name,quality=90);pages.append(name)
    reviewpages=[]
    for start in range(0,len(sample),8):
        sheet=Image.new('RGB',(1800,760),'white');d=ImageDraw.Draw(sheet)
        for z,s in enumerate(sample[start:start+8]):
            row=rows[s['image_id']];x=(z%4)*450;y=(z//4)*380
            d.text((x+6,y+5),s['sample_id']+f'  {s["status"]}  s={s["score"]:.3f}',font=font(16),fill='black')
            d.text((x+6,y+28),row['image_id'],font=font(13),fill='black')
            with Image.open(source_path(row)) as source:
                source=source.convert('RGB');context=ImageOps.contain(source,(210,260));gt(context,row)
                rect(ImageDraw.Draw(context),s['box'],context.size,'#ed1515',3);sheet.paste(context,(x+5,y+62))
                b=s['box'];padx=max(.04,(b[2]-b[0])*.2);pady=max(.04,(b[3]-b[1])*.2)
                frame=[max(0,b[0]-padx),max(0,b[1]-pady),min(1,b[2]+padx),min(1,b[3]+pady)]
                crop=source.crop(tuple(round(v*t) for v,t in zip(frame,(source.width,source.height,source.width,source.height))))
                crop=ImageOps.contain(crop,(220,260));cd=ImageDraw.Draw(crop)
                mapped=[(b[0]-frame[0])/(frame[2]-frame[0]),(b[1]-frame[1])/(frame[3]-frame[1]),(b[2]-frame[0])/(frame[2]-frame[0]),(b[3]-frame[1])/(frame[3]-frame[1])]
                rect(cd,mapped,crop.size,'#ed1515',2);sheet.paste(crop,(x+225,y+62))
            bucket=s['primary_geometry_bucket'].replace('_',' ')
            d.text((x+5,y+326),bucket,font=font(12),fill='#333')
            d.text((x+5,y+349),'Visual hypothesis only; human decision remains blank.',font=font(12),fill='#555')
        name=f'review-contact-{start//8+1:02d}.jpg';sheet.save(OUT/name,quality=92);reviewpages.append(name)
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.3));classes=['sofa','chair','bed'];x=range(3)
    for offset,key,label in ((-.25,'original_category_counts','Original'),(0,'lr1_category_counts','LR1e-5 E5'),(.25,'recovered_category_counts','LR3e-5 E5')):
        axs[0].bar([i+offset for i in x],[a[key][c]['fp'] for c in classes],.25,label=label)
    axs[0].set(xticks=list(x),xticklabels=classes,ylabel='Source-native FP at 0.15',title='Category contribution');axs[0].legend(fontsize=8)
    bins=a['category_aggregates']['chair']['confidence_bins'];b=['0.15–0.3','0.3–0.5','0.5–0.75','0.75–0.9','0.9–1']
    axs[1].bar(b,[bins.get(q,0) for q in b],color='#bd3333');axs[1].set(title='Recovered Chair FP confidence',ylabel='Scored unmatched predictions');axs[1].tick_params(axis='x',rotation=25)
    bins=a['category_aggregates']['chair']['area_bins'];b=['<1%','1–5%','5–20%','>=20%']
    axs[2].bar(b,[bins.get(q,0) for q in b],color='#3377aa');axs[2].set(title='Recovered Chair FP area',xlabel='Fraction of photograph area')
    fig.tight_layout();fig.savefig(OUT/'breakdown.png',dpi=150);plt.close(fig)
    data={ident:dict(image=files[ident],width=r['width'],height=r['height'],GT=r['annotations'],predictions=[dict(p,index=j,status=lookup.get((ident,j),'below_floor')) for j,p in enumerate(predictions[ident]) if p['score']>=.15]) for ident,r in rows.items()}
    summarytable=''.join('<tr><td>'+html.escape(r['image_id'])+'</td><td>'+str(r['chair_FP'])+'</td><td>'+str(r['chair_TP'])+'</td><td>'+str(r['chair_unscored'])+'</td></tr>' for r in a['photographs'])
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Chair false-positive investigation</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1500px;padding:0 20px;color:#182333}h1{font-size:28px}img{max-width:100%}.warning{background:#fff4d6;padding:18px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}svg{max-height:750px;width:100%;background:#eee}label{margin:12px}section{margin:30px 0}.missing{background:#eceff3;padding:16px}a{color:#135ea8}</style>
<h1>Chair false-positive investigation — saved results only</h1>
<p>No Phase10D training, GPU inference, downloads or checkpoint changes. Native unmatched counts are not automatically confirmed visual errors.</p>
<div class="warning">Only recovered LR3e-5 E5 retains per-image boxes. Original and LR1e-5 exact19-photo boxes were never saved; their visual overlays and per-photo/size/score breakdowns are unavailable. No substitute predictions are shown.</div>
<p><a href="report.md">Detailed findings</a> · <a href="photograph-summary.csv">Photograph counts</a> · <a href="predictions-operating.csv">All operating predictions</a> · <a href="review-template.csv">Blank human-review CSV (56 examples)</a></p>
<img src="breakdown.png" alt="Category, confidence and box area counts">
<h2>Four photographs account for all 1,094 scored Chair FPs</h2><table><tr><th>Photo</th><th>Chair FP</th><th>Chair TP</th><th>Unscored Chair</th></tr>'''+summarytable+'''</table>
<h2>Interactive saved-box viewer</h2><p>Blue: active native Chair GT. Grey: inactive native annotations. Red: scored unmatched. Green: TP. Orange: unscored because coverage is incomplete. Display filters do not change evaluation or production thresholds. Hover boxes for indices/scores.</p>
<label>Photo <select id="photo"></select></label><label>Display score ≥ <input id="threshold" type="number" min="0.15" max="1" value="0.15" step="0.05"></label>
<label>Maximum boxes <input id="limit" type="number" min="1" max="300" value="30"></label><label><input id="gt" type="checkbox" checked>GT</label><label>Status <select id="status"><option value="all">all</option><option value="fp">scored FP</option><option value="tp">TP</option><option value="unscored_incomplete">unscored</option></select></label>
<p id="counts"></p><svg id="scene" xmlns="http://www.w3.org/2000/svg"></svg>
<div class="missing">Original / LR1e-5 overlay: unavailable from saved artifacts. Aggregate Chair FP counts are19 and641; no per-image allocation is invented.</div>
<h2>Native GT / missing historical overlays / recovered predictions</h2>'''+''.join('<section><img src="'+p+'"></section>' for p in pages)+'''
<h2>Human-review sample:48 native-scored FPs +8 unscored controls</h2><p>Deterministic geometry, confidence and size strata,12FPs/photo. This is a targeted sample, not a random prevalence estimate. Complete review-template.csv; reviewer flags and decisions are blank. Suggested mechanisms: duplicate, fragment, localization mismatch, other furniture, background confusion, possible unannotated Chair, mixed, unresolved. Preserve multiple mechanisms and explain evidence. No automatic review acceptance occurs.</p>'''+''.join('<section><img src="'+p+'"></section>' for p in reviewpages)+'''
<script>const data='''+json.dumps(data,separators=(',',':'))+''';
const $=s=>document.getElementById(s),ns='http://www.w3.org/2000/svg';Object.keys(data).forEach(k=>{let o=document.createElement('option');o.value=k;o.textContent=k;$('photo').append(o)});
function element(name,attrs,parent){let e=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.append(e);return e}
function draw(){let r=data[$('photo').value],svg=$('scene');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);element('image',{href:r.image,width:r.width,height:r.height},svg);
if($('gt').checked)r.GT.forEach(g=>{let b=g.box;let e=element('rect',{x:b[0]*r.width,y:b[1]*r.height,width:(b[2]-b[0])*r.width,height:(b[3]-b[1])*r.height,fill:'none',stroke:g.ignore?'#aaa':'#00bfff','stroke-width':3,'stroke-dasharray':g.ignore?'8 5':'none'},svg);element('title',{},e).textContent=`Native ${g.label}; inactive=${g.ignore}`});
let all=r.predictions.filter(p=>p.label==='chair'&&p.score>=Number($('threshold').value)&&($('status').value==='all'||p.status===$('status').value)),ps=all.slice(0,Math.max(1,Math.min(300,Number($('limit').value)||30)));
ps.forEach(p=>{let b=p.box;let e=element('rect',{x:b[0]*r.width,y:b[1]*r.height,width:(b[2]-b[0])*r.width,height:(b[3]-b[1])*r.height,fill:'none',stroke:p.status==='fp'?'#ef2020':p.status==='tp'?'#00c96a':'#ffae00','stroke-width':2},svg);element('title',{},e).textContent=`index=${p.index} ${p.label} score=${p.score.toFixed(4)} ${p.status}`});$('counts').textContent=`Showing ${ps.length} of ${all.length} matching saved Chair predictions; native FP status is not visual confirmation.`}
$('photo').value='22a5848e375335e3';$('status').value='fp';['photo','threshold','limit','gt','status'].forEach(k=>$(k).addEventListener('change',draw));draw();</script></html>'''
    with (OUT/'index.html').open('x') as f:f.write(document)
    save(OUT/'visual-provenance.json',dict(source_images={r['image_id']:r['sha256'] for r in rows.values()},photos_decoded=19,
        source_images_modified=False,original_prediction_overlays_available=False,LR1_prediction_overlays_available=False,model_forwards=0,
        contact_pages=reviewpages,comparison_pages=pages,manual_review_flags_filled=False))
    print('Visual report:',OUT/'index.html','review pages',len(reviewpages))

if __name__=='__main__':run()
