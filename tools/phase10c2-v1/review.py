"""Explicit human scope review, separate from predictions and model approvals."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
from n2 import *

DECISIONS=('genuine_chair','other_furniture','background_non_furniture','partial_or_ambiguous','missing_ground_truth','uncertain_defer')
SCOPE_DECISIONS=('complete_for_chair','missing_or_ambiguous_chairs','uncertain_defer')

def validate_submission(payload,queue,rows):
    if payload.get('schema')!='phase10c2-human-review-v1' or payload.get('queue_sha256')!=digest(queue):raise ValueError('Review queue/schema changed')
    q={r['item_id']:r for r in queue['records']};index={r['image_id']:r for r in rows};seen=set();trusted=[];diagnostic=[]
    for d in payload.get('records',[]):
        ident=d['item_id']
        if ident in seen or ident not in q:raise ValueError('Unknown/duplicate review item')
        seen.add(ident);item=q[ident]
        if any(d.get(k)!=item.get(k) for k in ('image_id','sha256','identity_group','historical_partition','internal_partition','kind')):raise ValueError('Review identity or partition changed')
        if not(d.get('human_reviewed') is True and d.get('confirmed') is True and isinstance(d.get('reviewer'),str) and d['reviewer'].strip() and isinstance(d.get('evidence'),str) and len(d['evidence'].strip())>=8):raise ValueError('Explicit human confirmation/reviewer/evidence required')
        date.fromisoformat(d['review_date'])
        if d['decision'] not in (SCOPE_DECISIONS if item['kind']=='training_scope_review' else DECISIONS):raise ValueError('Unknown human decision')
        if item['kind']=='prediction_diagnostic':
            diagnostic.append(d);continue # No validation prediction can ever become a training negative.
        if not item['permitted_training_evidence'] or not optimizer_member(index[item['image_id']]):raise PermissionError('No evaluation item promotion')
        if d['decision']!='complete_for_chair':continue
        r=index[item['image_id']]
        e={k:r[k] for k in ('image_id','sha256','identity_group','historical_partition','internal_partition')}
        e.update(category='chair',native_category=item['native_category'],kind='complete_with_positives',complete_for_chair=True,
            reviewer=d['reviewer'],review_date=d['review_date'],evidence=d['evidence'],human_reviewed=True,confirmed=True,
            trust_basis='human_category_scope_review',potential_unannotated_chair_regions=[],source_review_item_id=ident,
            native_annotation_digest=digest(r['annotations']))
        if any(x['image_id']==e['image_id'] for x in trusted):raise ValueError('Duplicate scope evidence')
        trusted.append(e)
    return dict(reviewed_items=len(seen),pending_items=len(q)-len(seen),diagnostic_records=diagnostic,trusted_records=trusted,
        training_authorized=False,configuration_automatically_enabled=False)

def build():
    from PIL import Image,ImageDraw,ImageOps,ImageFont
    queue=read(RAW/'review-queue.json');rows={r['image_id']:r for r in manifest()['rows']};folder=RAW/'review'
    if folder.exists():raise FileExistsError('Review material already exists')
    (folder/'images').mkdir(parents=True)
    images={};nominated=[]
    for item in queue['records']:
        r=rows[item['image_id']]
        if item['image_id'] not in images:
            path=ROOT/r['path']
            if sha(path)!=r['sha256']:raise ValueError('Source photograph changed')
            with Image.open(path) as im:
                im=ImageOps.contain(im.convert('RGB'),(1300,1100));name=f'images/scene-{len(images):03d}.jpg';im.save(folder/name,quality=94)
            images[r['image_id']]=name
        if item['kind']=='training_scope_review':nominated.append(r)
    # Bounded contact sheet permits quick inspection without inferring completeness.
    sheet=Image.new('RGB',(1350,1080),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for i,r in enumerate(nominated):
        with Image.open(folder/images[r['image_id']]) as im:
            im=ImageOps.contain(im,(440,310));d=ImageDraw.Draw(im)
            for a in r['annotations']:
                if a['label']=='chair':d.rectangle(tuple(round(x*y) for x,y in zip(a['box'],(im.width,im.height,im.width,im.height))),outline='cyan',width=2)
            x=i%3*450;y=i//3*360;sheet.paste(im,(x,y+35));draw.text((x+5,y+5),r['image_id'],font=font,fill='black')
    sheet.save(folder/'scope-contact.jpg',quality=93)
    data=[dict(item,width=rows[item['image_id']]['width'],height=rows[item['image_id']]['height'],row_annotations=rows[item['image_id']]['annotations'],image=images[item['image_id']]) for item in queue['records']]
    source='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase10C.2 human review</title>
<style>body{font:17px system-ui;margin:25px auto;max-width:1200px;padding:15px}label{display:block;margin:12px 0}select,textarea,input,button{font:inherit;padding:8px}textarea{width:95%;height:85px}svg{width:100%;max-height:650px;background:#eee}.warn{background:#fff0bd;padding:15px}.ok{color:#067b2d}button{margin:7px}#info{white-space:pre-wrap}</style>
<h1>Chair evidence and category-scope review</h1><p class="warn">No item is preapproved. The56 prediction examples are INTERNAL VALIDATION: their decisions cannot become training negatives. Only the9 optimization-photo scope reviews can supply proposed completeness evidence, after your explicit confirmation. This interface never authorizes training.</p>
<label>Reviewer <input id="reviewer" autocomplete="name"></label><label>Review date (YYYY-MM-DD) <input id="date" type="date"></label>
<label>Item <select id="items"></select></label><p id="info"></p><svg id="scene" xmlns="http://www.w3.org/2000/svg"></svg>
<p>Blue=active Chair native boxes, grey=inactive native labels, red=reviewed prediction. An absent box is not proof of absence. Zoom the image before confirming completeness.</p><a id="full" target="_blank">Open photograph copy at full size</a>
<label>Decision <select id="decision"></select></label><label>Evidence: explain what the photograph/box shows and any missing or ambiguous Chairs <textarea id="evidence"></textarea></label>
<label><input type="checkbox" id="confirm">I inspected this item and explicitly confirm this decision and explanation.</label>
<button id="save">Confirm this item</button><button id="prev">Previous</button><button id="next">Next</button><button id="export">Export NEW versioned review JSON</button>
<p id="status"></p><p>Download progress regularly; refreshing loses unsaved browser state. Nothing is sent over a network.</p><label>Restore a previously exported review <input type="file" id="restore" accept="application/json"></label>
<script>const queue='''+json.dumps(queue,separators=(',',':'))+''',data='''+json.dumps(data,separators=(',',':'))+''',queueHash='''+json.dumps(digest(queue))+''';
const $=s=>document.getElementById(s),ns='http://www.w3.org/2000/svg',answers={};let cursor=0;
const options={genuine_chair:'Genuine Chair',other_furniture:'Other furniture incorrectly identified as Chair',background_non_furniture:'Background / non-furniture',partial_or_ambiguous:'Partial furniture / ambiguous object',missing_ground_truth:'Missing ground-truth annotation',uncertain_defer:'Uncertain / defer',complete_for_chair:'All visible native Chair instances are boxed; category scope complete',missing_or_ambiguous_chairs:'Missing or ambiguous Chairs; keep negative supervision inactive'};
data.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${r.item_id} | ${r.kind} | ${r.image_id}${r.score===undefined?'':' | score '+r.score.toFixed(3)}`;$('items').append(o)});
function el(tag,attrs,parent){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.append(e);return e}
function render(){let r=data[cursor];$('items').value=cursor;$('scene').replaceChildren();$('scene').setAttribute('viewBox',`0 0 ${r.width} ${r.height}`);el('image',{href:r.image,width:r.width,height:r.height},$('scene'));
r.row_annotations.forEach(g=>{let b=g.box;el('rect',{x:b[0]*r.width,y:b[1]*r.height,width:(b[2]-b[0])*r.width,height:(b[3]-b[1])*r.height,fill:'none',stroke:g.ignore?'#aaa':'#00bfff','stroke-width':3},$('scene'))});if(r.box){let b=r.box;el('rect',{x:b[0]*r.width,y:b[1]*r.height,width:(b[2]-b[0])*r.width,height:(b[3]-b[1])*r.height,fill:'none',stroke:'red','stroke-width':4},$('scene'))}
$('full').href=r.image;$('info').textContent=`${r.item_id}: ${r.kind}\n${r.historical_partition}/${r.internal_partition}; training evidence permitted=${r.permitted_training_evidence}\n${r.image_id}`;$('decision').replaceChildren();let empty=document.createElement('option');empty.value='';empty.textContent='Choose after inspection';$('decision').append(empty);
let choices=r.kind==='training_scope_review'?['complete_for_chair','missing_or_ambiguous_chairs','uncertain_defer']:['genuine_chair','other_furniture','background_non_furniture','partial_or_ambiguous','missing_ground_truth','uncertain_defer'];choices.forEach(k=>{let o=document.createElement('option');o.value=k;o.textContent=options[k];$('decision').append(o)});
let a=answers[r.item_id];$('decision').value=a?a.decision:'';$('evidence').value=a?a.evidence:'';$('confirm').checked=false;$('status').textContent=`${Object.keys(answers).length}/65 explicitly confirmed; exports do not authorize training.`}
$('items').onchange=()=>{cursor=Number($('items').value);render()};$('prev').onclick=()=>{cursor=Math.max(0,cursor-1);render()};$('next').onclick=()=>{cursor=Math.min(data.length-1,cursor+1);render()};
$('save').onclick=()=>{let r=data[cursor];if(!$('confirm').checked||!$('reviewer').value.trim()||!/^\\d{4}-\\d{2}-\\d{2}$/.test($('date').value)||!$('decision').value||$('evidence').value.trim().length<8){alert('Reviewer, date, decision, concrete evidence and explicit confirmation are required.');return}
let a={};['item_id','kind','image_id','sha256','identity_group','historical_partition','internal_partition'].forEach(k=>a[k]=r[k]);Object.assign(a,{decision:$('decision').value,evidence:$('evidence').value.trim(),reviewer:$('reviewer').value.trim(),review_date:$('date').value,human_reviewed:true,confirmed:true});answers[r.item_id]=a;render()};
$('export').onclick=()=>{if(!Object.keys(answers).length){alert('No confirmed decisions to export.');return}let p={schema:'phase10c2-human-review-v1',queue_sha256:queueHash,records:Object.values(answers)},blob=new Blob([JSON.stringify(p,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='phase10c2-review-'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('restore').onchange=async()=>{try{let p=JSON.parse(await $('restore').files[0].text());if(p.schema!=='phase10c2-human-review-v1'||p.queue_sha256!==queueHash)throw Error('Wrong queue/version');for(let a of p.records){let q=data.find(r=>r.item_id===a.item_id);if(!q||['image_id','sha256','identity_group','historical_partition','internal_partition','kind'].some(k=>q[k]!==a[k])||a.human_reviewed!==true||a.confirmed!==true)throw Error('Invalid restored identity/confirmation');answers[a.item_id]=a}render()}catch(e){alert(e.message)}};render();</script></html>'''
    with (folder/'index.html').open('x') as f:f.write(source)
    print('Local interface:',folder/'index.html')

def submit(path,destination):
    if destination.exists():raise FileExistsError('New versioned submission destination required')
    q=read(RAW/'review-queue.json');payload=read(path);result=validate_submission(payload,q,manifest()['rows'])
    destination=destination.resolve()
    if not destination.is_relative_to(RAW/'submissions'):raise PermissionError('Only isolated versioned submissions allowed')
    destination.mkdir(parents=True)
    save(destination/'human-submission.json',payload);save(destination/'validation.json',result)
    save(destination/'trusted-evidence-proposed.json',dict(records=result['trusted_records'],training_authorized=False,requires_separate_activation_approval=True))
    print('Validated explicit human review; no training configuration enabled')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','validate']);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='build':build()
    else:
        if a.input is None or a.output is None:p.error('--input and --output required')
        submit(a.input,a.output)
