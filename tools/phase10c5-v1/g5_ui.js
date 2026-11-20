'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',clone=x=>JSON.parse(JSON.stringify(x)),equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
let reg,hash,token,schema,absence,state,history=[],index=0,visible=[],draftKey,edited={},drawing=false,drag=null;
const task=()=>visible[index],photo=()=>reg.photos.find(p=>p.photo_key===task()?.photo_key),review=()=>state.reviews.find(r=>r.task_id===task()?.task_id);
function message(t){$('message').textContent=t;}
function point(x,y,p,inverse=false){const w=p.width,h=p.height,o=p.orientation;
 if(inverse)return {1:[x,y],2:[w-x,y],3:[w-x,h-y],4:[x,h-y],5:[y,x],6:[y,h-x],7:[w-y,h-x],8:[w-y,x]}[o];
 return {1:[x,y],2:[w-x,y],3:[w-x,h-y],4:[x,h-y],5:[y,x],6:[h-y,x],7:[h-y,w-x],8:[y,w-x]}[o];}
function mapped(b,p,inverse=false){let a=[[b[0],b[1]],[b[0],b[3]],[b[2],b[1]],[b[2],b[3]]].map(v=>point(...v,p,inverse));return [Math.min(...a.map(v=>v[0])),Math.min(...a.map(v=>v[1])),Math.max(...a.map(v=>v[0])),Math.max(...a.map(v=>v[1]))];}
const pixels=(b,p)=>b.map((v,i)=>v*(i%2?p.height:p.width));
function currentBox(){return edited[task().task_id]||review()?.box||task().box;}
function overlap(a,b){return Math.min(a[2],b[2])>Math.max(a[0],b[0])&&Math.min(a[3],b[3])>Math.max(a[1],b[1]);}
function reasons(){const p=photo(),b=currentBox(),out=[];if(p.role!=='optimization')out.push('Validation diagnostic: training negatives prohibited');if(equal(b,[0,0,1,1]))out.push('Whole-photo absence not established');for(const r of p.protected)if(overlap(b,r.box))out.push(r.reason);
 for(const q of state.proposals)if(q.photo_key===p.photo_key&&overlap(b,q.box))out.push('Overlaps an unresolved proposed object');
 for(const q of state.reviews)if(q.task_id!==task().task_id&&q.photo_key===p.photo_key&&['correct_chair','missing_annotation','uncertain'].includes(q.decision)&&overlap(b,q.box))out.push('Overlaps a Chair/missing/uncertain review');return [...new Set(out)];}
function checkpoint(){history.push(clone({state,edited}));if(history.length>40)history.shift();}
function persist(){try{localStorage.setItem(draftKey,JSON.stringify(state));localStorage.setItem(draftKey+'-reviewer',$('reviewer').value);}catch(e){message('Browser draft unavailable; save a versioned review regularly.');}}
function payload(){return clone(state);}
function identity(){const p=photo();return Object.fromEntries(['photo_key','image_id','sha256','identity_group','width','height','orientation'].map(k=>[k,p[k]]));}
function who(){if(!$('reviewer').value.trim()||!$('date').value)throw Error('Enter your name and date once in the header.');return {reviewer:$('reviewer').value.trim(),review_date:$('date').value};}
function decide(decision,negative=false){try{const person=who();if(negative&&reasons().length)throw Error(reasons().join('; '));checkpoint();const t=task(),b=clone(currentBox()),old=review();
 const sourceConfirmed=decision==='not_chair'&&t.source_grounded&&t.group!=='protected_furniture'&&equal(b,t.box)&&(!negative||old?.source_label_confirmed===true);
 const note=$('note').value.trim(),action=negative?absence:`Reviewer selected ${decision} for ${t.kind}: ${t.label}.`;
 const r={...identity(),...person,task_id:t.task_id,box:b,geometry_kind:equal(b,t.box)?'source':'human_corrected',decision,human_reviewed:true,confirmed:true,
 evidence:note||action,evidence_kind:note?'human_note':'explicit_action_log',action_log:action,hard_negative:negative,source_label_confirmed:sourceConfirmed};
 if(negative)Object.assign(r,{explicit_absence_statement:absence,confirmed_box:clone(b),review_covers_seating_stools:true});
 state.reviews=state.reviews.filter(x=>x.task_id!==t.task_id);state.reviews.push(r);persist();render();message(negative?'Explicit Chair-free box recorded. Training remains disabled.':'Your object decision was recorded. It does not establish Chair absence.');
 }catch(e){message(e.message);}}
function svg(tag,attrs,parent){let e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
function rectangle(b,p,color,parent,label,fill=.02){b=mapped(pixels(b,p),p);let e=svg('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke',fill:color,'fill-opacity':fill},parent);svg('title',{},e).textContent=label;return e;}
function render(){const t=task();if(!t){$('object-title').textContent='No objects in this filter';$('photo-image').removeAttribute('href');$('active').replaceChildren();$('boxes').replaceChildren();for(const id of ['correct','other','missing','uncertain','negative','draw','reset-box'])$(id).disabled=true;return;}
 for(const id of ['correct','other','missing','uncertain','draw','reset-box'])$(id).disabled=false;
 const p=photo(),r=review(),w=p.orientation>=5?p.height:p.width,h=p.orientation>=5?p.width:p.height;
 $('scene').setAttribute('viewBox',`0 0 ${w} ${h}`);const fit=Math.min(1,Math.max(1,$('viewport').clientWidth-2)/w,740/h);$('scene').style.width=(w*fit*Number($('zoom').value))+'px';
 $('photo-image').setAttribute('href','/api/photo/'+p.photo_key);$('photo-image').setAttribute('width',w);$('photo-image').setAttribute('height',h);$('original').href='/api/photo/'+p.photo_key;
 $('boxes').replaceChildren();$('active').replaceChildren();$('ghost').replaceChildren();
 for(const a of p.native){let type=a.category==='protected_furniture'?'furniture':a.category==='people'||a.category==='people_parts'?'people':a.category==='plants_flowers'?'plants':'other';if(a.ignored?!$('show-ignored').checked:!$('show-'+type).checked)continue;
 rectangle(a.box,p,a.ignored?'#777':type==='furniture'?'#087fc4':'#8239ad',$('boxes'),`Dataset annotation: ${a.label} · ${a.dataset}`);}
 for(const q of state.proposals.filter(x=>x.photo_key===p.photo_key))rectangle(q.box,p,'#b77a00',$('boxes'),'Inactive proposed object',.10);
 let color=r?.hard_negative?'#007b42':r&&['uncertain','missing_annotation'].includes(r.decision)?'#b77a00':t.model_prediction?'#d33740':t.group==='protected_furniture'?'#087fc4':'#8239ad';
 rectangle(currentBox(),p,color,$('active'),`ACTIVE: ${t.kind} · ${t.label}`, .09);
 $('object-title').textContent=`${t.model_prediction?'Saved model prediction':'Dataset annotation'}: ${t.label}`;
 $('context').textContent=`${t.source_dataset}${t.score!==undefined?' · score '+t.score.toFixed(3):''}. ${t.model_prediction?'Diagnostic review only; never used for training. Previous decision: '+t.previous_decision+'; evidence: '+t.previous_evidence:'Native evidence is a candidate label. Your review is not pre-filled.'}`;
 $('label-meaning').textContent=t.source_grounded&&t.group!=='protected_furniture'?'“Other object / Not Chair” confirms the displayed native object label, without claiming its surrounding box is Chair-free.':'“Correct Chair” confirms Chair semantics only. Source annotations remain unchanged; uncertain seating stays ignored.';
 $('photo-info').textContent=`${p.image_id} · ${p.width}×${p.height} · EXIF ${p.orientation} · ${p.role} · group ${p.identity_group}`;
 $('progress').textContent=`${index+1} / ${visible.length} objects`;const blocked=reasons();$('blocked').textContent=blocked.length?'Negative approval blocked: '+blocked.join('; '):'No known protected-box overlap. Look carefully for unannotated or partial seating before confirming.';$('negative').disabled=!!blocked.length;
 $('decision-state').textContent=r?`Recorded: ${r.decision}${r.hard_negative?' · explicitly Chair-free':''}${r.source_label_confirmed?' · native label confirmed':''}`:'No human decision recorded for this object.';
 const neg=state.reviews.filter(x=>x.hard_negative);$('counts').textContent=`${state.reviews.length} object decisions · ${neg.length} Chair-free boxes · ${new Set(neg.map(x=>x.identity_group)).size} source groups · ${state.proposals.length} inactive proposals`;
 $('note').value=r?.evidence_kind==='human_note'?r.evidence:'';
}
function filter(){const group=$('group').value;visible=reg.tasks.filter(t=>(t.eligible_for_training_review===($('scope').value==='training'))&&(group==='all'||t.group===group)&&(!$('unreviewed').checked||!state.reviews.some(r=>r.task_id===t.task_id)));index=0;render();}
function move(n){if(!visible.length)return;index=(index+n+visible.length)%visible.length;drawing=false;render();}
function undo(){if(!history.length)return;const old=history.pop();state=old.state;edited=old.edited;persist();render();message('Last action undone. Restored confirmations apply only to their original unchanged boxes.');}
async function api(path,data){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(data)}),v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function local(e){let q=new DOMPoint(e.clientX,e.clientY).matrixTransform($('scene').getScreenCTM().inverse()),p=photo();return [Math.max(0,Math.min(p.orientation>=5?p.height:p.width,q.x)),Math.max(0,Math.min(p.orientation>=5?p.width:p.height,q.y))];}
$('scene').addEventListener('pointerdown',e=>{if(!drawing||e.button!==0)return;e.preventDefault();drag={start:local(e),end:local(e)};$('scene').setPointerCapture(e.pointerId);});
$('scene').addEventListener('pointermove',e=>{if(!drag)return;drag.end=local(e);$('ghost').replaceChildren();let b=[Math.min(drag.start[0],drag.end[0]),Math.min(drag.start[1],drag.end[1]),Math.max(drag.start[0],drag.end[0]),Math.max(drag.start[1],drag.end[1])];svg('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],stroke:'#f3b334',fill:'none','stroke-width':2},$('ghost'));});
$('scene').addEventListener('pointerup',e=>{if(!drag)return;drag.end=local(e);let p=photo(),b=mapped([Math.min(drag.start[0],drag.end[0]),Math.min(drag.start[1],drag.end[1]),Math.max(drag.start[0],drag.end[0]),Math.max(drag.start[1],drag.end[1])],p,true);drag=null;drawing=false;
 // Browser CSS/PointerEvent transforms have subpixel float noise. Snap only
 // within 0.0001 original pixel of an integer, then round outward.
 b=b.map(x=>Math.abs(x-Math.round(x))<1e-4?Math.round(x):x);
 b=[Math.max(0,Math.floor(b[0])),Math.max(0,Math.floor(b[1])),Math.min(p.width,Math.ceil(b[2])),Math.min(p.height,Math.ceil(b[3]))];if(b[2]<=b[0]||b[3]<=b[1])return message('Draw a non-empty rectangle.');checkpoint();let normalized=b.map((v,i)=>v/(i%2?p.height:p.width));
 if($('draw-mode').value==='correct'){edited[task().task_id]=normalized;state.reviews=state.reviews.filter(r=>r.task_id!==task().task_id);}else state.proposals.push({...identity(),proposal_id:crypto.randomUUID(),box:normalized,original_pixel_edges:b,approved_for_training:false,kind:'missing_or_uncertain_object'});
 persist();render();message('Box saved as an unapproved change. Review again; no confirmation carries onto edited geometry.');});
for(const [id,decision]of [['correct','correct_chair'],['other','not_chair'],['missing','missing_annotation'],['uncertain','uncertain']])$(id).onclick=()=>decide(decision);
$('negative').onclick=()=>decide('not_chair',true);$('undo').onclick=undo;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
for(const id of ['scope','group','unreviewed'])$(id).onchange=filter;
for(const id of ['show-furniture','show-people','show-plants','show-other','show-ignored','zoom'])$(id).onchange=render;
$('draw').onclick=()=>{if(!task())return;drawing=true;message('Draw a rectangle on the photograph. This does not approve it.');};
$('reset-box').onclick=()=>{checkpoint();delete edited[task().task_id];state.reviews=state.reviews.filter(r=>r.task_id!==task().task_id);persist();render();};
$('remove-proposal').onclick=()=>{let i=state.proposals.findLastIndex(p=>p.photo_key===photo()?.photo_key);if(i<0)return;checkpoint();state.proposals.splice(i,1);persist();render();};
$('validate').onclick=async()=>{try{const v=await api('/api/validate',payload());message(`Checked: ${v.accepted_hard_negatives} accepted Chair-free boxes / ${v.negative_source_groups} groups; ${v.blocked.length} blocked. No training authorized.`);}catch(e){message(e.message);}};
$('export').onclick=async()=>{try{const p=payload(),v=await api('/api/export',p),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],{type:'application/json'}));a.download='object-review-'+v.export_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);message(`Saved local version ${v.export_id}: ${v.hard_negatives} Chair-free boxes / ${v.groups} groups. CPU fixture changed ${v.mask_changed_terms} terms. Training disabled.`);}catch(e){message('Export stopped: '+e.message);}};
$('restore').onchange=async()=>{try{const p=JSON.parse(await $('restore').files[0].text()),v=await api('/api/validate',p);if(v.blocked.length)throw Error('Blocked claims in export; retain original and resolve separately.');checkpoint();state=p;edited={};persist();filter();message('Restored existing human decisions after source/taxonomy validation.');}catch(e){message('Restore rejected: '+e.message);}};
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();return undo();}if(!task())return;let d={'1':'correct_chair','2':'not_chair','3':'missing_annotation','4':'uncertain'}[e.key];if(d)decide(d);if(['ArrowRight','n'].includes(e.key)){e.preventDefault();move(1);}if(['ArrowLeft','p'].includes(e.key)){e.preventDefault();move(-1);}});
async function init(){try{let r=await(await fetch('/api/registry')).json();reg=r.registry;hash=r.registry_sha256;token=r.token;schema=r.schema;absence=r.absence_statement;$('absence').textContent=absence;draftKey='furniture-ai-fast-'+hash;state={schema,registry_sha256:hash,taxonomy_sha256:reg.taxonomy_sha256,reviews:[],proposals:[]};$('date').value=new Date().toLocaleDateString('en-CA');$('reviewer').value=localStorage.getItem(draftKey+'-reviewer')||'';
 for(const group of [...new Set(reg.tasks.map(t=>t.group))]){let o=document.createElement('option');o.value=group;o.textContent=group.replaceAll('_',' ');$('group').append(o);}
 let saved=localStorage.getItem(draftKey);if(saved){const p=JSON.parse(saved),v=await api('/api/validate',p);if(v.blocked.length)throw Error('Browser draft contains blocked claims. Export and resolve in a separate version.');state=p;}
 filter();message('Ready. Pick an object, then one of the four actions. Chair-free confirmation is a separate explicit action.');window.reviewReady=true;
 }catch(e){message('Startup stopped: '+e.message);}}
window.addEventListener('resize',()=>{if(reg&&task()&&!drag)render();});init();
