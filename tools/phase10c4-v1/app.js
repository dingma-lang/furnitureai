'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
let registry, registryHash, token, schema, current=0, selected=null, drag=null, history=[], state, draftKey;
const clone=x=>JSON.parse(JSON.stringify(x)), equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const photo=()=>registry.photos[current], region=()=>state.regions.find(r=>r.region_id===selected);
function point(x,y,p,inverse=false){const w=p.width,h=p.height,o=p.orientation;
 if(inverse)return {1:[x,y],2:[w-x,y],3:[w-x,h-y],4:[x,h-y],5:[y,x],6:[y,h-x],7:[w-y,h-x],8:[w-y,x]}[o];
 return {1:[x,y],2:[w-x,y],3:[w-x,h-y],4:[x,h-y],5:[y,x],6:[h-y,x],7:[h-y,w-x],8:[y,w-x]}[o];}
function mappedBox(b,p,inverse=false){let a=[[b[0],b[1]],[b[0],b[3]],[b[2],b[1]],[b[2],b[3]]].map(v=>point(...v,p,inverse));return [Math.min(...a.map(v=>v[0])),Math.min(...a.map(v=>v[1])),Math.max(...a.map(v=>v[0])),Math.max(...a.map(v=>v[1]))];}
function rawBox(display,p){const b=mappedBox(display,p,true);return [Math.max(0,Math.floor(b[0]+1e-8)),Math.max(0,Math.floor(b[1]+1e-8)),Math.min(p.width,Math.ceil(b[2]-1e-8)),Math.min(p.height,Math.ceil(b[3]-1e-8))];}
function intersects(a,b){return Math.min(a[2],b[2])>Math.max(a[0],b[0])&&Math.min(a[3],b[3])>Math.max(a[1],b[1]);}
function validBox(b,p){return b.length===4&&b.every(Number.isInteger)&&0<=b[0]&&b[0]<b[2]&&b[2]<=p.width&&0<=b[1]&&b[1]<b[3]&&b[3]<=p.height;}
function blocking(b,p,exclude=null){let reasons=[];for(const r of p.restricted)if(intersects(b,r.box_pixels))reasons.push(r.reason);
 for(const r of state.regions)if(r.photo_key===p.key&&r.region_id!==exclude&&r.status!=='rejected'&&intersects(b,r.box_pixels))reasons.push(r.kind==='potential_positive'||r.status==='uncertain'?'potential seating/uncertain region':'another candidate rectangle');
 if(equal(b,[0,0,p.width,p.height]))reasons.push('whole-image absence requires a separate completeness review');return [...new Set(reasons)];}
function msg(s,ok=false){$('message').textContent=s;$('message').style.color=ok?'#075b60':'#9a2020';}
function checkpoint(){history.push(clone(state));if(history.length>30)history.shift();}
function revoke(r){r.status='draft';r.human_reviewed=false;r.confirmed=false;delete r.confirmed_box_pixels;delete r.confirmed_revision;delete r.review_covers_seating_stools;delete r.no_chair_stool_or_ambiguous_part;}
function saveDraft(){try{localStorage.setItem(draftKey,JSON.stringify(state));}catch(e){msg('Browser draft storage unavailable; export progress regularly.');}}
function geometryChanged(r,b){if(equal(r.box_pixels,b))return;r.box_pixels=b;r.revision++;revoke(r);}
function payload(){return clone(state);}
async function api(path,data){let response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':token},body:JSON.stringify(data)});let body=await response.json();if(!response.ok)throw Error(body.error||response.statusText);return body;}
function svg(tag,attrs,parent){let e=document.createElementNS(NS,tag);for(let [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
function drawRect(box,p,color,parent,attrs={}){let b=mappedBox(box,p);return svg('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'none',stroke:color,'stroke-width':2,...attrs},parent);}
function draw(){const p=photo();$('scene').setAttribute('viewBox',`0 0 ${p.display_width} ${p.display_height}`);
 const fit=Math.min(1,Math.max(1,$('viewport').clientWidth-2)/p.display_width,740/p.display_height);$('scene').style.width=(p.display_width*fit*Number($('zoom').value))+'px';
 if($('photo-image').getAttribute('href')!=='/api/photo/'+p.key)$('photo-image').setAttribute('href','/api/photo/'+p.key);$('photo-image').setAttribute('width',p.display_width);$('photo-image').setAttribute('height',p.display_height);
 $('native').replaceChildren();$('regions').replaceChildren();$('ghost').replaceChildren();
 for(const a of p.annotations){if(a.ignore&&!$('show-ignored').checked)continue;if(!a.ignore&&!$('show-positive').checked)continue;
  let e=drawRect(a.box_pixels,p,a.ignore?'#808080':a.label==='chair'?'#008ce0':'#397b59',$('native'),{fill:a.ignore?'url(#hatch)':'none','fill-opacity':a.ignore?.16:1,'pointer-events':'none'});svg('title',{},e).textContent=`Native ${a.label}; ${a.ignore?'restricted/ignored':'positive'}; ${a.native_dataset} ${a.native_id}`;}
 for(const r of state.regions.filter(r=>r.photo_key===p.key)){let color=r.status==='rejected'?'#999':r.kind==='potential_positive'||r.status==='uncertain'?'#d68a04':r.status==='confirmed_negative'?'#078642':'#087c99';
  let e=drawRect(r.box_pixels,p,color,$('regions'),{fill:color,'fill-opacity':.08,'stroke-dasharray':r.status==='draft'?'7 4':'none','data-region':r.region_id,'stroke-width':r.region_id===selected?3:2});e.style.cursor='move';svg('title',{},e).textContent=`${r.region_id}: ${r.status}`;
  if(r.region_id===selected){let b=mappedBox(r.box_pixels,p),size=10*p.display_width/Math.max($('scene').getBoundingClientRect().width,1);
   for(const [handle,x,y]of [['nw',b[0],b[1]],['ne',b[2],b[1]],['se',b[2],b[3]],['sw',b[0],b[3]]])svg('rect',{x:x-size/2,y:y-size/2,width:size,height:size,fill:'white',stroke:color,'data-handle':handle,'data-region':r.region_id,class:'handle'},$('regions'));}}
 $('photo').value=p.key;$('photo-info').textContent=`${p.image_id} · group ${p.identity_group} · original ${p.width}×${p.height}px · EXIF ${p.orientation} · ${p.priority_scope_review?'previous completeness declined; regional evidence only':'optimization training only'}`;
 $('full-image').href='/api/photo/'+p.key;drawList();updateCounts();}
function updateCounts(){const confirmed=state.regions.filter(r=>r.status==='confirmed_negative'),groups=new Set(confirmed.map(r=>r.identity_group));$('counts').textContent=`${confirmed.length} locally confirmed rectangles · ${groups.size} source groups`;}
function drawList(){let list=$('region-list');list.replaceChildren();for(const r of state.regions.filter(r=>r.photo_key===photo().key)){let e=document.createElement('div');e.className='region-row'+(r.region_id===selected?' selected':'');e.textContent=`${r.region_id.slice(0,10)} · ${r.status} · ${r.kind} · [${r.box_pixels.join(', ')}]`;e.onclick=()=>{selected=r.region_id;draw();populate()};list.append(e);}if(!list.children.length)list.textContent='No rectangles on this photo. Drawing never approves one.';}
function populate(){let r=region();$('selected-label').textContent=r?`${r.kind} · ${r.status}`:'Draw or select a rectangle.';
 ['x1','y1','x2','y2'].forEach((id,i)=>$(id).value=r?r.box_pixels[i]:'');$('evidence').value=r?r.evidence||'':'';
 $('coordinate-info').textContent=r?`Original integer pixel edges; normalized XYXY: ${r.box_pixels.map((v,i)=>(v/(i%2?photo().height:photo().width)).toFixed(8)).join(', ')}`:'';
 for(const id of ['taxonomy-confirm','absence-confirm','explicit-confirm'])$(id).checked=false;}
function newRegion(b,kind){const p=photo();return {region_id:crypto.randomUUID(),photo_key:p.key,image_id:p.image_id,sha256:p.sha256,identity_group:p.identity_group,width:p.width,height:p.height,orientation:p.orientation,native_annotation_digest:p.native_annotation_digest,
 box_pixels:b,kind,status:'draft',revision:0,human_reviewed:false,confirmed:false,evidence:'',reviewer:'',review_date:'',taxonomy_sha256:registry.taxonomy_sha256,historical_partition:'train',internal_partition:'optimization'};}
function localPoint(event){let p=photo(),q=new DOMPoint(event.clientX,event.clientY).matrixTransform($('scene').getScreenCTM().inverse());return [Math.max(0,Math.min(p.display_width,q.x)),Math.max(0,Math.min(p.display_height,q.y))];}
$('scene').addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();const hit=e.target.closest('[data-region]'),p=photo();
 if(hit&&$('mode').value!=='potential_positive'){selected=hit.dataset.region;const r=region();drag={mode:e.target.dataset.handle?'resize':'move',handle:e.target.dataset.handle,start:localPoint(e),original:clone(r),before:clone(state)};draw();populate();}
 else if($('mode').value==='select'){selected=null;draw();populate();return;}
 else {let a=localPoint(e);drag={mode:'new',start:a,current:a,before:clone(state)};}
 $('scene').setPointerCapture(e.pointerId);});
$('scene').addEventListener('pointermove',e=>{if(!drag)return;let p=photo(),v=localPoint(e),a=drag.start,b;
 if(drag.mode==='new')b=[Math.min(a[0],v[0]),Math.min(a[1],v[1]),Math.max(a[0],v[0]),Math.max(a[1],v[1])];
 else {b=mappedBox(drag.original.box_pixels,p);if(drag.mode==='move'){let dx=v[0]-a[0],dy=v[1]-a[1];dx=Math.max(-b[0],Math.min(p.display_width-b[2],dx));dy=Math.max(-b[1],Math.min(p.display_height-b[3],dy));b=b.map((q,i)=>q+(i%2?dy:dx));}
  else {if(drag.handle.includes('w'))b[0]=v[0];else b[2]=v[0];if(drag.handle.includes('n'))b[1]=v[1];else b[3]=v[1];b=[Math.min(b[0],b[2]),Math.min(b[1],b[3]),Math.max(b[0],b[2]),Math.max(b[1],b[3])];}}
 drag.proposed=rawBox(b,p);$('ghost').replaceChildren();drawRect(drag.proposed,p,'#b72a16',$('ghost'),{'stroke-dasharray':'4 3','pointer-events':'none'});});
$('scene').addEventListener('pointerup',e=>{if(!drag)return;const p=photo(),d=drag;drag=null;$('ghost').replaceChildren();if($('scene').hasPointerCapture(e.pointerId))$('scene').releasePointerCapture(e.pointerId);
 let b=d.proposed;if(!b||!validBox(b,p)){draw();return;}let kind=d.mode==='new'?$('mode').value:d.original.kind;
 let blocked=kind==='negative'?blocking(b,p,d.original?.region_id):[];if(blocked.length){msg('Rectangle not applied: '+blocked.join('; ')+'. Mark potential seating instead where appropriate.');draw();return;}
 history.push(d.before);if(history.length>30)history.shift();let r;
 if(d.mode==='new'){r=newRegion(b,kind);state.regions.push(r);selected=r.region_id;}else{r=region();geometryChanged(r,b);}
 if(kind==='potential_positive')for(const other of state.regions)if(other.photo_key===p.key&&other.kind==='negative'&&other.status==='confirmed_negative'&&intersects(b,other.box_pixels))revoke(other);
 saveDraft();draw();populate();msg('Rectangle saved as unconfirmed draft. Inspect and explicitly confirm it.',true);});
$('scene').addEventListener('pointercancel',()=>{drag=null;draw();});
$('apply-coordinates').onclick=()=>{let r=region();if(!r)return msg('Select a rectangle.');if(['x1','y1','x2','y2'].some(id=>$(id).value.trim()===''))return msg('All four pixel coordinates are required.');let b=['x1','y1','x2','y2'].map(id=>Number($(id).value)),p=photo();if(!validBox(b,p))return msg('Use valid original-image integer pixel edges.');let reasons=r.kind==='negative'?blocking(b,p,r.region_id):[];if(reasons.length)return msg('Blocked: '+reasons.join('; '));checkpoint();geometryChanged(r,b);
 if(r.kind==='potential_positive')for(const q of state.regions)if(q.photo_key===r.photo_key&&q.status==='confirmed_negative'&&intersects(b,q.box_pixels))revoke(q);saveDraft();draw();populate();msg('Edited geometry is unconfirmed.',true);};
$('evidence').addEventListener('input',()=>{let r=region();if(r&&r.evidence!==$('evidence').value){if(r.status==='confirmed_negative')revoke(r);r.evidence=$('evidence').value;saveDraft();draw();$('selected-label').textContent=`${r.kind} · ${r.status}`;}});
$('approve').onclick=async()=>{let r=region(),p=photo();if(!r||r.kind!=='negative')return msg('Select a negative candidate, not a potential-positive rectangle.');
 if(!$('reviewer').value.trim()||!$('review-date').value||$('evidence').value.trim().length<8||!$('taxonomy-confirm').checked||!$('absence-confirm').checked||!$('explicit-confirm').checked)return msg('Reviewer, date, concrete evidence and all three explicit confirmations are required.');
 let reasons=blocking(r.box_pixels,p,r.region_id);if(reasons.length)return msg('Blocked: '+reasons.join('; '));checkpoint();Object.assign(r,{status:'confirmed_negative',human_reviewed:true,confirmed:true,confirmed_revision:r.revision,confirmed_box_pixels:clone(r.box_pixels),reviewer:$('reviewer').value.trim(),review_date:$('review-date').value,evidence:$('evidence').value.trim(),review_covers_seating_stools:true,no_chair_stool_or_ambiguous_part:true});
 try{let result=await api('/api/validate',payload());if(result.blocked.length)throw Error(JSON.stringify(result.blocked));saveDraft();draw();populate();msg('Server validated your explicit rectangle confirmation. Training is still disabled.',true);}catch(e){revoke(r);saveDraft();draw();populate();msg(e.message);}};
function mark(status){let r=region();if(!r)return;checkpoint();revoke(r);r.status=status;r.evidence=$('evidence').value.trim();r.reviewer=$('reviewer').value.trim();r.review_date=$('review-date').value;saveDraft();draw();populate();msg(status==='uncertain'?'Region remains ignored, not a negative.':'Candidate rejected; no negative target.',true);}
$('uncertain').onclick=()=>mark('uncertain');$('reject').onclick=()=>mark('rejected');
$('delete').onclick=()=>{if(!region())return;checkpoint();state.regions=state.regions.filter(r=>r.region_id!==selected);selected=null;saveDraft();draw();populate();};
$('undo').onclick=()=>{if(!history.length)return;let old=state,restored=history.pop();for(const r of restored.regions){let q=old.regions.find(x=>x.region_id===r.region_id);if(r.status==='confirmed_negative'&&(!q||!equal(q,r)))revoke(r);}state=restored;selected=null;saveDraft();draw();populate();msg('Undone. Any changed/restored approval must be explicitly confirmed again.',true);};
function changePhoto(i){current=Math.max(0,Math.min(registry.photos.length-1,i));selected=null;drag=null;draw();populate();msg('Whole photograph remains incomplete; approve only inspected rectangles.',true);}
$('photo').onchange=()=>changePhoto(registry.photos.findIndex(p=>p.key===$('photo').value));$('prev').onclick=()=>changePhoto(current-1);$('next').onclick=()=>changePhoto(current+1);
for(const id of ['zoom','show-positive','show-ignored'])$(id).onchange=draw;
$('validate').onclick=async()=>{try{let r=await api('/api/validate',payload());msg(`${r.confirmed_negative_rectangles} accepted rectangles, ${r.distinct_source_groups} groups; ${r.blocked.length} blocked. Training remains disabled.`,r.blocked.length===0);}catch(e){msg(e.message);}};
$('export').onclick=async()=>{const p=payload();try{let r=await api('/api/export',p);let blob=new Blob([JSON.stringify(p,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='chair-negative-'+r.export_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 msg(`Saved workstation version ${r.export_id} and downloaded JSON. ${r.validation.confirmed_negative_rectangles} accepted regions / ${r.validation.distinct_source_groups} groups. CPU fixture mask differences: ${r.mask_changed_terms}. No training approved.`,true);}catch(e){msg(e.message);}};
$('restore').onchange=async()=>{try{let p=JSON.parse(await $('restore').files[0].text()),v=await api('/api/validate',p);if(v.blocked.length)throw Error('Restored confirmed rectangles overlap protected regions; resolve them in a separate review.');checkpoint();state=p;selected=null;saveDraft();draw();populate();renderClarifications();msg('Restored existing exported decisions; server identity and coordinate checks passed.',true);}catch(e){msg('Restore rejected: '+e.message);}};
function renderClarifications(){let container=$('clarification-list');container.replaceChildren();for(const c of registry.clarifications){let card=document.createElement('section');card.className='clarification';let title=document.createElement('h3');title.textContent=`${c.item_id} · ${c.partition==='optimization'?'training-scope context':'validation diagnostic only'}`;card.append(title);
 let old=document.createElement('p');old.className='original';old.textContent=`Original decision: ${c.decision}\nOriginal evidence: ${c.evidence}\nQuestions: ${c.clarification_flags.join('; ')}`;card.append(old);
 let answer=document.createElement('textarea');answer.placeholder='Clarification or reason to defer; this does not change the original decision or approve any region.';card.append(answer);
 let action=document.createElement('select');for(const [v,t]of [['clarified','Provide clarification'],['deferred','Defer / remains unresolved']]){let o=document.createElement('option');o.value=v;o.textContent=t;action.append(o);}card.append(action);
 let previous=state.clarifications.find(x=>x.item_id===c.item_id);if(previous){answer.value=previous.evidence;action.value=previous.action;}
 let button=document.createElement('button');button.textContent='Explicitly record this response';button.onclick=()=>{if(!$('reviewer').value.trim()||!$('review-date').value||answer.value.trim().length<8)return msg('Set reviewer/date and a concrete clarification response.');checkpoint();state.clarifications=state.clarifications.filter(x=>x.item_id!==c.item_id);state.clarifications.push({item_id:c.item_id,action:action.value,evidence:answer.value.trim(),reviewer:$('reviewer').value.trim(),review_date:$('review-date').value,confirmed:true});saveDraft();msg('Separate clarification recorded; original review and training masks unchanged.',true);};card.append(button);container.append(card);}}
$('regions-tab').onclick=()=>{$('editor').hidden=false;$('clarifications').hidden=true;$('regions-tab').classList.add('active');$('clarifications-tab').classList.remove('active');};
$('clarifications-tab').onclick=()=>{$('editor').hidden=true;$('clarifications').hidden=false;$('clarifications-tab').classList.add('active');$('regions-tab').classList.remove('active');};
async function init(){try{const response=await fetch('/api/registry');if(!response.ok)throw Error('Registry unavailable');let r=await response.json();registry=r.registry;registryHash=r.registry_sha256;token=r.csrf_token;schema=r.schema;
 state={schema,registry_sha256:registryHash,taxonomy_sha256:registry.taxonomy_sha256,regions:[],clarifications:[]};draftKey='furniture-ai-negative-'+registryHash;
 for(const p of registry.photos){let o=document.createElement('option');o.value=p.key;o.textContent=`${p.priority_scope_review?'★ ':''}${p.image_id} · ${p.width}×${p.height}`;$('photo').append(o);}
 let stored=localStorage.getItem(draftKey);if(stored){let s=JSON.parse(stored),v=await api('/api/validate',s);for(const b of v.blocked){let q=s.regions.find(r=>r.region_id===b.region_id);if(q)revoke(q);}state=s;}
 draw();populate();renderClarifications();msg('Ready. No rectangle is approved automatically.',true);
 }catch(e){msg('Startup stopped: '+e.message);}}
init();
window.addEventListener('resize',()=>{if(registry&&state&&!drag)draw();});
