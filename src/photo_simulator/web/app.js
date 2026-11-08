'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const state={project:null,assets:[],box:null,selectedId:null,mask_edits:[],protected_edits:[],editHistory:[],redoEdits:[],placement:{x:0,y:0,width:200,height:140,rotation:0,skew:0,flip:false,lock_aspect:true,anchor:"center"},options:{depth_assist:true,shadow:false,color_match:false,inpaint_radius:5},reference:{value:null,units:'m',note:''},mode:'select',polygon:[],drag:null,resultVisible:true,busy:false,camera:null,unsaved:false};
let jobToken=0,uploadToken=0;
function status(text,kind=''){ $('status').textContent=text;$('status').className=kind; }
function errorMessage(error){return error instanceof Error?error.message:String(error);}
async function request(path,body){
 const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 let result;try{result=await response.json();}catch(e){throw Error('The local server returned an unreadable response ('+response.status+').');}
 if(!response.ok)throw Error(typeof result.error==='string'?result.error:result.error?.message||result.message||'Request failed ('+response.status+').');
 return result;
}
function busy(on,text='Working…'){
 state.busy=on;document.querySelectorAll('#detection-list button').forEach(b=>b.disabled=on);$('busy').hidden=!on;$('progress-text').textContent=text;
 const has=!!state.project,selected=!!state.box;
 $('background-file').disabled=on||!has;$('clear-background').disabled=on||!state.project?.background_override;
 const outputs=state.project?.outputs||[],version=outputs.findIndex(o=>o.url===state.project?.latest_result_url);$('undo-render').disabled=on||version<=0;$('redo-render').disabled=on||version<0||version>=outputs.length-1;$('redo-polygon').disabled=on||!selected||!state.redoEdits.length;
 $('room-file').disabled=on;$('open-camera').disabled=on;$('project-picker').disabled=on;
 $('detect').disabled=on||!has;$('manual-select').disabled=on||!has;$('save-settings').disabled=on||!has;
 $('asset-picker').disabled=on||!has;$('asset-file').disabled=on||!has;
 $('placement-fields').disabled=on||!selected;$('options-fields').disabled=on||!has;
 $('generate').disabled=on||!selected||!state.project?.asset;
 for(const id of ['mask-add','mask-subtract','mask-protect','undo-polygon','build-mask'])$(id).disabled=on||!selected||(id==='undo-polygon'&&!state.editHistory.length);
 $('finish-polygon').disabled=on||state.polygon.length<3;
 $('show-mask').disabled=on||!state.project?.mask_url;
 $('restore-original').disabled=on||!has;$('show-result').disabled=on||!state.project?.latest_result_url;
 $('compare-toggle').disabled=on||!state.project?.latest_result_url;$('compare-slider').disabled=on||!state.project?.latest_result_url;
 $('download-result').setAttribute('aria-disabled',String(!state.project?.latest_result_url||on));
}
function markDirty(){state.unsaved=true;status('Placement changed. Generate to update the image, or save settings to continue later.');}
function dimensions(){return [Number(state.project?.image_width)||1,Number(state.project?.image_height)||1];}
function xml(tag,attrs={}){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;}
function point(event,element=$('selection-overlay')){
 const bounds=element.getBoundingClientRect(),[w,h]=dimensions();
 return [Math.max(0,Math.min(w,(event.clientX-bounds.left)/bounds.width*w)),Math.max(0,Math.min(h,(event.clientY-bounds.top)/bounds.height*h))];
}
function payload(){
 readPlacement();
 return {box:state.box,mask_edits:state.mask_edits,protected_edits:state.protected_edits,placement:{...state.placement},options:{...state.options},reference:{...state.reference},selected_detection_id:state.selectedId};
}
function readPlacement(){
 for(const [key,id] of Object.entries({x:'place-x',y:'place-y',width:'place-width',height:'place-height',rotation:'place-rotation',skew:'place-skew'})){
  const n=Number($(id).value);if(Number.isFinite(n))state.placement[key]=n;
 }
 state.placement.flip=$('place-flip').checked;state.placement.lock_aspect=$('lock-aspect').checked;state.placement.anchor=$('floor-anchor').checked?'floor':'center';if(state.placement.lock_aspect&&assetRatio())state.placement.height=state.placement.width/assetRatio();state.placement.source_angle=Number($('source-angle').value);state.placement.room_angle=Number($('room-angle').value);
 state.options={...state.options,shadow_mode:$('shadow-mode').value,depth_assist:$('depth-assist').checked,shadow:$('shadow').checked,color_match:$('color-match').checked,inpaint_radius:Number($('inpaint-radius').value)||5};
 readRestoration();
 const ref=$('reference-value').value;state.reference={value:ref===''?null:Number(ref),units:'m',note:$('reference-note').value.trim()};
}
function fillPlacement(){
 for(const [key,id] of Object.entries({x:'place-x',y:'place-y',width:'place-width',height:'place-height',rotation:'place-rotation',skew:'place-skew'}))$(id).value=Number(state.placement[key]||0).toFixed(2);
 $('lock-aspect').checked=!!state.placement.lock_aspect;$('floor-anchor').checked=state.placement.anchor==='floor';$('source-angle').value=state.placement.source_angle||0;$('room-angle').value=state.placement.room_angle||0;$('shadow-mode').value=state.options.shadow_mode||'legacy';
 $('place-flip').checked=!!state.placement.flip;$('rotation-value').textContent=Number(state.placement.rotation||0).toFixed(1)+'°';$('skew-value').textContent=Number(state.placement.skew||0).toFixed(2);
 $('depth-assist').checked=!!state.options.depth_assist;$('shadow').checked=!!state.options.shadow;$('color-match').checked=!!state.options.color_match;$('inpaint-radius').value=state.options.inpaint_radius||5;
 fillRestoration();
 $('reference-value').value=state.reference.value??'';$('reference-note').value=state.reference.note||'';
}
const restorationFields={visibility_margin:['visibility-margin',2],visibility_context:['visibility-context',64],mask_expand:['mask-expand',0],edge_feather:['edge-feather',0],edge_contract:['edge-contract',0],edge_decontam:['edge-decontam',0],shadow_opacity:['shadow-opacity',.18],shadow_softness:['shadow-softness',3],shadow_offset_x:['shadow-offset-x',0],shadow_offset_y:['shadow-offset-y',2],shadow_coverage:['shadow-coverage',.025]};
function readRestoration(){for(const [key,[id]] of Object.entries(restorationFields))state.options[key]=Number($(id).value);state.options.restoration_scope=$('restoration-scope').value;state.options.repair_method=$('repair-method').value;state.options.repair_resolution=Number($('repair-resolution').value);state.options.exclude_foreground_donors=$('exclude-donors').checked;}
function fillRestoration(){for(const [key,[id,value]] of Object.entries(restorationFields))$(id).value=state.options[key]??value;$('restoration-scope').value=state.options.restoration_scope||'full';$('repair-method').value=state.options.repair_method||'telea';$('repair-resolution').value=state.options.repair_resolution||1000;$('exclude-donors').checked=!!state.options.exclude_foreground_donors;}
function assetRatio(){const a=state.project?.asset;return a&&a.width>1&&a.height>1?(a.width-1)/(a.height-1):null;}
function fitSelection(){
 if(!state.box)return;const [x1,y1,x2,y2]=state.box;let width=Math.max(5,x2-x1),height=Math.max(5,y2-y1);
 if(state.placement.lock_aspect&&assetRatio()){width=Math.min(width,height*assetRatio());height=width/assetRatio();}
 state.placement={...state.placement,x:(x1+x2)/2,y:state.placement.anchor==='floor'?y2:y2-height/2,width,height};fillPlacement();drawPlacement();
}
function placementCorners(p){
 const h=p.lock_aspect&&assetRatio()?p.width/assetRatio():p.height,w=p.width,f=1-Math.abs(p.skew||0)*.55,shift=(p.skew||0)*w*.3;
 const t=(p.rotation||0)*Math.PI/180,c=Math.cos(t),s=Math.sin(t),offset=p.anchor==='floor'?h/2:0;
 return [[-w/2*f+shift,-h/2],[w/2*f+shift,-h/2],[w/2,h/2],[-w/2,h/2]].map(([x,y])=>[x*c-(y-offset)*s+p.x,x*s+(y-offset)*c+p.y]);
}
function selectBox(box,det=null){
 if(!Array.isArray(box)||box.length!==4)return;
 state.box=box.map(Number);state.selectedId=det?.id??null;state.mask_edits=[];state.protected_edits=[];state.editHistory=[];state.polygon=[];state.mode='select';state.redoEdits=[];
 $('show-mask').checked=false;$('mask-image').hidden=true;
 $('selected-title').textContent=det?.label?det.label.charAt(0).toUpperCase()+det.label.slice(1):'Manual selection';
 $('selected-details').textContent=det?`${Number.isFinite(Number(det.score))?(100*Number(det.score)).toFixed(1)+'% detection confidence · ':''}Box ${state.box.map(Math.round).join(', ')} pixels`:`User-drawn box · ${state.box.map(Math.round).join(', ')} pixels`;
 $('mask-tools').hidden=false;fitSelection();drawSelections();busy(false);state.unsaved=true;
 status('Selection ready. Choose replacement furniture, then generate. You can preview or correct the mask first.');
}
function drawSelections(){
 const svg=$('selection-overlay');svg.replaceChildren();if(!state.project)return;
 const [w,h]=dimensions();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.removeAttribute('hidden');
 const sw=Math.max(2,w/260),font=Math.max(12,w/48);
 for(const [index,d] of (state.project.detections||[]).entries()){
  const b=d.box||d.bbox;if(!b||b.length!==4)continue;
  const chosen=state.selectedId!=null&&String(state.selectedId)===String(d.id??index),g=xml('g',{'data-detection-id':d.id??index,role:'button',tabindex:0,'aria-label':`Select ${d.label||'furniture'}`});
  g.style.cursor='pointer';g.append(xml('rect',{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:chosen?'#d4ff9a25':'transparent',stroke:chosen?'#d5ff8c':'#fff','stroke-width':chosen?sw*1.5:sw}));
  const label=`${d.label||'furniture'} ${Number.isFinite(Number(d.score))?Math.round(Number(d.score)*100)+'%':''}`,ty=Math.max(font+6,b[1]);
  g.append(xml('rect',{x:b[0],y:ty-font-5,width:Math.min(w-b[0],label.length*font*.62+12),height:font+8,fill:chosen?'#245d46':'#1e3425cc',rx:3}));
  const text=xml('text',{x:b[0]+6,y:ty,fill:'#fff','font-size':font,'font-family':'system-ui'});text.textContent=label;g.append(text);
  const pick=e=>{if(state.busy||state.mode!=='select')return;e.stopPropagation();selectBox(b,{...d,id:d.id??index});};g.addEventListener('click',pick);g.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();pick(e);}});svg.append(g);
 }
 if(state.box){const [x1,y1,x2,y2]=state.box;svg.append(xml('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:'#c4fa7b','stroke-width':sw*1.5,'stroke-dasharray':`${sw*4} ${sw*2}`,'pointer-events':'none'}));}
 for(const edit of [...state.mask_edits,...state.protected_edits.map(p=>({...p,mode:'protect'}))]){
  const color=edit.mode==='subtract'?'#ee785b':edit.mode==='protect'?'#63b6ff':'#acf461';svg.append(xml('polygon',{points:edit.points.map(p=>p.join(',')).join(' '),fill:color+'44',stroke:color,'stroke-width':sw,'pointer-events':'none'}));
 }
 if(state.polygon.length){const color=state.mode==='subtract'?'#ee785b':state.mode==='protect'?'#63b6ff':'#acf461';svg.append(xml('polyline',{points:state.polygon.map(p=>p.join(',')).join(' '),fill:color+'22',stroke:color,'stroke-width':sw,'pointer-events':'none'}));for(const p of state.polygon)svg.append(xml('circle',{cx:p[0],cy:p[1],r:sw*2,fill:color,'pointer-events':'none'}));}
 if(state.drag?.type==='box'){const [a,b]=[state.drag.start,state.drag.end];svg.append(xml('rect',{x:Math.min(a[0],b[0]),y:Math.min(a[1],b[1]),width:Math.abs(b[0]-a[0]),height:Math.abs(b[1]-a[1]),fill:'#c4fa7b22',stroke:'#c4fa7b','stroke-width':sw,'pointer-events':'none'}));}
 svg.style.cursor=state.mode==='box'?'crosshair':['add','subtract','protect'].includes(state.mode)?'crosshair':'default';
 $('manual-select').classList.toggle('active',state.mode==='box');for(const id of ['add','subtract','protect'])$('mask-'+id).classList.toggle('active',state.mode===id);
 $('detection-list').replaceChildren();for(const [index,d] of (state.project.detections||[]).entries()){
  const button=document.createElement('button');button.type='button';button.textContent=`${d.label||'Furniture'} ${Number.isFinite(Number(d.score))?Math.round(Number(d.score)*100)+'%':''}`;button.disabled=state.busy;button.classList.toggle('active',String(d.id??index)===String(state.selectedId));button.onclick=()=>selectBox(d.box||d.bbox,{...d,id:d.id??index});$('detection-list').append(button);
 }
}
function drawPlacement(){
 const svg=$('placement-overlay');svg.replaceChildren();if(!state.project||!state.box){svg.setAttribute('hidden','');return;}const [w,h]=dimensions();svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.toggleAttribute('hidden',$('compare-toggle').checked||!state.resultVisible);
 const p=state.placement,sw=Math.max(2,w/250),corners=placementCorners(p);
 const poly=xml('polygon',{points:corners.map(p=>p.join(',')).join(' '),fill:'#e7ffb80a',stroke:'#eaffb6','stroke-width':sw,'stroke-dasharray':`${sw*4} ${sw*2}`});poly.style.cursor='move';svg.append(poly);
 svg.append(xml('circle',{cx:p.x,cy:p.y,r:sw*3,fill:'#f7ffe1',stroke:'#245d46','stroke-width':sw}));
 const stretch=assetRatio()?p.height*assetRatio()/p.width:1;
 $('geometry-warning').textContent=(Math.abs(stretch-1)>.15&&!p.lock_aspect?'Source proportions are stretched. Enable the aspect lock. ': '')+(Math.abs(p.skew)>.15?'Large perspective distortion: prefer a better-matched source photograph. ':'')+'2D alignment only. Source camera, hidden surfaces and physical fit remain unknown.';

}
function applyComparison(){
 const active=$('compare-toggle').checked&&!!state.project?.latest_result_url&&state.resultVisible,n=Number($('compare-slider').value);
 $('before-layer').hidden=!active;$('comparison-line').hidden=!active;$('before-layer').style.clipPath=`inset(0 ${100-n}% 0 0)`;$('comparison-line').style.left=n+'%';$('compare-value').textContent=n+'%';drawPlacement();
}
function updateVisuals(){
 const p=state.project;if(!p)return;const [w,h]=dimensions();
 for(const id of ['original-stage','result-stage']){$(id).style.aspectRatio=`${w} / ${h}`;$(id).classList.remove('empty');}
 $('original-image').hidden=false;$('original-image').src=p.original_url;$('before-image').src=p.original_url;$('source-size').textContent=`${w} × ${h}`;
 $('result-image').hidden=false;$('result-image').src=state.resultVisible&&p.latest_result_url?p.latest_result_url:p.original_url;
 $('mask-image').src=p.mask_url||'';$('mask-image').hidden=!p.mask_url||!$('show-mask').checked;
 if(p.latest_result_url){$('download-result').href=p.latest_result_url;$('download-result').download=`${p.id}-replacement.png`;}
 if(p.asset){$('asset-preview-wrap').hidden=false;$('asset-preview').src=p.asset.url;$('asset-details').textContent=[p.asset.name||'Replacement image',p.asset.license||p.asset.extraction?.provenance?.license||p.asset.source||'User supplied'].filter(Boolean).join(' · ')+(p.asset.extraction?.warnings?.length?' · '+p.asset.extraction.warnings.join(' '):'');}else{$('asset-preview-wrap').hidden=true;}
 $('layer-links').replaceChildren();for(const [key,title] of Object.entries({'restoration-mask':'Actual restoration mask','exposed':'Exposed-region mask','replacement-coverage':'Opaque replacement coverage',background:'Background PNG (estimated)',unknown:'Unknown-region mask',shadow:'Editable shadow layer','refined-asset':'Refined asset PNG'})){if(p.layers?.[key]){const a=document.createElement('a');a.href=p.layers[key];a.download=key+'.png';a.textContent=title;a.className='button';$('layer-links').append(a);}}
 const visibility=p.last_diagnostics?.background_repair?.visibility;$('visibility-note').textContent=visibility?`${visibility.exposed_pixels} exposed pixels; ${visibility.restoration_pixels} repair pixels including margin; ${visibility.hidden_skipped_pixels} hidden pixels skipped. Partial background: generate again after moving or resizing. Hidden geometry stays unknown.`:'Exposed-only backgrounds retain old furniture under opaque replacement pixels. Generate again after moving or resizing. Hidden geometry remains unknown.';
 $('background-note').textContent=(p.background_override?'Manual background override active. ':'')+'Hidden regions remain estimated. Correct an exported background and upload a same-size PNG; only removal-mask pixels are used. '+(state.options.background_patches?.length?state.options.background_patches.length+' explicit same-photo donor patches saved.':'');
 $('diagnostics').textContent=JSON.stringify({analysis:p.analysis||{},latest_output:p.outputs?.find(o=>o.url===p.latest_result_url)||null,geometry:{image_coordinates:'pixels',physical_scale:'unknown unless independently calibrated',reference:state.reference,physical_fit_verified:false}},null,2);
 const stages=p.analysis?.stages,items=[];if(Array.isArray(stages))for(const item of stages)items.push(`${item.name||item.model||'Stage'}: ${item.status||'complete'}`);else if(stages)for(const [name,item] of Object.entries(stages))items.push(`${name}: ${typeof item==='string'?item:item.status||'complete'}`);
 if(p.detections)items.unshift(`${p.detections.length} furniture detection${p.detections.length===1?'':'s'}`);$('analysis-summary').replaceChildren(...items.map(t=>{const d=document.createElement('div');d.textContent=t;return d;}));
 const warnings=p.analysis?.warnings||[];if(warnings.length){const d=document.createElement('div');d.textContent=warnings.join(' ');$('analysis-summary').append(d);}
 $('result-note').textContent=state.resultVisible&&p.latest_result_url?'Rough composite · Visual preview — physical dimensions not verified.':'Original shown. Generate a replacement to create an edited image.';
 drawSelections();drawPlacement();applyComparison();busy(state.busy);
}
function useProject(p,hydrate=true){
 state.project=p;state.resultVisible=true;if(hydrate){state.redoEdits=[];const s=p.settings||{};state.box=s.box||null;state.selectedId=s.selected_detection_id??null;state.mask_edits=s.mask_edits||[];state.protected_edits=s.protected_edits||[];state.editHistory=[...state.mask_edits.map(()=> 'mask'),...state.protected_edits.map(()=> 'protected')];state.placement={...state.placement,lock_aspect:s.placement?false:true,anchor:"center",...s.placement};state.options={depth_assist:true,shadow:false,color_match:false,inpaint_radius:5,...s.options};state.reference={value:null,units:'m',note:'',...s.reference};state.polygon=[];state.mode='select';state.unsaved=false;$('project-name').value=p.name||'';fillPlacement();$('mask-tools').hidden=!state.box;$('selected-title').textContent=state.box?'Saved selection':'Choose an object';$('selected-details').textContent=state.box?'Original image coordinates · '+state.box.map(Math.round).join(', '):'Click a detection or draw a manual box.';}
 updateVisuals();$('project-picker').value=p.id;
}
async function refreshProjects(){
 const result=await request('/api/projects');const picker=$('project-picker');picker.replaceChildren(new Option('Choose a saved project',''));
 for(const p of result.projects||[])picker.append(new Option(p.name||p.id,p.id));if(state.project)picker.value=state.project.id;
}
async function runJob(action,body,label){
 if(!state.project)throw Error('Upload a room photograph first.');const id=state.project.id,token=++jobToken;busy(true,label);status(label);
 try{
  const started=await request(`/api/projects/${encodeURIComponent(id)}/${action}`,body||{});
  if(!started.job_id){useProject(started.project||started,false);return;}
  let elapsed=0;
  while(token===jobToken){
   const job=await request('/api/jobs/'+encodeURIComponent(started.job_id));
   $('progress-text').textContent=typeof job.progress==='string'?job.progress:typeof job.progress==='number'?`${label} · ${job.progress}%`:label;
   if(job.status==='complete'){const project=job.project||await request('/api/projects/'+encodeURIComponent(id));useProject(project,false);state.unsaved=false;status(action==='analyze'?`${project.detections?.length||0} furniture detections ready. Select an object, or draw a manual selection.`:action==='mask'?'Mask preview ready. Add or subtract polygons to refine the selection.':'Replacement generated. Drag the placement outline or adjust its size to try another arrangement.','success');if(action==='mask'){$('show-mask').checked=true;updateVisuals();}return;}
   if(job.status==='failed')throw Error(typeof job.error==='string'?job.error:job.error?.message||'Processing failed. You can keep your photo and use a manual selection.');
   if(elapsed>1200000)throw Error('This operation is taking longer than expected. Your project is saved; reopen it to check its results.');
   await new Promise(r=>setTimeout(r,800));elapsed+=800;
  }
 }finally{busy(false);await refreshProjects().catch(()=>{});}
}
async function fileData(file){
 if(!file)throw Error('Choose an image first.');if(!['image/jpeg','image/png','image/webp'].includes(file.type))throw Error('Choose a JPG, PNG or WebP image.');if(file.size>18*1024*1024)throw Error('Choose an image smaller than 18 MB.');
 return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=()=>reject(Error('Could not read this image.'));r.readAsDataURL(file);});
}
async function uploadRoom(image,name){
 const token=++uploadToken;busy(true,'Saving photograph…');
 try{const p=await request('/api/projects',{name:name||$('project-name').value.trim()||'Room simulation',image});if(token!==uploadToken)return;state.box=null;state.project=null;useProject(p);await refreshProjects();status('Photo saved. Detect furniture, or draw a selection to begin.','success');}finally{busy(false);}
}
function handle(fn){return async(...args)=>{try{await fn(...args);}catch(e){busy(false);status(errorMessage(e),'error');}};}
function activateMode(mode){if(state.busy||!state.project)return;state.mode=state.mode===mode?'select':mode;state.polygon=[];drawSelections();$('finish-polygon').disabled=true;$('selection-help').textContent=state.mode==='box'?'Drag a rectangle around the furniture you want to replace.':state.mode==='protect'?'Click corners around a foreground object to preserve it, then finish the polygon.':state.mode==='add'?'Click corners to add a region to the removal mask, then finish the polygon.':state.mode==='subtract'?'Click corners to subtract a region from the removal mask, then finish the polygon.':'Click a detected object or draw a manual box.';}
function finishPolygon(){if(state.polygon.length<3)return;state.redoEdits=[];const edit={points:state.polygon.map(p=>p.map(v=>Math.round(v)))};if(state.mode==='protect'){state.protected_edits.push(edit);state.editHistory.push('protected');}else{state.mask_edits.push({...edit,mode:state.mode});state.editHistory.push('mask');}state.polygon=[];state.mode='select';drawSelections();busy(false);state.unsaved=true;status('Polygon saved in the selection. Preview mask or generate to apply it.');}
function stopCamera(){if(state.camera){state.camera.getTracks().forEach(t=>t.stop());state.camera=null;}$('camera-video').srcObject=null;$('capture-camera').disabled=true;}
async function startCamera(){
 stopCamera();$('camera-status').textContent='Requesting camera permission…';
 try{
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('Camera access needs a secure browser context. Open this application through localhost, or upload a photograph instead.');
  const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1600},height:{ideal:1200}},audio:false});
  if(!$('camera-dialog').open){stream.getTracks().forEach(t=>t.stop());return;}
  state.camera=stream;$('camera-video').srcObject=stream;await $('camera-video').play();$('capture-camera').disabled=false;$('camera-status').textContent='Frame your room, then capture a still photograph.';
 }catch(e){$('camera-status').textContent=e.name==='NotAllowedError'?'Camera permission was denied. Allow access in browser settings, or upload a photograph.':errorMessage(e);}
}
function wire(){
 $('room-file').onchange=handle(async()=>{const file=$('room-file').files[0];if(!file)return;busy(true,'Reading photograph…');const image=await fileData(file);await uploadRoom(image,$('project-name').value.trim()||file.name.replace(/\.[^.]+$/,''));$('room-file').value='';});
 $('project-picker').onchange=handle(async()=>{const id=$('project-picker').value;if(!id)return;if(state.unsaved&&!confirm('Open another simulation? Unsaved placement changes will be left behind.')){$('project-picker').value=state.project?.id||'';return;}busy(true,'Opening simulation…');try{useProject(await request('/api/projects/'+encodeURIComponent(id)));status('Saved simulation opened.','success');}finally{busy(false);}});
 $('detect').onclick=handle(()=>runJob('analyze',{},'Detecting furniture and estimating room structure…'));
 $('manual-select').onclick=()=>activateMode('box');for(const mode of ['add','subtract','protect'])$('mask-'+mode).onclick=()=>activateMode(mode);
 $('finish-polygon').onclick=finishPolygon;$('undo-polygon').onclick=()=>{const type=state.editHistory.pop();if(type){const edit=(type==='protected'?state.protected_edits:state.mask_edits).pop();state.redoEdits.push({type,edit});}state.polygon=[];drawSelections();busy(false);state.unsaved=true;};
 $('redo-polygon').onclick=()=>{const item=state.redoEdits.pop();if(item){(item.type==='protected'?state.protected_edits:state.mask_edits).push(item.edit);state.editHistory.push(item.type);}drawSelections();busy(false);state.unsaved=true;};
 for(const direction of ['undo','redo'])$(direction+'-render').onclick=handle(async()=>{busy(true,'Opening saved render version…');try{useProject(await request(`/api/projects/${state.project.id}/history`,{direction}));status('Saved render version restored; all version files retained.');}finally{busy(false);}});
 $('background-file').onchange=handle(async()=>{const file=$('background-file').files[0];if(!file)return;busy(true,'Saving corrected background…');try{const image=await fileData(file);useProject(await request(`/api/projects/${state.project.id}/background`,{image}),false);state.unsaved=true;status('Background correction saved. Generate to apply it only inside the removal mask.');}finally{$('background-file').value='';busy(false);}});
 $('clear-background').onclick=handle(async()=>{busy(true,'Clearing background override…');try{useProject(await request(`/api/projects/${state.project.id}/background`,{clear:true}),false);state.unsaved=true;status('Override cleared. Generate to return to classical restoration; upload history retained.');}finally{busy(false);}});
 for(const [id] of Object.values(restorationFields))$(id).oninput=()=>{readRestoration();markDirty();};for(const id of ['repair-method','repair-resolution','exclude-donors','restoration-scope'])$(id).onchange=()=>{readRestoration();markDirty();};

 const svg=$('selection-overlay');
 svg.addEventListener('pointerdown',e=>{if(state.busy||state.mode!=='box')return;e.preventDefault();state.drag={type:'box',start:point(e),end:point(e)};svg.setPointerCapture(e.pointerId);drawSelections();});
 svg.addEventListener('pointermove',e=>{if(state.drag?.type!=='box')return;state.drag.end=point(e);drawSelections();});
 svg.addEventListener('pointerup',e=>{if(state.drag?.type!=='box')return;const {start:a,end:b}=state.drag;state.drag=null;const box=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])].map(Math.round);if(box[2]-box[0]<5||box[3]-box[1]<5){drawSelections();status('Draw a larger box around the furniture.','error');return;}selectBox(box);});
 svg.addEventListener('click',e=>{if(state.busy||!['add','subtract','protect'].includes(state.mode))return;state.polygon.push(point(e));drawSelections();$('finish-polygon').disabled=state.polygon.length<3;});
 svg.addEventListener('dblclick',e=>{if(['add','subtract','protect'].includes(state.mode)&&state.polygon.length>=3){e.preventDefault();finishPolygon();}});
 $('build-mask').onclick=handle(()=>runJob('mask',payload(),'Preparing selection mask…'));
 $('show-mask').onchange=()=>{$('mask-image').hidden=!$('show-mask').checked||!state.project?.mask_url;};
 $('asset-picker').onchange=handle(async()=>{const id=$('asset-picker').value;if(!id||!state.project)return;busy(true,'Loading replacement furniture…');try{useProject(await request(`/api/projects/${state.project.id}/asset`,{asset_id:id}),false);if(state.placement.lock_aspect)fitSelection();status('Replacement furniture loaded. Generate to preview it in your room.','success');}finally{busy(false);}});
 $('asset-file').onchange=handle(async()=>{const f=$('asset-file').files[0];if(!f)return;busy(true,'Reading replacement image…');const image=await fileData(f);busy(true,'Isolating replacement furniture…');try{useProject(await request(`/api/projects/${state.project.id}/asset`,{image,name:f.name}),false);$('asset-picker').value='';if(state.placement.lock_aspect)fitSelection();status('Replacement image loaded. Generate to preview it in your room.','success');}finally{busy(false);$('asset-file').value='';}});
 for(const id of ['place-x','place-y','place-width','place-height','place-rotation','place-skew','place-flip','lock-aspect','source-angle','room-angle'])$(id).oninput=()=>{
  readPlacement();
  if(id==='place-height'&&state.placement.lock_aspect&&assetRatio()){state.placement.height=Number($('place-height').value);state.placement.width=state.placement.height*assetRatio();}
  fillPlacement();drawPlacement();markDirty();
 };
 $('floor-anchor').onchange=()=>{const p=state.placement,delta=p.height/2*($('floor-anchor').checked?1:-1),t=p.rotation*Math.PI/180;p.x-=Math.sin(t)*delta;p.y+=Math.cos(t)*delta;p.anchor=$('floor-anchor').checked?'floor':'center';fillPlacement();drawPlacement();markDirty();};
 $('reset-transform').onclick=()=>{state.placement={...state.placement,rotation:0,skew:0,flip:false,lock_aspect:true,source_angle:0,room_angle:0};fitSelection();markDirty();};
 $('align-reference').onclick=handle(()=>{readPlacement();const angle=state.placement.room_angle-state.placement.source_angle;if(!Number.isFinite(angle)||Math.abs(angle)>12)throw Error('These directions require more than 12° rotation. Use a better-matched source view instead of forcing alignment.');state.placement.rotation=angle;fillPlacement();drawPlacement();markDirty();});
 for(const id of ['depth-assist','color-match','shadow','shadow-mode','inpaint-radius'])$(id).oninput=()=>{readPlacement();markDirty();};
 $('fit-selection').onclick=()=>{fitSelection();markDirty();};
 const placement=$('placement-overlay');placement.addEventListener('pointerdown',e=>{if(state.busy||!state.box||$('compare-toggle').checked)return;e.preventDefault();const p=point(e,placement);state.drag={type:'placement',start:p,center:[state.placement.x,state.placement.y]};placement.setPointerCapture(e.pointerId);});
 placement.addEventListener('pointermove',e=>{if(state.drag?.type!=='placement')return;const p=point(e,placement);state.placement.x=Math.round(state.drag.center[0]+p[0]-state.drag.start[0]);state.placement.y=Math.round(state.drag.center[1]+p[1]-state.drag.start[1]);fillPlacement();drawPlacement();});
 placement.addEventListener('pointerup',()=>{if(state.drag?.type==='placement'){state.drag=null;markDirty();}});
 $('generate').onclick=handle(()=>{const p=payload();if(!p.box)throw Error('Select furniture first.');if(p.placement.width<5||p.placement.height<5)throw Error('Replacement width and height must be at least five pixels.');if(p.reference.value!==null&&(!Number.isFinite(p.reference.value)||p.reference.value<=0))throw Error('Use a positive reference measurement or leave it unknown.');return runJob('render',p,'Removing selected furniture and compositing the replacement…');});
 $('save-settings').onclick=handle(async()=>{if(!state.project)return;busy(true,'Saving simulation settings…');try{const p=await request(`/api/projects/${state.project.id}/settings`,{...payload(),name:$('project-name').value.trim()});useProject(p,false);state.unsaved=false;await refreshProjects();status('Simulation settings saved locally. Reopen it from the project list.','success');}finally{busy(false);}});
 $('restore-original').onclick=()=>{state.resultVisible=false;$('compare-toggle').checked=false;updateVisuals();status('Original photograph shown. Your generated result remains available.');};$('show-result').onclick=()=>{state.resultVisible=true;updateVisuals();status('Latest replacement shown. Physical fit remains unverified.');};
 $('compare-toggle').onchange=applyComparison;$('compare-slider').oninput=applyComparison;
 $('download-result').onclick=e=>{if($('download-result').getAttribute('aria-disabled')==='true')e.preventDefault();};
 $('project-name').oninput=()=>{state.unsaved=true;};
 $('open-camera').onclick=()=>{$('camera-dialog').showModal();startCamera();};$('close-camera').onclick=()=>{$('camera-dialog').close();stopCamera();};$('camera-dialog').addEventListener('close',stopCamera);$('retry-camera').onclick=startCamera;
 $('capture-camera').onclick=handle(async()=>{const v=$('camera-video');if(!v.videoWidth||!v.videoHeight)throw Error('Wait for a camera frame before capturing.');const canvas=document.createElement('canvas');canvas.width=v.videoWidth;canvas.height=v.videoHeight;canvas.getContext('2d').drawImage(v,0,0);const image=canvas.toDataURL('image/jpeg',.94);$('camera-dialog').close();stopCamera();await uploadRoom(image,$('project-name').value.trim()||'Camera room capture');status('Camera frame saved. Detect furniture or draw a manual selection.','success');});
 window.addEventListener('pagehide',stopCamera);window.addEventListener('beforeunload',e=>{if(state.unsaved){e.preventDefault();e.returnValue='';}});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){state.polygon=[];state.mode='select';state.drag=null;drawSelections();$('finish-polygon').disabled=true;}});
}
async function start(){
 wire();busy(false);fillPlacement();
 const outcomes=await Promise.allSettled([refreshProjects(),request('/api/assets')]);
 if(outcomes[1].status==='fulfilled'){state.assets=outcomes[1].value.assets||[];for(const a of state.assets)$('asset-picker').append(new Option(a.name||a.id,a.id));}
 else status('Local asset list unavailable. Upload a replacement image to continue.','error');
 if(outcomes[0].status==='rejected')status('Could not load saved simulations: '+errorMessage(outcomes[0].reason),'error');
 const openId=new URLSearchParams(location.search).get('project');if(openId&&/^[a-f0-9]{32}$/.test(openId)){try{useProject(await request('/api/projects/'+openId));status('Saved simulation opened.','success');}catch(e){status(errorMessage(e),'error');}}
 window.photoSimulator={placementCorners:()=>placementCorners(state.placement),snapshot:()=>JSON.parse(JSON.stringify({project:state.project,box:state.box,selectedId:state.selectedId,mask_edits:state.mask_edits,protected_edits:state.protected_edits,placement:state.placement,mode:state.mode,busy:state.busy,unsaved:state.unsaved,resultVisible:state.resultVisible})),ready:true};
}
start().catch(e=>status(errorMessage(e),'error'));
