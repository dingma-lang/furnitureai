'use strict';
(async()=>{
 const out=document.querySelector('#result'),frame=document.querySelector('#intake');
 const results={actual_photos:[],source_boxes_test:'not_run',model_calls:0,real_annotations:0,grants_or_reviews_fabricated:false};
 const sleep=ms=>new Promise(r=>setTimeout(r,ms));
 const wait=async fn=>{for(let n=0;n<300;n++){if(fn())return;await sleep(30);}throw Error('Bounded DOM wait expired');};
 try{
  await wait(()=>frame.contentDocument?.querySelector('#completeness')?.children.length===7);
  const doc=frame.contentDocument,win=frame.contentWindow;
  const choose=(input,file)=>{const dt=new win.DataTransfer();dt.items.add(file);input.files=dt.files;input.dispatchEvent(new win.Event('change',{bubbles:true}));};
  const entries=await (await fetch('/smoke-data.json')).json();
  for(const row of entries){
   const n=row.source_id.split(':')[1],body=await (await fetch('/images/'+n+'.jpg')).arrayBuffer();
   choose(doc.querySelector('#file'),new win.File([body],n+'.jpg',{type:'image/jpeg'}));
   await wait(()=>doc.querySelector('#identity').textContent.includes(row.sha256));
   doc.querySelector('[name=photo_id]').value=row.source_id;doc.querySelector('[name=creator_id]').value=row.creator;
   doc.querySelector('[name=source_group_id]').value='commons-user:'+row.creator;doc.querySelector('[name=source]').value=row.source;
   const dimensions=[doc.querySelector('#photo').naturalWidth,doc.querySelector('#photo').naturalHeight];
   if(JSON.stringify(dimensions)!==JSON.stringify(row.dimensions))throw Error('Actual browser dimensions disagree');
   if([...doc.querySelectorAll('input[type=checkbox]')].some(c=>c.checked))throw Error('Unexpected real-photo confirmation');
   if([...doc.querySelectorAll('[name^=complete_]')].some(c=>c.value!=='unknown'))throw Error('Unexpected completeness');
   results.actual_photos.push({source_id:row.source_id,sha256_matches:true,dimensions,confirmations:false,labels_active:false});
  }
  // Import/correction tests ONLY on newly generated synthetic pixels, not the three photos.
  const canvas=doc.createElement('canvas');canvas.width=100;canvas.height=80;canvas.getContext('2d').fillRect(0,0,100,80);
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
  choose(doc.querySelector('#file'),new win.File([blob],'SYNTHETIC-NOT-A-PHOTO.png',{type:'image/png'}));
  await wait(()=>doc.querySelector('#photo').naturalWidth===100&&doc.querySelector('#identity').textContent.startsWith('100'));
  const hash=doc.querySelector('#identity').textContent.match(/[a-f0-9]{64}/)[0];
  const source={schema:'modern-room-source-annotations-v1',image_sha256:hash,width:100,height:80,box_frame:'exif-oriented-xyxy-pixels',source_id:'SYNTHETIC-UI-TEST-ONLY',source_file_sha256:'c'.repeat(64),annotation_license:'SYNTHETIC EXAMPLE ONLY',annotations:[{id:'synthetic-box',box:[10,10,40,40],native_label:'chair',native_annotation:{synthetic:true},suggested_category:'chair'}]};
  choose(doc.querySelector('#sourceBoxes'),new win.File([JSON.stringify(source)],'synthetic-boxes.json',{type:'application/json'}));
  await wait(()=>doc.querySelector('#objects article'));
  const article=doc.querySelector('#objects article');if(!article.textContent.includes('proposal'))throw Error('Imported box was not inactive');
  const x=article.querySelector('input[type=number]');x.value='12';x.dispatchEvent(new win.Event('change'));
  if(doc.querySelector('#objects article input[type=number]').value!=='12')throw Error('Coordinate edit failed');
  if(!doc.querySelector('#objects article').textContent.includes('proposal'))throw Error('Edit auto-approved');
  doc.querySelector('#undo').click();
  if(doc.querySelector('#objects article input[type=number]').value!=='10')throw Error('Undo failed');
  results.source_boxes_test='synthetic import/edit/undo PASS; no confirmation/export';results.status='PASS';
 }catch(e){results.status='INCOMPLETE_OR_FAIL';results.error=String(e);}
 out.textContent=JSON.stringify(results);out.dataset.done='true';
})();
