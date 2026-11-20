/* Appended only to an isolated synthetic-browser fixture. Never served by CLI. */
(async()=>{
 const checks=[];function check(value,message){if(!value)throw Error(message);checks.push(message);}
 async function until(fn){for(let i=0;i<250;i++){if(fn())return;await new Promise(r=>setTimeout(r,20));}throw Error('Browser fixture timed out');}
 function fire(target,type,raw,p=photo()){let d=point(raw[0],raw[1],p),screen=new DOMPoint(...d).matrixTransform($('scene').getScreenCTM());target.dispatchEvent(new PointerEvent(type,{clientX:screen.x,clientY:screen.y,pointerId:7,button:0,bubbles:true}));}
 function drawBox(b,kind='negative'){ $('mode').value=kind;fire($('scene'),'pointerdown',[b[0],b[1]]);fire($('scene'),'pointermove',[b[2],b[3]]);fire($('scene'),'pointerup',[b[2],b[3]]);}
 function confirmations(){ $('reviewer').value='SYNTHETIC BROWSER FIXTURE';$('review-date').value='2026-11-19';$('evidence').value='Synthetic blank floor area in fixture, not real human evidence';$('evidence').dispatchEvent(new Event('input'));for(let id of ['taxonomy-confirm','absence-confirm','explicit-confirm'])$(id).checked=true;}
 try{
  await until(()=>registry&&state&&$('photo').options.length===3);localStorage.removeItem(draftKey);
  check(state.regions.length===0,'No prefilled or preapproved real regions');
  // Synthetic DOM events have no native active pointer. Only this test fixture
  // stubs capture; production keeps the actual browser pointer-capture API.
  $('scene').setPointerCapture=()=>{};$('scene').hasPointerCapture=()=>false;
  HTMLAnchorElement.prototype.click=function(){if(this.download)window.fixtureDownload=this.download;};
  $('scene').scrollIntoView();drawBox([20,500,250,700]);
  check(state.regions.length===1&&region().status==='draft','Draw creates only an unconfirmed draft');
  check(equal(region().box_pixels,[20,500,250,700]),'CSS display coordinates map to exact original pixels');
  $('approve').click();check(region().status==='draft','Cannot approve without human fields and confirmations');
  confirmations();$('approve').click();await until(()=>region().status==='confirmed_negative'&&$('message').textContent.includes('Server validated'));
  check(region().confirmed_revision===region().revision,'Explicit confirmation binds current revision');
  $('x2').value=260;$('apply-coordinates').click();check(region().status==='draft'&&!region().confirmed,'Numeric edit revokes confirmation');
  confirmations();$('approve').click();await until(()=>$('message').textContent.includes('Server validated'));
  let handle=$('regions').querySelector('[data-handle="se"]');$('mode').value='select';fire(handle,'pointerdown',[260,700]);fire($('scene'),'pointermove',[280,720]);fire($('scene'),'pointerup',[280,720]);
  check(equal(region().box_pixels,[20,500,280,720])&&region().status==='draft','Corner resize maps and revokes approval');
  $('undo').click();check(state.regions.length===1&&state.regions[0].status==='draft','Undo does not reapprove a restored rectangle');
  selected=state.regions[0].region_id;draw();populate();$('zoom').value='2';$('zoom').dispatchEvent(new Event('change'));$('viewport').scrollTop=200;$('viewport').scrollLeft=70;
  let before=state.regions.length;drawBox([130,140,190,210]);check(state.regions.length===before,'Known Chair overlap is prevented');
  check($('message').textContent.includes('not applied'),'Overlap rejection is visible');
  selected=state.regions[0].region_id;draw();populate();confirmations();$('approve').click();await until(()=>$('message').textContent.includes('Server validated'));
  drawBox([40,550,100,650],'potential_positive');check(state.regions[0].status==='draft','Potential seating revokes an overlapping approved negative');
  check(state.regions.length===2,'Potential seating can be drawn even over an existing region');
  $('delete').click();check(state.regions.length===1,'Remove selected rectangle works');$('undo').click();check(state.regions.length===2,'Undo restores deleted draft');
  $('clarifications-tab').click();check($('editor').hidden&&document.querySelectorAll('.clarification').length===24,'All24 clarification items presented separately');
  check(document.querySelectorAll('.clarification h3')[4].textContent.includes('validation'),'Evaluation clarification remains diagnostic only');
  let card=document.querySelector('.clarification');card.querySelector('textarea').value='Synthetic fixture clarification; no real evidence';card.querySelector('select').value='deferred';card.querySelector('button').click();
  check(state.clarifications.length===1&&state.regions.every(r=>r.status!=='confirmed_negative'),'Clarification does not approve rectangles');
  $('export').click();await until(()=>window.fixtureDownload);check(window.fixtureDownload.startsWith('chair-negative-SYNTHETIC'),'Versioned download emitted only to fixture interception');
  for(let o=1;o<=8;o++){let p={width:800,height:600,orientation:o};let b=[13,17,799,599];check(equal(mappedBox(mappedBox(b,p),p,true),b),'Browser orientation roundtrip '+o);}
  check(state.regions.every(r=>!r.confirmed),'Fixture ends with no confirmed regions');
  let el=document.createElement('pre');el.id='browser-test-result';el.textContent=JSON.stringify({status:'PASS',checks,real_user_confirmations:0,real_exports_written:0,synthetic_pointer_capture_stubbed:true});document.body.append(el);
 }catch(e){let el=document.createElement('pre');el.id='browser-test-result';el.textContent=JSON.stringify({status:'FAIL',error:e.stack,checks});document.body.append(el);}
})();
