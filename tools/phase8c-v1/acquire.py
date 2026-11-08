"""Exactly27 pinned HTTPS objects. Conservative ledger and range recovery."""
import argparse
import fcntl
import ssl
import urllib.request
from pilot_core import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None

def validate_headers(response,row,offset=0):
 obj=row['remote_object'];h=response.headers;remaining=obj['bytes']-offset
 if response.status!=(206 if offset else 200) or int(h.get('Content-Length','-1'))!=remaining or h.get('ETag')!=obj['etag'] or h.get('Content-Type','').split(';')[0]!='image/jpeg' or h.get('Content-Encoding','identity') not in ('','identity'):raise ValueError('Source object status/size/version/type changed')
 if offset and h.get('Content-Range')!=f"bytes {offset}-{obj['bytes']-1}/{obj['bytes']}":raise ValueError('Unsafe range recovery response')

def publish(part,target):
 if target.exists():raise FileExistsError('Do not overwrite a photograph')
 os.link(part,target);part.unlink()

def inspect_download(row,partial,filename,attempts,digest):
 from PIL import Image
 import re
 etag=row['remote_object']['etag'].strip('"')
 if not re.fullmatch('[0-9a-fA-F]{32}',etag):raise ValueError('No pinned single-part source MD5; stop for integrity review')
 md5=hashlib.md5(partial.read_bytes()).hexdigest()
 if md5!=etag.lower():raise ValueError('Downloaded body does not match pinned source ETag MD5')
 with Image.open(partial) as im:im.verify()
 with Image.open(partial) as im:im.load();dimensions=im.size;orientation=im.getexif().get(274,1)
 if dimensions!=(row['width'],row['height']):raise ValueError('Decoded dimensions differ from frozen COCO metadata')
 return dict(image_id=row['image_id'],coco_image_id=row['coco_image_id'],source_identity=row['source_identity'],identity_group=row['identity_group'],proposed_split=row['proposed_split'],file=filename,bytes=row['remote_object']['bytes'],sha256=digest,source_etag_md5_verified=md5,encoding='JPEG',width=dimensions[0],height=dimensions[1],exif_orientation=orientation,source_url=row['remote_object']['url'],source_etag=row['remote_object']['etag'],downloaded_utc=now(),attempts=attempts,complete=True)

def validate_state(state,rows):
 if state['manifest_sha256']!=PIN:raise ValueError('Resume binding changed')
 ids={r['image_id'] for r in rows};requests=state['requests'];gets=[r for r in requests if r['method']=='GET']
 if any(r['image_id'] not in ids for r in requests):raise ValueError('Journal contains unapproved ID')
 if sum(r['actual_bytes'] for r in gets)!=state['actual_body_bytes'] or sum(r['reserved'] for r in gets)!=state['conservative_charge'] or not 0<=state['actual_body_bytes']<=state['conservative_charge']<=CAP:raise ValueError('Transfer accounting mismatch')
 for ident in ids:
  attempts=[r for r in gets if r['image_id']==ident]
  if len(attempts)>2 or [r['attempt'] for r in attempts]!=list(range(1,len(attempts)+1)):raise ValueError('Attempt limit or ordering mismatch')

def run(resume=False):
 m=manifest();auth=read(OUT/'authorization.json')
 if not auth['image_acquisition_approved'] or auth['manifest_sha256']!=PIN:raise PermissionError('No matching user approval')
 journal=OUT/'network-ledger.json'
 if journal.exists() and not resume:raise FileExistsError('Explicit --resume required for existing journal')
 with (OUT/'acquisition.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(journal) if journal.exists() else dict(manifest_sha256=PIN,actual_body_bytes=0,conservative_charge=0,requests=[],objects={})
  validate_state(state,m['images'])
  context=ssl.create_default_context();assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED
  opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context));atomic(journal,state)
  if (OUT/'acquisition-complete.json').exists():
   for r in read(OUT/'acquisition-complete.json')['records']:assert sha(OUT/r['file'])==r['sha256']
   print('SKIP VERIFIED COMPLETE',flush=True);return
  rows=[]
  for row in m['images']:
   ident=row['image_id'];size=row['remote_object']['bytes'];url=row['remote_object']['url'];filename=f"photos/{row['coco_image_id']:012d}.jpg";target=OUT/filename;partial=target.with_suffix('.jpg.part');target.parent.mkdir(exist_ok=True)
   old=state['objects'].get(ident)
   if old and old.get('complete'):
    if not target.exists() and partial.exists() and sha(partial)==old['sha256']:publish(partial,target)
    if sha(target)!=old['sha256']:raise ValueError('Previously completed photo changed')
    rows.append(old);continue
   # Recheck saved evidence freshness/hashes on every entry/resume.
   if sha(ROOT/row['rights_record'])!=row['rights_record_sha256'] or sha(ROOT/row['source_document'])!=row['source_document_sha256']:raise ValueError('Rights evidence changed')
   age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(row['rights_checked_utc'])).total_seconds()/86400
   if not 0<=age<=7:raise ValueError('Rights evidence stale; stop before downloading')
   with opener.open(urllib.request.Request(url,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=30) as h:
    validate_headers(h,row);state['requests'].append(dict(image_id=ident,url=url,method='HEAD',status=h.status,body_bytes=0,checked_utc=now(),etag=h.headers['ETag'],bytes=size,tls_verified=True));atomic(journal,state)
   prior=[r for r in state['requests'] if r['image_id']==ident and r['method']=='GET'];offset=partial.stat().st_size if partial.exists() else 0
   if offset:
    if not prior or prior[-1]['prefix_bytes']!=offset or sha(partial)!=prior[-1]['prefix_sha256']:raise ValueError('Uncertain partial prefix; refusing resume')
   if offset==size:
    if not prior[-1].get('headers_verified'):raise ValueError('Full partial has no committed source-header proof')
    data=inspect_download(row,partial,filename,len(prior),sha(partial));prior[-1].update(complete=True,sha256=data['sha256']);state['objects'][ident]=data;atomic(journal,state);publish(partial,target);rows.append(data);continue
   if not offset<size:raise ValueError('Partial exceeds expected source size')
   if len(prior)>=2:raise RuntimeError('Two attempts exhausted')
   remaining=size-offset
   if state['conservative_charge']+remaining>CAP:raise RuntimeError('Approved transfer cap')
   space(remaining+32*2**20)
   record=dict(image_id=ident,url=url,method='GET',attempt=len(prior)+1,offset=offset,reserved=remaining,actual_bytes=0,prefix_bytes=offset,prefix_sha256=sha(partial) if offset else hashlib.sha256(b'').hexdigest(),complete=False,checked_utc=now(),tls_verified=True)
   state['conservative_charge']+=remaining;state['requests'].append(record);atomic(journal,state)
   digest=hashlib.sha256()
   if offset:
    with partial.open('rb') as f:
     for chunk in iter(lambda:f.read(65536),b''):digest.update(chunk)
   headers={'Accept-Encoding':'identity','If-Match':row['remote_object']['etag'],'User-Agent':'FurnitureAI-approved-photo-acquisition/1.0'}
   if offset:headers['Range']=f'bytes={offset}-'
   try:
    with opener.open(urllib.request.Request(url,headers=headers),timeout=45) as response:
     validate_headers(response,row,offset)
     record['headers_verified']=True;atomic(journal,state)
     with partial.open('ab' if offset else 'xb') as f:
      while record['actual_bytes']<remaining:
       b=response.read(min(65536,remaining-record['actual_bytes']))
       if not b:break
       f.write(b);f.flush();os.fsync(f.fileno());digest.update(b);record['actual_bytes']+=len(b);state['actual_body_bytes']+=len(b);record['prefix_bytes']=offset+record['actual_bytes'];record['prefix_sha256']=digest.hexdigest();atomic(journal,state)
    if record['actual_bytes']!=remaining or partial.stat().st_size!=size:raise ValueError('Incomplete photo body; retain bounded partial for resume')
    data=inspect_download(row,partial,filename,len(prior)+1,digest.hexdigest())
    record.update(complete=True,sha256=data['sha256']);state['objects'][ident]=data;atomic(journal,state);publish(partial,target);rows.append(data)
    print('DOWNLOADED',len(rows),'/27',ident,'bytes',state['actual_body_bytes'],flush=True)
   except BaseException as exc:record['error']=repr(exc);atomic(journal,state);raise
  commit(OUT/'acquisition-complete.json',dict(manifest_sha256=PIN,records=rows,images=len(rows),actual_photo_body_bytes=state['actual_body_bytes'],conservative_charge=state['conservative_charge'],photo_get_attempts=sum(r['method']=='GET' for r in state['requests']),training_or_inference=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.resume)
