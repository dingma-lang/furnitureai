"""Pinned TLS, durable transfer reservations, <=2 GET attempts/object and safe range resume."""
import argparse
import fcntl
import hashlib
import os
import ssl
import urllib.request
from core import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None

def headers_valid(response,row,offset=0):
 h=response.headers;n=row['bytes']-offset
 if response.status!=(206 if offset else 200) or int(h.get('Content-Length','-1'))!=n or h.get('ETag')!=row['etag'] or h.get('Content-Encoding','identity') not in ('','identity'):raise ValueError('Source size/version/encoding changed')
 if h.get('Content-Type','').split(';')[0] not in ('application/zip','application/x-zip-compressed'):raise ValueError('Unexpected archive MIME')
 if offset and h.get('Content-Range')!=f"bytes {offset}-{row['bytes']-1}/{row['bytes']}":raise ValueError('Unsafe partial response')

def reserve(state,key,n):
 prior=[x for x in state['requests'] if x['object']==key]
 if len(prior)>=2:raise RuntimeError('Two attempts exhausted')
 if state['conservative_body_charge']+n>CAP:raise RuntimeError('850MB cumulative transfer cap')
 state['conservative_body_charge']+=n
 record=dict(object=key,attempt=len(prior)+1,reserved_body_bytes=n,actual_body_bytes=0,complete=False,started_utc=now())
 state['requests'].append(record);return record

def publish(part,target):
 if target.exists():raise FileExistsError(target)
 os.link(part,target);part.unlink()

def run(resume=False):
 p=proposal();auth=read(OUT/'authorization.json')
 if auth.get('proposal_sha256')!=sha(PROPOSAL) or auth.get('annotation_acquisition_approved') is not True:raise PermissionError('Matching explicit approval required')
 context=ssl.create_default_context();assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED
 opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context))
 with (OUT/'acquisition.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  journal=OUT/'transfer-ledger.json'
  if journal.exists() and not resume:raise FileExistsError('Use --resume for an existing ledger')
  state=read(journal) if journal.exists() else dict(proposal_sha256=sha(PROPOSAL),actual_body_bytes=read(OUT/'preflight.json')['document_body_bytes'],conservative_body_charge=read(OUT/'preflight.json')['document_body_bytes'],requests=[],completed={})
  if state['proposal_sha256']!=sha(PROPOSAL):raise ValueError('Resume proposal binding changed')
  atomic(journal,state)
  for row in p['objects']:
   key=row['expected_local_filename'];target=OUT/key;part=OUT/(key+'.part')
   if key in state['completed']:
    old=state['completed'][key]
    if not target.exists() and part.exists() and sha(part)==old['sha256']:publish(part,target)
    if sha(target)!=old['sha256']:raise ValueError('Completed archive changed')
    print('SKIP',key,flush=True);continue
   if target.exists():raise FileExistsError('Unrecorded target; no overwrite')
   with opener.open(urllib.request.Request(row['url'],method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=40) as h:headers_valid(h,row)
   old_requests=[r for r in state['requests'] if r['object']==key];offset=part.stat().st_size if part.exists() else 0
   if offset:
    if not old_requests or old_requests[-1].get('prefix_bytes')!=offset or sha(part)!=old_requests[-1].get('prefix_sha256'):raise ValueError('Uncertain partial prefix; refusing to overwrite/discard it')
   if offset>row['bytes']:raise ValueError('Oversize partial')
   if offset==row['bytes']:
    if not old_requests[-1].get('headers_verified'):raise ValueError('No header proof for full partial')
    info=dict(bytes=offset,sha256=sha(part),url=row['url'],etag=row['etag'],completed_utc=now(),attempts=len(old_requests));state['completed'][key]=info;atomic(journal,state);publish(part,target);continue
   n=row['bytes']-offset;space(n)
   record=reserve(state,key,n);record.update(offset=offset,prefix_bytes=offset,prefix_sha256=sha(part) if offset else hashlib.sha256(b'').hexdigest());atomic(journal,state)
   headers={'Accept-Encoding':'identity','If-Match':row['etag'],'User-Agent':'FurnitureAI-approved-LVIS-metadata/1.0'}
   if offset:headers['Range']=f'bytes={offset}-'
   digest=hashlib.sha256()
   if offset:
    with part.open('rb') as f:
     for b in iter(lambda:f.read(1048576),b''):digest.update(b)
   try:
    with opener.open(urllib.request.Request(row['url'],headers=headers),timeout=60) as response:
     headers_valid(response,row,offset);record['headers_verified']=True;atomic(journal,state)
     with part.open('ab' if offset else 'xb') as f:
      while record['actual_body_bytes']<n:
       b=response.read(min(1048576,n-record['actual_body_bytes']))
       if not b:raise RuntimeError('Incomplete body; retain partial and stop')
       f.write(b);f.flush();os.fsync(f.fileno());digest.update(b)
       record['actual_body_bytes']+=len(b);state['actual_body_bytes']+=len(b);record.update(prefix_bytes=offset+record['actual_body_bytes'],prefix_sha256=digest.hexdigest());atomic(journal,state)
       if record['actual_body_bytes']%(32*1048576)<1048576:print(key,record['prefix_bytes'],'/',row['bytes'],flush=True)
       space()
    if part.stat().st_size!=row['bytes']:raise ValueError('Archive length mismatch')
    info=dict(bytes=row['bytes'],sha256=digest.hexdigest(),url=row['url'],etag=row['etag'],completed_utc=now(),attempts=record['attempt'])
    expected=read(OUT/'preflight.json').get('published_sha256',{}).get(key)
    if expected and info['sha256']!=expected:raise ValueError('Published archive SHA256 mismatch')
    record['complete']=True;state['completed'][key]=info;atomic(journal,state);publish(part,target);print('COMPLETE',key,info['sha256'],flush=True)
   except BaseException as exc:
    record['error']=repr(exc);atomic(journal,state);raise
  if not (OUT/'acquisition-complete.json').exists():commit(OUT/'acquisition-complete.json',dict(archives=state['completed'],actual_body_bytes=state['actual_body_bytes'],conservative_body_charge=state['conservative_body_charge'],completed_utc=now(),photo_bodies=0))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args();run(args.resume)
