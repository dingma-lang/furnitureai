"""Approval-gated official ABO downloader. No network in dry-run mode."""
import argparse
import fcntl
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.request
from PIL import Image
from common import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise ValueError('Redirects are not approved')

class Downloader:
 def __init__(self,plan,dest,opener=None):
  self.plan=plan;self.dest=Path(dest);self.limits=plan['limits'];self.opener=opener or urllib.request.build_opener(NoRedirect())
  self.dest.mkdir(parents=True,exist_ok=True);self.statepath=self.dest/'transfer-ledger.json'
  self.binding=fingerprint(plan)
  self.state=read(self.statepath) if self.statepath.exists() else {'plan_binding':self.binding,'charged_bytes':0,'observed_body_bytes':0,'objects':{},'documentation':{}}
  if self.state['plan_binding']!=self.binding:raise ValueError('Destination belongs to another acquisition plan')
  self.stop=False
 def commit(self):save(self.statepath,self.state)
 def disk(self,extra=0):
  seen=set();used=0
  for path in self.dest.rglob('*'):
   if path.is_file():
    s=path.stat();key=(s.st_dev,s.st_ino)
    if key not in seen:used+=s.st_blocks*512;seen.add(key)
  if used+extra+16*1024**2>self.limits['new_allocated_storage_bytes']:raise RuntimeError('Acquisition storage cap')
  return used
 def check_url(self,obj):
  path=obj['metadata_path']
  if not re.fullmatch(r'[0-9a-f]{2}/[0-9a-f]+\.(jpg|png)',path):raise ValueError('Unsafe official object path')
  expected='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/small/'+path
  if obj['url']!=expected or obj['destination']!='images/'+path:raise ValueError('Unapproved source/destination')
 def request(self,url,method='GET',headers=None):
  return self.opener.open(urllib.request.Request(url,method=method,headers={'Accept-Encoding':'identity',**(headers or {})}),timeout=30)
 def head(self,obj):
  self.check_url(obj)
  with self.request(obj['url'],'HEAD') as response:
   if response.status!=200 or response.geturl()!=obj['url']:raise ValueError('Unexpected official source response')
   size=int(response.headers.get('Content-Length','-1'));mime=response.headers.get('Content-Type','').split(';')[0]
   if not 0<size<=self.limits['object_bytes'] or mime not in ('image/jpeg','image/png'):raise ValueError('Remote size/type outside approved limits')
   return {'bytes':size,'etag':response.headers.get('ETag'),'content_type':mime}
 def documentation(self):
  sources=[('LICENSE-CC-BY-4.0.txt',META/'LICENSE-CC-BY-4.0.txt'),('README.md',META/'README.md')]
  for name,local in sources:
   url='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/'+name
   reservation=65536
   if self.state['charged_bytes']+reservation>self.limits['download_bytes']:raise RuntimeError('Documentation transfer budget')
   self.state['charged_bytes']+=reservation;self.commit()
   data=b''
   with self.request(url) as response:
    if response.status!=200 or response.geturl()!=url:raise ValueError('Unexpected licence source')
    size=int(response.headers.get('Content-Length','-1'))
    if not 0<size<=reservation:raise ValueError('Unexpected licence size')
    while len(data)<size:
     block=response.read(min(8192,size-len(data)))
     if not block:raise ValueError('Incomplete licence transfer')
     data+=block;self.state['observed_body_bytes']+=len(block);self.commit()
   if hashlib.sha256(data).hexdigest()!=sha(local):raise ValueError('Official licence/README changed; stop for review before images')
   path=self.dest/'documentation'/name
   if path.exists() and path.read_bytes()!=data:raise ValueError('Documentation conflict')
   if not path.exists():atomic(path,data)
   self.state['documentation'][name]={'url':url,'sha256':sha(local),'bytes':len(data)};self.commit()
 def get(self,obj,remote):
  self.check_url(obj);key=obj['image_id'];file=self.dest/obj['destination'];record=self.state['objects'].setdefault(key,{'url':obj['url'],'attempts':[]})
  if record['url']!=obj['url']:raise ValueError('Object identity changed')
  if record.get('status')=='downloaded':
   if not file.is_file() or sha(file)!=record['sha256']:raise ValueError('Saved image changed')
   return record
  if file.exists():raise ValueError('Uncommitted final image exists; audit before recovery, never overwrite')
  if len(record['attempts'])>=self.limits['attempts_per_object']:raise RuntimeError('Per-object attempt limit exhausted')
  # Worst-case precharge survives process death. It is intentionally never refunded.
  reservation=self.limits['object_bytes']
  if self.state['charged_bytes']+reservation>self.limits['download_bytes']:raise RuntimeError('Cumulative download budget')
  self.disk(reservation);self.state['charged_bytes']+=reservation
  attempt={'expected':remote,'observed_bytes':0,'status':'in_progress'};record['attempts'].append(attempt);self.commit()
  file.parent.mkdir(parents=True,exist_ok=True);partial=file.with_name(file.name+f'.attempt-{len(record["attempts"])}.part')
  try:
   if self.stop:raise InterruptedError('Stop requested')
   with self.request(obj['url'],headers={'If-Match':remote['etag']} if remote['etag'] else {}) as response:
    if response.status!=200 or response.geturl()!=obj['url']:raise ValueError('Unexpected download response')
    if int(response.headers.get('Content-Length','-1'))!=remote['bytes']:raise ValueError('Object changed after HEAD')
    if remote['etag'] and response.headers.get('ETag')!=remote['etag']:raise ValueError('Object ETag changed')
    with partial.open('xb') as sink:
     while attempt['observed_bytes']<remote['bytes']:
      if self.stop:raise InterruptedError('Graceful stop; partial transfer charged and retained')
      block=response.read(min(65536,remote['bytes']-attempt['observed_bytes']))
      if not block:raise ValueError('Truncated body')
      if len(block)>remote['bytes']-attempt['observed_bytes']:raise ValueError('Oversized body')
      sink.write(block);sink.flush();os.fsync(sink.fileno())
      attempt['observed_bytes']+=len(block);self.state['observed_body_bytes']+=len(block);self.commit()
   with Image.open(partial) as image:image.verify()
   with Image.open(partial) as image:dimensions=list(image.size)
   checksum=sha(partial)
   # Commit ledger before final pathname so crash recovery never overwrites a final image.
   record.update(status='ready_to_commit',sha256=checksum,bytes=remote['bytes'],dimensions=dimensions,partial=str(partial.relative_to(self.dest)))
   self.commit();os.link(partial,file)
   record['status']='downloaded';attempt['status']='complete';self.commit()
   partial.unlink()  # Only this newly created successful transfer's transient hardlink.
   return record
  except Exception as exc:
   attempt['status']='interrupted_or_failed';attempt['error']=str(exc);self.commit();raise
 def recover_commits(self):
  for key,record in self.state['objects'].items():
   if record.get('status')!='ready_to_commit':continue
   matches=[o for o in self.plan['objects'] if o['image_id']==key]
   if len(matches)!=1:raise ValueError('Unknown pending commit')
   file=self.dest/matches[0]['destination'];partial=self.dest/record['partial']
   if sha(partial)!=record['sha256']:raise ValueError('Pending partial changed')
   if file.exists():
    if sha(file)!=record['sha256']:raise ValueError('Conflicting pending final')
   else:os.link(partial,file)
   record['status']='downloaded';record['attempts'][-1]['status']='complete';self.commit()
   partial.unlink()


def approval(plan,path,out):
 if path is None:raise PermissionError('Explicit user approval file is required before any network access')
 approved=read(path)
 if approved.get('approved') is not True or not approved.get('approved_by','').strip() or approved.get('acquisition_plan_sha256')!=sha(out/'acquisition-plan.json') or approved.get('limits')!=plan['limits']:raise PermissionError('Approval is missing or does not match the exact frozen plan/limits')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preparation',type=Path,default=OUT);parser.add_argument('--approval-file',type=Path);parser.add_argument('--execute',action='store_true');args=parser.parse_args();verify_plan(args.preparation)
 plan=read(args.preparation/'acquisition-plan.json')
 if not args.execute:
  print(json.dumps({'dry_run':True,'network_requests':0,'products':plan['product_count'],'images':plan['image_count'],'destination':plan['destination'],'limits':plan['limits']},indent=2));return
 approval(plan,args.approval_file,args.preparation)
 dest=ROOT/plan['destination']
 if dest.exists() and not (dest/'transfer-ledger.json').exists():raise FileExistsError('Destination exists without this plan ledger; do not modify')
 dest.mkdir(parents=True,exist_ok=True)
 with (dest/'acquisition.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);d=Downloader(plan,dest)
  def stop(*_):d.stop=True
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  d.recover_commits();d.documentation();remote={}
  for obj in plan['objects']:
   if d.stop:raise InterruptedError('Preflight stopped')
   remote[obj['image_id']]=d.head(obj)
  save(dest/'remote-preflight.json',remote)
  for obj in plan['objects']:
   if d.stop:break
   while True:
    try:d.get(obj,remote[obj['image_id']]);break
    except InterruptedError:raise
    except (OSError,ValueError,http.client.HTTPException):
     if len(d.state['objects'][obj['image_id']]['attempts'])>=plan['limits']['attempts_per_object']:raise
  save(dest/'acquisition-summary.json',{'products_requested':plan['product_count'],'images_downloaded':sum(r.get('status')=='downloaded' for r in d.state['objects'].values()),'observed_body_bytes':d.state['observed_body_bytes'],'conservative_charged_bytes':d.state['charged_bytes'],'crash_may_leave_charged_bytes_above_known_actual':True,'allocated_storage_bytes':d.disk(),'no_images_are_automatically_valid_training_examples':True})

if __name__=='__main__':main()
