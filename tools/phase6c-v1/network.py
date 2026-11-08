"""Certificate-verified source HTML only; photo host categorically HEAD-only."""
import fcntl
import gzip
import io
import ssl
import threading
import time
import urllib.request
import urllib.error
from common import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None
class ResourceLimit(RuntimeError):pass
class SourceRateLimited(RuntimeError):pass

class Transport:
 def __init__(self,resume=False,root=OUT,cap=CAP,disk=DISK,retry_429=False):
  self.root=Path(root)
  if self.root.exists() and not resume:raise FileExistsError('Fresh rights-audit destination required')
  self.root.mkdir(parents=True,exist_ok=resume);self.lock=(self.root/'audit.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.mutex=threading.RLock();self.path=self.root/'network-ledger.json';self.cap=cap;self.disk=disk
  self.state=read(self.path) if self.path.exists() else dict(manifest_sha256=PIN,cap=cap,disk=disk,actual_body_bytes=0,conservative_charge=0,requests=[])
  if self.state['manifest_sha256']!=PIN or self.state['cap']!=cap or self.state['disk']!=disk:raise ValueError('Journal scope changed')
  self.save()
  self.retry_429=retry_429;self.source_blocked=False;self.last_source_request=0.
 def save(self):atomic(self.path,self.state)
 def close(self):self.lock.close()
 def request(self,row,url,method='HEAD'):
  photo=row['source_identity'].removeprefix('flickr-photo:');is_photo=url==row['proposed_secure_image_url']
  if method not in ('HEAD','GET') or (is_photo and method!='HEAD') or (not is_photo and not allowed_page(url,photo)):raise ValueError('Unapproved host, photo ID, redirect or photo-body GET')
  context=ssl.create_default_context()
  if context.verify_mode!=ssl.CERT_REQUIRED or not context.check_hostname:raise RuntimeError('TLS verification required')
  opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context))
  with self.mutex:
   prior=[r for r in self.state['requests'] if r['url']==url and r['method']==method]
   complete=[r for r in prior if r.get('complete')]
   if complete:
    record=complete[-1]
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(record['checked_utc'])).total_seconds()
    delay=max(120,int(record.get('headers',{}).get('Retry-After','0'))) if record.get('headers',{}).get('Retry-After','0').isdigit() else 120
    retry=self.retry_429 and not is_photo and record.get('status')==429 and len(prior)<2 and elapsed>=delay
    if not retry:
     if method=='GET' and record.get('file') and sha(self.root/record['file'])!=record['sha256']:raise ValueError('Cached rights response changed')
     return record
   if len(prior)>=(2 if method=='GET' or self.retry_429 else 1):raise ValueError('Request attempt bound reached; do not bypass access restrictions')
   if self.retry_429 and not is_photo:
    if self.source_blocked:raise SourceRateLimited('Rate limit persisted; no more source requests this pass')
    wait=max(0.,1.25-(time.monotonic()-self.last_source_request))
    if wait:time.sleep(wait) # <=1.25s, single-worker courteous cooldown retry.
    self.last_source_request=time.monotonic()
   reserved=PAGE if method=='GET' else 0
   if self.state['conservative_charge']+reserved>self.cap or allocated(self.root)+reserved+16*2**20>self.disk:raise ResourceLimit('Approved transfer/storage cap; candidate remains unverified')
   record=dict(image_id=row['image_id'],source_identity=row['source_identity'],url=url,method=method,photo_object=is_photo,checked_utc=now(),attempt=len(prior)+1,reserved=reserved,actual_bytes=0,complete=False,tls_certificate_chain_verified=True)
   self.state['conservative_charge']+=reserved;self.state['requests'].append(record);self.save()
  response=None
  try:
   try:response=opener.open(urllib.request.Request(url,method=method,headers={'Accept-Encoding':'gzip' if method=='GET' else 'identity','User-Agent':'FurnitureAI-source-rights-audit/1.0'}),timeout=25)
   except urllib.error.HTTPError as exc:response=exc
   with response:
    headers={k:response.headers.get(k) for k in ('Content-Type','Content-Length','Content-Encoding','Location','ETag','Last-Modified','Retry-After') if response.headers.get(k) is not None}
    record.update(status=response.status,headers=headers)
    if response.status==429 and self.retry_429 and not is_photo:self.source_blocked=True
    if method=='GET':
     if response.status!=200:
      # Do not read error/redirect bodies. Only429 is eligible for one later,
      # cooldown-respecting retry; never retry forbidden/missing sources.
      with self.mutex:self.state['conservative_charge']-=reserved;record.update(complete=True,body_not_read=True);self.save()
      return record
     if headers.get('Content-Type','').split(';')[0] not in ('text/html','application/xhtml+xml'):raise ValueError('Expected source HTML; refusing non-document response')
     n=headers.get('Content-Length')
     if n is not None and not 0<=int(n)<=PAGE:raise ValueError('Source page exceeds approved per-body cap')
     relative='responses/'+row['image_id'].replace(':','-')+f"-a{record['attempt']}.html-response";target=self.root/relative;target.parent.mkdir(exist_ok=True)
     record['file']=relative
     with target.open('xb') as f:
      while record['actual_bytes']<PAGE:
       chunk=response.read(min(65536,PAGE-record['actual_bytes']))
       if not chunk:break
       f.write(chunk);f.flush();os.fsync(f.fileno())
       with self.mutex:record['actual_bytes']+=len(chunk);self.state['actual_body_bytes']+=len(chunk);self.save()
     if (n is not None and int(n)!=record['actual_bytes']) or (n is None and record['actual_bytes']==PAGE):raise ValueError('Incomplete/unbounded source page')
     record['sha256']=sha(target)
    with self.mutex:
     if method=='GET':self.state['conservative_charge']-=reserved-record['actual_bytes']
     record['complete']=True;self.save()
    return record
  except BaseException as exc:
   with self.mutex:record['error']=repr(exc);self.save()
   raise
 def decoded(self,record):
  data=(self.root/record['file']).read_bytes();encoding=record['headers'].get('Content-Encoding','')
  if encoding=='gzip':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(DECODED+1)
  elif encoding not in ('','identity'):raise ValueError('Unknown document encoding')
  if len(data)>DECODED:raise ValueError('Decoded source document cap')
  return data.decode('utf-8')
