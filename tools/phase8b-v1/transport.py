"""Single-worker paced HTTPS, one candidate attempt, durable per-provider circuit."""
import email.utils
import gzip
import io
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from audit_core import *
from source_evidence import allowed_page

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None

def provider(url):
 host=(urlparse(url).hostname or '').casefold()
 return 'flickr.com' if host in ('flickr.com','www.flickr.com') else host

def retry_not_before(value,instant=None):
 instant=instant or dt.datetime.now(dt.timezone.utc)
 try:
  if value is not None and str(value).strip().isdigit():return (instant+dt.timedelta(seconds=int(value))).isoformat()
  parsed=email.utils.parsedate_to_datetime(value) if value else None
  if parsed:
   if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=dt.timezone.utc)
   return max(instant,parsed.astimezone(dt.timezone.utc)).isoformat()
 except (ValueError,TypeError,OverflowError):pass
 return (instant+dt.timedelta(seconds=120)).isoformat()

def allowed_request(row,url,method):
 if method not in ('HEAD','GET'):return False
 if url==row['official_image_object_url']:return method=='HEAD'
 return allowed_page(url,row['source_photo_id'].split(':')[1])

def begin_candidate(state,iid):
 if str(iid) in state['candidates']:raise ValueError('Candidate already attempted; no second evidence attempt')
 state['candidates'][str(iid)]=dict(attempt=1,started_utc=now(),complete=False)

class Transport:
 def __init__(self,resume=False):
  self.path=OUT/'network-ledger.json'
  if self.path.exists() and not resume:raise FileExistsError('Existing journal requires --resume')
  self.state=read(self.path) if self.path.exists() else dict(manifest_sha256=PIN,actual_body_bytes=0,conservative_charge=0,requests=[],candidates={},circuits={},last_source_utc=None)
  if self.state['manifest_sha256']!=PIN:raise ValueError('Resume scope changed')
  if self.state['actual_body_bytes']!=sum(r['actual_body_bytes'] for r in self.state['requests']) or not 0<=self.state['actual_body_bytes']<=self.state['conservative_charge']<=CAP:
   raise ValueError('Durable transfer ledger does not reconcile')
  if any(r['attempt']!=1 for r in self.state['candidates'].values()):raise ValueError('Candidate attempt bound changed')
  context=ssl.create_default_context();assert context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED
  self.opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context));self.save()
 def save(self):atomic(self.path,self.state)
 def circuit_open(self,host='flickr.com'):return self.state['circuits'].get(host,{}).get('open',False)
 def begin(self,row):
  if self.circuit_open():raise HostSuspended('Source provider suspended; no attempt consumed')
  begin_candidate(self.state,row['coco_image_id']);self.save()
 def request(self,row,url,method):
  if not allowed_request(row,url,method):raise ValueError('Unapproved host/photo identity or prohibited photograph-body GET')
  host=provider(url);is_source=host=='flickr.com'
  if self.circuit_open(host):raise HostSuspended('Provider circuit remains open; separate future approval required')
  candidate=self.state['candidates'].get(str(row['coco_image_id']))
  if not candidate or candidate['attempt']!=1 or candidate.get('complete'):raise ValueError('No active single candidate attempt')
  previous=[r for r in self.state['requests'] if r['coco_image_id']==row['coco_image_id']]
  if any(r['url']==url and r['method']==method for r in previous):raise ValueError('Repeated HTTP transaction is not allowed')
  if method=='GET' and any(r['method']=='GET' for r in previous):raise ValueError('Only one source-document GET per candidate')
  if is_source:
   previous_time=self.state.get('last_source_utc')
   if previous_time:
    delay=max(0,3-(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(previous_time)).total_seconds())
    if delay:time.sleep(min(delay,3))
   self.state['last_source_utc']=now()
  reserve=PAGE if method=='GET' else 0
  if self.state['conservative_charge']+reserve>CAP:raise ResourceLimit('32,000,000-byte cumulative response-body limit')
  space(reserve+PAGE)
  record=dict(coco_image_id=row['coco_image_id'],source_photo_id=row['source_photo_id'],url=url,method=method,provider=host,
   checked_utc=now(),reserved_body_bytes=reserve,actual_body_bytes=0,complete=False,tls_verified=False,candidate_attempt=1)
  self.state['conservative_charge']+=reserve;self.state['requests'].append(record);self.save()
  try:
   try:response=self.opener.open(urllib.request.Request(url,method=method,headers={'Accept-Encoding':'gzip' if method=='GET' else 'identity','User-Agent':'FurnitureAI-source-rights-audit/1.0'}),timeout=30)
   except urllib.error.HTTPError as e:response=e
   with response:
    headers={k:response.headers[k] for k in ('Content-Type','Content-Length','Content-Encoding','Location','ETag','Last-Modified','Retry-After') if k in response.headers}
    record.update(status=response.status,headers=headers,tls_verified=True)
    if response.status in (401,403,429):
     self.state['circuits'][host]=dict(open=True,reason=f'HTTP{response.status}',opened_utc=now(),retry_after=headers.get('Retry-After'),not_before=retry_not_before(headers.get('Retry-After')),
      same_phase_retry_permitted=False,separate_future_approval_required=True)
     self.save()
    if method=='GET' and response.status==200:
     if headers.get('Content-Type','').split(';')[0] not in ('text/html','application/xhtml+xml'):raise ValueError('Non-HTML response; body not read')
     size=headers.get('Content-Length')
     if size is not None and not 0<=int(size)<=PAGE:raise ValueError('Source document length bound')
     filename=f'responses/{row["coco_image_id"]:012d}.html-response';target=OUT/filename;record['file']=filename
     with target.open('xb') as f:
      while record['actual_body_bytes']<PAGE:
       b=response.read(min(65536,PAGE-record['actual_body_bytes']))
       if not b:break
       f.write(b);f.flush();os.fsync(f.fileno());record['actual_body_bytes']+=len(b);self.state['actual_body_bytes']+=len(b);self.save()
     if size is not None and record['actual_body_bytes']!=int(size) or size is None and record['actual_body_bytes']==PAGE:raise ValueError('Incomplete or unbounded document')
     record['sha256']=sha(target);self.state['conservative_charge']-=reserve-record['actual_body_bytes']
    elif method=='GET':record['body_not_read']=True # Keep full reservation for a conservative error-body upper bound.
    record['complete']=True;self.save();return record
  except BaseException as e:
   record['error']=repr(e)
   if isinstance(e,OSError):
    self.state['circuits'][host]=dict(open=True,reason='Transport/TLS access failure',opened_utc=now(),separate_future_approval_required=True)
   self.save();raise
 def decoded(self,record):
  p=OUT/record['file']
  if sha(p)!=record['sha256']:raise ValueError('Saved source response hash changed')
  data=p.read_bytes();encoding=record['headers'].get('Content-Encoding','')
  if encoding=='gzip':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(DECODED+1)
  elif encoding not in ('','identity'):raise ValueError('Unsupported source encoding')
  if len(data)>DECODED:raise ValueError('Decoded source document bound')
  return data.decode('utf-8')
