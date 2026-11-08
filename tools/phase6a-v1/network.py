"""Durable bounded transport; interrupted bodies retained and fully charged.

Only rights-page verification and HEAD preflight are authorized now. Photo GETs
require a separate approval file binding the exact finalized manifest digest.
"""
import fcntl
import gzip
import io
import urllib.error
import urllib.request
from core import *

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

class Budget:
 def __init__(self,root,cap=TRANSFER,disk_cap=STORAGE):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  self.lock=(self.root/'network.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.path=self.root/'network-ledger.json';self.cap=cap;self.disk_cap=disk_cap
  self.state=read(self.path) if self.path.exists() else dict(cap=cap,disk_cap=disk_cap,charged=0,actual_bytes=0,requests=[])
  if self.state['cap']!=cap or self.state['disk_cap']!=disk_cap:raise ValueError('Resource binding changed')
  self.opener=urllib.request.build_opener(NoRedirect());atomic(self.path,self.state)
 def close(self):self.lock.close()
 def request(self,url,relative,limit,kind,expected=None,etag=None):
  u=urlparse(url)
  if kind=='rights':
   if not photo_id(url):raise ValueError('Only canonical individual Flickr rights pages')
  elif kind=='image':
   if u.scheme!='https' or u.hostname!='open-images-dataset.s3.amazonaws.com' or not re.fullmatch(r'/validation/[0-9a-f]{16}\.jpg',u.path) or u.query or u.fragment or u.port or u.username:raise ValueError('Unapproved photo source')
  else:raise ValueError('Unexpected transfer kind')
  target=(self.root/relative).resolve()
  if not target.is_relative_to(self.root.resolve()) or Path(relative).is_absolute():raise ValueError('Unsafe path')
  cached=[r for r in self.state['requests'] if r['url']==url and r.get('complete')]
  if cached:
   r=cached[-1]
   if r['file']!=relative or sha(target)!=r['sha256']:raise ValueError('Changed cache')
   return r
  attempts=sum(r['url']==url for r in self.state['requests'])
  if attempts>=2:raise RuntimeError('Maximum two attempts reached')
  if target.exists():raise FileExistsError('Never overwrite a response')
  if self.state['charged']+limit>self.cap or allocated(self.root)+limit+8388608>self.disk_cap:raise RuntimeError('Resource cap')
  rec=dict(url=url,file=relative,kind=kind,attempt=attempts+1,utc=now(),reserved=limit,actual_bytes=0,complete=False)
  self.state['charged']+=limit;self.state['requests'].append(rec);atomic(self.path,self.state)
  headers={'User-Agent':'FurnitureAI-license-audit/1.0','Accept-Encoding':'gzip' if kind=='rights' else 'identity'}
  if etag:headers['If-Match']=etag
  try:
   try:response=self.opener.open(urllib.request.Request(url,headers=headers),timeout=25)
   except urllib.error.HTTPError as exc:response=exc
   with response:
    rec.update(status=response.status,content_type=response.headers.get('Content-Type',''),encoding=response.headers.get('Content-Encoding',''),etag=response.headers.get('ETag'))
    length=response.headers.get('Content-Length');n=int(length) if length is not None else None
    if n is not None and (n<0 or n>limit):raise ValueError('Unexpected response size')
    if kind=='image' and (response.status!=200 or n!=expected or rec['etag']!=etag or rec['content_type'].split(';')[0]!='image/jpeg'):raise ValueError('Image version/size/type/status changed')
    if kind=='rights' and response.status!=200:raise ValueError('Rights page unavailable HTTP '+str(response.status))
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
     while rec['actual_bytes']<limit:
      b=response.read(min(65536,limit-rec['actual_bytes']))
      if not b:break
      f.write(b);f.flush();os.fsync(f.fileno());rec['actual_bytes']+=len(b);self.state['actual_bytes']+=len(b);atomic(self.path,self.state)
    if (n is not None and n!=rec['actual_bytes']) or (n is None and rec['actual_bytes']==limit):raise ValueError('Incomplete/unbounded body')
   rec.update(complete=True,sha256=sha(target));self.state['charged']-=limit-rec['actual_bytes'];atomic(self.path,self.state);return rec
  except BaseException as exc:
   rec['error']=repr(exc);atomic(self.path,self.state);raise
 def decoded(self,rec):
  data=(self.root/rec['file']).read_bytes()
  if rec['encoding']=='gzip':
   with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(4194305)
  elif rec['encoding'] not in ('','identity'):raise ValueError('Unknown encoding')
  if len(data)>4194304:raise ValueError('Decoded document cap')
  return data.decode('utf-8')

def head(url):
 # Never redirects or reads a photograph response body.
 u=urlparse(url)
 if u.hostname!='open-images-dataset.s3.amazonaws.com' or not re.fullmatch(r'/validation/[0-9a-f]{16}\.jpg',u.path) or u.scheme!='https' or u.query or u.fragment:raise ValueError('Unexpected HEAD source')
 opener=urllib.request.build_opener(NoRedirect())
 with opener.open(urllib.request.Request(url,method='HEAD'),timeout=25) as response:
  size=int(response.headers.get('Content-Length','-1'));etag=response.headers.get('ETag','')
  if response.status!=200 or not 0<size<=IMAGE_MAX or not etag or response.headers.get('Content-Type','').split(';')[0]!='image/jpeg':raise ValueError('Invalid remote photo object')
  return dict(url=url,bytes=size,etag=etag,method='HEAD',response_body_bytes=0,checked_utc=now())

def approval(manifest_path,approval_path):
 record=read(approval_path)
 if record.get('image_acquisition_approved') is not True or record.get('manifest_sha256')!=sha(manifest_path) or record.get('destination')!=str(DEST) or record.get('max_transfer')!=TRANSFER or record.get('max_storage')!=STORAGE:raise PermissionError('Separate explicit image approval bound to this exact manifest is required')
 return record
