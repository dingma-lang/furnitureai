"""Single pinned archive acquisition; default TLS, bounded safe range recovery."""
import argparse
import fcntl
import ssl
import urllib.request
from common import *

DOCS={
 'published-checksums.txt':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/coco.txt',
 'coco-terms.html':'https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm',
 'coco-download.html':'https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm'}
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def opener():
 context=ssl.create_default_context()
 if context.verify_mode!=ssl.CERT_REQUIRED or not context.check_hostname:raise RuntimeError('TLS verification required')
 return urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=context))

def validate_headers(status,headers,offset):
 expected=SIZE-offset
 if status!=(206 if offset else 200) or int(headers.get('Content-Length','-1'))!=expected or headers.get('ETag')!=ETAG or headers.get('Content-Encoding','identity') not in ('identity',''):raise ValueError('Archive status/version/size changed')
 if headers.get('Content-Type','').split(';')[0]!='application/zip':raise ValueError('Unexpected archive type')
 if offset and headers.get('Content-Range')!=f'bytes {offset}-{SIZE-1}/{SIZE}':raise ValueError('Unsafe range response')

def complete(target,state,recovered=False):
 if target.stat().st_size!=SIZE or sha(target)!=SHA:raise ValueError('Completion requires pinned full SHA256')
 commit(OUT/'acquisition-complete.json',dict(archive=target.name,source_url=URL,expected_sha256=SHA,local_sha256=SHA,archive_bytes=SIZE,actual_total_body_bytes=state['actual_body_bytes'],conservative_transfer_charge=state['conservative_charged_bytes'],archive_attempts=state['archive_attempts'],downloaded_utc=now(),recovered_completion=recovered,annotation_creator='COCO Consortium',annotation_license='https://creativecommons.org/licenses/by/4.0/',photographs_downloaded=0,training=False))

def download(resume=False):
 manifest=approved()
 if OUT.exists() and not resume:raise FileExistsError('Fresh destination required; explicit --resume only')
 if resume and not OUT.exists():raise FileNotFoundError('No acquisition to resume')
 OUT.mkdir(parents=True,exist_ok=resume)
 auth=OUT/'authorization.json'
 if not auth.exists():write(auth,dict(metadata_acquisition_approved=True,user_order='APPROVED — Furniture AI Phase6B COCO2017 metadata acquisition and audit',proposal_sha256=sha(PROPOSAL),recorded_utc=now(),archive_count=1,photographs=0,models=0,max_transfer=TRANSFER,max_storage=STORAGE,max_attempts=2))
 if read(auth)['proposal_sha256']!=sha(PROPOSAL):raise ValueError('Approval binding changed')
 with (OUT/'acquisition.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  journal=OUT/'network-ledger.json';state=read(journal) if journal.exists() else dict(proposal_sha256=sha(PROPOSAL),actual_body_bytes=0,conservative_charged_bytes=0,archive_attempts=0,requests=[])
  if state['proposal_sha256']!=sha(PROPOSAL):raise ValueError('Journal binding changed')
  transport=opener()
  for filename,url in DOCS.items():
   dest=OUT/'provenance'/filename
   if dest.exists():
    rec=next((r for r in state['requests'] if r['url']==url and r.get('complete')),None)
    if rec is None or sha(dest)!=rec['sha256']:raise ValueError('Untracked/changed provenance document')
    continue
   limit=65536
   if state['conservative_charged_bytes']+limit>TRANSFER:raise RuntimeError('Transfer cap')
   rec=dict(url=url,kind='source_document',utc=now(),reserved=limit,actual_bytes=0,complete=False)
   state['conservative_charged_bytes']+=limit;state['requests'].append(rec);atomic(journal,state)
   dest.parent.mkdir(exist_ok=True)
   with transport.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=30) as r:
    if r.status!=200 or int(r.headers.get('Content-Length','0'))>limit:raise ValueError('Source documentation unavailable/oversize')
    declared=r.headers.get('Content-Length')
    with dest.open('xb') as f:
     while rec['actual_bytes']<limit:
      chunk=r.read(min(8192,limit-rec['actual_bytes']))
      if not chunk:break
      f.write(chunk);f.flush();os.fsync(f.fileno());rec['actual_bytes']+=len(chunk);state['actual_body_bytes']+=len(chunk);atomic(journal,state)
    if rec['actual_bytes']==limit:raise ValueError('Document reaches cap')
    if declared is not None and rec['actual_bytes']!=int(declared):raise ValueError('Incomplete source document')
   rec.update(complete=True,sha256=sha(dest));state['conservative_charged_bytes']-=limit-rec['actual_bytes'];atomic(journal,state)
  registry=(OUT/'provenance/published-checksums.txt').read_text()
  matches=[line.split() for line in registry.splitlines() if line.startswith('http://images.cocodataset.org/annotations/annotations_trainval2017.zip\t')]
  if len(matches)!=1 or matches[0][1:3]!=[str(SIZE),SHA]:raise ValueError('Independent published checksum/size changed')
  if 'creativecommons.org/licenses/by/4.0/legalcode' not in (OUT/'provenance/coco-terms.html').read_text() or 'COCO Consortium does not own the copyright' not in (OUT/'provenance/coco-terms.html').read_text():raise ValueError('Annotation/photo licence terms inconsistent')
  if 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip' not in (OUT/'provenance/coco-download.html').read_text():raise ValueError('Official source publication changed')
  target=OUT/'annotations_trainval2017.zip';part=OUT/'annotations_trainval2017.zip.part'
  if target.exists():
   if target.stat().st_size!=SIZE or sha(target)!=SHA:raise ValueError('Existing archive corrupt')
   if not (OUT/'acquisition-complete.json').exists():
    committed=[r for r in state['requests'] if r['kind']=='archive' and r.get('complete') and r.get('sha256')==SHA]
    if not committed:raise ValueError('Untracked archive; cannot invent transfer history')
    complete(target,state,recovered=True)
   saved=read(OUT/'acquisition-complete.json')
   if saved['local_sha256']!=SHA or saved['archive_bytes']!=SIZE or saved['source_url']!=URL:raise ValueError('Completion marker does not match archive')
   print('SKIP COMPLETE',flush=True);return
  with transport.open(urllib.request.Request(URL,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=30) as response:
   validate_headers(response.status,response.headers,0)
   head=dict(url=URL,method='HEAD',body_bytes=0,checked_utc=now(),bytes=SIZE,etag=response.headers['ETag'],tls_default_certificate_chain_verified=True)
  if not (OUT/'source-preflight.json').exists():write(OUT/'source-preflight.json',dict(head=head,independent_checksum_matches=True,source_documents={name:sha(OUT/'provenance'/name) for name in DOCS},maximum_planned_storage=SIZE+EXTRACT_MAX+DERIVED_MAX+16*2**20))
  # A single partial file is range-resumed, avoiding two stored archive copies.
  offset=part.stat().st_size if part.exists() else 0
  if offset:
   prior=[r for r in state['requests'] if r['kind']=='archive'][-1]
   if offset!=prior['prefix_bytes'] or sha(part)!=prior['prefix_sha256']:raise ValueError('Partial prefix differs from committed ledger; refusing uncertain resume')
  if offset==SIZE:
   if sha(part)!=SHA:raise ValueError('Full partial archive checksum mismatch')
   prior.update(complete=True,sha256=SHA,finished_utc=now());atomic(journal,state);os.replace(part,target);complete(target,state,recovered=True);print('RECOVERED FULL ARCHIVE',flush=True);return
  if state['archive_attempts']>=2:raise RuntimeError('Two archive attempts exhausted')
  remaining=SIZE-offset
  if not 0<remaining<=SIZE:raise ValueError('Invalid partial archive size')
  if state['conservative_charged_bytes']+remaining>TRANSFER:raise RuntimeError('Transfer cap')
  space(remaining+EXTRACT_MAX+DERIVED_MAX)
  state['archive_attempts']+=1;state['conservative_charged_bytes']+=remaining
  rec=dict(url=URL,kind='archive',attempt=state['archive_attempts'],utc=now(),offset=offset,reserved=remaining,actual_bytes=0,prefix_bytes=offset,prefix_sha256=sha(part) if offset else hashlib.sha256(b'').hexdigest(),complete=False)
  state['requests'].append(rec);atomic(journal,state)
  headers={'Accept-Encoding':'identity','If-Match':ETAG,'User-Agent':'FurnitureAI-metadata-audit/1.0'}
  if offset:headers['Range']=f'bytes={offset}-'
  digest=hashlib.sha256()
  if offset:
   with part.open('rb') as f:
    for chunk in iter(lambda:f.read(1048576),b''):digest.update(chunk)
  try:
   with transport.open(urllib.request.Request(URL,headers=headers),timeout=60) as response:
    validate_headers(response.status,response.headers,offset)
    with part.open('ab' if offset else 'xb') as f:
     while rec['actual_bytes']<remaining:
      chunk=response.read(min(1048576,remaining-rec['actual_bytes']))
      if not chunk:break
      f.write(chunk);f.flush();os.fsync(f.fileno());digest.update(chunk)
      rec['actual_bytes']+=len(chunk);state['actual_body_bytes']+=len(chunk);rec['prefix_bytes']=offset+rec['actual_bytes'];rec['prefix_sha256']=digest.hexdigest();atomic(journal,state)
      if rec['prefix_bytes']//(50*2**20)!=(rec['prefix_bytes']-len(chunk))//(50*2**20):print('ARCHIVE BYTES',rec['prefix_bytes'],'/',SIZE,flush=True)
   if rec['actual_bytes']!=remaining or part.stat().st_size!=SIZE:raise ValueError('Interrupted/incomplete archive; retain prefix for explicit resume')
   if digest.hexdigest()!=SHA:raise ValueError('Archive SHA256 mismatch; stop without extraction')
   rec.update(complete=True,sha256=SHA,finished_utc=now());atomic(journal,state)
   os.replace(part,target)
   complete(target,state)
   print('ARCHIVE VERIFIED',SIZE,SHA,flush=True)
  except BaseException as exc:
   rec['error']=repr(exc);atomic(journal,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args();download(a.resume)
