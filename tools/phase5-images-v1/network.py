"""Bounded audited network requests for the explicitly approved Phase 5 image pilot."""
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase5-metadata-v1'))
from acquire import atomic, allocated, NoRedirect
DEST=ROOT/'data/open-images/image-pilots/phase5-v1'
MANIFEST=ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json'
PIN='b07ee0e045a7d005d26778b8c27e76bbaa78129dd05e113540efe28094e3e131'
CAP=100000000
DISK=536870912
PHOTO_MAX=3145728

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest():
    if sha(MANIFEST)!=PIN:raise ValueError('Frozen acquisition manifest changed')
    m=json.loads(MANIFEST.read_text())
    if len(m['images'])!=150:raise ValueError('Expected 150 approved identities')
    return m

class Budget:
    def __init__(self,root=DEST,resume=False):
        self.root=Path(root)
        if (self.root/'completion.json').exists():raise RuntimeError('Completed acquisition is immutable; use read-only verification/tests')
        if self.root.exists() and not resume:raise FileExistsError('Destination exists: explicit resume required')
        self.root.mkdir(parents=True,exist_ok=resume)
        self.lock=(self.root/'acquisition.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path=self.root/'network-ledger.json'
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(manifest_sha256=PIN,cap=CAP,disk_cap=DISK,actual_body_bytes=0,charged_bytes=0,requests=[],image_attempts={})
        if self.state['manifest_sha256']!=PIN or self.state['cap']!=CAP:raise ValueError('Budget/manifest binding changed')
        self.opener=urllib.request.build_opener(NoRedirect());self.save()
    def close(self):self.lock.close()
    def save(self):atomic(self.path,self.state)
    def fetch(self,url,relative,limit,kind='licence_html',expected=None,etag=None,reserve_after=0):
        target=self.root/relative
        if not target.resolve().is_relative_to(self.root.resolve()):raise ValueError('Unsafe output path')
        old=next((r for r in self.state['requests'] if r.get('file')==relative and r.get('complete')),None)
        if old:
            if old['url']!=url or not target.is_file() or sha(target)!=old['sha256']:raise ValueError('Cached response changed')
            return old
        if target.exists():raise FileExistsError('Untracked response file')
        if self.state['charged_bytes']+limit+reserve_after>CAP or allocated(self.root)+limit+8388608>DISK:raise RuntimeError('Insufficient approved transfer/storage budget')
        headers={'Accept-Encoding':'gzip' if kind=='licence_html' else 'identity','User-Agent':'FurnitureAI-research-pilot/1.0'}
        if etag:headers['If-Match']=etag
        rec=dict(url=url,file=relative,kind=kind,method='GET',utc=dt.datetime.now(dt.timezone.utc).isoformat(),complete=False,actual_body_bytes=0,reserved_bytes=limit,runner_sha256=sha(__file__))
        if kind=='image':
            attempts=self.state['image_attempts'].get(url,0)
            if attempts>=2:raise RuntimeError('Image two-attempt limit')
            self.state['image_attempts'][url]=attempts+1
        self.state['charged_bytes']+=limit;self.state['requests'].append(rec);self.save()
        response=None
        try:
            try:response=self.opener.open(urllib.request.Request(url,headers=headers),timeout=45)
            except urllib.error.HTTPError as e:response=e
            with response:
                rec.update(status=response.status,headers={k:v for k,v in response.headers.items() if k.lower() not in ('set-cookie',)})
                n=response.headers.get('Content-Length')
                if n is not None and int(n)>limit:raise ValueError('Response exceeds reserved object limit')
                if kind=='image' and (response.status!=200 or response.headers.get('ETag')!=etag or int(n or '-1')!=expected):raise ValueError('Image status/version/size changed')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:
                    left=limit
                    while left:
                        chunk=response.read(min(left,65536))
                        if not chunk:break
                        f.write(chunk);f.flush();os.fsync(f.fileno());left-=len(chunk)
                        self.state['actual_body_bytes']+=len(chunk);rec['actual_body_bytes']+=len(chunk);self.save()
                if n is not None and int(n)!=rec['actual_body_bytes']:raise ValueError('Incomplete response body')
                if rec['actual_body_bytes']==limit and n is None:raise ValueError('Unknown-length response reached hard limit')
            rec.update(complete=True,sha256=sha(target));self.state['charged_bytes']-=limit-rec['actual_body_bytes'];self.save()
            return rec
        except BaseException as exc:
            rec['error']=repr(exc);self.save() # Keep unconsumed reservation charged on uncertain failures.
            raise
    def decoded(self,rec,max_decoded=4194304):
        data=(self.root/rec['file']).read_bytes()
        encoding=next((v for k,v in rec['headers'].items() if k.lower()=='content-encoding'),'')
        if encoding.lower()=='gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(max_decoded+1)
        if len(data)>max_decoded:raise ValueError('Decoded HTML limit')
        return data.decode('utf-8')
