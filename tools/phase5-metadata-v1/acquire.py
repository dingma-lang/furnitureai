"""Acquire only the eight approved metadata objects, with a durable bounded ledger.

No image support. Run with the project interpreter. Explicit --resume only; altered
sources, unverifiable partials and checksum failures stop without replacing files.
"""
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
PROPOSAL=ROOT/'reports/phase5-detection-preparation-v1/metadata-acquisition-proposal.json'
DEST=ROOT/'data/open-images/metadata/phase5-v1'
CAP=220_000_000
DISK=1_073_741_824
BLOCK=262144


def digest(path,alg='sha256'):
    h=hashlib.new(alg)
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def allocated(root):
    return sum(p.stat().st_blocks*512 for p in [root,*root.rglob('*')])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Unapproved redirect')


def approved():
    plan=json.loads(PROPOSAL.read_text());objects=plan['objects']
    if len(objects)!=8 or sum(r['bytes'] for r in objects)!=205865596:raise ValueError('Approved object list changed')
    if plan['cumulative_body_cap']!=CAP or plan['new_allocated_storage_cap']!=DISK:raise ValueError('Approval budget changed')
    names=set()
    for r in objects:
        u=urllib.parse.urlparse(r['url']);name=Path(u.path).name
        if u.scheme!='https' or u.netloc!='storage.googleapis.com' or not u.path.startswith('/openimages/') or u.query or u.fragment or not name.endswith(('.csv','.json')) or name in names:raise ValueError('Unapproved metadata path')
        names.add(name)
    return objects


def preflight(opener,objects):
    result=[]
    for r in objects:
        with opener.open(urllib.request.Request(r['url'],method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=45) as response:
            headers=dict(response.headers.items())
            if response.status!=200 or int(response.headers.get('Content-Length','-1'))!=r['bytes'] or response.headers.get('ETag')!=r['etag'] or response.headers.get('Last-Modified')!=r['last_modified']:raise ValueError('Source size/ETag/version changed: '+r['url'])
            hashes=dict(item.split('=',1) for value in response.headers.get_all('x-goog-hash',[]) for item in value.split(',') if '=' in item)
            hashes={k.strip():v.strip() for k,v in hashes.items()}
            if 'md5' not in hashes or base64.b64decode(hashes['md5']).hex()!=r['etag'].strip('"'):raise ValueError('No verifiable source MD5/ETag')
            result.append(dict(**r,verified_headers=headers,source_md5_hex=base64.b64decode(hashes['md5']).hex(),local_name=Path(urllib.parse.urlparse(r['url']).path).name))
    return result


def run(resume=False):
    objects=approved();opener=urllib.request.build_opener(NoRedirect())
    # Complete HEAD verification for all eight BEFORE creating the destination/body requests.
    sources=preflight(opener,objects)
    if DEST.exists() and not resume:raise FileExistsError('Destination exists: explicit --resume required')
    if not DEST.exists():
        if shutil.disk_usage(ROOT).free < DISK:raise OSError('Insufficient free storage headroom')
        DEST.mkdir(parents=True)
    with (DEST/'acquisition.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=dict(proposal_sha256=digest(PROPOSAL),runner_sha256=digest(__file__),expected_bytes=205865596,cap=CAP,allocated_storage_cap=DISK)
        state_path=DEST/'transfer-ledger.json'
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['binding']!=binding:raise ValueError('Resume code/approval binding changed')
            # Reserved chunks from a killed process remain charged conservatively.
            state['conservative_unknown_bytes']+=state.get('outstanding_reservation',0)
            state['outstanding_reservation']=0
        else:
            if resume:raise ValueError('No existing acquisition ledger')
            state=dict(binding=binding,actual_body_bytes=0,charged_bytes=0,conservative_unknown_bytes=0,outstanding_reservation=0,objects={},complete=False)
        atomic(state_path,state)
        pre=DEST/'source-preflight.json'
        if not pre.exists():atomic(pre,dict(sources=sources,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),licensing=dict(annotations='CC BY 4.0; Google LLC',images='Not acquired; per-image licensing verification required',source='https://storage.googleapis.com/openimages/web/factsfigures_v7.html',verified_public_documentation=True)))
        for source in sources:
            name=source['local_name'];target=DEST/name;partial=DEST/(name+'.part')
            item=state['objects'].setdefault(name,dict(attempts=0,complete=False,partial_size=0,partial_sha256=None))
            if target.exists():
                if not item['complete'] or target.stat().st_size!=source['bytes'] or digest(target)!=item['sha256'] or digest(target,'md5')!=source['source_md5_hex']:raise ValueError('Completed object integrity mismatch')
                continue
            if item['complete']:raise ValueError('Completed object missing')
            if partial.exists() and (partial.stat().st_size!=item['partial_size'] or digest(partial)!=item['partial_sha256']):raise ValueError('Partial not durably checkpointed; preserve it and stop')
            if not partial.exists() and item['partial_size']:raise ValueError('Missing partial file')
            while not item['complete']:
                if item['attempts']>=2:raise RuntimeError('Two-attempt ceiling reached')
                if allocated(DEST)+source['bytes']-item['partial_size']+8*1048576>DISK:raise RuntimeError('Allocated storage budget insufficient')
                offset=item['partial_size'];remaining=source['bytes']-offset
                if state['charged_bytes']+remaining>CAP:raise RuntimeError('Insufficient cumulative transfer budget')
                headers={'Accept-Encoding':'identity','If-Match':source['etag']}
                if offset:headers['Range']=f'bytes={offset}-'
                item['attempts']+=1;atomic(state_path,state)
                print('GET',name,'offset',offset,'attempt',item['attempts'],flush=True)
                try:
                    response=opener.open(urllib.request.Request(source['url'],headers=headers),timeout=60)
                    with response:
                        if response.status!=(206 if offset else 200) or response.headers.get('ETag')!=source['etag'] or int(response.headers.get('Content-Length','-1'))!=remaining or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('Unexpected GET status/version/length/encoding')
                        if offset and response.headers.get('Content-Range')!=f"bytes {offset}-{source['bytes']-1}/{source['bytes']}":raise ValueError('Unsafe Range response')
                        with partial.open('ab' if offset else 'xb') as f:
                            while remaining:
                                size=min(BLOCK,remaining)
                                if state['charged_bytes']+size>CAP or allocated(DEST)+size+2*1048576>DISK:raise RuntimeError('Hard resource limit')
                                # Reserve BEFORE network read: a process kill cannot reset the cap.
                                state['charged_bytes']+=size;state['outstanding_reservation']=size;atomic(state_path,state)
                                chunk=response.read(size)
                                state['actual_body_bytes']+=len(chunk)
                                state['charged_bytes']-=size-len(chunk);state['outstanding_reservation']=0
                                if not chunk:atomic(state_path,state);raise OSError('Premature body EOF')
                                f.write(chunk);f.flush();os.fsync(f.fileno());remaining-=len(chunk)
                                # Persist durable byte offset/hash after each received chunk.
                                item['partial_size']=partial.stat().st_size;item['partial_sha256']=digest(partial)
                                atomic(state_path,state)
                    if partial.stat().st_size!=source['bytes'] or digest(partial,'md5')!=source['source_md5_hex']:raise ValueError('Downloaded source checksum mismatch')
                    item.update(complete=True,sha256=digest(partial),md5_hex=digest(partial,'md5'),bytes=source['bytes'],url=source['url'],etag=source['etag'])
                    # Use hard-link creation: never replace a pre-existing final object.
                    os.link(partial,target);partial.unlink();atomic(state_path,state)
                    print('VERIFIED',name,item['sha256'],flush=True)
                except (TimeoutError,ConnectionError,OSError) as exc:
                    # Only transport failures are retryable. Unknown read reservations stay charged.
                    state['conservative_unknown_bytes']+=state['outstanding_reservation'];state['outstanding_reservation']=0
                    item['last_transport_error']=repr(exc);atomic(state_path,state)
                    if isinstance(exc,urllib.error.HTTPError):raise # no alternate sources/status repairs
                    if item['attempts']>=2:raise
                    print('TRANSPORT STOP/RESUME',repr(exc),flush=True)
        state['complete']=True;state['allocated_bytes_at_completion']=allocated(DEST);atomic(state_path,state)
        checks=DEST/'SHA256SUMS'
        if not checks.exists():
            with checks.open('x') as f:
                for name,item in sorted(state['objects'].items()):f.write(item['sha256']+'  '+name+'\n')
        print(json.dumps(dict(complete=True,actual_body_bytes=state['actual_body_bytes'],charged_bytes=state['charged_bytes'],allocated_bytes=allocated(DEST)),indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');args=ap.parse_args();run(args.resume)
