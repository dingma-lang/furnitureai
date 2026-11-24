"""Bounded, exact-source, verified TLS acquisition. No retries/redirects automatically."""
import http.client,time,urllib.parse,urllib.robotparser
from common11 import *
sys.path.insert(0,str(ROOT/'tools/phase10e9-v1'))
from transport9 import TLSStream
OVERHEAD=32768
ROBOTS=('https://api.wellcomecollection.org/robots.txt','https://iiif.wellcomecollection.org/robots.txt')
def allowed(url,kind):
    rs=rows()
    if kind=='robots':return url in ROBOTS
    if kind=='metadata':return url in {metadata_url(id) for id in IDS}|{r['image_metadata_URL'] for r in rs}
    if kind=='image':return url in {r['exact_download_URL'] for r in rs}
    return False
class Client:
    def __init__(self):
        if (OUT/'acquisition-closed.json').exists():raise PermissionError('Closed pilot cannot download')
        self.receipts=[read(p) for p in sorted((OUT/'requests').glob('*.json'))];intents=list((OUT/'intents').glob('*.json'))
        if len(intents)!=len(self.receipts):raise PermissionError('Interrupted request; preserve reservation and stop')
        self.used=sum(r['budget_charge'] for r in self.receipts);self.last=time.monotonic() if self.receipts else 0
        self.blocked={r['host'] for r in self.receipts if r.get('stop_provider')}
    def fetch(self,url,kind):
        if not allowed(url,kind):raise PermissionError('Only exact authorized sources')
        u=urllib.parse.urlsplit(url);host=u.hostname
        if host in self.blocked:raise PermissionError('Provider stopped; no alternate endpoint')
        prior=[r for r in self.receipts if r['url']==url]
        # Approval permits at most2, but this one-shot implementation never retries automatically.
        if prior:raise PermissionError('Existing transaction preserved; no repeated successful/failed requests')
        body_cap=IMAGE_CAP if kind=='image' else 65536;tls_cap=body_cap+131072;reservation=tls_cap+OVERHEAD
        if self.used+reservation>CAP:raise RuntimeError('8,000,000 transfer reserve before request')
        space(2*body_cap+65536)
        wait=max(0,2-(time.monotonic()-self.last))
        if wait:time.sleep(wait)
        self.last=time.monotonic();num=len(self.receipts)+1
        emit(OUT/'intents'/f'{num:03}.json',dict(number=num,url=url,kind=kind,reserved=reservation,time_UTC=now()))
        r=dict(number=num,url=url,host=host,kind=kind,checked_UTC=now(),TLS_sent=0,TLS_received=0,body_bytes=0,complete=False,attempt=1)
        self.receipts.append(r);self.used+=reservation;s=None;data=bytearray()
        def wire(event,n):
            if r['TLS_sent']+r['TLS_received']+n>tls_cap:raise RuntimeError('TLS socket limit before operation')
            if event=='sent':r['TLS_sent']+=n
            elif event=='received':r['TLS_received']+=n
            with (OUT/'wire-events.jsonl').open('ab') as f:
                f.write((json.dumps(dict(request=num,event=event,bytes=n,UTC=now()))+'\n').encode());f.flush();os.fsync(f.fileno())
        try:
            s=TLSStream(host,wire);r.update(TLS_verified=True,TLS_version=s.ssl.version(),certificate_SHA256=hashlib.sha256(s.ssl.getpeercert(binary_form=True)).hexdigest())
            target=u.path+('?' +u.query if u.query else '')
            s.sendall(('GET '+target+' HTTP/1.1\r\nHost: '+host+'\r\nUser-Agent: '+UA+'\r\nAccept: '+('image/jpeg' if kind=='image' else 'application/json,text/plain')+'\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n').encode('ascii'))
            response=http.client.HTTPResponse(s);response.begin();r['status']=response.status
            r['headers']={k:v for k,v in response.getheaders() if k.lower() in ('content-type','content-length','content-encoding','location','retry-after','etag','last-modified')}
            if response.status in (401,403,429):r['stop_provider']=True;self.blocked.add(host)
            if response.status!=200:
                r['body_not_read']=True
                if kind=='robots' and response.status==404:return r,b''
                raise RuntimeError('HTTP '+str(response.status)+'; no redirects/retries')
            mime=response.getheader('Content-Type','').split(';')[0].lower()
            accepted=('image/jpeg',) if kind=='image' else ('application/json','text/plain','application/ld+json')
            if mime not in accepted:raise ValueError('Unexpected MIME; no body processing')
            if response.getheader('Content-Encoding','identity').lower() not in ('','identity'):raise ValueError('Unexpected encoding')
            lengths=response.headers.get_all('Content-Length') or []
            if len(set(lengths))>1:raise ValueError('Conflicting Content-Length')
            length=int(lengths[0]) if lengths else None
            if length is not None and not 0<=length<=body_cap:raise RuntimeError('Object exceeds byte cap; stop before body')
            while len(data)<body_cap:
                chunk=response.read(min(16384,body_cap-len(data)))
                if not chunk:break
                data.extend(chunk);r['body_bytes']=len(data)
            if (length is not None and len(data)!=length) or (length is None and len(data)==body_cap):raise ValueError('Truncated or unbounded object; no extra cap-probe byte')
            r['complete']=True;r['SHA256']=hashlib.sha256(data).hexdigest()
            return r,bytes(data)
        except BaseException as exc:
            r['error']=type(exc).__name__+': '+str(exc);r['stop_provider']=True;self.blocked.add(host)
            if data:atomic(OUT/'quarantine'/f'{num:03}-{kind}.partial',bytes(data))
            raise
        finally:
            if s is not None:s.close()
            charge=r['TLS_sent']+r['TLS_received']+OVERHEAD if 'status' in r else reservation
            self.used-=reservation-charge;r['budget_charge']=charge
            emit(OUT/'requests'/f'{num:03}.json',r)
    def summary(self):return dict(requests=len(self.receipts),response_body_bytes=sum(r['body_bytes'] for r in self.receipts),
       image_body_bytes=sum(r['body_bytes'] for r in self.receipts if r['kind']=='image'),TLS_bytes=sum(r['TLS_sent']+r['TLS_received'] for r in self.receipts),
       conservative_charge=self.used,transfer_cap=CAP,per_image_cap=IMAGE_CAP,max_attempts_per_object=2,actual_max_attempts=1 if self.receipts else 0,
       redirects_followed=0,retries=0,physical_IP_TCP_DNS_retransmissions_not_measured=True)
def check_robots(client):
    evidence=[]
    for url in ROBOTS:
        receipt,body=client.fetch(url,'robots');r=dict(receipt)
        if receipt['status']==404:r['decision']='No robots policy published at this host; documented API/IIIF access still required'
        else:
            parser=urllib.robotparser.RobotFileParser();parser.parse(body.decode('utf-8-sig').splitlines())
            urls=[metadata_url(id) for id in IDS] if 'api.wellcome' in url else [u for row in rows() for u in (row['image_metadata_URL'],row['exact_download_URL'])]
            if not all(parser.can_fetch(UA,u) for u in urls):raise PermissionError('Robots denies a requested source path; do not bypass')
            atomic(OUT/'evidence'/('robots-api.txt' if 'api.wellcome' in url else 'robots-IIIF.txt'),body);r['decision']='Approved paths allowed for identified client'
        evidence.append(r)
    emit(OUT/'robots-preflight.json',dict(records=evidence))
def run():
    if OUT.exists():raise FileExistsError('Approved destination already exists; stop before modifying it')
    # Each inherited body/document checksum has already been sealed by E10.
    rs=rows();OUT.mkdir(parents=True)
    emit(OUT/'authorization.json',dict(exact_ids=IDS,approved_by='User Phase10E.11 attachment, not the earlier proposal false-approval flag',manifest_SHA256=MANIFEST_SHA,
        cap=CAP,disk_cap=DISK,per_image_cap=IMAGE_CAP,max_attempts=2,no_GPU=True,no_training=True,checked_UTC=now()))
    client=Client();accepted=[];stopped=None
    try:
        check_robots(client)
        # Reverify all7 rights and dimensions before requesting any photograph.
        for row in rs:
            ident=row['source_image_id'];receipt,body=client.fetch(metadata_url(ident),'metadata');value=json.loads(body)
            rights=rights_validate(value,row);atomic(OUT/'evidence'/(ident+'-catalogue.json'),body)
            info_receipt,info_body=client.fetch(row['image_metadata_URL'],'metadata');info=json.loads(info_body)
            if (info.get('width'),info.get('height'))!=(row['width'],row['height']) or info.get('@id')+'/full/!1600,1600/0/default.jpg'!=row['exact_download_URL']:raise ValueError('Changed image dimensions/service identity')
            atomic(OUT/'evidence'/(ident+'-IIIF.json'),info_body)
            emit(OUT/'rights'/(ident+'.json'),dict(image_id=ident,rights=rights,catalogue_evidence=receipt,IIIF_evidence=info_receipt,source_record=row))
        from screening11 import inspect_bytes
        for row in rs:
            ident=row['source_image_id'];receipt,body=client.fetch(row['exact_download_URL'],'image')
            quality=inspect_bytes(body,row)
            path=OUT/'images'/(ident+'.jpg');atomic(path,body)
            entry=dict(image_id=ident,file=str(path.relative_to(ROOT)),sha256=sha(path),source_url=row['exact_download_URL'],byte_size=len(body),
                quality=quality,rights_evidence=str((OUT/'rights'/(ident+'.json')).relative_to(ROOT)),request_number=receipt['number'],
                source_group_id=row['source_group_id'],final_partition=None,training_active=False)
            emit(OUT/'integrity'/(ident+'.json'),entry);accepted.append(entry);print('Acquired',ident,len(body),quality['display_dimensions'],flush=True)
    except BaseException as exc:
        stopped=dict(error=type(exc).__name__+': '+str(exc),UTC=now());print('STOP',stopped,flush=True)
    finally:
        emit(OUT/'acquisition-closed.json',dict(records=accepted,downloaded_count=len(accepted),stop=stopped,network=client.summary(),allocated_bytes_at_close=allocated(),training_updates=0,GPU_calls=0))
    if stopped:raise RuntimeError('Acquisition stopped safely; inspect closed report, do not retry automatically')
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    run()
