"""Metered verified-TLS acquisition: exact sources, one attempt, no redirects."""
import html
import http.client
import re
import time
import urllib.parse

from common16 import *
sys.path.insert(0,str(ROOT/'tools/phase10e9-v1'))
from transport9 import TLSStream

OVERHEAD=65536
LICENSES=('https://creativecommons.org/publicdomain/zero/1.0/',
          'https://creativecommons.org/licenses/by/4.0/',
          'https://creativecommons.org/licenses/by/2.0/')


def metadata_url():return read(ROOT/'reports/phase10e15-v1/commons-metadata-access.json')['source_url']
def allowed(url,kind,method):
    if method not in ('HEAD','GET'):return False
    if kind=='metadata':return url==metadata_url() and method=='GET'
    if kind=='licence':return url in LICENSES and method=='GET'
    if kind=='image':return url in {r['source_object_url'] for r in rows()}
    return False


def rights_validate(document,rs):
    pages=document.get('query',{}).get('pages',{})
    if {'commons-page:'+str(v.get('pageid')) for v in pages.values()}!=set(IDS):
        raise ValueError('Changed source IDs or unavailable metadata')
    verified=[]
    for old in rs:
        p=pages[old['source_id'].split(':')[1]];i=p['imageinfo'][0];m=i['extmetadata']
        if p['title']!=old['native_file_title'] or any(i[k]!=old[v] for k,v in (
            ('url','source_object_url'),('size','expected_bytes'),('width','width'),('height','height'),('sha1','source_sha1'))):
            raise ValueError('Changed object identity, URL, dimensions, size or SHA1')
        if html.unescape(re.sub('<[^>]+>','',m['Artist']['value']))!=old['creator_credit']:
            raise ValueError('Changed attribution')
        if 'own work' not in html.unescape(re.sub('<[^>]+>','',m['Credit']['value'])).lower():
            raise ValueError('Own-work evidence absent or changed')
        normalize=lambda s:s.replace('http://','https://').rstrip('/')
        if m['LicenseShortName']['value']!=old['license'] or normalize(m['LicenseUrl']['value'])!=normalize(old['license_url']):
            raise ValueError('Changed photograph-specific licence')
        if m.get('Restrictions',{}).get('value'):raise ValueError('Source restrictions require review')
        verified.append(dict(source_id=old['source_id'],creator_credit=old['creator_credit'],license=old['license'],
            license_url=old['license_url'],source_page=old['source_page'],source_object_url=old['source_object_url'],
            status='current_file_specific_copyright_declaration_supported',checked_utc=now(),
            ownership_privacy_property_releases_not_inferred=True,training_active=False))
    return verified


class Client:
    def __init__(self):
        if (OUT/'acquisition-closed.json').exists():raise PermissionError('Closed acquisition; no new requests')
        self.receipts=[read(p) for p in sorted((OUT/'requests').glob('*.json'))]
        if len(list((OUT/'intents').glob('*.json')))!=len(self.receipts):
            raise PermissionError('Interrupted transaction: retain reservation; no automatic retry')
        self.used=sum(r['budget_charge'] for r in self.receipts);self.last=0.
        self.blocked={r['host'] for r in self.receipts if r.get('stop_provider')}

    def fetch(self,url,kind,method='GET'):
        if not allowed(url,kind,method):raise PermissionError('Unapproved source/method')
        u=urllib.parse.urlsplit(url);host=u.hostname
        if host in self.blocked:raise PermissionError('Provider stopped; no bypass or alternate')
        if any(r['url']==url and r['method']==method for r in self.receipts):
            raise PermissionError('One attempt only; successful or failed transactions are never repeated')
        expected=next((r['expected_bytes'] for r in rows() if r['source_object_url']==url),None)
        body_cap=expected if kind=='image' and method=='GET' else 1048576
        tls_cap=(body_cap if method=='GET' else 0)+196608;reservation=tls_cap+OVERHEAD
        if self.used+reservation>CAP:raise RuntimeError('60,000,000-byte cumulative transfer reservation exceeded')
        space(2*body_cap+262144)
        delay=max(0,1.5-(time.monotonic()-self.last))
        if delay:time.sleep(delay)
        self.last=time.monotonic();number=len(self.receipts)+1
        r=dict(number=number,url=url,method=method,kind=kind,host=host,time_utc=now(),
               TLS_sent=0,TLS_received=0,body_bytes=0,complete=False,attempt=1,reserved_charge=reservation)
        emit(OUT/'intents'/f'{number:03}.json',r)
        self.receipts.append(r);self.used+=reservation
        data=bytearray();stream=None
        def wire(event,n):
            if r['TLS_sent']+r['TLS_received']+n>tls_cap:raise RuntimeError('TLS transfer limit before socket operation')
            if event not in ('sent','received'):return
            r['TLS_sent' if event=='sent' else 'TLS_received']+=n
            space(65536)
            with (OUT/'wire-events.jsonl').open('ab') as f:
                f.write((json.dumps(dict(request=number,event=event,bytes=n,utc=now()))+'\n').encode());f.flush();os.fsync(f.fileno())
        try:
            stream=TLSStream(host,wire)
            r.update(TLS_verified=True,TLS_version=stream.ssl.version(),certificate_SHA256=hashlib.sha256(stream.ssl.getpeercert(binary_form=True)).hexdigest())
            target=u.path+('?' +u.query if u.query else '')
            stream.sendall((method+' '+target+' HTTP/1.1\r\nHost: '+host+'\r\nUser-Agent: '+UA+
                '\r\nAccept: '+('image/jpeg' if kind=='image' else 'application/json,text/html')+
                '\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n').encode('ascii'))
            response=http.client.HTTPResponse(stream,method=method);response.begin()
            r.update(status=response.status,headers={k:v for k,v in response.getheaders() if k.lower() in
                ('content-type','content-length','content-encoding','location','retry-after','etag','last-modified')})
            if response.status!=200:raise RuntimeError('HTTP '+str(response.status)+'; no redirect following or retries')
            mime=response.getheader('Content-Type','').split(';')[0].lower()
            permitted=('image/jpeg',) if kind=='image' else ('application/json',) if kind=='metadata' else ('text/html','application/xhtml+xml')
            if mime not in permitted:raise ValueError('Unexpected MIME; refuse body')
            if response.getheader('Content-Encoding','identity').lower() not in ('','identity'):raise ValueError('Unexpected content encoding')
            lengths=response.headers.get_all('Content-Length') or []
            if len(set(lengths))>1:raise ValueError('Conflicting length headers')
            length=int(lengths[0]) if lengths else None
            if kind=='image' and length!=expected:raise ValueError('Remote image length changed or missing')
            if method=='GET':
                if length is not None and not 0<=length<=body_cap:raise RuntimeError('Object size cap before body')
                while len(data)<body_cap:
                    chunk=response.read(min(16384,body_cap-len(data)))
                    if not chunk:break
                    data.extend(chunk);r['body_bytes']=len(data)
                if length is not None and len(data)!=length or length is None and len(data)==body_cap:
                    raise ValueError('Truncated/unbounded source body')
            r.update(complete=True,body_SHA256=hashlib.sha256(data).hexdigest())
            return r,bytes(data)
        except BaseException as exc:
            r.update(error=type(exc).__name__+': '+str(exc),stop_provider=True);self.blocked.add(host)
            if data:save(OUT/'quarantine'/f'{number:03}-{kind}.partial',bytes(data))
            raise
        finally:
            if stream is not None:stream.close()
            charge=r['TLS_sent']+r['TLS_received']+OVERHEAD if 'status' in r else reservation
            self.used-=reservation-charge;r['budget_charge']=charge
            emit(OUT/'requests'/f'{number:03}.json',r)

    def summary(self):
        return dict(requests=len(self.receipts),response_body_bytes=sum(r['body_bytes'] for r in self.receipts),
            image_body_bytes=sum(r['body_bytes'] for r in self.receipts if r['kind']=='image'),
            observed_bidirectional_TLS_bytes=sum(r['TLS_sent']+r['TLS_received'] for r in self.receipts),
            conservative_transfer_charge=self.used,cap=CAP,actual_max_attempts=1,redirects_followed=0,retries=0,
            physical_IP_DNS_TCP_retransmissions_not_directly_measured=True,
            overhead_allowance_per_request=OVERHEAD)


def run():
    rs=rows()
    if OUT.exists():raise FileExistsError('Destination already exists; do not modify')
    OUT.mkdir(parents=True)
    emit(OUT/'authorization.json',dict(exact_ids=IDS,manifest_sha256=PIN,
        user_attachment_sha256=sha('/home/tristary/.codex/attachments/38164d67-b45f-4360-af72-2eb8d9333a72/Pasted text.txt'),
        approval='Explicit Phase10E.16 user attachment supersedes proposal acquisition_authorized=false',
        transfer_cap=CAP,storage_cap=DISK,image_cap=IMAGE_CAP,max_attempts=2,automatic_retries=False))
    client=Client();acquired=[];stop=None
    try:
        receipt,body=client.fetch(metadata_url(),'metadata')
        save(OUT/'evidence/current-commons-imageinfo.json',body)
        rights=rights_validate(json.loads(body),rs)
        for url in LICENSES:
            receipt,body=client.fetch(url,'licence')
            if (b'Creative Commons' not in body or b'creativecommons.org' not in body):raise ValueError('Unexpected licence document identity')
            name=('cc0' if '/zero/' in url else 'by4' if '/4.0/' in url else 'by2')+'.html'
            save(OUT/'evidence'/name,body)
        emit(OUT/'rights-ledger.json',dict(records=rights,metadata_SHA256=sha(OUT/'evidence/current-commons-imageinfo.json'),
             licence_documents={u:sha(OUT/'evidence'/n) for u,n in zip(LICENSES,('cc0.html','by4.html','by2.html'))}))
        # All three source checks must pass before any photograph GET.
        heads=[]
        for row in rs:
            receipt,_=client.fetch(row['source_object_url'],'image','HEAD');heads.append(receipt)
        emit(OUT/'source-preflight.json',dict(heads=heads,rights=rights))
        from screen16 import inspect_image
        for row in rs:
            receipt,body=client.fetch(row['source_object_url'],'image')
            quality=inspect_image(body,row)
            ident=row['source_id'].split(':')[1];path=OUT/'images'/(ident+'.jpg');save(path,body)
            record=dict(source_id=row['source_id'],file=str(path.relative_to(ROOT)),sha256=sha(path),sha1=sha(path,'sha1'),
                bytes=len(body),quality=quality,source=row,rights=next(r for r in rights if r['source_id']==row['source_id']),
                request=receipt['number'],training_active=False,evaluation_active=False,annotations=[])
            emit(OUT/'integrity'/(ident+'.json'),record);acquired.append(record)
            print('Acquired',row['native_file_title'],len(body),quality['display_dimensions'],flush=True)
    except BaseException as exc:
        stop=dict(error=type(exc).__name__+': '+str(exc),utc=now());print('STOP',stop,flush=True)
    finally:
        emit(OUT/'acquisition-closed.json',dict(records=acquired,downloaded_count=len(acquired),stop=stop,
            network=client.summary(),allocated_bytes_at_close=allocated(),new_labels=0,training_updates=0,model_calls=0))
    if stop:raise RuntimeError('Stopped safely; no automatic retry/substitution')


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    run()
