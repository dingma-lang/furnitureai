"""Allowlisted, TLS-metered metadata transport; no redirects, retries or image bodies."""
import gzip,http.client,io,time,urllib.parse
from common10 import *
sys.path.insert(0,str(ROOT/'tools/phase10e9-v1'))
from transport9 import TLSStream
MAX_TLS=2**20*2;MAX_BODY=2**20;ALLOWANCE=65536
HOSTS={'commons.wikimedia.org','www.mediawiki.org','foundation.wikimedia.org','raw.githubusercontent.com','api.smithsonianopenaccess.si.edu','www.si.edu','www.flickrhelp.com','api.flickr.com','creativecommons.org','www.metmuseum.org','collectionapi.metmuseum.org','www.loc.gov','api.wellcomecollection.org','developers.wellcomecollection.org','images-api.nasa.gov','www.nasa.gov','iiif.wellcomecollection.org'}
def allowed(url):
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.hostname not in HOSTS or u.username or u.password or u.port or u.fragment:return False
    q=urllib.parse.parse_qs(u.query)
    if any(k.lower() in ('api_key','access_token','token','key','oauth_token') for k in q):return False
    if u.hostname=='api.flickr.com':return False # No credentials or permission; documentation only elsewhere.
    if u.hostname=='raw.githubusercontent.com':return u.path.startswith(('/flickr/flickr-sdk/','/Smithsonian/OpenAccess/','/wellcomecollection/catalogue-api/')) and u.path.endswith(('.md','.json','.txt'))
    if u.path.lower().endswith(('.jpg','.jpeg','.png','.webp','.gif','.tif','.tiff','.zip','.gz','.mp4')):return False
    if u.hostname=='commons.wikimedia.org':return u.path=='/robots.txt' or u.path.startswith('/wiki/Commons:')
    if u.hostname=='collectionapi.metmuseum.org':return False # Met provider returned429; no alternate interface.
    if u.hostname=='iiif.wellcomecollection.org':
        import re
        return bool(re.fullmatch(r'/image/[A-Za-z0-9_-]+/info\.json',u.path)) and not u.query
    if u.hostname=='api.wellcomecollection.org':return u.path=='/robots.txt' or u.path.startswith(('/catalogue/v2/works','/catalogue/v2/images'))
    if u.hostname=='images-api.nasa.gov':return u.path in ('/robots.txt','/search')
    if u.hostname=='www.loc.gov':return u.path=='/robots.txt' or u.path.startswith(('/apis/','/pictures/collection/highsm/','/pictures/collection/hh/','/collections/carol-m-highsmith/','/collections/historic-american-buildings-landscapes-and-engineering-records/','/resource/','/item/'))
    return True
class Client:
    def __init__(self):
        if (OUT/'discovery-closed.json').exists():raise PermissionError('Audit closed; no further requests')
        self.rows=[read(p) for p in sorted((OUT/'requests').glob('*.json'))];self.last=time.monotonic() if self.rows else 0
        self.blocked={r['host'] for r in self.rows if r.get('stop_provider')}
        intents=[read(p) for p in (OUT/'intents').glob('*.json')]
        self.used=sum(r.get('budget_charge',MAX_TLS+ALLOWANCE) for r in self.rows)
        self.used+=sum(MAX_TLS+ALLOWANCE for i in intents if i['number'] not in {r['number'] for r in self.rows})
        if len(intents)!=len(self.rows):raise PermissionError('Interrupted request; no automatic resumption/retry')
    def fetch(self,url):
        if not allowed(url):raise PermissionError('Unapproved metadata URL or credential/image request')
        host=urllib.parse.urlsplit(url).hostname
        if host in self.blocked:raise PermissionError('Provider circuit open; no alternate endpoint/identity')
        if any(r['url']==url for r in self.rows):raise PermissionError('No repeated requests')
        if self.used+MAX_TLS+ALLOWANCE>CAP:raise RuntimeError('Transfer reserve exceeds cap')
        if allocated()+4*MAX_BODY+8*2**20>DISK:raise RuntimeError('Disk reserve exceeds cap')
        delay=max(0,2-(time.monotonic()-self.last))
        if delay:time.sleep(delay)
        self.last=time.monotonic();num=len(self.rows)+1
        emit(OUT/'intents'/f'{num:03}.json',dict(number=num,url=url,reserved=MAX_TLS+ALLOWANCE,checked_utc=now()))
        r=dict(number=num,url=url,host=host,checked_utc=now(),TLS_sent=0,TLS_received=0,body_bytes=0,complete=False)
        self.rows.append(r);self.used+=MAX_TLS+ALLOWANCE;s=None
        def meter(kind,n):
            if r['TLS_sent']+r['TLS_received']+n>MAX_TLS:raise RuntimeError('Per-request TLS limit')
            if kind=='sent':r['TLS_sent']+=n
            elif kind=='received':r['TLS_received']+=n
            # Persist observed traffic even on a process-level interruption.
            with (OUT/'wire-events.jsonl').open('ab') as f:
                f.write((json.dumps(dict(request=num,event=kind,bytes=n,UTC=now()))+'\n').encode());f.flush();os.fsync(f.fileno())
        try:
            s=TLSStream(host,meter);r.update(TLS_verified=True,TLS_version=s.ssl.version(),certificate_sha256=hashlib.sha256(s.ssl.getpeercert(binary_form=True)).hexdigest())
            u=urllib.parse.urlsplit(url);target=u.path or '/'
            if u.query:target+='?'+u.query
            s.sendall(('GET '+target+' HTTP/1.1\r\nHost: '+host+'\r\nUser-Agent: '+UA+'\r\nAccept: application/json,text/html,text/plain\r\nAccept-Encoding: gzip\r\nConnection: close\r\n\r\n').encode('ascii'))
            response=http.client.HTTPResponse(s);response.begin();r.update(status=response.status)
            r['headers']={k:v for k,v in response.getheaders() if k.lower() in ('content-type','content-length','content-encoding','retry-after','location','last-modified','etag')}
            if response.status in (401,403,429):
                r['stop_provider']=True;self.blocked.add(host);r['Retry_After']=response.getheader('Retry-After')
            if response.status!=200:r['body_not_read']=True;return r,None
            mime=response.getheader('Content-Type','').split(';')[0].lower()
            if mime not in ('application/json','application/ld+json','text/html','text/plain','application/xhtml+xml'):raise ValueError('Non-metadata MIME; refuse body')
            lengths=response.headers.get_all('Content-Length') or []
            if len(set(lengths))>1:raise ValueError('Conflicting length')
            length=int(lengths[0]) if lengths else None
            if length is not None and not 0<=length<=MAX_BODY:raise RuntimeError('Body size cap')
            chunks=[]
            while r['body_bytes']<MAX_BODY:
                b=response.read(min(16384,MAX_BODY-r['body_bytes']))
                if not b:break
                chunks.append(b);r['body_bytes']+=len(b)
            if (length is not None and length!=r['body_bytes']) or (length is None and r['body_bytes']==MAX_BODY):raise RuntimeError('Truncated/unbounded metadata')
            raw=b''.join(chunks);p=OUT/'responses'/f'{num:03}.document';atomic(p,raw);r.update(file=str(p.relative_to(ROOT)),sha256=sha(p))
            if response.getheader('Content-Encoding','').lower()=='gzip':
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:raw=f.read(4*MAX_BODY+1)
            elif response.getheader('Content-Encoding','').lower() not in ('','identity'):raise ValueError('Unsupported compression')
            if len(raw)>4*MAX_BODY:raise ValueError('Decompression cap')
            r['complete']=True
            return r,raw.decode('utf-8')
        except BaseException as e:
            r['error']=type(e).__name__+': '+str(e);r['stop_provider']=True;self.blocked.add(host);raise
        finally:
            if s is not None:s.close()
            charge=r['TLS_sent']+r['TLS_received']+ALLOWANCE if r.get('complete') or 'status' in r else MAX_TLS+ALLOWANCE
            self.used-=MAX_TLS+ALLOWANCE-charge;r['budget_charge']=charge
            emit(OUT/'requests'/f'{num:03}.json',r)
    def summary(self):return dict(requests=len(self.rows),response_body_bytes=sum(r['body_bytes'] for r in self.rows),TLS_bytes=sum(r['TLS_sent']+r['TLS_received'] for r in self.rows),conservative_charge=self.used,cap=CAP,blocked_hosts=sorted(self.blocked),physical_IP_DNS_retransmissions_not_measured=True,photo_requests=0,retries=0)
if __name__=='__main__':
    c=Client();r,text=c.fetch(sys.argv[1]);print(json.dumps(r,indent=2));print((text or '')[:2000]);print(json.dumps(c.summary()))
