"""Paced certificate-verified HTTPS with measured TLS socket bytes and no photo GET."""
import email.utils,gzip,http.client,io,socket,ssl,time,urllib.parse
from common9 import *
from evidence9 import allowed_page
POLICIES={'https://www.flickr.com/robots.txt','https://www.flickr.com/help/terms/','https://www.flickr.com/help/terms','https://creativecommons.org/licenses/by/2.0/legalcode.en','https://creativecommons.org/licenses/by/2.0/legalcode'}
class ProviderStopped(RuntimeError):pass
class TransferLimit(RuntimeError):pass
def provider(url):
    host=urllib.parse.urlsplit(url).hostname
    return 'flickr.com' if host in ('flickr.com','www.flickr.com') else host
def allowed(row,url,method,policy=False):
    if method not in ('HEAD','GET'):return False
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' or u.username or u.password or u.fragment or u.port:return False
    if policy:return url in POLICIES
    if row is None:return False
    if url==row['official_image_url']:return method=='HEAD'
    return allowed_page(url,row['source_photo_id'].split(':')[1])
def not_before(value):
    instant=dt.datetime.now(dt.timezone.utc)
    try:
        if value and str(value).isdigit():return (instant+dt.timedelta(seconds=int(value))).isoformat()
        parsed=email.utils.parsedate_to_datetime(value) if value else None
        if parsed:return max(instant,parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)).isoformat()
    except (ValueError,TypeError,OverflowError):pass
    return (instant+dt.timedelta(seconds=120)).isoformat()

class TLSStream:
    """MemoryBIO keeps certificate verification in ssl while metering encrypted
    TCP-stream bytes in both directions,including handshakes and read-ahead.
    Physical IP/TCP/DNS overhead and retransmission bytes are not observable here;
    a separate64KiB/request allowance is charged conservatively,not called measured.
    """
    def __init__(self,host,on_wire):
        self.on_wire=on_wire;self.raw=None;self.incoming=ssl.MemoryBIO();self.outgoing=ssl.MemoryBIO()
        self.context=ssl.create_default_context()
        if not self.context.check_hostname or self.context.verify_mode!=ssl.CERT_REQUIRED:raise ValueError('TLS verification cannot be disabled')
        self.ssl=self.context.wrap_bio(self.incoming,self.outgoing,server_hostname=host)
        self.raw=socket.create_connection((host,443),timeout=20);self.raw.settimeout(20)
        try:
            while True:
                try:self.ssl.do_handshake();self.flush();break
                except ssl.SSLWantReadError:self.flush();self.feed()
        except BaseException:self.close();raise
    def flush(self):
        while self.outgoing.pending:
            data=self.outgoing.read(min(16384,self.outgoing.pending));self.on_wire('reserve_send',len(data))
            sent=0
            while sent<len(data):
                n=self.raw.send(data[sent:])
                if not n:raise OSError('TLS socket send interrupted')
                self.on_wire('sent',n);sent+=n
    def feed(self):
        self.on_wire('reserve_receive',16384)
        data=self.raw.recv(16384);self.on_wire('received',len(data))
        if data:self.incoming.write(data)
        else:self.incoming.write_eof()
    def sendall(self,data):
        offset=0
        while offset<len(data):
            try:offset+=self.ssl.write(data[offset:]);self.flush()
            except ssl.SSLWantReadError:self.flush();self.feed()
    def recv(self,size):
        while True:
            try:return self.ssl.read(size)
            except ssl.SSLWantReadError:self.flush();self.feed()
            except ssl.SSLZeroReturnError:return b''
    def makefile(self,*_):
        stream=self
        class Reader(io.RawIOBase):
            def readable(self):return True
            def readinto(self,b):
                data=stream.recv(len(b));b[:len(data)]=data;return len(data)
        return io.BufferedReader(Reader(),buffer_size=8192)
    def close(self):
        if self.raw is not None:self.raw.close();self.raw=None

class Transport:
    def __init__(self,resume=False):
        marker=OUT/'network-begun.json'
        if (OUT/'audit-complete.json').exists():raise PermissionError('Completed audit cannot issue new requests')
        if marker.exists() and not resume:raise FileExistsError('Rights audit already began;no automatic retry')
        if not marker.exists():
            if resume:raise FileNotFoundError('No same-audit journal to resume')
            emit(marker,dict(manifest_sha256=PIN,checked_utc=now(),one_worker=True,automatic_retry=False))
        if read(marker)['manifest_sha256']!=PIN:raise ValueError('Network scope changed')
        self.requests=[read(p) for p in sorted((OUT/'requests').glob('*.json'))]
        for p in sorted((OUT/'request-intents').glob('*.json')):
            r=read(p)
            if r['number'] not in {a['number'] for a in self.requests}:
                self.requests.append(dict(r,conservative_charge=r['reserved_charge'],error='Interrupted transaction;reserved charge retained andnever retried'))
        self.requests.sort(key=lambda r:r['number'])
        self.circuits={p.stem:read(p) for p in (OUT/'circuits').glob('*.json')}
        self.used=sum(r['conservative_charge'] for r in self.requests);self.attempts={read(p)['image_id'] for p in (OUT/'attempts').glob('*.json')}
        self.last=time.monotonic() if self.requests else 0.
        if self.used>CAP:raise TransferLimit('Existing conservative transfer charge exceeds cap')
        self.journal=(OUT/'network-events.jsonl').open('ab' if resume else 'xb')
    def event(self,value):
        value=dict(value,time_utc=now());data=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode();space(len(data)+65536)
        self.journal.write(data);self.journal.flush();os.fsync(self.journal.fileno())
    def begin_candidate(self,row):
        if self.circuits.get('flickr.com'):raise ProviderStopped('Source provider circuit is open')
        if row['image_id'] in self.attempts:raise ValueError('One evidence attempt per exact candidate')
        self.attempts.add(row['image_id']);emit(OUT/'attempts'/(row['image_id'].split(':')[1]+'.json'),dict(image_id=row['image_id'],source_photo_id=row['source_photo_id'],started_utc=now(),attempt=1))
    def stop(self,host,reason,retry=None):
        if host not in self.circuits:
            self.circuits[host]=dict(reason=reason,checked_utc=now(),Retry_After=retry,not_before=not_before(retry),retry_in_this_phase=False)
            emit(OUT/'circuits'/(host+'.json'),self.circuits[host])
    def request(self,row,url,method='HEAD',policy=False):
        if not allowed(row,url,method,policy):raise PermissionError('Unapproved source/identity/method or forbidden photo-body request')
        host=provider(url)
        if host in self.circuits:raise ProviderStopped('Provider stopped;no subsequent requests')
        ident=row['image_id'] if row else None
        if ident and ident not in self.attempts:raise PermissionError('Candidate attempt must begin before requests')
        if any(r['url']==url and r['method']==method for r in self.requests):raise PermissionError('No repeated transaction or retry')
        if method=='GET' and ident and any(r['image_id']==ident and r['method']=='GET' for r in self.requests):raise PermissionError('One source-document GET per candidate')
        reservation=TLS_REQUEST_CAP+TRANSACTION_OVERHEAD_RESERVE
        if self.used+reservation>CAP:raise TransferLimit('32,000,000-byte conservative transfer cap before request')
        space(2*PAGE+65536)
        delay=max(0,3-(time.monotonic()-self.last))
        if delay:time.sleep(delay)
        self.last=time.monotonic();number=len(self.requests)+1
        rec=dict(number=number,image_id=ident,url=url,method=method,policy_document=policy,provider=host,checked_utc=now(),
             sent_TLS_bytes=0,received_TLS_bytes=0,actual_body_bytes=0,protocol_overhead_allowance=TRANSACTION_OVERHEAD_RESERVE,complete=False)
        self.requests.append(rec);self.used+=reservation
        emit(OUT/'request-intents'/f'{number:03}.json',dict(rec,maximum_TLS_bytes=TLS_REQUEST_CAP,reserved_charge=reservation))
        def wire(kind,n):
            actual=rec['sent_TLS_bytes']+rec['received_TLS_bytes']
            if actual+n>TLS_REQUEST_CAP:raise TransferLimit('Per-request encrypted transfer cap before socket operation')
            if kind=='sent':rec['sent_TLS_bytes']+=n
            elif kind=='received':rec['received_TLS_bytes']+=n
            else:return
            self.event(dict(request=number,kind=kind,bytes=n,sent_TLS_bytes=rec['sent_TLS_bytes'],received_TLS_bytes=rec['received_TLS_bytes']))
        stream=None
        try:
            stream=TLSStream(urllib.parse.urlsplit(url).hostname,wire)
            rec.update(TLS_certificate_chain_verified=True,TLS_version=stream.ssl.version(),peer_certificate_SHA256=hashlib.sha256(stream.ssl.getpeercert(binary_form=True)).hexdigest())
            u=urllib.parse.urlsplit(url);target=u.path or '/'
            if u.query:target+='?'+u.query
            request=(method+' '+target+' HTTP/1.1\r\nHost: '+u.hostname+'\r\nUser-Agent: '+UA+'\r\nAccept: text/html,application/xhtml+xml,text/plain\r\nAccept-Encoding: gzip\r\nConnection: close\r\n\r\n').encode('ascii')
            rec['plaintext_request_bytes']=len(request);stream.sendall(request)
            response=http.client.HTTPResponse(stream,method=method);response.begin()
            rec.update(status=response.status,headers=dict(response.getheaders()))
            if response.status in (401,403,429):self.stop(host,'HTTP '+str(response.status),response.getheader('Retry-After'))
            if method=='GET' and response.status==200:
                mime=response.getheader('Content-Type','').split(';')[0].lower();allowed_types=('text/plain',) if url.endswith('/robots.txt') else ('text/html','application/xhtml+xml')
                if mime not in allowed_types:raise ValueError('Unexpected content type;refuse body before parsing/saving')
                lengths=response.headers.get_all('Content-Length') or []
                if len(set(lengths))>1:raise ValueError('Conflicting document Content-Length')
                length=int(lengths[0]) if lengths else None
                if length is not None and not 0<=length<=PAGE:raise TransferLimit('Source document exceeds1MiBbody limit')
                p=OUT/'responses'/f'{number:03}.document';p.parent.mkdir(exist_ok=True)
                with p.open('xb') as f:
                    while rec['actual_body_bytes']<PAGE:
                        chunk=response.read(min(16384,PAGE-rec['actual_body_bytes']))
                        if not chunk:break
                        space(len(chunk)+65536);f.write(chunk);f.flush();os.fsync(f.fileno());rec['actual_body_bytes']+=len(chunk)
                    if length is not None and rec['actual_body_bytes']!=length or length is None and rec['actual_body_bytes']==PAGE:raise TransferLimit('Incomplete/unbounded rights document')
                rec.update(file=str(p.relative_to(OUT)),sha256=sha(p))
            else:rec['body_not_read']=True
            response.close();rec['complete']=True
            self.used-=TLS_REQUEST_CAP-(rec['sent_TLS_bytes']+rec['received_TLS_bytes'])
            rec['conservative_charge']=rec['sent_TLS_bytes']+rec['received_TLS_bytes']+TRANSACTION_OVERHEAD_RESERVE
            return rec
        except BaseException as exc:
            rec['error']=type(exc).__name__+': '+str(exc);rec['conservative_charge']=reservation
            self.stop(host,'Transport/resource/content validation stop;no retry')
            raise
        finally:
            if stream is not None:stream.close()
            emit(OUT/'requests'/f'{number:03}.json',rec)
    def decoded(self,rec):
        p=OUT/rec['file']
        if sha(p)!=rec['sha256']:raise ValueError('Evidence checksum mismatch')
        data=p.read_bytes();headers={k.lower():v for k,v in rec['headers'].items()};encoding=headers.get('content-encoding','').lower()
        if encoding=='gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:data=f.read(DECODED+1)
        elif encoding not in ('','identity'):raise ValueError('Unexpected evidence encoding')
        if len(data)>DECODED:raise ValueError('4MiBdecoded metadata limit')
        return data.decode('utf-8')
    def finish(self):
        self.journal.close()
        value=dict(requests=self.requests,circuits=self.circuits,candidate_evidence_attempts=sorted(self.attempts),
          actual_response_body_bytes=sum(r['actual_body_bytes'] for r in self.requests),measured_TLS_sent_bytes=sum(r['sent_TLS_bytes'] for r in self.requests),
          measured_TLS_received_bytes=sum(r['received_TLS_bytes'] for r in self.requests),conservative_transfer_charge=self.used,transfer_cap=CAP,
          measurement='TLS bytes observed on TCP stream include handshake,HTTPheaders,body and read-ahead. Physical IP/TCP/DNS/retransmission bytes are not measured;64KiB/request allowance and full reservation on interrupted requests are charged separately.',
          photograph_body_GET_requests=0,no_retries=True)
        version=1;path=OUT/'network-ledger.json'
        while path.exists():version+=1;path=OUT/f'network-ledger-v{version}.json'
        emit(path,value);return value
