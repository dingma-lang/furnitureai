"""Read-only local results dashboard with optional append-only exclusion notes."""
import argparse,io,json,secrets
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from PIL import Image,ImageDraw
from common6 import *
import uuid

def validate_correction(data,known):
    if data.get('candidate_id') not in known or data.get('decision') not in ('exclude','needs_recheck','note') or not isinstance(data.get('reviewer'),str) or not data['reviewer'].strip() or not isinstance(data.get('note'),str) or not 8<=len(data['note'].strip())<=2000:raise ValueError('Known candidate, explicit reviewer, decision and concrete note required')
    return True

def correction(data,known):
    validate_correction(data,known)
    folder=RAW/'corrections';folder.mkdir(exist_ok=True);ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10]
    r=dict(data,correction_id=ident,recorded_utc=datetime.now(timezone.utc).isoformat(),training_approval=False,source_evidence_sha256=sha(RAW/'source-negative-evidence.json'))
    save(folder/(ident+'.json'),r);return dict(correction_id=ident,training_activated=False)

def load():
    raw=read(RAW/'automatic-evidence.json')['results'];brief=[]
    for r in raw:
        brief.append({k:r[k] for k in ('candidate_id','native_label','category','evidence_level','image_id','identity_group','explanation','eligibility_gate_failure')}|dict(source_dataset=r['annotation']['dataset'],evidence_summary=dict(native_category_id=r['annotation']['native_category_id'],source_annotation_key=r['annotation']['annotation_key'],source_file=r['annotation']['annotation_source'],source_file_sha256=r['annotation']['annotation_source_sha256'],original_box=r['annotation']['box'],proposed_box=r['proposal']['box'] if r['proposal'] else None,photo_sha256=r['sha256'],source_polygon_interior=bool(r['proposal']),human_reviewed=False,visual_artifact=r['automatic_observation']['plate'],visual_artifact_sha256=r['automatic_observation']['plate_sha256'],licence_url=r['rights']['license_url'],creator=r['rights']['creator'],source_url=r['rights']['source_url'])))
    return raw,dict(summary=read(REPORT/'summary.json'),masks=read(REPORT/'mask-summary.json'),results=brief)

def make_server(port=8770,fixture=None,assets=None,correction_fn=None):
    raw,brief=load() if fixture is None else fixture;known={r['candidate_id']:r for r in raw};token=secrets.token_urlsafe(32);assets=HERE if assets is None else Path(assets);fn=correction if correction_fn is None else correction_fn
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def reply(self,code,value,kind='application/json'):
            b=json.dumps(value,allow_nan=False).encode() if isinstance(value,(dict,list)) else value.encode() if isinstance(value,str) else value
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");self.end_headers()
            try:self.wfile.write(b)
            except (BrokenPipeError,ConnectionResetError):pass
        def local(self):return self.headers.get('Host') in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')
        def do_GET(self):
            if not self.local():return self.reply(403,dict(error='Loopback Host required'))
            path=urlparse(self.path).path;routes={'/':('dashboard.html','text/html; charset=utf-8'),'/app.js':('dashboard.js','text/javascript; charset=utf-8'),'/style.css':('dashboard.css','text/css')}
            if path in routes:
                name,kind=routes[path];return self.reply(200,(assets/name).read_bytes(),kind)
            if path=='/api/results':return self.reply(200,dict(brief,token=token))
            if path.startswith('/preview/') and path.endswith('.jpg'):
                ident=path[len('/preview/'):-4];r=known.get(ident)
                if r is None:return self.reply(404,dict(error='Unknown candidate'))
                try:
                    p=ROOT/r['path']
                    if sha(p)!=r['sha256']:raise ValueError('Photograph changed')
                    with Image.open(p) as opened:
                        if opened.size!=(r['width'],r['height']) or opened.getexif().get(274,1)!=1:raise ValueError('Source registration changed')
                        im=opened.convert('RGB')
                    d=ImageDraw.Draw(im)
                    for q in r['protected']:
                        b=q['box'];d.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#db3636',width=2)
                    b=r['annotation']['box'];d.rectangle([b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height],outline='#215acb',width=2)
                    if r['proposal']:d.rectangle(r['proposal']['box_pixels'],outline='#008342' if r['evidence_level']=='A' else '#e89500',width=3)
                    im.thumbnail((900,700));buf=io.BytesIO();im.save(buf,format='JPEG',quality=88);return self.reply(200,buf.getvalue(),'image/jpeg')
                except (ValueError,OSError) as e:return self.reply(409,dict(error=str(e)))
            return self.reply(404,dict(error='No inference or training route'))
        def do_POST(self):
            origins=(f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}')
            if not self.local() or self.headers.get('X-Review-Token')!=token or self.headers.get('Origin',origins[0]) not in origins:return self.reply(403,dict(error='Local session required'))
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192:return self.reply(413,dict(error='Correction limit'))
                if self.path!='/api/correction' or self.headers.get('Content-Type')!='application/json':return self.reply(404,dict(error='No training endpoint'))
                return self.reply(200,fn(json.loads(self.rfile.read(n)),known))
            except (ValueError,KeyError,TypeError) as e:return self.reply(422,dict(error=str(e)))
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770);args=p.parse_args();s=make_server(args.port);print('Automatic results http://localhost:'+str(s.server_port)+' ;training disabled',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
