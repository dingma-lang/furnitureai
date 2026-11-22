"""Loopback-only GET diagnostic server;no train/review/export endpoints."""
import argparse,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from common14 import *
from dashboard14 import ViewStore


def make_server(port=8774,fixture=None,assets=HERE14):
    store=ViewStore() if fixture is None else fixture;data=store.data;photos=store.photos;assets=Path(assets)
    if fixture is None:
        _,train,dev=verify_inputs13();canonical={r['image_id']:r for r in train}
        for photo in photos.values():
            row=canonical.get(photo['image_id'])
            if row is None or photo['role']!='optimization' or any(photo[k]!=row[k] for k in ('path','sha256','identity_group','width','height')):raise PermissionError('Dashboard includes non-optimization or changed photograph')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,value,kind='application/json; charset=utf-8'):
            payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode() if isinstance(value,(dict,list)) else value.encode() if isinstance(value,str) else value
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self';img-src 'self';script-src 'self';style-src 'self';connect-src 'self';frame-ancestors 'none';base-uri 'none'");self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            if self.headers.get('Host') not in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}'):return self.reply(403,dict(error='Loopback Host required'))
            path=urlparse(self.path).path;routes={'/':('dashboard.html','text/html; charset=utf-8'),'/app.js':('dashboard.js','text/javascript; charset=utf-8'),'/style.css':('dashboard.css','text/css; charset=utf-8')}
            if fixture is not None:routes['/fixture.js']=('browser_fixture.js','text/javascript; charset=utf-8')
            if path in routes:
                name,kind=routes[path];return self.reply(200,(assets/name).read_bytes(),kind)
            if path=='/api/data':return self.reply(200,data)
            parts=path.strip('/').split('/')
            try:
                if len(parts)==2 and parts[0]=='photo':
                    photo=photos[parts[1]];p=(ROOT/photo['path']).resolve()
                    if not p.is_relative_to(ROOT/'data') or sha(p)!=photo['sha256']:raise ValueError('Training photograph hash/path changed')
                    return self.reply(200,p.read_bytes(),'image/png' if p.suffix=='.png' else 'image/jpeg')
                if len(parts)==4 and parts[:2]==['api','view']:return self.reply(200,store.view(parts[2],parts[3]))
                if len(parts)==5 and parts[:2]==['api','detail'] and parts[4].isdigit():return self.reply(200,store.detail(parts[2],parts[3],int(parts[4])))
            except (KeyError,IndexError):return self.reply(404,dict(error='Unknown optimization photo/state/query;no inference fallback'))
            except (OSError,ValueError,PermissionError):return self.reply(409,dict(error='Source identity/hash/partition integrity failure'))
            return self.reply(404,dict(error='Unknown read-only route'))
        def do_POST(self):return self.reply(405,dict(error='Read-only;no review,supervision,training,inference or export writes'))
        do_PUT=do_POST;do_DELETE=do_POST;do_PATCH=do_POST
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8774);s=make_server(p.parse_args().port);print('C14 read-only training-query dashboard http://localhost:'+str(s.server_port),flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
