"""Exact-file loopback gallery and unchanged E15 intake; no uploads or model calls."""
import argparse
import http.server
from common16 import *

E15=ROOT/'tools/phase10e15-v1'


def routes():
    files={'/':(OUT/'gallery.html','text/html'),'/gallery.html':(OUT/'gallery.html','text/html'),
        '/intake.html':(E15/'intake.html','text/html'),'/intake.js':(E15/'intake.js','text/javascript'),
        '/smoke.html':(HERE/'smoke.html','text/html'),'/smoke.js':(HERE/'smoke.js','text/javascript')}
    for ident in IDS:
        n=ident.split(':')[1]
        files['/images/'+n+'.jpg']=(OUT/'images'/(n+'.jpg'),'image/jpeg')
        files['/previews/'+n+'.jpg']=(OUT/'previews'/(n+'.jpg'),'image/jpeg')
    return files


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
            self.send_error(403);return
        if self.path=='/smoke-data.json':
            if not getattr(self.server,'smoke_enabled',False):self.send_error(404);return
            records=read(OUT/'acquisition-closed.json')['records']
            body=json.dumps([dict(source_id=r['source_id'],sha256=r['sha256'],dimensions=r['quality']['display_dimensions'],
                creator=r['rights']['creator_credit'],source=r['rights']['source_page']) for r in records]).encode();mime='application/json'
        else:
            item=routes().get(self.path)
            if not item or self.path.startswith('/smoke') and not getattr(self.server,'smoke_enabled',False):
                self.send_error(404);return
            p,mime=item
            if not p.is_file():self.send_error(404);return
            body=p.read_bytes()
        self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8' if not mime.startswith('image/') else mime)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
        smoke=self.path.startswith('/smoke')
        self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'unsafe-inline'; script-src 'self'; img-src 'self' blob:; "+
                         ("frame-src 'self'; connect-src 'self'; " if smoke else "connect-src 'none'; ")+"form-action 'none'; base-uri 'none'")
        self.end_headers();self.wfile.write(body)
        if getattr(self.server,'smoke_enabled',False):
            with (OUT/'browser-local-http.jsonl').open('ab') as f:
                f.write((json.dumps(dict(path=self.path,body_bytes=len(body),utc=now()))+'\n').encode())
    def do_POST(self):self.send_error(405,'Read only; no uploads')
    def log_message(self,*args):pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8784);a=p.parse_args()
    with http.server.ThreadingHTTPServer(('127.0.0.1',a.port),Handler) as server:
        print(f'Read-only pilot gallery: http://localhost:{a.port}/gallery.html',flush=True);server.serve_forever()
