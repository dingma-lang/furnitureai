"""Loopback-only training-photo annotation server. No model or training routes."""
import argparse
import io
import json
import secrets
import os
from threading import Lock
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from core import *
os.environ['CUDA_VISIBLE_DEVICES']='' # Local process only; annotation/mask checks are CPU-only.

MAX_BODY=2*2**20

def make_server(port=8766,reg=None,export_function=None):
    reg=registry() if reg is None else reg;photos={p['key']:p for p in reg['photos']};token=secrets.token_urlsafe(32);export_lock=Lock()
    if export_function is None:
        from exporter import save_export
        export_function=save_export
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def response(self,status,body,kind='application/json'):
            if isinstance(body,(dict,list)):body=json.dumps(body,allow_nan=False).encode()
            if isinstance(body,str):body=body.encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def allowed(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.allowed():return self.response(403,dict(error='Loopback Host required'))
            path=urlparse(self.path).path
            if path in ('/','/app.js','/style.css'):
                name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[path]
                return self.response(200,(HERE/name).read_bytes(),{'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}[path])
            if path=='/api/registry':return self.response(200,dict(registry=reg,registry_sha256=digest(reg),csrf_token=token,schema=SCHEMA))
            if path.startswith('/api/photo/'):
                key=path[len('/api/photo/'):];p=photos.get(key)
                if p is None:return self.response(404,dict(error='No eligible training photo'))
                try:
                    from PIL import Image,ImageOps
                    with Image.open(photo_path(p)) as im:
                        if im.size!=(p['width'],p['height']) or im.getexif().get(274,1)!=p['orientation']:raise ValueError('Source frame changed')
                        out=ImageOps.exif_transpose(im).convert('RGB');data=io.BytesIO();out.save(data,format='PNG')
                    return self.response(200,data.getvalue(),'image/png')
                except (ValueError,PermissionError) as e:return self.response(409,dict(error=str(e)))
            return self.response(404,dict(error='No such route'))
        def do_POST(self):
            origin=self.headers.get('Origin');allowed_origins=(f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}')
            if not self.allowed() or self.headers.get('X-Annotation-Token')!=token or (origin is not None and origin not in allowed_origins):return self.response(403,dict(error='Local annotation session required'))
            if self.headers.get('Content-Type')!='application/json':return self.response(415,dict(error='JSON required'))
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=MAX_BODY:return self.response(413,dict(error='Annotation body exceeds limit'))
                payload=json.loads(self.rfile.read(n))
                if self.path=='/api/validate':return self.response(200,{k:v for k,v in validate(payload,reg).items() if k!='accepted'})
                if self.path=='/api/export':
                    if not export_lock.acquire(blocking=False):return self.response(409,dict(error='An export is already running; no automatic retry'))
                    try:return self.response(200,export_function(payload,reg))
                    finally:export_lock.release()
                return self.response(404,dict(error='No model/training endpoint'))
            except (ValueError,KeyError,TypeError,PermissionError,FileExistsError) as e:return self.response(422,dict(error=str(e)))
            except Exception as e:return self.response(500,dict(error=type(e).__name__+': '+str(e),automatic_retry=False))
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);a=parser.parse_args()
    server=make_server(a.port);print(f'Chair-negative annotation ready: http://localhost:{server.server_port}/ ;0 regions preapproved',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
