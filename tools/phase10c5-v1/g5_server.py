"""Simple loopback review server; original C.4 service remains unchanged."""
import argparse
import io
import json
import os
import secrets
from threading import Lock
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from g5_review import *
os.environ['CUDA_VISIBLE_DEVICES']=''

def make_server(port=8768,reg=None,export_fn=None,photo_loader=None,assets=None):
    reg=registry() if reg is None else reg;photos={p['photo_key']:p for p in reg['photos']};token=secrets.token_urlsafe(32);lock=Lock();assets=HERE if assets is None else Path(assets)
    export_fn=export_review if export_fn is None else export_fn;photo_loader=source_photo if photo_loader is None else photo_loader
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value,kind='application/json'):
            b=json.dumps(value,allow_nan=False).encode() if isinstance(value,(dict,list)) else value.encode() if isinstance(value,str) else value
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");self.end_headers()
            try:self.wfile.write(b)
            except (BrokenPipeError,ConnectionResetError):pass
        def local(self):return self.headers.get('Host') in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')
        def do_GET(self):
            if not self.local():return self.reply(403,dict(error='Loopback Host required'))
            path=urlparse(self.path).path
            routes={'/':('g5_index.html','text/html; charset=utf-8'),'/app.js':('g5_ui.js','text/javascript; charset=utf-8'),'/style.css':('g5_style.css','text/css; charset=utf-8')}
            if path in routes:
                name,kind=routes[path];return self.reply(200,(assets/name).read_bytes(),kind)
            if path=='/api/registry':return self.reply(200,dict(registry=reg,registry_sha256=digest(reg),token=token,schema=SCHEMA,absence_statement=ACTION_STATEMENT))
            if path.startswith('/api/photo/'):
                p=photos.get(path[len('/api/photo/'):])
                if p is None:return self.reply(404,dict(error='Unknown photo'))
                try:
                    from PIL import Image,ImageOps
                    with Image.open(photo_loader(p)) as im:
                        if im.size!=(p['width'],p['height']) or im.getexif().get(274,1)!=p['orientation']:raise ValueError('Orientation/dimensions changed')
                        data=io.BytesIO();ImageOps.exif_transpose(im).convert('RGB').save(data,format='PNG')
                    return self.reply(200,data.getvalue(),'image/png')
                except (ValueError,PermissionError) as e:return self.reply(409,dict(error=str(e)))
            return self.reply(404,dict(error='No route'))
        def do_POST(self):
            origins=(f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}')
            if not self.local() or self.headers.get('X-Review-Token')!=token or self.headers.get('Origin',origins[0]) not in origins:return self.reply(403,dict(error='Local session required'))
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=3*2**20:return self.reply(413,dict(error='Review body limit'))
                if self.headers.get('Content-Type')!='application/json':return self.reply(415,dict(error='JSON required'))
                data=json.loads(self.rfile.read(n))
                if self.path=='/api/validate':return self.reply(200,{k:v for k,v in validate(data,reg).items() if k not in ('accepted','auxiliary_source_confirmations')})
                if self.path=='/api/export':
                    if not lock.acquire(False):return self.reply(409,dict(error='Export already running'))
                    try:return self.reply(200,export_fn(data,reg))
                    finally:lock.release()
                return self.reply(404,dict(error='No training or inference endpoint'))
            except (ValueError,PermissionError,KeyError,TypeError,FileExistsError) as e:return self.reply(422,dict(error=str(e)))
            except Exception as e:return self.reply(500,dict(error=type(e).__name__+': '+str(e),automatic_retry=False))
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8768);a=p.parse_args();s=make_server(a.port)
    print(f'Fast object review: http://localhost:{s.server_port}/ ; no inference/training',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
