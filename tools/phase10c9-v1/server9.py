"""Loopback-only GET dashboard; no review, training, inference or export writes."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from common9 import *

def make_server(port=8772,fixture=None,assets=HERE):
    data=read(OUT/'dashboard-live.json') if fixture is None else fixture;photos={p['photo_key']:p for p in data['photos']};assets=Path(assets)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def reply(self,code,value,kind='application/json; charset=utf-8'):
            b=json.dumps(value,allow_nan=False).encode() if isinstance(value,(dict,list)) else value.encode() if isinstance(value,str) else value
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");self.end_headers()
            try:self.wfile.write(b)
            except (BrokenPipeError,ConnectionResetError):pass
        def local(self):return self.headers.get('Host') in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')
        def do_GET(self):
            if not self.local():return self.reply(403,dict(error='Loopback Host required'))
            path=urlparse(self.path).path
            routes={'/':('dashboard.html','text/html; charset=utf-8'),'/app.js':('dashboard.js','text/javascript; charset=utf-8'),'/style.css':('dashboard.css','text/css; charset=utf-8')}
            if fixture is not None:routes['/fixture.js']=('browser_fixture.js','text/javascript; charset=utf-8')
            if path in routes:
                name,kind=routes[path];return self.reply(200,(assets/name).read_bytes(),kind)
            if path=='/api/data':return self.reply(200,data)
            if path.startswith('/photo/'):
                row=photos.get(path[7:])
                if row is None:return self.reply(404,dict(error='Unknown registered photo key'))
                try:return self.reply(200,source_path(row).read_bytes(),'image/png' if row['path'].endswith('.png') else 'image/jpeg')
                except (OSError,ValueError):return self.reply(409,dict(error='Source identity/hash verification failed'))
            return self.reply(404,dict(error='Unknown read-only route'))
        def do_POST(self):return self.reply(405,dict(error='Read-only diagnostics; no supervision, inference or training endpoint'))
        do_PUT=do_POST;do_DELETE=do_POST
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8772);s=make_server(p.parse_args().port);print('C9 read-only diagnostics http://localhost:'+str(s.server_port),flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
