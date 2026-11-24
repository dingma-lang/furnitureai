"""Read-only exact-file gallery server, loopback only, no uploads or directory lists."""
import argparse,http.server,urllib.parse
from common11 import *
ALLOWED={'/gallery.html':'text/html; charset=utf-8','/creator-intake.html':'text/html; charset=utf-8','/visual-review.json':'application/json',**{'/images/'+id+'.jpg':'image/jpeg' for id in IDS}}
class Handler(http.server.BaseHTTPRequestHandler):
    def serve(self,head=False):
        path=urllib.parse.urlsplit(self.path).path
        if path=='/':path='/gallery.html'
        if path not in ALLOWED:self.send_error(404);return
        p=OUT/path.lstrip('/')
        if not p.is_file():self.send_error(404);return
        body=p.read_bytes();self.send_response(200);self.send_header('Content-Type',ALLOWED[path]);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'")
        self.end_headers()
        if not head:self.wfile.write(body)
    def do_GET(self):self.serve()
    def do_HEAD(self):self.serve(True)
    def do_POST(self):self.send_error(405,'Read only')
    def log_message(self,*_):pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8782);a=p.parse_args()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    print('Read-only gallery: http://localhost:'+str(a.port)+'/gallery.html',flush=True);server.serve_forever()
