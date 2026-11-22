"""Existing Windows Chrome,CPU-only browser checks on readonly report metadata."""
import argparse,html,re,subprocess,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from common16 import *


def server(port=8778,browser=False):
    class H(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def reply(self,code,body,kind='application/json; charset=utf-8'):
            if isinstance(body,dict):body=json.dumps(body).encode()
            elif isinstance(body,str):body=body.encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('Content-Security-Policy',"default-src 'self';script-src 'self';style-src 'self' 'unsafe-inline';connect-src 'self';frame-ancestors 'none'");self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.headers.get('Host') not in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}'):return self.reply(403,dict(error='Loopback only'))
            p=urlparse(self.path).path
            if p=='/data':return self.reply(200,dict(audit=read(REPORT16/'source-group-audit.json'),comparison=read(REPORT16/'comparison.json')))
            if p=='/':
                text=(HERE16/'viewer.html').read_text()
                if browser:text=text.replace('</html>','<script defer src="/viewercheck.js"></script></html>')
                return self.reply(200,text,'text/html; charset=utf-8')
            if p=='/viewer.js' or (browser and p=='/viewercheck.js'):return self.reply(200,(HERE16/p[1:]).read_bytes(),'text/javascript; charset=utf-8')
            return self.reply(404,dict(error='No data/model file endpoints'))
        def do_POST(self):return self.reply(405,dict(error='Read-only diagnostics,no training,inference or human labels'))
        do_PUT=do_POST;do_PATCH=do_POST;do_DELETE=do_POST
    return ThreadingHTTPServer(('127.0.0.1',port),H)


def test_browser():
    from browser14 import command,windows_temp
    folder=DATA16/('browser-'+uuid.uuid4().hex[:8]);folder.mkdir();s=server(browser=True);thread=threading.Thread(target=s.serve_forever,daemon=True);thread.start()
    try:
        for _ in range(12):
            probe=subprocess.run(['/mnt/c/Windows/System32/curl.exe','--noproxy','*','--max-time','2','-sS','http://127.0.0.1:8778/'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=4)
            if not probe.returncode:break
            time.sleep(1)
        else:raise RuntimeError('Windows-to-WSL localhost unavailable;no external fallback')
        profile=windows_temp()+'\\furniture-ai-c16-'+uuid.uuid4().hex[:8];r=subprocess.run(command('http://127.0.0.1:8778/',profile),capture_output=True,timeout=60);(folder/'browser-dom.html').write_bytes(r.stdout);(folder/'browser.log').write_bytes(r.stderr)
    finally:s.shutdown();s.server_close();thread.join()
    match=re.search(r'<pre id="browser-result">(.*?)</pre>',r.stdout.decode(errors='replace'),re.S)
    if not match:raise RuntimeError('Browser result unavailable;no fallback or install')
    result=json.loads(html.unescape(match[1]));emit16(REPORT16/'browser-tests.json',result)
    if result['status']!='PASS':raise RuntimeError('Browser checks failed')
    print('CPU BROWSER PASS',len(result['checks']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');a=parser.parse_args()
    if a.serve:
        s=server();print('Read-only C16 report viewer http://localhost:8778/',flush=True)
        try:s.serve_forever()
        except KeyboardInterrupt:pass
        finally:s.server_close()
    else:test_browser()
