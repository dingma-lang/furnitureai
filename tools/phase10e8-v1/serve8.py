"""Read-only localhost dashboard;only the exact21candidates/10training photos."""
import argparse,csv,http.server,urllib.parse
from common8q import *

def load_dashboard():
    binding=read(DATA/'input-binding.json')
    for rel,h in binding['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Dashboard source provenance changed:'+rel)
    decisions_path=DATA/'automatic-decisions-v1.json';manifest=read(DATA/'hard-negative-dataset-v1.json')
    if sha(decisions_path)!=manifest['decisions_sha256'] or sha(DATA/'candidate-source-audit.json')!=manifest['source_audit_sha256']:
        raise ValueError('Dashboard decision/source manifest checksum changed')
    decisions=read(decisions_path)['records'];summary=read(REPORT/'qualification-summary.json')
    from collections import Counter
    if dict(Counter(r['decision'] for r in decisions))!=summary['decisions'] or any(r['human_reviewed'] or r['human_confirmed'] or r['training_activated'] for r in decisions):
        raise ValueError('Dashboard counts/human flags/activation inconsistent')
    if {r['candidate_id'] for r in decisions}!={r['candidate_id'] for r in exact_candidates()} or len(decisions)!=21:raise ValueError('Dashboard candidate membership changed')
    _,train,_=context();source={}
    for r in decisions:
        require_train(r['image_id'],train)
        p=(ROOT/r['evidence_plate']).resolve()
        if not p.is_relative_to(DATA/'dashboard/assets') or sha(p)!=r['evidence_plate_sha256']:raise ValueError('Dashboard plate integrity mismatch')
        source[r['image_id']]=(ROOT/train[r['image_id']]['path'],train[r['image_id']]['sha256'])
    return dict(records=decisions,summary=summary),source

def handler_for(payload,sources):
    encoded=json.dumps(payload,allow_nan=False,separators=(',',':')).encode()
    assets={r['sample_id']+'.jpg':ROOT/r['evidence_plate'] for r in payload['records']}
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def do_POST(self):self.send_error(405,'Read-only dashboard')
        def do_PUT(self):self.send_error(405,'Read-only dashboard')
        def do_DELETE(self):self.send_error(405,'Read-only dashboard')
        def do_HEAD(self):self.do_GET(head=True)
        def do_GET(self,head=False):
            host=self.headers.get('Host','').split(':',1)[0].lower()
            if host not in ('127.0.0.1','localhost'):self.send_error(403,'Localhost Host required');return
            parsed=urllib.parse.urlsplit(self.path);path=urllib.parse.unquote(parsed.path)
            if parsed.query or '..' in path.split('/') or '\\' in path:self.send_error(404);return
            value=None;kind=None
            if path=='/api/data':value=encoded;kind='application/json; charset=utf-8'
            elif path in ('/','/index.html','/app.js','/style.css'):
                name='index.html' if path in ('/','/index.html') else path[1:];value=(HERE/name).read_bytes();kind={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name]
            elif path.startswith('/assets/') and path[len('/assets/'):] in assets:value=assets[path[len('/assets/'):]].read_bytes();kind='image/jpeg'
            elif path.startswith('/sources/') and path[len('/sources/'):] in sources:
                p,h=sources[path[len('/sources/'):]]
                if sha(p)!=h:self.send_error(409,'Source image hash changed');return
                value=p.read_bytes();kind='image/jpeg'
            elif path=='/decisions.csv':
                stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['sample_id','candidate_id','image_id','query_index','decision','object_assessment','E5_Chair_score','original_Chair_score','decision_reason','human_reviewed','training_activated'])
                for r in payload['records']:writer.writerow([r[k] for k in ('sample_id','candidate_id','image_id','query_index','decision','object_assessment','student_Chair_score','original_same_query_Chair_score','decision_reason','human_reviewed','training_activated')])
                value=stream.getvalue().encode();kind='text/csv; charset=utf-8'
            if value is None:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(value)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'none'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            if path=='/decisions.csv':self.send_header('Content-Disposition','attachment; filename="phase10e8-decisions.csv"')
            self.end_headers()
            if not head:self.wfile.write(value)
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8778);a=p.parse_args()
    payload,sources=load_dashboard()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',a.port),handler_for(payload,sources))
    print(f'Furniture AI E8 dashboard: http://localhost:{server.server_port} · read only ·21candidates · no human decisions or training',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
