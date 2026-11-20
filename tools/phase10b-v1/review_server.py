"""Loopback-only annotation review; submissions never activate training targets."""
import argparse
import secrets
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from p10 import *

DECISIONS=('valid_complete_candidate','partial_or_fragmented','taxonomy_conflict','visually_ambiguous','multi_object_or_unit_ambiguous','source_completeness_review')

def validate_submission(value,known):
    if set(value)-{'id','decision','reviewer','evidence','confirmed'}:raise ValueError('Unexpected fields')
    if value.get('id') not in known or value.get('decision') not in DECISIONS:raise ValueError('Unknown annotation/decision')
    if value.get('confirmed') is not True or not isinstance(value.get('reviewer'),str) or not value['reviewer'].strip() or not isinstance(value.get('evidence'),str) or len(value['evidence'].strip())<12:raise ValueError('Explicit confirmation, reviewer and evidence required')
    if len(value['evidence'])>4000 or len(value['reviewer'])>120:raise ValueError('Review text bound')
    return dict(value,human_reviewed=True,training_active=False,source_sha256=known[value['id']]['source_sha256'])

PAGE='''<!doctype html><meta charset="utf-8"><title>Furniture AI annotation recovery</title>
<style>body{font:16px system-ui;max-width:1320px;margin:auto;padding:20px}img{max-width:100%}textarea{width:95%;height:80px}button,input,select{font:inherit;margin:6px;padding:8px}pre{white-space:pre-wrap}</style>
<h1>Phase 10B: annotation recovery</h1><p>Human review is optional for the prepared Sofa/Chair/Bed diagnostics. Decisions here never activate targets automatically.</p>
<button id="prev">Previous</button><button id="next">Next</button><span id="position"></span><select id="jump"></select>
<img id="overlay"><p><a id="original" target="_blank" rel="noopener">Open original photograph</a></p><pre id="details"></pre>
<label>Decision <select id="decision"></select></label><label>Reviewer <input id="reviewer" autocomplete="name"></label><br>
<label>Evidence: describe object unit, visible boundaries, category and unresolved issues.<textarea id="evidence"></textarea></label>
<label><input type="checkbox" id="confirm">I inspected this photograph and annotation and confirm this explanation.</label><br>
<button id="save">Save new versioned review record</button><p id="status"></p>
<script>
let rows=[],pos=0,token='';const $=s=>document.getElementById(s);
function show(){let r=rows[pos];$('position').textContent=(pos+1)+' / '+rows.length;$('overlay').src='/view/'+r.id;$('original').href='/photo/'+r.id;$('jump').value=pos;$('details').textContent=JSON.stringify({category:r.category,native_dataset:r.native_dataset,native_id:r.native_id,reason:r.exclusion_reason,scope:r.native_scope,assistant_triage:r.assistant_triage,human_reviewed:false,training_active:false},null,2);$('confirm').checked=false;$('decision').value='';$('evidence').value='';$('status').textContent='';}
fetch('/api/queue').then(r=>r.json()).then(d=>{rows=d.records;token=d.csrf;for(let [i,r] of rows.entries()){let o=document.createElement('option');o.value=i;o.textContent=r.index+' '+r.category;$('jump').append(o)}for(let v of ['',...d.decisions]){let o=document.createElement('option');o.value=v;o.textContent=v||'Choose a decision';$('decision').append(o)}show()});
$('prev').onclick=()=>{pos=Math.max(0,pos-1);show()};$('next').onclick=()=>{pos=Math.min(rows.length-1,pos+1);show()};$('jump').onchange=()=>{pos=Number($('jump').value);show()};
$('save').onclick=async()=>{let body={id:rows[pos].id,decision:$('decision').value,reviewer:$('reviewer').value,evidence:$('evidence').value,confirmed:$('confirm').checked};let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-CSRF':token},body:JSON.stringify(body)});$('status').textContent=await r.text()};
</script>'''

def make_server(port=8786):
    queue=read(RAW/'review/queue.json')['records'];known={r['id']:r for r in queue};token=secrets.token_urlsafe(32)
    triage_path=RAW/'review/assistant-triage.json'
    if triage_path.exists():
        triage={r['id']:r for r in read(triage_path)['records']}
        queue=[dict(r,assistant_triage=triage[r['id']]['classification']) for r in queue]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def reply(self,status,data,kind='text/plain; charset=utf-8'):
            if isinstance(data,str):data=data.encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'");self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.reply(200,PAGE,'text/html; charset=utf-8')
            if path=='/api/queue':return self.reply(200,json.dumps(dict(records=queue,csrf=token,decisions=DECISIONS)),'application/json')
            prefix,_,ident=path.strip('/').partition('/')
            if ident not in known or prefix not in ('photo','view'):return self.reply(404,'Unknown exact annotation')
            r=known[ident]
            if prefix=='photo':
                p=(ROOT/r['source_photo']).resolve()
                if not p.is_relative_to(ROOT/'data') or sha(p)!=r['source_sha256']:return self.reply(409,'Source integrity mismatch')
            else:p=RAW/'review/views'/(ident+'.jpg')
            self.reply(200,p.read_bytes(),'image/jpeg')
        def do_POST(self):
            if self.path!='/api/review':return self.reply(404,'Unknown endpoint')
            host=self.headers.get('Host','');origin=self.headers.get('Origin','')
            if host not in ('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port)) or origin!='http://'+host or not secrets.compare_digest(self.headers.get('X-Review-CSRF',''),token):return self.reply(403,'Local origin and review session required')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<12000:raise ValueError('Bounded review request required')
                value=validate_submission(json.loads(self.rfile.read(length)),known)
                path=RAW/'review/submissions'/(uuid.uuid4().hex+'.json');save(path,value)
                self.reply(201,'Saved '+path.name+'; remains inactive pending eligibility audit.')
            except (ValueError,TypeError) as exc:self.reply(400,str(exc))
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8786);a=p.parse_args()
    server=make_server(a.port);print(f'Local review: http://127.0.0.1:{server.server_port}; Ctrl-C stops',flush=True);server.serve_forever()
