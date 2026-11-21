"""CPU-only installed Windows Chrome; synthetic dashboard and correction handler."""
import html,json,re,subprocess,threading,time,uuid
from common6 import *
from server6 import make_server,validate_correction

def run():
    folder=RAW/('browser-fixture-'+uuid.uuid4().hex[:8]);folder.mkdir(exist_ok=False)
    for name in ('dashboard.html','dashboard.css'):(folder/name).write_bytes((HERE/name).read_bytes())
    (folder/'dashboard.js').write_bytes((HERE/'dashboard.js').read_bytes()+b'\n'+(HERE/'browser_fixture.js').read_bytes())
    from PIL import Image
    Image.new('RGB',(100,100),'#cde5e8').save(folder/'synthetic.png');raw=[];results=[];cats=[]
    for i,(level,category) in enumerate((('A','people'),('B','plants_flowers'),('C','animals')),1):
        ident='C'+str(i).zfill(3);raw.append(dict(candidate_id=ident,path=str((folder/'synthetic.png').relative_to(ROOT)),sha256=sha(folder/'synthetic.png'),width=100,height=100,protected=[],annotation=dict(box=[.1,.1,.9,.9]),proposal=dict(box_pixels=[20,20,60,60]),evidence_level=level))
        results.append(dict(candidate_id=ident,native_label='SYNTHETIC '+category,category=category,evidence_level=level,image_id='synthetic-photo'+str(i),identity_group='fixture-group'+str(i),explanation='Synthetic visual fixture only',eligibility_gate_failure=None,source_dataset='fixture',evidence_summary=dict(human_reviewed=False)))
        cats.append(dict(category=category,total=1,source_verified=int(level=='A'),automatic=int(level=='B'),uncertain=int(level=='C'),known_groups=1))
    summary=dict(photographs_inspected=3,objects_inspected=3,source_verified_regions=1,automatically_screened_candidates=1,uncertain_or_rejected=1,source_verified_groups=1,by_category=cats)
    def corrected(data,known):validate_correction(data,known);return dict(correction_id='SYNTHETIC-NO-REAL-CORRECTION',training_activated=False)
    srv=make_server(8767,fixture=(raw,dict(summary=summary,masks=dict(changed_mask_terms=352,region_interior_probes_added=41),results=results)),assets=folder,correction_fn=corrected);thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    try:
        chrome=Path('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe')
        if not chrome.exists():raise RuntimeError('Existing Chrome unavailable; no download allowed')
        temp=subprocess.check_output(['/mnt/c/Windows/System32/cmd.exe','/c','echo','%TEMP%'],stderr=subprocess.DEVNULL,text=True).strip().splitlines()[-1];profile=temp.rstrip('\\')+r'\furniture-ai-phase10c6-browser-v1'
        for _ in range(8):
            p=subprocess.run(['/mnt/c/Windows/System32/curl.exe','--noproxy','*','--max-time','2','-sS','http://127.0.0.1:8767/'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=4)
            if p.returncode==0:break
            time.sleep(1)
        else:raise RuntimeError('Windows loopback unavailable')
        cmd=[str(chrome),'--headless','--disable-gpu','--no-proxy-server','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--metrics-recording-only','--user-data-dir='+profile,'--virtual-time-budget=15000','--dump-dom','http://127.0.0.1:8767/']
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);(folder/'dom.html').write_bytes(p.stdout);(folder/'browser.log').write_bytes(p.stderr)
    finally:srv.shutdown();srv.server_close();thread.join()
    m=re.search(r'<pre id="browser-test-result">(.*?)</pre>',p.stdout.decode(errors='replace'),re.S)
    if not m:raise RuntimeError('No completed browser result '+str(folder))
    status=json.loads(html.unescape(m[1]));save(folder/'result.json',status)
    if status['status']!='PASS':raise RuntimeError(str(status))
    save(REPORT/('browser-tests-'+uuid.uuid4().hex[:8]+'.json'),status);print('BROWSER PASS',len(status['checks']),'checks;0real corrections')

if __name__=='__main__':run()
