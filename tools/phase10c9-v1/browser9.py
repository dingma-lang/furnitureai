"""Installed Windows Chrome; synthetic browser tests, zero downloads or inference."""
import html,re,subprocess,threading,time,uuid
from PIL import Image
from common9 import *
from server9 import make_server

def fixture(folder):
    p=folder/'synthetic.png';Image.new('RGB',(200,150),'#b9d7e5').save(p);photos=[]
    for role,ident,key in (('internal_development_diagnostic_only','SYNTHETIC-DEV','synthetic-dev'),('optimization_training_only','SYNTHETIC-TRAIN','synthetic-train')):
        models={}
        for model in MODELS:
            models[model]=[dict(box=[.1,.1,.4,.5],score=s,label='chair',query_index=i,prediction_id=model+':'+ident+':'+str(i),status=status,geometry=dict(size_bin='5-20%')) for i,(s,status) in enumerate(((.9,'fp'),(.85,'unknown_unmatched'),(.6,'fp'),(.2,'fp')),1)]
        photos.append(dict(image_id=ident,photo_key=key,role=role,width=200,height=150,path=str(p.relative_to(ROOT)),sha256=sha(p),models=models,annotations=[dict(label='chair',ignore=False,box=[.5,.5,.8,.8])],native=[],coverage_warning='Synthetic unverified diagnostic, never a training label'))
    summary={model:dict(thresholds={str(t):dict(chair=dict(predictions=4,tp=0,scored_fp=3,unknown_unmatched=1,recall=0)) for t in THRESHOLDS}) for model in MODELS}
    obs=dict(sample_id='SYNTHETIC-OBSERVATION',scope='development_diagnostic_only',taxonomy_primary='F',evidence_tier='synthetic_fixture',image_id='SYNTHETIC-DEV',query_index=1,observation='Synthetic unresolved fixture; no real reviewer decision',human_reviewed=False)
    return dict(photos=photos,summary=summary,observations=[obs],training_summary=dict(existing_source_regions=0,saved_query_photos=1,thresholds_top_Chair_query_counts={'0.5':1},unverified_unprotected_queries=1,new_spatial_candidates_pending_visual=0))

def run():
    folder=OUT/('browser-fixture-'+uuid.uuid4().hex[:8]);folder.mkdir()
    for name in ('dashboard.css','dashboard.js','browser_fixture.js'):(folder/name).write_bytes((HERE/name).read_bytes())
    (folder/'dashboard.html').write_text((HERE/'dashboard.html').read_text().replace('</body>','<script src="/fixture.js" defer></script></body>'))
    s=make_server(8767,fixture=fixture(folder),assets=folder);thread=threading.Thread(target=s.serve_forever,daemon=True);thread.start()
    try:
        chrome=Path('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe')
        if not chrome.exists():raise RuntimeError('Installed Chrome unavailable; no browser installation authorized')
        temp=subprocess.check_output(['/mnt/c/Windows/System32/cmd.exe','/c','echo','%TEMP%'],stderr=subprocess.DEVNULL,text=True).strip().splitlines()[-1];profile=temp.rstrip('\\')+r'\furniture-ai-phase10c9-browser'
        for _ in range(8):
            p=subprocess.run(['/mnt/c/Windows/System32/curl.exe','--noproxy','*','--max-time','2','-sS','http://127.0.0.1:8767/'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=4)
            if p.returncode==0:break
            time.sleep(1)
        else:raise RuntimeError('Windows-to-WSL localhost unavailable')
        cmd=[str(chrome),'--headless','--disable-gpu','--no-proxy-server','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--metrics-recording-only','--user-data-dir='+profile,'--virtual-time-budget=15000','--dump-dom','http://127.0.0.1:8767/']
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);(folder/'dom.html').write_bytes(result.stdout);(folder/'browser.log').write_bytes(result.stderr)
    finally:s.shutdown();s.server_close();thread.join()
    found=re.search(r'<pre id="browser-test-result">(.*?)</pre>',result.stdout.decode(errors='replace'),re.S)
    if not found:raise RuntimeError('No completed browser result; inspect '+str(folder))
    status=json.loads(html.unescape(found[1]));save(folder/'result.json',status)
    if status['status']!='PASS':raise RuntimeError(str(status))
    save(REPORT/'browser-tests-final.json',status);print('BROWSER PASS',len(status['checks']),'checks;no real human decisions',flush=True)

if __name__=='__main__':run()
