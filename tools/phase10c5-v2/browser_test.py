"""Use the already installed Windows Chrome with synthetic photos only."""
import html,json,re,subprocess,threading,time,uuid
from review2 import *
from test_g5 import fixture_registry
from review2 import validate
from server2 import make_server

def run():
    folder=RAW/('browser-fixture-'+uuid.uuid4().hex[:10]);folder.mkdir(exist_ok=False)
    for name in ('index.html','style.css'):(folder/name).write_bytes((HERE/name).read_bytes())
    (folder/'app.js').write_bytes((HERE/'app.js').read_bytes()+b'\n'+(HERE/'browser_fixture.js').read_bytes())
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(1000,800),'#f5f5f5');d=ImageDraw.Draw(image);d.rectangle((100,80,300,240),fill='#b6ddf1');image.save(folder/'synthetic.png')
    reg=fixture_registry();reg['synthetic_browser_run']=uuid.uuid4().hex
    def exported(payload,registry):
        result=validate(payload,registry)
        if result['blocked']:raise ValueError('Blocked synthetic fixture')
        return dict(export_id='SYNTHETIC-NO-REAL-EXPORT',hard_negatives=result['accepted_hard_negatives'],groups=result['negative_source_groups'],mask_changed_terms=0)
    srv=make_server(8767,reg,exported,lambda p:folder/'synthetic.png',folder);thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    try:
        chrome=Path('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe')
        if not chrome.exists():raise RuntimeError('Existing Chrome unavailable; no browser download permitted')
        temp=subprocess.check_output(['/mnt/c/Windows/System32/cmd.exe','/c','echo','%TEMP%'],stderr=subprocess.DEVNULL,text=True).strip().splitlines()[-1]
        profile=temp.rstrip('\\')+r'\furniture-ai-phase10c5-browser-v2'
        for attempt in range(8):
            probe=subprocess.run(['/mnt/c/Windows/System32/curl.exe','--noproxy','*','--max-time','2','-sS',f'http://127.0.0.1:{srv.server_port}/api/registry'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=4)
            if probe.returncode==0:break
            time.sleep(1)
        else:raise RuntimeError('Windows-to-WSL loopback unavailable')
        cmd=[str(chrome),'--headless','--disable-gpu','--no-proxy-server','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--metrics-recording-only','--user-data-dir='+profile,'--virtual-time-budget=15000','--dump-dom',f'http://127.0.0.1:{srv.server_port}/']
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
        (folder/'browser-dom.html').write_bytes(result.stdout);(folder/'browser-stderr.log').write_bytes(result.stderr)
    finally:srv.shutdown();srv.server_close();thread.join()
    match=re.search(r'<pre id="browser-test-result">(.*?)</pre>',result.stdout.decode(errors='replace'),re.S)
    if not match:raise RuntimeError('No completed browser fixture result; inspect '+str(folder))
    status=json.loads(html.unescape(match[1]));save(folder/'result.json',status)
    if status['status']!='PASS':raise RuntimeError(str(status))
    save(REPORT/('browser-tests-'+uuid.uuid4().hex[:8]+'.json'),status)
    print('BROWSER PASS',len(status['checks']),'checks;0real human reviews/exports')

if __name__=='__main__':run()
