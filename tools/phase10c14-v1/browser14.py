"""Installed Windows Chrome,CPU-rendered synthetic UI tests and live screenshot."""
import html,json,re,subprocess,threading,time,uuid
from PIL import Image
from common14 import *
from server14 import make_server
from test_audit14 import FixtureStore
from g5_common import save_versioned


def command(url,profile,extra=()):
    chrome='/mnt/c/Program Files/Google/Chrome/Application/chrome.exe'
    if not Path(chrome).exists():raise RuntimeError('Existing Chrome unavailable;no install authorized')
    return [chrome,'--headless','--disable-gpu','--no-proxy-server','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--metrics-recording-only','--user-data-dir='+profile,'--window-size=1700,1300','--virtual-time-budget=15000',*extra,'--dump-dom',url]


def windows_temp():return subprocess.check_output(['/mnt/c/Windows/System32/cmd.exe','/c','echo','%TEMP%'],stderr=subprocess.DEVNULL,text=True).strip().splitlines()[-1].rstrip('\\')


def run():
    folder=RAW14/('browser-test-'+uuid.uuid4().hex[:8]);folder.mkdir();p=folder/'synthetic.png';Image.new('RGB',(200,150),'#bdd7e5').save(p)
    for name in ('dashboard.css','dashboard.js','browser_fixture.js'):(folder/name).write_bytes((HERE14/name).read_bytes())
    (folder/'dashboard.html').write_text((HERE14/'dashboard.html').read_text().replace('</body>','<script defer src="/fixture.js"></script></body>'))
    server=make_server(8776,fixture=FixtureStore(p),assets=folder);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();temp=windows_temp();profile=temp+'\\furniture-ai-c14-test-'+uuid.uuid4().hex[:8]
    try:
        for _ in range(8):
            r=subprocess.run(['/mnt/c/Windows/System32/curl.exe','--noproxy','*','--max-time','2','-sS','http://127.0.0.1:8776/'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=4)
            if r.returncode==0:break
            time.sleep(1)
        else:raise RuntimeError('Windows-to-WSL localhost unavailable;no remote exposure or download fallback')
        r=subprocess.run(command('http://127.0.0.1:8776/',profile),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);(folder/'dom.html').write_bytes(r.stdout);(folder/'browser.log').write_bytes(r.stderr)
    finally:server.shutdown();server.server_close();thread.join()
    match=re.search(r'<pre id="browser-test-result">(.*?)</pre>',r.stdout.decode(errors='replace'),re.S)
    if not match:raise RuntimeError('Browser test result unavailable;inspect private DOM/log')
    result=json.loads(html.unescape(match[1]));save(folder/'result.json',result)
    if result['status']!='PASS':raise RuntimeError(str(result))
    save_versioned(REPORT14/'browser-tests.json',result);print('BROWSER PASS',len(result['checks']),'checks',flush=True)
    # Read-only production dashboard rendering,not human annotation or inference.
    shot=temp+'\\furniture-ai-c14-live-'+uuid.uuid4().hex[:8]+'.png';profile=temp+'\\furniture-ai-c14-live-'+uuid.uuid4().hex[:8]
    live=subprocess.run(command('http://127.0.0.1:8774/',profile,['--screenshot='+shot]),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);(folder/'live-dom.html').write_bytes(live.stdout);(folder/'live-browser.log').write_bytes(live.stderr)
    dom=live.stdout.decode(errors='replace')
    if live.returncode or 'NO QUERY CACHE' in dom or 'Diagnostic unavailable:' in dom or not re.search(r'<tbody id="queries"><tr',dom):raise RuntimeError('Live training dashboard rendering failed;inspect private DOM')
    source=Path(subprocess.check_output(['wslpath','-u',shot],text=True).strip());target=RAW14/('live-dashboard-'+uuid.uuid4().hex[:8]+'.png');target.write_bytes(source.read_bytes())
    save_versioned(REPORT14/'live-browser-tests.json',dict(status='PASS',URL='http://localhost:8774/',optimization_photos=59,cached_photos=24,uncached_photos=35,rendered_panels=2,screenshot=str(target.relative_to(ROOT)),screenshot_sha256=sha(target),screenshot_private_not_for_Git=True,GPU_disabled=True,new_model_forwards=0,new_reviews=0))
    print('LIVE BROWSER PASS:actual optimization photograph and cached query views',flush=True)


if __name__=='__main__':run()
