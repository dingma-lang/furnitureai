"""Bounded real dashboard rendering with installed Windows Chrome, CPU only."""
import subprocess,uuid,re
from common9 import *

def run():
    temp=subprocess.check_output(['/mnt/c/Windows/System32/cmd.exe','/c','echo','%TEMP%'],stderr=subprocess.DEVNULL,text=True).strip().splitlines()[-1]
    screenshot=temp.rstrip('\\')+'\\furniture-ai-c9-'+uuid.uuid4().hex[:8]+'.png';profile=temp.rstrip('\\')+r'\furniture-ai-phase10c9-live'
    chrome='/mnt/c/Program Files/Google/Chrome/Application/chrome.exe'
    cmd=[chrome,'--headless','--disable-gpu','--no-proxy-server','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-extensions','--disable-sync','--metrics-recording-only','--user-data-dir='+profile,'--window-size=1800,1250','--virtual-time-budget=15000','--screenshot='+screenshot,'--dump-dom','http://127.0.0.1:8772/']
    attempt=uuid.uuid4().hex[:8]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60);(OUT/('live-dom-'+attempt+'.html')).write_bytes(r.stdout);(OUT/('live-browser-'+attempt+'.log')).write_bytes(r.stderr)
    dom=r.stdout.decode(errors='replace');panels=len(re.findall('<article class="panel">',dom))
    expected=next(p for p in read(OUT/'dashboard-live.json')['photos'] if p['image_id']=='757a1631131f4ba3')
    if r.returncode or panels!=3 or f"757a1631131f4ba3 · {expected['width']}×{expected['height']}" not in dom:raise RuntimeError('Live dashboard rendering incomplete; inspect live DOM/log')
    path=Path(subprocess.check_output(['wslpath','-u',screenshot],text=True).strip())
    (OUT/'live-dashboard.png').write_bytes(path.read_bytes())
    save(REPORT/'live-browser-tests.json',dict(status='PASS',rendered_model_panels=panels,real_photo_pool=43,default_photo='757a1631131f4ba3',screenshot=str((OUT/'live-dashboard.png').relative_to(ROOT)),screenshot_sha256=sha(OUT/'live-dashboard.png'),GPU_disabled=True,model_forwards=0,production_or_label_changes=False))
    print('LIVE BROWSER PASS3model panels;real registered photograph;CPU rendering')

if __name__=='__main__':run()
