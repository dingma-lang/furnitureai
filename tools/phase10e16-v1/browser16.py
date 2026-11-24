"""One bounded isolated Windows-headless-browser attempt; no installation or models."""
import argparse
import html
import re
import subprocess
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from common16 import *
from serve16 import Handler


def run(label=''):
    chrome=Path('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe')
    result=dict(attempts=1,maximum_runtime_seconds=30,profile_increment_cap=64*2**20,
                browser='existing Windows Chrome, isolated fresh profile',real_model_calls=0)
    if not chrome.is_file() or not Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists():
        result.update(status='UNAVAILABLE',reason='No usable installed browser/WSL interoperability');emit(REPORT/('browser-smoke'+label+'.json'),result);return
    space(64*2**20)
    key='phase10e16-profile-'+uuid.uuid4().hex
    profile=OUT/key
    winprofile=subprocess.check_output(['wslpath','-w',str(profile)],text=True).strip()
    baseline=allocated();peak=baseline;started=time.monotonic();proc=None
    local_before=sum(json.loads(l)['body_bytes'] for l in (OUT/'browser-local-http.jsonl').read_text().splitlines()) if (OUT/'browser-local-http.jsonl').exists() else 0
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        server.smoke_enabled=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        command=[str(chrome),'--headless=new','--disable-gpu','--disable-gpu-compositing','--disable-accelerated-2d-canvas',
            '--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions',
            '--no-first-run','--no-default-browser-check','--disable-default-apps','--disable-breakpad','--noerrdialogs',
            '--disable-gpu-shader-disk-cache','--disk-cache-size=1048576','--media-cache-size=1048576',
            '--disable-features=OptimizationHints,MediaRouter,AutofillServerCommunication',
            '--user-data-dir='+winprofile,'--dump-dom','--virtual-time-budget=20000',f'http://localhost:{server.server_port}/smoke.html']
        try:
            with (OUT/('browser-stdout'+label+'.txt')).open('xb') as stdout,(OUT/('browser-stderr'+label+'.txt')).open('xb') as stderr:
                proc=subprocess.Popen(command,stdout=stdout,stderr=stderr)
                while proc.poll() is None:
                    peak=max(peak,allocated())
                    if time.monotonic()-started>30:raise TimeoutError('30-second browser bound')
                    if peak>baseline+64*2**20:raise RuntimeError('Browser profile exceeded 64MiB reserved increment')
                    time.sleep(.1)
            text=(OUT/('browser-stdout'+label+'.txt')).read_text(errors='replace')
            match=re.search(r'<pre id="result"[^>]*>(.*?)</pre>',text,re.S)
            if not match:result.update(status='UNVERIFIED',reason='No browser smoke result found',exit_code=proc.returncode)
            else:
                payload=html.unescape(match.group(1))
                if payload=='RUNNING':result.update(status='UNVERIFIED',reason='Browser async checks did not finish',exit_code=proc.returncode)
                else:result.update(status='PASS' if json.loads(payload).get('status')=='PASS' else 'INCOMPLETE_OR_FAIL',checks=json.loads(payload),exit_code=proc.returncode)
        except (OSError,RuntimeError,TimeoutError) as e:
            result.update(status='UNVERIFIED',reason=type(e).__name__+': '+str(e))
        finally:
            if proc is not None and proc.poll() is None:proc.kill();proc.wait(timeout=5)
            # Only this freshly generated profile GUID; never touch a personal browser process.
            ps=Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
            if proc is not None and ps.is_file():
                cleanup="Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"+key+"') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
                try:subprocess.run([str(ps),'-NoProfile','-NonInteractive','-Command',cleanup],capture_output=True,timeout=8,check=True)
                except (OSError,subprocess.SubprocessError) as e:result['isolated_process_cleanup_warning']=type(e).__name__
            server.shutdown();thread.join(3)
    result.update(runtime_seconds=time.monotonic()-started,peak_observed_allocated_bytes=peak,
        final_allocated_bytes=allocated(),source_requests_during_browser=0,
        local_HTTP_response_body_bytes=(sum(json.loads(l)['body_bytes'] for l in (OUT/'browser-local-http.jsonl').read_text().splitlines()) if (OUT/'browser-local-http.jsonl').exists() else 0)-local_before)
    emit(REPORT/('browser-smoke'+label+'.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='');a=p.parse_args()
    if a.label not in ('','-v2'):raise ValueError('Only initial or one versioned browser smoke')
    run(a.label)
