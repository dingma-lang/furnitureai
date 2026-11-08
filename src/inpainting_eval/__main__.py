"""Bounded single-case CLI. All model paths are local; no acquisition side effects."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image
from .evaluate import evaluate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True);parser.add_argument('--output',required=True);parser.add_argument('--timeout',type=int,default=180);parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    if not 1<=a.timeout<=600:parser.error('Timeout must be 1–600 seconds')
    case=json.loads(Path(a.case).read_text());out=Path(a.output)
    if out.exists():parser.error('Output already exists; use a new directory')
    if a.worker:
        report=evaluate(Image.open(case['original']),Image.open(case['removal_mask']),Image.open(case['protected_mask']),case['configuration'],out)
        print(json.dumps({'status':report['status'],'output':str(out)}));return 0 if report['status']=='complete' else 2
    start=time.perf_counter();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    try:
        p=subprocess.run([sys.executable,'-m','inpainting_eval','--worker','--case',a.case,'--output',a.output],env=env,timeout=a.timeout,capture_output=True,text=True)
        failure=None if (out/'report.json').exists() else 'Worker exited without report'
        detail={'returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-8000:]}
    except subprocess.TimeoutExpired as exc:
        failure='Worker timeout; no automatic retry';detail={'timeout_seconds':a.timeout}
    out.mkdir(parents=True,exist_ok=True)
    if failure:(out/'report.json').write_text(json.dumps({'status':'failed','error':failure,'neural_inference_executed':None},indent=2))
    detail['wall_seconds_including_startup']=time.perf_counter()-start;(out/'process.json').write_text(json.dumps(detail,indent=2))
    print((out/'report.json').read_text());return 2 if failure else p.returncode

if __name__=='__main__':raise SystemExit(main())
