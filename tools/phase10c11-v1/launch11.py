"""Disabled-by-default one-arm launcher. Approval gate precedes any worker/CUDA."""
import argparse,subprocess,time,signal,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase10c10-v1'))
from common10 import *
from config10 import require_training_approval
HERE11=Path(__file__).resolve().parent

def run(config_path,approval_path=None):
    config=read(config_path);approval=read(approval_path) if approval_path else None
    require_training_approval(config,approval)
    if approval.get('config_file_sha256')!=sha(config_path):raise PermissionError('Approval file-byte hash mismatch')
    verify_inputs()
    if FUTURE.exists():raise FileExistsError('Fresh experiment destination required; no replay of completed/uncertain updates')
    FUTURE.mkdir(parents=True)
    save(FUTURE/'authorized-config.json',config);save(FUTURE/'approval-record.json',approval)
    start=time.monotonic();save(FUTURE/'queue-started.json',dict(monotonic_start=start,max_seconds=600,experiment_count=1,max_updates=75))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (FUTURE/'worker.log').open('x') as f:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE11/'worker11.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=600-(time.monotonic()-start))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();save(FUTURE/'timeout.json',dict(limit_seconds=600,automatic_retry=False));raise RuntimeError('Hard ten-minute boundary; preserve partial checkpoints')
    if code:raise RuntimeError('Worker stopped; no automatic retry or additional variant')
    save(FUTURE/'queue-complete.json',dict(seconds=time.monotonic()-start,experiment_count=1,updates=read(FUTURE/'complete.json')['updates'],additional_training_authorized=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(REPORT/'experiment-disabled.json'));p.add_argument('--approval-record');a=p.parse_args();run(a.config,a.approval_record)
