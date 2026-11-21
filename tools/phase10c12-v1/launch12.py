"""Disabled-by-default launcher; approval is checked before any worker import."""
import argparse,subprocess,time,signal
from common12 import *
from config12 import require_training_approval


def run(config_path,approval_path=None):
    cfg=read(config_path);approval=read(approval_path) if approval_path else None;require_training_approval(cfg,approval)
    if approval.get('config_file_sha256')!=sha(config_path):raise PermissionError('Exact approved configuration file SHA required')
    if FUTURE.exists():raise FileExistsError('Fresh destination required; no automatic replay')
    FUTURE.mkdir(parents=True);save(FUTURE/'authorized-config.json',cfg);save(FUTURE/'approval-record.json',approval)
    start=time.monotonic();save(FUTURE/'queue-started.json',dict(monotonic_start=start,max_seconds=600,experiment_count=1,max_updates=75))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (FUTURE/'worker.log').open('x') as f:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE12/'future_worker12.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=600-(time.monotonic()-start))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();save(FUTURE/'timeout.json',dict(limit_seconds=600,automatic_retry=False));raise RuntimeError('Cumulative ten-minute boundary')
    if code:raise RuntimeError('C12 stopped; preserve partial results without automatic retry')
    save(FUTURE/'queue-complete.json',dict(seconds=time.monotonic()-start,experiment_count=1,updates=read(FUTURE/'complete.json')['updates'],additional_training_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(REPORT12/'experiment-disabled.json'));p.add_argument('--approval-record');a=p.parse_args();run(a.config,a.approval_record)
