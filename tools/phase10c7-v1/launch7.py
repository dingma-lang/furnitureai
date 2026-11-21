"""Exactly one GPU worker, hard process deadline, no automatic retries."""
import subprocess,time,signal
from common7 import *

def run():
    validate_binding(inputs())
    save(RUN/'launch-once.json',dict(max_forward_passes=24,deadline_seconds=600,training_authorized=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    begin=time.monotonic()
    with (RUN/'gpu.log').open('x') as f:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'gpu7.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();save(RUN/'parent-timeout.json',dict(status='STOPPED',seconds=time.monotonic()-begin,no_automatic_retry=True));raise RuntimeError('600-second hard worker deadline; partial raw receipts preserved')
    save(RUN/'launch-result.json',dict(exit_code=code,seconds=time.monotonic()-begin,automatic_retry=False))
    if code:raise RuntimeError('GPU worker failed; inspectgpu.log; no automatic retry')
    print('GPU SMOKE COMPLETE;raw predictions persisted;no training or backward passes')

if __name__=='__main__':run()
