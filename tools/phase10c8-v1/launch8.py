"""Two sequential authorized arms under one hard hour; no additional variants."""
import subprocess,signal,time
from common8 import *

def run():
    verify_binding(read(RUN/'binding.json'));start=time.monotonic();save(RUN/'queue-started.json',dict(monotonic_start=start,max_seconds=3600,arms=list(ARMS),max_total_updates=150))
    for arm in ARMS:
        remaining=3600-(time.monotonic()-start)
        if remaining<=0:raise RuntimeError('Queue budget consumed')
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
        with (RUN/(arm+'.log')).open('x') as f:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'worker8.py'),'--arm',arm],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait();save(RUN/'timeout.json',dict(arm=arm,no_automatic_retry=True));raise RuntimeError('Hard one-hour queue deadline; partial snapshots preserved')
        if code:raise RuntimeError('Arm failed; stop queue, inspect '+arm+'.log;no automatic retry')
        done=read(RUN/arm/'complete.json')
        if done['updates']!=75 or sha(RUN/done['final_checkpoint']['file'])!=done['final_checkpoint']['sha256']:raise ValueError('Arm completion/checkpoint integrity failed')
    save(RUN/'queue-complete.json',dict(arms=list(ARMS),optimizer_updates=150,seconds=time.monotonic()-start,no_more_training_authorized=True));print('A/B QUEUE COMPLETE150updates;no additional variants')

if __name__=='__main__':run()
