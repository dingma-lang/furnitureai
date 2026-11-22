"""One launch only, parent hard deadline, offline and no automatic retry."""
import signal,subprocess
from common15 import *


def run():
    cfg,_,_,_=inputs15();emit(RUN15/'launch-once.json',dict(config_sha256=sha(RUN15/'authorized-config.json'),max_worker_seconds=180,automatic_retry=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    begin=time.monotonic()
    with (RUN15/'gpu.log').open('x') as log:
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE15/'gpu15.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();emit(RUN15/'parent-timeout.json',dict(seconds=time.monotonic()-begin,no_retry=True));raise RuntimeError('180-second worker deadline;stop with partial records')
    emit(RUN15/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,automatic_retry=False))
    if code:raise RuntimeError('GPU smoke failed;inspect immutable failure/log;no retry')
    print('C15 GPU COMPLETE;run CPU analysis separately;training remains disabled',flush=True)


if __name__=='__main__':run()
