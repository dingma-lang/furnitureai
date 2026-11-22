"""One offline worker,175s graceful stop and180s absolute process deadline."""
import signal,subprocess
from common17 import *


def latest():
    records=[]
    for p in (RUN17/'telemetry').glob('event-*.json'):
        try:records.append(read(p))
        except (ValueError,OSError):pass
    return max(records,key=lambda r:r['elapsed_seconds']) if records else None


def run():
    cfg,_,_,_=inputs17();checks=read(REPORT17/'CPU-preflight-tests.json')
    if checks.get('status')!='PASS' or checks['configuration_sha256']!=sha(RUN17/'authorized-config.json'):raise PermissionError('CPU preflight required before GPU')
    begin=time.monotonic();emit17(RUN17/'launch-once.json',dict(monotonic_start=begin,configuration_sha256=sha(RUN17/'authorized-config.json'),automatic_retry=False,maximum_worker_seconds=180))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');reason=None
    with (RUN17/'gpu.log').open('x') as log:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE17/'gpu17.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=max(.1,175-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=max(.1,180-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
    emit17(RUN17/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,last_saved_telemetry=latest(),automatic_retry=False))
    if code or not (RUN17/'complete.json').exists():raise RuntimeError('Single smoke stopped;partial records preserved,no retry')
    print('C17 GPU complete;only CPU analysis next;no training',flush=True)


if __name__=='__main__':run()
