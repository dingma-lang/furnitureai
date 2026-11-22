"""One durable launch token;595s graceful,600s hard wall-clock limit."""
import signal
from commonD import *


def run():
    cfg,*_=inputs();pre=read(REPORT/'CPU-tests.json')
    if pre['status']!='PASS' or pre['binding_sha256']!=sha(RUN/'binding.json'):
        raise PermissionError('Bound CPU tests must pass first')
    begin=time.monotonic()
    emit(RUN/'launch-once.json',dict(monotonic_start=begin,binding_sha256=sha(RUN/'binding.json'),max_seconds=600,automatic_retry=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONFAULTHANDLER='1')
    with (RUN/'gpu.log').open('x') as log:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'workerD.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        reason=None
        try:code=process.wait(timeout=max(.1,595-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=max(.1,600-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    emit(RUN/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,automatic_retry=False))
    if code or not (RUN/'complete.json').exists():raise RuntimeError('Single authorized attempt stopped;do not retry')
    print('Phase10D single GPU arm finished;CPU analysis next',flush=True)


if __name__=='__main__':run()
