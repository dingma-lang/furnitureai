"""One supervised GPU diagnostic;177-second graceful and180-second hard deadline."""
import signal
from common2 import *

def run():
    cfg,*_=inputs();checks=read(REPORT/'CPU-tests.json')
    if checks['status']!='PASS' or checks['binding_sha256']!=sha(RUN/'binding.json'):raise PermissionError('Exact passing CPU preflight required')
    begin=time.monotonic();emit(RUN/'launch-once.json',dict(monotonic_start=begin,binding_sha256=sha(RUN/'binding.json'),maximum_seconds=180,automatic_retry=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (RUN/'gpu.log').open('x') as log:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'worker2.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        reason=None
        try:code=process.wait(timeout=max(.1,177-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=max(.1,180-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    emit(RUN/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,reason=reason,automatic_retry=False))
    if code or not (RUN/'complete.json').exists():raise RuntimeError('Single E2 attempt stopped;no retry authorized')
    print('E2 GPU diagnostic complete;CPU analysis only next',flush=True)

if __name__=='__main__':run()
