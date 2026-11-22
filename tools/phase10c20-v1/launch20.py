"""Exactly one offline attempt:510s graceful stop,515s absolute deadline."""
from common20 import *


def run():
    cfg,*_=inputs20();checks=read(REPORT20/'CPU-preflight-tests.json')
    if checks['status']!='PASS' or checks['binding_sha256']!=sha(RUN20/'binding.json'):raise PermissionError('Pinned CPU preflight required')
    begin=time.monotonic();emit20(RUN20/'launch-once.json',dict(monotonic_start=begin,binding_sha256=sha(RUN20/'binding.json'),automatic_retry=False,maximum_seconds=515))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');reason=None
    with (RUN20/'gpu.log').open('x') as log:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE20/'worker20.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=max(.1,510-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=max(.1,515-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
    emit20(RUN20/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,automatic_retry=False))
    if code or not (RUN20/'complete.json').exists():raise RuntimeError('Single recovery stopped;no retry or restart authorized')
    print('C20 GPU recovery finished;CPU analysis only next',flush=True)


if __name__=='__main__':run()
