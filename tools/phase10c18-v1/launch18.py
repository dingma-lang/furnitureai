"""One-shot offline queue; 595-second graceful stop and600-second absolute bound."""
from common18 import *


def run():
    cfg,*_=inputs18();checks=read(REPORT18/'CPU-preflight-tests.json')
    if checks['status']!='PASS' or checks['binding_sha256']!=sha(RUN18/'binding.json'):
        raise PermissionError('Pinned CPU tests required before GPU')
    begin=time.monotonic();emit18(RUN18/'launch-once.json',dict(monotonic_start=begin,binding_sha256=sha(RUN18/'binding.json'),automatic_retry=False,maximum_seconds=600))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    reason=None
    with (RUN18/'gpu.log').open('x') as log:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE18/'worker18.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=max(.1,595-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=max(.1,600-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
    emit18(RUN18/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,automatic_retry=False))
    if code or not (RUN18/'complete.json').exists():raise RuntimeError('Single authorized arm stopped;no retry or extra variant')
    print('C18 GPU completed. CPU metrics only from here.',flush=True)


if __name__=='__main__':run()
