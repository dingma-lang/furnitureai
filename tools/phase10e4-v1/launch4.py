"""One-shot supervised recovery;235s graceful /240s hard wall deadline."""
import signal
from common4 import *


def run():
    cfg,*_=inputs()
    if read(REPORT/'CPU-tests.json')['status']!='PASS':raise PermissionError('Passing CPU preflight required')
    begin=time.monotonic();nonce=uuid.uuid4().hex
    emit(RUN/'launch-once.json',dict(monotonic_start=begin,nonce=nonce,binding_sha256=sha(RUN/'binding.json'),automatic_retry=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',FURNITURE_E4_NONCE=nonce)
    reason=None
    with (RUN/'gpu.log').open('x') as log:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'worker4.py')],cwd=ROOT,env=env,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=max(.1,235-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=max(.1,240-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    emit(RUN/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,reason=reason,automatic_retry=False))
    if code or not (RUN/'complete.json').exists():raise RuntimeError('One E4 attempt stopped;no retry authorized')
    print('GPU recovery complete;CPU analysis only next',flush=True)


if __name__=='__main__':run()
