"""Single no-retry bounded worker;whole subprocess deadline stricter than GPU budget."""
import signal
from common7 import *

def run():
    inputs()
    if read(REPORT/'CPU-tests.json')['status']!='PASS':raise PermissionError('Passing CPU tests required')
    begin=time.monotonic();nonce=uuid.uuid4().hex
    emit(RUN/'launch-once.json',dict(monotonic_start=begin,nonce=nonce,automatic_retry=False,binding_sha256=sha(RUN/'binding.json')))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONFAULTHANDLER='1',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',FURNITURE_E7_NONCE=nonce)
    reason=None
    with (RUN/'console.log').open('x') as log:
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'worker7.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=max(.1,175-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=max(.1,180-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=proc.returncode
    emit(RUN/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,automatic_retry=False,
         last_telemetry_files=[str(p.relative_to(RUN)) for p in sorted((RUN/'telemetry').glob('*complete.json'))[-1:]]))
    if code or not (RUN/'complete.json').exists():raise RuntimeError('Single E7attempt stopped;do not automatically retry')
    print('Bounded E7GPU worker finished;CPU-only analysis next')

if __name__=='__main__':run()
