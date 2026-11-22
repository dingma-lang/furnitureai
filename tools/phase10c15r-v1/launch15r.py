"""One offline recovery child;graceful stop175s,absolute process deadline180s."""
import signal,subprocess
from common15r import *


def latest_telemetry():
    records=[]
    for p in RUNR.glob('telemetry-*.json'):
        try:records.append(read(p))
        except (ValueError,OSError):pass
    return max(records,key=lambda r:r['elapsed_seconds']) if records else None


def run():
    cfg,_,_,_=inputs_r();tests=read(REPORTR/'CPU-preflight-tests.json')
    if tests.get('status')!='PASS' or not tests.get('all24_cached_equality_passed') or tests['configuration_sha256']!=sha(RUNR/'authorized-config.json'):raise PermissionError('CPU equality tests must pass before GPU')
    begin=time.monotonic();emit_r(RUNR/'launch-once.json',dict(monotonic_start=begin,configuration_sha256=sha(RUNR/'authorized-config.json'),max_seconds=180,automatic_retry=False))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    reason=None
    with (RUNR/'gpu.log').open('x') as log:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERER/'gpu15r.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=max(.1,175-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            reason=type(exc).__name__;os.killpg(p.pid,signal.SIGTERM)
            try:code=p.wait(timeout=max(.1,180-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
    terminal=read(RUNR/'complete.json') if (RUNR/'complete.json').exists() else read(RUNR/'failure.json').get('terminal',{}) if (RUNR/'failure.json').exists() else {}
    emit_r(RUNR/'launch-result.json',dict(exit_code=code,wall_seconds=time.monotonic()-begin,stop_reason=reason,last_saved_telemetry=latest_telemetry(),terminal_live_hash_available=bool(terminal.get('model_state_after')),automatic_retry=False))
    if code or not (RUNR/'complete.json').exists():raise RuntimeError('Recovery stopped;use preserved telemetry/results,no retry')
    print('C15R GPU COMPLETE;CPU combined analysis only next',flush=True)


if __name__=='__main__':run()
