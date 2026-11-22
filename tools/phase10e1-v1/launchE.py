"""Prepared hard-deadline supervisor. No default approval and no automatic retry."""
import argparse,signal
from commonE import *
from configE import require_execution

def launch(approval_path,mode):
    approval=read(approval_path);require_execution(approval,mode)
    root=FUTURE if mode=='training' else SMOKE;limit=approval['limits']['seconds']
    if root.exists():raise FileExistsError('Single authorized attempt only')
    root.mkdir(parents=True);nonce=uuid.uuid4().hex;begin=time.monotonic()
    receipt=dict(nonce=nonce,approval_sha256=sha(approval_path),monotonic_start=begin,maximum_seconds=limit,mode=mode,automatic_retry=False)
    with (root/'launch-once.json').open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
    env=dict(os.environ,FURNITURE_E_LAUNCH_NONCE=nonce,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (root/'console.log').open('x') as log:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'workerE.py'),'--mode',mode,'--authorization',str(approval_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=max(.1,limit-3-(time.monotonic()-begin)))
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=max(.1,limit-(time.monotonic()-begin)))
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    result=dict(exit_code=code,wall_seconds=time.monotonic()-begin,automatic_retry=False)
    p=root/'launch-result.json';pending=root/('.launch-result-'+nonce)
    with pending.open('x') as f:json.dump(result,f);f.flush();os.fsync(f.fileno())
    os.link(pending,p)
    if code or not (root/'complete.json').exists():raise RuntimeError('Single attempt stopped;no retry/repair authorized')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['training','gradient_smoke'],required=True);p.add_argument('--authorization',type=Path);a=p.parse_args()
    if a.authorization is None:raise PermissionError('Phase10E.1 authorizes preparation only')
    launch(a.authorization,a.mode)
