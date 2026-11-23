"""CPU pre-network tests with no model construction or Internet calls."""
import re,subprocess
from common9 import *
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only tests')
    r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'test_rights9.py'),'-v'],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,timeout=90)
    p=REPORT/('CPU-rights-'+uuid.uuid4().hex[:8]+'.log');atomic(p,r.stdout+r.stderr);m=re.search(r'Ran (\d+) tests?',p.read_text())
    if r.returncode or not m or '\nOK' not in p.read_text():raise RuntimeError('CPU tests failed:'+str(p))
    emit(REPORT/'CPU-tests.json',dict(status='PASS',passed=int(m[1]),log=str(p.relative_to(ROOT)),sha256=sha(p),network_calls=0,GPU_calls=0))
    print('CPU rights/transport PASS',m[1],flush=True)
if __name__=='__main__':run()
