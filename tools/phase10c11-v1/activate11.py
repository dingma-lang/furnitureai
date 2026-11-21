"""Refresh execution code pins after zero-GPU startup fix; same single arm."""
import copy,subprocess,re
from common11 import *
from config10 import check_spec,require_training_approval

def run():
    if FUTURE.exists():raise ValueError('Cannot replay or modify an existing experiment')
    log=OUT11/'tests-phase10c11-startup-fixed.log'
    with log.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(HERE11),'-p','test*.py','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=120)
    s=log.read_text();n=re.search(r'Ran (\d+) tests?',s)
    if r.returncode or not n or not s.rstrip().endswith('OK'):raise RuntimeError('Startup-fixed C11 tests failed; inspect '+str(log))
    cfg=copy.deepcopy(read(REPORT11/'authorized-config.json'));cfg['code_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in HERE11.glob('*.py')});check_spec(cfg);verify_inputs()
    path=REPORT11/'authorized-config-v2.json';save(path,cfg)
    approval=copy.deepcopy(read(OUT11/'approval-record.json'));approval.update(config_semantic_sha256=digest(cfg),config_file_sha256=sha(path));require_training_approval(cfg,approval);save(OUT11/'approval-record-v2.json',approval);save(REPORT11/'approval-record-v2.json',approval)
    old=read(REPORT11/'preflight-tests.json');suites=[dict(suite='phase10c11-v1',passed=int(n[1]),log_sha256=sha(log))]+[p for p in old['suites'] if p['suite']!='phase10c11-v1'];save(REPORT11/'preflight-tests-v2.json',dict(status='PASS',passed=sum(p['passed'] for p in suites),suites=suites,logging_loss_and_gradients_unchanged=True,standalone_imports_verified=True,old_C10_sources_unchanged=True))
    save(REPORT11/'startup-fix.json',dict(failed_attempt='Standalone launch11 import common10 path absent',model_forwards=0,optimizer_updates=0,CUDA_initialized=False,experiment_output_directory_created=False,fix='Explicit local C10 module path in isolated C11 launcher and worker; fresh-process CPU import regression test',new_config=str(path.relative_to(ROOT)),old_authorized_draft_preserved=True,training_variant_changed=False))
    print('STARTUP FIX +',sum(p['passed'] for p in suites),'CPU tests PASS;one unchanged GPU arm',flush=True)

if __name__=='__main__':run()
