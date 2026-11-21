"""Verify CPU comparability and bind the user's explicit one-arm authorization."""
import subprocess,re,copy
from common11 import *
from config10 import check_spec,require_training_approval

def run():
    if not (REPORT11/'input-preflight.json').exists() or FUTURE.exists():raise ValueError('Input preflight/fresh destination required')
    tests=[]
    for suite in ('phase10c11-v1','phase10c10-v1','phase10c8-v1','phase10c6-v1','phase10a-v1'):
        path=OUT11/('tests-'+suite+'.log')
        with path.open('x') as f:r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/suite),'-p','test*.py','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=path.read_text();n=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not n or not s.rstrip().endswith('OK'):raise RuntimeError('Preflight tests failed: '+suite)
        tests.append(dict(suite=suite,passed=int(n[1]),log_sha256=sha(path)));print('CPU PREFLIGHT PASS',suite,n[1],flush=True)
    old=read(ROOT/'reports/phase10c10-v1/experiment-disabled-v3.json');cfg=copy.deepcopy(old);cfg['training_authorized']=True;cfg['source_configuration'].update(enabled=True,approved=True,approval_reference='User explicitly authorizes Phase10C.11 one-arm frozen-calibration experiment')
    cfg['code_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in HERE11.glob('*.py')})
    check_spec(cfg)
    before=copy.deepcopy(old);after=copy.deepcopy(cfg)
    for d in (before,after):
        d.pop('code_hashes');d['training_authorized']=False;d['source_configuration'].update(enabled=False,approved=False,approval_reference='metadata')
    if before!=after:raise ValueError('Approved candidate differs from prepared specification')
    verify_inputs();config_path=REPORT11/'authorized-config.json';save(config_path,cfg)
    approval=dict(user_training_authorized=True,user_reference='Explicit current user instruction: Phase10C.11 Frozen-Calibration GPU Ablation; exactly one arm at1e-5,5epochs/75updates,600GPUseconds including inference,6.5GiB;no extra variants/Phase10D',
        config_semantic_sha256=digest(cfg),config_file_sha256=sha(config_path),max_optimizer_updates=75,max_epochs=5,max_GPU_seconds=600,max_VRAM_bytes=int(6.5*2**30),Phase10D_authorized=False)
    require_training_approval(cfg,approval);save(OUT11/'approval-record.json',approval);save(REPORT11/'approval-record.json',approval)
    save(REPORT11/'preflight-tests.json',dict(status='PASS',suites=tests,passed=sum(t['passed'] for t in tests),logging_loss_and_gradients_unchanged=True,training_semantics_match_prepared_config=True,original_C10_files_unmodified=True,GPU_initialized=False))
    print('ONE ARM AUTHORIZED AND COMPARABLE;prepared config unchanged except activation/code-provenance metadata',flush=True)

if __name__=='__main__':run()
