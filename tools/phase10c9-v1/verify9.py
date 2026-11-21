"""Final CPU suites and read-only historical preservation after investigation."""
import subprocess,re,importlib.util,ast
from common9 import *

def run():
    tests=[]
    for suite in ('phase10c9-v1','phase10c8-v1','phase10c6-v1','phase10a-v1'):
        p=OUT/('tests-'+suite+'.log')
        with p.open('x') as f:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/suite),'-p','test*.py','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        log=p.read_text();n=re.search(r'Ran (\d+) tests?',log)
        if result.returncode or not n or not log.rstrip().endswith('OK'):raise RuntimeError('CPU suite failed: '+suite)
        tests.append(dict(suite=suite,passed=int(n[1]),log_sha256=sha(p)));print('CPU PASS',suite,n[1],flush=True)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));spec=importlib.util.spec_from_file_location('c9_historical_check',ROOT/'tools/phase10c-v1/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);historical=m.preservation()
    public=verify_seal(ROOT/'reports/phase10c8-v2/SHA256SUMS');private=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    from common8 import verify_binding
    verify_binding(read(C8/'binding.json'));train,dev=subsets()
    proposed=read(OUT/'hard-negative-proposals.json')
    if proposed['candidates']:raise ValueError('Expected zero new source certificates; audit changed inputs')
    allowed={r['image_id'] for r in train}
    if any(r['image_id'] not in allowed for r in proposed['unverified_queries']):raise ValueError('Development-to-training candidate leakage')
    main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
    if main!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Main changed')
    browser=REPORT/'browser-tests-final.json'
    save(REPORT/'tests.json',dict(status='PASS',suites=tests,passed=sum(t['passed'] for t in tests),browser=read(browser if browser.exists() else REPORT/'browser-tests.json')))
    save(REPORT/'preservation.json',dict(status='PASS',historical=historical,C8_public_sealed_files=public,C8_private_sealed_files=private,C8_original_checkpoint_sha256=read(REPORT/'preflight.json')['original_checkpoint_sha256'],
        new_model_forwards=0,optimizer_updates=0,photo_or_model_downloads=0,new_human_reviews=0,new_activated_training_labels=0,development_mining=False,main_commit=main,Phase10D_started=False))
    print('PRESERVATION PASS',historical['historical_data_model_files'],'old files;',private,'C8 files',flush=True)

if __name__=='__main__':run()
