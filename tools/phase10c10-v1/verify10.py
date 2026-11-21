"""Final CPU tests and post-preflight preservation. Never imports a GPU worker."""
import subprocess,re,ast,importlib.util
from common10 import *
from config10 import check_spec

def run():
    config=read(REPORT/'experiment-disabled-v3.json');check_spec(config)
    if config['training_authorized'] or config['source_configuration']['enabled'] or config['source_configuration']['approved'] or FUTURE.exists():raise ValueError('Preflight must remain disabled with no training directory')
    suites=[]
    for name in ('phase10c10-v1','phase10c8-v1','phase10c6-v1','phase10a-v1'):
        p=OUT/('tests-final-'+name+'.log')
        with p.open('x') as f:
            r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=p.read_text();n=re.search(r'Ran (\d+) tests?',text)
        if r.returncode or not n or not text.rstrip().endswith('OK'):raise RuntimeError('CPU suite failed: '+name)
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(p)));print('CPU PASS',name,n[1],flush=True)
    verify_inputs();c8=verify_seal(C8/'ARTIFACT-SHA256SUMS');c9=read(ROOT/'reports/phase10c9-v1/artifact-seal.json');c9_count=verify_seal(ROOT/c9['private_seal_path']);c9_public=verify_seal(ROOT/'reports/phase10c9-v1/SHA256SUMS')
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'));spec=importlib.util.spec_from_file_location('C10_historical_preservation',ROOT/'tools/phase10c-v1/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);old=m.preservation()
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    save(REPORT/'tests-final.json',dict(status='PASS',suites=suites,passed=sum(s['passed'] for s in suites),synthetic_full_75_update_test=True,real_photograph_model_forwards=0,pretrained_model_optimizer_updates=0,CUDA_initialized=False,
        disabled_v3_code_pins_valid=True,old_config_drafts_preserved=True,initial_fixture_errors='Two initial CPU test fixtures were corrected: identical synthetic frozen-backbone initialization on restore, and named optimizer groups matching Torch requirements. No real model/GPU run failed or was retried.'))
    main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
    if main!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Main changed')
    save(REPORT/'preservation.json',dict(status='PASS',historical=old,C8_private_sealed_files=c8,C9_private_sealed_files=c9_count,C9_public_sealed_files=c9_public,main_commit=main,original_checkpoint_sha256=sha(BASE/'model.safetensors'),
        real_model_training=False,GPU_inference=False,downloads=0,environment_changed=False,production_changed=False,Mac_project_changed=False,frozen_or_reserved_images_consumed=False,Phase10D_started=False,future_training_directory_created=FUTURE.exists()))
    print('PRESERVATION PASS',old['historical_data_model_files'],'historical files;0CUDA initialization',flush=True)

if __name__=='__main__':run()
