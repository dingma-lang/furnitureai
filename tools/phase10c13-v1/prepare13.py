"""Verify exact comparability, run CPU tests, bind the one current user approval."""
import copy,re,subprocess
import numpy as np
import torch
from common13 import *
from config12 import check_spec,require_training_approval


def run():
    if OUT13.exists() or REPORT13.exists() or FUTURE.exists():raise FileExistsError('Fresh C13 preparation and unused prepared GPU destination required')
    parent=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if parent!='423dfd0de60b685dccff271c97fe89fa53549412' or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='experiment/phase10c13-projection-preservation-ablation':raise ValueError('Unexpected Git parent/branch')
    public=verify_seal(ROOT/'reports/phase10c12-v1/SHA256SUMS');r=read(ROOT/'reports/phase10c12-v1/artifact-seal.json');seal=ROOT/r['private_seal']
    if sha(seal)!=r['private_seal_sha256']:raise ValueError('C12 private seal changed')
    private=verify_seal(seal);historical,_,_,_=verify_history();c8=verify_seal(C8/'ARTIFACT-SHA256SUMS');prepared,train,dev=verify_inputs13()
    reference=original_heads();initials=list((C11_RUN/'checkpoints').glob('update-000-*.pt'))
    if len(initials)!=1:raise ValueError('Missing C11 initial checkpoint')
    initial=torch.load(initials[0],map_location='cpu',weights_only=True)
    if any(not torch.equal(reference[n],initial['weights'][n]) for n in HEAD_KEYS):raise ValueError('Original/C11 head initialization differs')
    old=read(ROOT/'reports/phase10c12-v1/experiment-disabled.json');stored=old['projection_preservation']['lambda_value']
    if np.float32(stored)!=np.float32(AUTHORIZED_DECIMAL):raise ValueError('Authorized/prepared lambda arithmetic differs')
    OUT13.mkdir(parents=True);REPORT13.mkdir(parents=True)
    save(REPORT13/'input-preflight.json',dict(status='PASS',parent_commit=parent,C12_public_files=public,C12_private_files=private,C8_private_files=c8,C11_preservation=historical,original_heads_match_C11_initialization=True,train_photos=len(train),known_source_groups=len({r['identity_group'] for r in train}),positive_boxes=sum(not a['ignore'] for r in train for a in r['annotations']),internal_development_photos=len(dev),manifest_sha256=sha(ROOT/old['data']['manifest']),training_rows_sha256=digest(train),development_rows_sha256=digest(dev),
        intentional_optimization_difference_from_C11='Only implemented L2-SP on class_head.dense0.weight+bias;4 calibration tensors frozen;eight trainable tensors unchanged',
        implementation_difference_from_prepared_C12='None;execute original sealed launch12.py/future_worker12.py/objective12.py',
        decimal_representation=dict(user_authorized=AUTHORIZED_DECIMAL,prepared_full_precision=stored,effective_FP32=float(np.float32(stored)),loss_and_gradient_FP32_identical=True,not_a_hyperparameter_change=True),
        reporting_differences=['C12 prepared diagnostic logging additionally partitions projection regularizer/data gradients and stores objective/reference metadata','Epoch2/4 summaries use already recorded training updates and snapshots;no extra model forwards'],new_source_annotations=0,Phase10D=False))
    suites=[]
    commands=[('C13',HERE13,None),('C12',HERE12,None),('C11',ROOT/'tools/phase10c11-v1',None),('frozen_scope',ROOT/'tools/phase10c10-v1','test_preflight10.ScopeTests'),('C8_loss',ROOT/'tools/phase10c8-v1',None),('C6_masks',ROOT/'tools/phase10c6-v1',None),('expanded_base',ROOT/'tools/phase10a-v1',None)]
    for name,path,test in commands:
        args=[str(ROOT/'.venv/bin/python'),'-B','-m','unittest']+([test,'-v'] if test else ['discover','-s',str(path),'-p','test*.py','-v']);log=OUT13/('tests-'+name+'.log')
        with log.open('x') as f:result=subprocess.run(args,cwd=path if test else ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();n=re.search(r'Ran (\d+) tests?',s)
        if result.returncode or not n or not s.rstrip().endswith('OK'):raise RuntimeError('Preflight failed;STOP before GPU;inspect '+str(log))
        suites.append(dict(suite=name,passed=int(n[1]),log_sha256=sha(log)));print('CPU PASS',name,n[1],flush=True)
    cfg=copy.deepcopy(old);cfg['training_authorized']=True;cfg['projection_preservation']['enabled']=True;cfg['source_configuration'].update(enabled=True,approved=True,approval_reference='Explicit user Phase10C.13 single-arm GPU authorization')
    check_spec(cfg);path=REPORT13/'authorized-config.json';save(path,cfg)
    approval=dict(user_training_authorized=True,user_reference='Current user attached Phase10C.13:exactly one original-init frozen-calibration projection-preservation arm;no extra variants/downloads/Phase10D',config_semantic_sha256=digest(cfg),config_file_sha256=sha(path),max_optimizer_updates=75,max_epochs=5,max_GPU_seconds=600,max_VRAM_bytes=int(6.5*2**30),user_decimal_lambda=AUTHORIZED_DECIMAL,prepared_decimal_lambda=stored,effective_FP32_lambda=float(np.float32(stored)),FP32_equivalence_verified=True)
    require_training_approval(cfg,approval);save(OUT13/'approval-record.json',approval);save(REPORT13/'approval-record.json',approval)
    save(REPORT13/'preflight-tests.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,GPU_forwards=0,training_updates=0,only_one_arm_authorized=True,prepared_C12_code_unchanged=True))
    print('C13 READY:all preflight checks PASS;one authorized arm;600sec/75updates/6.5GiB',flush=True)


if __name__=='__main__':run()
