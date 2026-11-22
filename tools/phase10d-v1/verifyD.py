"""CPU preflight and postrun verification; never launches inference/training."""
import argparse
import re
from commonD import *


def tests(postrun=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU tests require hidden CUDA')
    cfg,*_=inputs();suites=[]
    specs=[('Phase10D',['discover','-s',str(HERE),'-p','test_boxD.py'],ROOT),('C21scope',['test_preparation21.ScopeTests21'],ROOT/'tools/phase10c21-v1'),('originalBox',['test_phase10c.BoxTests'],ROOT/'tools/phase10c-v1'),('unchangedMetrics',['test_phase10c.MetricTests'],ROOT/'tools/phase10c-v1')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,args,cwd in specs:
        path=RUN/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with path.open('x') as f:p=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        output=path.read_text();match=re.search(r'Ran (\d+) tests?',output)
        if p.returncode or not match or '\nOK' not in output or 'skipped=' in output:raise RuntimeError('CPU test failed: '+str(path))
        suites.append(dict(suite=name,passed=int(match[1]),log=str(path.relative_to(ROOT)),sha256=sha(path)));print('CPU PASS',name,match[1],flush=True)
    result=dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,binding_sha256=sha(RUN/'binding.json'),new_GPU_calls=0)
    emit(REPORT/('postrun-tests.json' if postrun else 'CPU-tests.json'),result)


def preservation():
    cfg,prepared,train,dev,_=inputs()
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from verify21 import check_seal
    from common21 import verify20_seals,verify19_seals,verify18_seals,verify17_seals,C8,verify_seal
    from verify import preservation as historical
    check_seal();p=historical()
    p.update(C20_seal=verify20_seals(),C19_seal=verify19_seals(),C18_seal=verify18_seals(),C17_seal=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),original_OWLv2_sha256=sha(BASE/'model.safetensors'))
    if p['original_OWLv2_sha256']!=BASE_SHA:raise ValueError('Original checkpoint changed')
    main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
    if main!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Main changed')
    p.update(main_unchanged=main,Phase10D_did_not_modify_historical_outputs=True)
    if (RUN/'complete.json').exists():
        import torch
        from scopeD import BOX_KEYS
        done=read(RUN/'complete.json');cps=[]
        for path in sorted((RUN/'checkpoints').glob('*.pt')):
            ref=read(path.with_suffix('.receipt.json'))
            if sha(path)!=ref['sha256']:raise ValueError('New checkpoint hash changed')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['frozen_sha256']!=cfg['frozen_sha256'] or saved['unregistered_tensor_sha256']!=cfg['unregistered_tensor_sha256'] or set(saved['weights'])!=BOX_KEYS or saved['binding_sha256']!=sha(RUN/'binding.json'):
                raise ValueError('New checkpoint scope/binding/frozen integrity failed')
            if any(not torch.isfinite(v).all() for v in saved['weights'].values()) or any(int(v['step'])!=saved['updates'] for v in saved['optimizer']['state'].values()):raise ValueError('New checkpoint numeric/cursor invalid')
            cps.append(ref)
        if [r['updates'] for r in cps]!=[0,15,30,45,60,75]:raise ValueError('Unexpected snapshot/update sequence')
        p['new_checkpoints_verified']=cps
        updates=[read(RUN/'updates'/('update-'+str(i).zfill(3)+'.json')) for i in range(1,76)]
        for i,u in enumerate(updates):
            if u['update']!=i+1 or u['source_order']!=cfg['orders'][i//15][i%15]:raise ValueError('Committed training order mismatch')
        if done['forwards']!=322 or done['actual_forwards']!=322 or done['updates']!=75 or done['training_backwards']!=295 or len(list((RUN/'forward-intents').glob('*.json')))!=322:
            raise ValueError('Actual execution counters mismatch')
        for i in range(6):
            d=read(RUN/('parameters-epoch-'+str(i)+'.json'))
            if d['frozen_sha256']!=cfg['frozen_sha256'] or d['unregistered_tensor_sha256']!=cfg['unregistered_tensor_sha256']:raise ValueError('Epoch frozen fingerprints changed')
        p.update(actual_epochs=5,actual_updates=75,actual_forwards=322,classification_frozen=True,ignored_examples_activated=0,actual_training_photos=len({r['image_id'] for u in updates for r in u['images']}),actual_training_groups=len({r['source_group'] for u in updates for r in u['images']}))
    emit(REPORT/'preservation.json',p);print('Phase10D historical preservation PASS',p['historical_data_model_files'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--postrun',action='store_true');a=parser.parse_args()
    tests(a.postrun)
    if a.postrun:preservation()
