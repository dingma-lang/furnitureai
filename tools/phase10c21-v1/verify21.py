"""Read-only historical verification, isolated CPU tests, then optional C21 sealing."""
import argparse
import re
from common21 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise PermissionError('Run with CUDA_VISIBLE_DEVICES empty')
    import torch
    suites=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    specs=[('C21-preparation',['discover','-s',str(HERE21),'-p','test_preparation21.py'],ROOT),
           ('original-box-adapter',['test_phase10c.BoxTests'],ROOT/'tools/phase10c-v1'),
           ('source-negative-protections',['test_auto6.GeometryTests','test_auto6.EvidenceTests','test_auto6.MaskTests'],ROOT/'tools/phase10c6-v1')]
    for name,args,cwd in specs:
        path=DATA21/('CPU-'+name+'-'+uuid.uuid4().hex[:8]+'.log')
        with path.open('x') as stream:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*args,'-v'],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        output=path.read_text();match=re.search(r'Ran (\d+) tests?',output)
        if p.returncode or not match or '\nOK' not in output or 'skipped=' in output:
            raise RuntimeError('CPU test failed: '+str(path))
        suites.append(dict(suite=name,passed=int(match[1]),log=str(path.relative_to(ROOT)),sha256=sha(path)))
        print('CPU PASS',name,match[1],flush=True)
    emit21(REPORT21/'tests-v2.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,real_model_forwards=0,GPU_calls=0,synthetic_CPU_fixture_checks_only=True))
    print('Rehashing preserved historical artifacts;no photograph decoding or inference',flush=True)
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    from verify import preservation
    history=preservation()
    history.update(C20_seal=verify20_seals(),C19_seal=verify19_seals(),C18_seal=verify18_seals(),C17_seal=verify17_seals(),C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'))
    original=sha(BASE/'model.safetensors');done=read(RUN20/'complete.json');final=done['final_checkpoint']
    if original!='e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7' or sha(RUN20/final['file'])!=final['sha256']:
        raise ValueError('Original or C20 final checkpoint changed')
    main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
    if main!='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':raise ValueError('Unexpected main change')
    if torch.cuda.is_initialized():raise RuntimeError('CPU verification initialized CUDA')
    history.update(original_OWLv2_sha256=original,C20_final_checkpoint=final,main_unchanged=main,
                   new_real_model_forwards=0,new_optimizer_updates=0,downloads=0,Phase10D_activated=False,
                   current_interpreter=sys.executable,environment_modified=False,backups_uploaded=False)
    emit21(REPORT21/'preservation-v2.json',history)
    print('Historical preservation PASS',history['historical_data_model_files'],'files',flush=True)


def public_seal_bytes(paths,root=ROOT):
    """Match historical verify_seal: paths are relative to repository root."""
    lines=[]
    for path in paths:
        path=path.resolve()
        if not path.is_relative_to(root.resolve()):raise PermissionError('Outside seal root')
        lines.append(sha(path)+'  '+str(path.relative_to(root.resolve()))+'\n')
    return ''.join(lines).encode()


def seal():
    if not (REPORT21/'report.md').exists() or read(REPORT21/'tests-v2.json')['status']!='PASS' or read(REPORT21/'preservation-v2.json')['status']!='PASS':
        raise ValueError('Complete report and passing checks required before sealing')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(DATA21.rglob('*')) if p.is_file()}
    path=DATA21/'artifact-sha256-v2.json'
    emit21(path,dict(files=files))
    # Retained pending hard links are conservatively counted by the writer.
    seen=set();actual=0
    for folder in (DATA21,REPORT21):
        for p in [folder,*folder.rglob('*')]:
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);actual+=st.st_blocks*512
    meta=dict(schema='phase10c21-artifact-seal-v1',private_manifest=str(path.relative_to(ROOT)),private_manifest_sha256=sha(path),private_files=len(files),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE21.iterdir()) if p.is_file()},conservative_allocated_bytes_before_public_seal=allocated21(),actual_unique_inode_allocated_bytes_before_public_seal=actual,local_bound_bytes=MAX_BYTES21,new_real_model_forwards=0,optimizer_updates=0,downloads=0,main_unchanged=subprocess.check_output(['git','rev-parse','main'],text=True).strip(),private_images_or_weights_for_Git=False)
    meta['supersedes']='artifact-seal.json; initial report-relative seal snapshot retained unchanged'
    emit21(REPORT21/'artifact-seal-v2.json',meta)
    public=[p for p in sorted(REPORT21.iterdir()) if p.is_file() and not p.name.startswith('.')]
    atomic21(REPORT21/'SHA256SUMS-v2',public_seal_bytes(public))
    print('C21 sealed',len(public),'public files;',len(files),'private entries;',actual,'unique-inode allocated bytes before public seal',flush=True)


def check_seal():
    n=verify_seal(REPORT21/'SHA256SUMS-v2');meta=read(REPORT21/'artifact-seal-v2.json');p=ROOT/meta['private_manifest']
    if sha(p)!=meta['private_manifest_sha256']:raise ValueError('C21 manifest changed')
    for rel,h in {**read(p)['files'],**meta['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('C21 changed: '+rel)
    print('C21 sealed bytes PASS',n,'public entries',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');parser.add_argument('--check-seal',action='store_true');args=parser.parse_args()
    if args.seal:seal()
    elif args.check_seal:check_seal()
    else:run()
