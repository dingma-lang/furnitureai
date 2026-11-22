"""Final immutable provenance/resource manifest and repository-relative public seal."""
from commonD import *


def storage():
    seen=set();count=0;logical=0
    for root in (RUN,REPORT):
        for p in [root,*root.rglob('*')]:
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:
                seen.add(key);count+=s.st_blocks*512
                if p.is_file():logical+=s.st_size
    return dict(unique_inode_allocated_bytes=count,unique_file_bytes=logical,conservative_path_allocated_bytes=allocated())


def run():
    cfg,*_=inputs();done=read(RUN/'complete.json')
    if read(REPORT/'preservation.json')['status']!='PASS' or read(REPORT/'postrun-tests.json')['status']!='PASS' or not (REPORT/'report.md').exists():
        raise ValueError('Final passing verification and report required')
    emit(REPORT/'resource-accounting.json',dict(launcher=read(RUN/'launch-result.json'),completed_worker=done,storage_before_final_seal=storage(),limits=LIMITS,downloads=0,automatic_retries=0,another_configuration=False))
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    path=RUN/'artifact-sha256.json';emit(path,dict(files=files))
    meta=dict(schema='phase10d-artifact-seal-v1',private_manifest=str(path.relative_to(ROOT)),private_manifest_sha256=sha(path),private_files=len(files),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},base_sha256=BASE_SHA,binding_sha256=sha(RUN/'binding.json'),updates=75,forwards=322,main_unchanged=subprocess.check_output(['git','rev-parse','main'],text=True).strip(),private_data_pushed=False)
    emit(REPORT/'artifact-seal.json',meta)
    public=[p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.')]
    atomic(REPORT/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in public).encode())
    print('Phase10D final sealed',len(public),'public entries;',len(files),'private entries',flush=True)


def check():
    seal=read(REPORT/'artifact-seal.json');p=ROOT/seal['private_manifest']
    if sha(p)!=seal['private_manifest_sha256']:raise ValueError('Private hash manifest changed')
    for rel,h in {**read(p)['files'],**seal['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('Sealed Phase10D file changed: '+rel)
    for line in (REPORT/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1);p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=h:raise ValueError('Public seal changed: '+rel)
    print('Phase10D final seal PASS',flush=True)


if __name__=='__main__':
    if '--check' in sys.argv:check()
    else:run()
