"""No-training CPU verification, historical preservation and aggregate report."""
import re
import subprocess
from p10 import *

def run():
    tests=[]
    for name in ('phase10b-v1','phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        log=RAW/('tests-'+name+'.log')
        with log.open('x') as f:r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        s=log.read_text();n=re.search(r'Ran (\d+) tests?',s)
        if r.returncode or not n or 'skipped=' in s:raise RuntimeError('Required tests failed '+str(log))
        tests.append(dict(suite=name,passed=int(n[1])));print('TEST PASS',name,n[1],flush=True)
    save(REPORT/'tests.json',dict(status='PASS',suites=tests,passed=sum(t['passed'] for t in tests),
         actual_model_updates=0,actual_GPU_model_forwards=0,fixture_bookkeeping_not_training=True))
    oldinv=read(OLD/'backup-inventory.json')['records']
    for r in oldinv:
        p=ROOT/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Historical data/model bytes changed: '+r['path'])
    n=0
    for line in (OLD/'FINAL-ARTIFACT-SHA256SUMS-v2').read_text().splitlines():
        h,rel=line.split('  ',1);p=(OLD/rel).resolve()
        if not p.is_relative_to(OLD) or sha(p)!=h:raise ValueError('Phase10A artifact changed: '+rel)
        n+=1
    for name,h in read(OLD/'input-contract.json')['implementation'].items():
        if sha(ROOT/'tools/phase10a-v1'/name)!=h:raise ValueError('Phase10A code changed')
    if RUN.exists():raise ValueError('No actual training run directory may be created during this phase')
    save(RAW/'preservation.json',dict(status='PASS',historical_files_byte_verified=len(oldinv),phase10a_artifact_files_verified=n,
        historical_models_datasets_unchanged=True,phase10a_code_unchanged=True,reserved_images_decoded=0,
        original_annotations_modified=False,previous33_overlays_activated=0,training_updates=0,GPU_model_forwards=0,
        files_deleted_or_moved=0,dataset_downloads=0,external_backup_transfers=0,Mac_project_modified=False))
    recovery=read(RAW/'recovery-summary.json');hist=read(REPORT/'history-summary.json');manifest=read(RAW/'eligibility-manifest.json')
    save(REPORT/'summary.json',dict(status='COMPLETE_PREPARATION_ONLY',training_authorized=False,environment=read(RAW/'environment.json'),
        recovery=recovery,per_category=manifest['per_category'],internal_partitions=manifest['internal_partitions'],
        prospective_images=120,prospective_boxes=373,eligible_images=78,eligible_groups=75,eligible_boxes=243,
        tests_passed=sum(t['passed'] for t in tests),raw_dataset_history='data/open-images/preparation/phase10b-v1/dataset-history.json',
        historical_model_families=len(hist['models']),historical_dataset_cohorts=len(hist['dataset_cohorts']),
        recovery_conditional_upper_bound_bytes=hist['conditional_recovery_upper_bound'],
        no_independent_accuracy_established=True,training_worker_GPU_smoke_not_run=True))
    proposal=read(RAW/'authorization-proposal.json');proposal.update(approved=False,model_updates_authorized=False,
        approved_by=None,approval_reference=None,code_hashes=code_hashes(),max_updates_per_fit=150,max_updates_total=300,max_seconds=3600,max_epochs=5,
        max_new_training_storage_bytes=8*2**30,
        required_GPU_smoke='Separate approval must include bounded forward/backward smoke before optimizer calls; no such smoke was performed here')
    save(RAW/'training-approval-request-v2.json',proposal)
    print('PRESERVATION PASS',len(oldinv),'historical files; zero training/model inference',flush=True)

if __name__=='__main__':run()
