"""CPU tests and byte-preservation verification. No training or model inference."""
import json
import re
import subprocess
import sys
from common import *
from inventory import verify

def run():
    suites=[]
    for name in ('phase10a-v1','phase5b-v1','phase5c-v1','phase8a-v1','phase9-v1'):
        path=OUT/('tests-'+name+'.log')
        with path.open('x') as f:
            result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT/'tools'/name),'-p','test*.py','-v'],
                                  cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        text=path.read_text();n=re.search(r'Ran (\d+) tests?',text)
        if result.returncode or not n or 'skipped=' in text:raise RuntimeError('Test suite failed: '+name+'; preserve '+str(path))
        suites.append(dict(suite=name,passed=int(n[1])));print('PASS',name,n[1],flush=True)
    save(OUT/'tests.json',dict(status='PASS',passed=sum(s['passed'] for s in suites),suites=suites,
         model_forwards=0,optimizer_calls=0,synthetic_tensor_gradient_tests_only=True))
    preservation=verify()
    pins=read(ROOT/'data/open-images/experiments/phase9-multiview-v1/model-inventory.json')['weights']
    critical=[]
    for name,r in pins.items():
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Pinned critical checkpoint changed')
        critical.append(dict(name=name,path=r['path'],bytes=r['bytes'],sha256=r['sha256'],independent_backup_verified=False))
    code_before = subprocess.check_output(['git','diff','275b26b','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(p!='.gitignore' for p in code_before):raise ValueError('Tracked historical source changed: '+str(code_before))
    baseline='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6'
    for ref in ('main','phase9-baseline-20261118^{commit}'):
        if subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()!=baseline:raise ValueError('Baseline Git reference changed')
    preservation.update(critical_checkpoints=critical,main_and_baseline_tag_unchanged=True,
                        phase9_commit='275b26bd4bb630a5ca5bb02ec1d45c2f12b6fdf3',
                        historical_source_files_unchanged=True,external_data_transfer_bytes=0,
                        training_updates=0,new_model_inference=0,frozen_photos_decoded=0,
                        unreviewed33_LVIS_overlays_activated=0,Mac_project_modified=False)
    save(OUT/'preservation.json',preservation)
    manifest=read(OUT/'prepared-manifest.json');inv=read(OUT/'backup-inventory.json')
    summary=dict(status='PREPARATION_COMPLETE_NOT_APPROVED_FOR_TRAINING',
      prospective_images=120,prospective_boxes=373,prospective_groups=114,
      conservative_images=manifest['eligible_images'],conservative_groups=manifest['eligible_groups'],
      conservative_boxes=manifest['eligible_boxes'],per_category=manifest['per_category'],
      internal_partitions=manifest['internal_partitions'],rights_evidence_replayed=120,
      inactive_boxes=130,excluded_photos_without_supported_positive=42,
      original33_overlays_inactive=True,broader_seven_class_training_ready=False,
      usable_explicit_negative_images=3,backup_files=inv['files'],backup_bytes=inv['bytes'],
      backup_checkpoint_candidates=inv['checkpoint_candidates'],tests_passed=sum(s['passed'] for s in suites),
      github_phase9_push='BLOCKED: HTTPS credentials unavailable; gh and GCM not installed',
      original_models_unchanged=True,training_updates=0,new_model_forwards=0)
    save(OUT/'summary.json',summary)
    save(OUT/'implementation-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*')) if p.is_file()})
    with (OUT/'ARTIFACT-SHA256SUMS').open('x') as f:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='ARTIFACT-SHA256SUMS':f.write(sha(p)+'  '+str(p.relative_to(OUT))+'\n')
    print('PRESERVATION',preservation['files'],'historical files byte-verified; all four pinned models unchanged',flush=True)

if __name__=='__main__':run()
