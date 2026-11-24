"""Targeted evidence verification, not another codebase audit or experiment."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from prepare15 import ROOT,OUT,sha,emit

PARENT='f9a06a6103428bee4915acbde03862c498b66c92'
MAIN='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6'


def preserve():
    inputs=json.loads((OUT/'input-preservation.json').read_text())
    for r in inputs['records']:
        if sha(ROOT/r['path']) != r['sha256']:raise ValueError('Historical input changed: '+r['path'])
    lineage=json.loads((ROOT/'data/open-images/preparation/phase10e6-v1/source-lineage-v1.json').read_text())['photos']
    eligible=json.loads((ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json').read_text())['rows']
    photos={r['image_id']:(ROOT/r['path'],r['sha256']) for r in lineage+eligible}
    wc=json.loads((ROOT/'reports/phase10e11-v1/image-and-rights-manifest.json').read_text())
    for r in wc.get('records',wc.get('images',[])):photos['wellcome:'+r['image_id']]=(ROOT/r['file'],r['sha256'])
    integration=json.loads((ROOT/'reports/phase4-room-comparison-v1/input-lock.json').read_text())['rooms']
    for r in integration:photos['integration:'+r['image_id']]=(Path(r['file']),r['sha256'])
    for image_id,(path,expected) in photos.items():
        if sha(path)!=expected:raise ValueError('Historical photo changed: '+image_id)
    previous=json.loads((ROOT/'reports/phase10e14-v1/artifact-seal.json').read_text())
    for rel,expected in previous['files'].items():
        original=(ROOT/rel).read_bytes() if '/private/' in rel else subprocess.check_output(['git','show',PARENT+':'+rel],cwd=ROOT)
        if hashlib.sha256(original).hexdigest()!=expected:raise ValueError('Parent evidence changed')
    # No pre-existing tracked file other than the new report allowlist may change.
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',PARENT],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',PARENT],cwd=ROOT,text=True).splitlines())
    if changed & (old-{'.gitignore'}):
        raise ValueError('Historical tracked content changed: '+repr(sorted(changed & (old-{'.gitignore'}))))
    if subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()!=MAIN:raise ValueError('main changed')
    return dict(status='PASS',historical_input_files=len(inputs['records']),historical_photos_verified=len(photos),
        historical_annotations_and_checkpoints_modified=False,all_preexisting_tracked_content_unchanged_except_gitignore=True,
        no_core_ML_code_changes=True,E14_baseline_not_rerun=True,new_images_downloaded=0,new_real_annotations=0,
        main=MAIN,note='Targeted source/photo and recorded parent-seal checks; not a new exhaustive checkpoint audit')


def seal():
    preservation=preserve();emit('preservation-final.json',preservation)
    paths=[ROOT/'.gitignore',*sorted((ROOT/'tools/phase10e15-v1').glob('*')),*sorted(OUT.glob('*'))]
    paths=[p for p in paths if p.is_file() and p.name not in ('artifact-seal.json','SHA256SUMS')]
    emit('artifact-seal.json',dict(parent=PARENT,main=MAIN,files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    with (OUT/'SHA256SUMS').open('x') as f:
        for p in paths+[OUT/'artifact-seal.json']:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print('Phase15 sealed;',preservation['historical_photos_verified'],'historical photo hashes preserved')


def check():
    value=json.loads((OUT/'artifact-seal.json').read_text())
    for p,expected in value['files'].items():
        if sha(ROOT/p)!=expected:raise ValueError('Changed E15 evidence: '+p)
    preserve();print('Phase15 targeted preservation PASS; no model operations')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check']);args=p.parse_args()
    seal() if args.action=='seal' else check()
