"""Isolated Phase10C. Default entry points are CPU-only and append-only."""
import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c-v1'
REPORT=ROOT/'reports/phase10c-v1'
PREP=ROOT/'data/open-images/preparation/phase10b-v1'
RUN=ROOT/'data/open-images/experiments/phase10b-heads-v1'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink()
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def manifest():
    p=PREP/'eligibility-manifest.json';m=read(p)
    if sha(p)!=read(RUN/'smoke-pass.json')['binding']['dataset_sha256']:raise ValueError('Historical dataset changed')
    return m
def internal_rows():
    rows=[r for r in manifest()['rows'] if r['historical_partition']=='train' and r['internal_partition']=='inner_development']
    if len(rows)!=19 or sum(sum(not a['ignore'] for a in r['annotations']) for r in rows)!=61:raise ValueError('Exact19/61 internal membership required')
    return rows
def source_path(row):
    if row not in internal_rows():raise PermissionError('No external or reserved photograph access')
    p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Source path/hash mismatch')
    return p
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*.py'))}
