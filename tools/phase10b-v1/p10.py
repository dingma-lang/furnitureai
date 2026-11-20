"""Versioned Phase10B preparation; no authorization implied by these files."""
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10b-v1'
REPORT=ROOT/'reports/phase10b-v1'
RUN=ROOT/'data/open-images/experiments/phase10b-heads-v1'
OLD=ROOT/'reports/phase10a-v1'
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from common import sha,read,CLASSES

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    with tmp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink()
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def code_hashes():
    files=[p for d in [HERE,ROOT/'tools/phase10a-v1'] for p in sorted(d.glob('*.py'))]
    files += [ROOT/'tools/phase9-v1'/n for n in ('core.py','fusion.py','metrics.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def image_path(row,part,manifest):
    if part not in ('optimization','inner_development') or row.get('internal_partition')!=part or row.get('historical_partition')!='train':raise PermissionError('Training-derived partitions only')
    if row!=next((r for r in manifest['rows'] if r['image_id']==row['image_id']),None):raise PermissionError('Forged row')
    p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Image identity changed')
    return p
