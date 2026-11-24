"""Exact approved three-photo scope; no modifications to historical datasets."""
import datetime
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/licensed-rooms/image-pilots/modern-room-screening-v1'
REPORT=ROOT/'reports/phase10e16-v1'
MANIFEST=ROOT/'reports/phase10e15-v1/proposed-photo-acquisition.json'
PIN='22daba25dec7227bfd7697f454f0cdb3f668e332710e856e972d4f514b488e17'
IDS=('commons-page:150832261','commons-page:178637758','commons-page:195256255')
CAP=60000000
IMAGE_CAP=12*2**20
DISK=128*2**20
UA='FurnitureAI-ModernRoomSuitability/10E.16 (https://github.com/dingma-lang/furnitureai; exact approved three-photo inspection)'
sys.path[:0]=[str(ROOT/'src')]
from furniture_bench.persistence import atomic_write


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p,kind='sha256'):
    h=hashlib.new(kind)
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rows():
    if sha(MANIFEST)!=PIN:raise ValueError('Approved manifest SHA256 mismatch')
    m=read(MANIFEST);r=m['records']
    if tuple(x['source_id'] for x in r)!=IDS or m['exact_photo_count']!=3:
        raise ValueError('Manifest differs from exact three approved IDs')
    if sum(x['expected_bytes'] for x in r)!=21251887 or any(x['expected_bytes']>IMAGE_CAP for x in r):
        raise ValueError('Approved object size differs')
    return r


def allocated():
    total=0;seen=set()
    for root in (OUT,REPORT,HERE):
        for p in ([root,*root.rglob('*')] if root.exists() else []):
            try:s=p.lstat()
            except FileNotFoundError:continue  # A live browser can remove its temporary between enumeration and stat.
            if stat.S_ISLNK(s.st_mode):raise ValueError('Symlinks are not allowed in pilot outputs')
            key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total


def space(extra=0):
    if allocated()+extra+4*2**20>DISK:raise RuntimeError('128MiB allocation limit/reserve')


def save(path,payload):
    path=Path(path)
    if not any(path.resolve().is_relative_to(root.resolve()) for root in (OUT,REPORT)):
        raise PermissionError('New E16 pilot/report outputs only')
    space(2*len(payload)+65536)
    atomic_write(path,lambda f:f.write(payload),overwrite=False)


def emit(path,value):save(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
