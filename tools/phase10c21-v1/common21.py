"""CPU-only C21 data preparation. No training/inference/downloader entry point."""
import sys,stat,zipfile,hashlib
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c20-v1'))
from common20 import *
HERE21=Path(__file__).resolve().parent
DATA21=ROOT/'data/open-images/preparation/phase10c21-v1'
REPORT21=ROOT/'reports/phase10c21-v1'
PARENT21='b0b74ddea184cc9065b37a6a70fe4d03210dbefa'
BRANCH21='experiment/phase10c21-hard-negatives-pretrained-preservation'
MAX_BYTES21=128*2**20
LOCK21=threading.RLock()


def allocated21():
    return sum(p.stat().st_blocks*512 for folder in (DATA21,REPORT21) if folder.exists() for p in folder.rglob('*'))


def atomic21(path,data):
    with LOCK21:
        path=Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (DATA21,REPORT21)):raise PermissionError('Outside C21 preparation roots')
        if path.exists():raise FileExistsError('Immutable C21 artifact exists')
        if allocated21()+2*((len(data)+4095)//4096)*4096+8192>MAX_BYTES21:raise RuntimeError('C21 128MiB output bound')
        p=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(p,path)


def emit21(path,value):atomic21(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())


def verify20_seals():
    count=verify_seal(REPORT20/'SHA256SUMS');meta=read(REPORT20/'artifact-seal.json');p=ROOT/meta['private_manifest']
    if sha(p)!=meta['private_manifest_sha256']:raise ValueError('C20 seal changed')
    for rel,h in {**read(p)['files'],**meta['code_sha256']}.items():
        if sha(ROOT/rel)!=h:raise ValueError('C20 historical artifact changed: '+rel)
    return dict(public_files=count,private_files=meta['private_files'])


def context21():
    cfg,prepared,train,dev,reg=inputs20();native=read(ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json');photos={p['image_id']:p for p in native['photos'] if p['image_id'] in {r['image_id'] for r in train}}
    if set(photos)!={r['image_id'] for r in train}:raise ValueError('Optimization inventory incomplete')
    for row in train:
        require_training_row(row,train,dev)
        if any(photos[row['image_id']][k]!=row[k] for k in ('sha256','identity_group','path','width','height')):raise ValueError('Source registration changed')
    return cfg,prepared,train,dev,reg,native,photos


def prepare21():
    if DATA21.exists() or REPORT21.exists():raise FileExistsError('Fresh C21 roots required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT21 or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH21:raise ValueError('C21 dedicated parent/branch required')
    seal=verify20_seals();cfg,*_=context21();DATA21.mkdir(parents=True);REPORT21.mkdir(parents=True);(DATA21/'contact-sheets').mkdir()
    emit21(REPORT21/'preflight.json',dict(status='PASS',C20_seal=seal,original_OWLv2_sha256=sha(BASE/'model.safetensors'),checkpoint28_sha256=sha(CHECKPOINT28),C20_final_checkpoint=read(RUN20/'complete.json')['final_checkpoint'],training_manifest_sha256=cfg['training_rows_sha256'],training_authorized=False,GPU_authorized=False,Phase10D_authorized=False,downloads=0))
    print('C21 CPU-only preflight PASS;C20/preserved model integrity verified',flush=True)


if __name__=='__main__':prepare21()
