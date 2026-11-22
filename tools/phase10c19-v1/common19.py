"""C19 CPU-only guard repair and immutable recovery preflight outputs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase10c18-v1'))
from common18 import *
HERE19 = Path(__file__).resolve().parent
DATA19 = ROOT / 'data/open-images/preparation/phase10c19-v1'
REPORT19 = ROOT / 'reports/phase10c19-v1'
PARENT19 = 'ee298e96a7c21196bd2df89ec142e16829531b05'
BRANCH19 = 'experiment/phase10c19-guard-recovery-preflight'
CHECKPOINT28 = RUN18 / 'checkpoints/update-028-0520b171.pt'
CHECKPOINT28_SHA256 = '77081cb4ef7775a8f83c2b17475bf275847f9c0305f1984b66354a9880a2eefa'
MAX_BYTES19 = 128 * 2**20
WRITE_LOCK19 = threading.RLock()


def allocated19():
    return sum(p.stat().st_blocks * 512 for folder in (DATA19, REPORT19) if folder.exists() for p in folder.rglob('*'))


def atomic19(path, data):
    with WRITE_LOCK19:
        path = Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (DATA19, REPORT19)):
            raise PermissionError('Outside C19 output roots')
        if path.exists():
            raise FileExistsError('Immutable C19 artifact exists')
        if allocated19() + 2 * ((len(data)+4095)//4096)*4096 + 8192 > MAX_BYTES19:
            raise RuntimeError('C19 128MiB local diagnostic bound')
        pending = path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex[:8])
        with pending.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(pending, path)


def emit19(path, value):
    atomic19(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def verify18_seals():
    public = verify_seal(REPORT18/'SHA256SUMS')
    meta = read(REPORT18/'artifact-seal.json'); private = ROOT/meta['private_manifest']
    if sha(private) != meta['private_manifest_sha256']:
        raise ValueError('C18 private seal changed')
    for rel, h in {**read(private)['files'], **meta['code_sha256']}.items():
        if sha(ROOT/rel) != h:
            raise ValueError('C18 historical artifact changed: '+rel)
    if sha(CHECKPOINT28) != CHECKPOINT28_SHA256:
        raise ValueError('Checkpoint28 changed')
    return dict(public_files=public, private_files=meta['private_files'])


def prepare19():
    if DATA19.exists() or REPORT19.exists():
        raise FileExistsError('Fresh C19 output roots required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT19 or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH19:
        raise ValueError('C19 dedicated branch/parent required')
    seal=verify18_seals(); cfg,*_=inputs18()
    DATA19.mkdir(parents=True); REPORT19.mkdir(parents=True)
    (DATA19/'telemetry').mkdir()
    emit19(REPORT19/'preflight.json',dict(status='PASS',CPU_only=True,training_authorized=False,GPU_authorized=False,Phase10D_authorized=False,C18_seal=seal,checkpoint28_sha256=CHECKPOINT28_SHA256,training_rows_sha256=cfg['training_rows_sha256'],source_manifest_sha256=sha(SOURCE/'source-negative-evidence.json')))
    print('C19 CPU-only preflight; C18 immutable seals and checkpoint verified',flush=True)


if __name__=='__main__':prepare19()
