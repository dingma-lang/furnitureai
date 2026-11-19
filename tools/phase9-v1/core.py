"""Phase9 inference-only boundaries and exclusive versioned artifacts."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/open-images/experiments/phase9-multiview-v1'
PUBLIC=ROOT/'reports/phase9-multiview-v1'
OWL=ROOT/'models/owlv2-original-baseline-20261103'
SIGLIP=ROOT/'models/siglip2-base-patch16-224-phase2a-v1'
HEAD=ROOT/'data/abo/experiments/phase3-gpu-v1/final-selected/best.pt'
REGISTRY=ROOT/'data/open-images/experiments/phase5c-v1/selected-checkpoint-registry.json'
OLD_LOCK=ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json'
READINESS=ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
SETTINGS=dict(prompts=['a photo of a '+c for c in CLASSES],ap_score_floor=.001,operating_score=.15,nms_iou=.5,max_detections=300,precision='float32',ap='101-point interpolated',iou_thresholds=[.5,.75],seed=20261105)
PINS={str((OWL/'model.safetensors').relative_to(ROOT)):'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7',
 str((SIGLIP/'model.safetensors').relative_to(ROOT)):'612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b',
 str(HEAD.relative_to(ROOT)):'51097ebf696a7765d023c8c02a9e4ea14965e0da04877cdd65be4416b9c13f0d'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def space():
 used=sum(p.stat().st_blocks*512 for b in (OUT,HERE,PUBLIC) for p in b.rglob('*') if p.is_file())
 if used>480*2**20:raise RuntimeError('Phase9 512MiB budget headroom exhausted; stop')
 return used
def save(p,value):
 space();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def code_hashes():return {p.name:sha(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')}
def checked(p):
 p=Path(p)
 if not p.exists():return None
 if sha(p)!=p.with_suffix(p.suffix+'.sha256').read_text().strip():raise ValueError('Incomplete or changed artifact '+str(p))
 return read(p)
def sealed(p,value):
 save(p,value)
 with Path(p).with_suffix(Path(p).suffix+'.sha256').open('x') as f:f.write(sha(p)+'\n')
