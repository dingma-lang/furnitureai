"""Offline preparation only. Native annotation coordinates and historical state remain immutable."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/phase8a-v1'
PILOT=ROOT/'data/lvis/image-pilots/phase7-v1'
META=ROOT/'data/lvis/metadata/phase7-v1'
RIGHTS=ROOT/'data/lvis/rights-audits/phase7-v1'
CLASSES=('sofa','chair','table','bed','desk','cabinet','lamp')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def commit(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def stable_box_id(image_id,annotation,occurrence=0):
 # OI has no native row ID. Preserve duplicate row occurrences instead of collapsing the old 'native-box' placeholders.
 payload=dict(source_mid=annotation['source_mid'],native_normalized_box=annotation['box'],flags=annotation.get('flags',{}))
 return 'OpenImages:'+image_id+':'+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]+':'+str(occurrence)
