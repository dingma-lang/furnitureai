"""Shared offline Phase 3 primitives; completed code is imported read-only."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/abo/experiments/phase3-preparation-v1'
WEAK=ROOT/'data/abo/experiments/weak-training-v1'
PREP=ROOT/'data/abo/training-prep/phase1d-v1/proposal-v2'
META=ROOT/'data/abo/metadata/phase1-v1'
spec=importlib.util.spec_from_file_location('phase3_frozen_weak_helpers',ROOT/'tools/weak-training-v1/pipeline.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
sha,read,save,atomic,fingerprint=w.sha,w.read,w.save,w.atomic,w.fingerprint
CLASSES=w.CLASSES
IMAGE_DEST=ROOT/'data/abo/image-pilots/phase3-expansion-v1'
SEED=20261105

def verify_plan(out=OUT):
 lock=read(out/'preparation-lock.json')
 for rel,h in lock['source_files'].items():
  if sha(ROOT/rel)!=h:raise ValueError('Pinned source changed: '+rel)
 for name,h in lock['artifacts'].items():
  if sha(out/name)!=h:raise ValueError('Frozen Phase 3 artifact changed: '+name)
 return lock

def safe_output(path,parent):
 path=Path(path).resolve()
 if not path.is_relative_to(parent.resolve()) or path==parent.resolve():raise ValueError('Output outside isolated root')
 return path
