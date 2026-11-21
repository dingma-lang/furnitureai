"""Phase C12 preparation paths; imports do not execute a model or touch CUDA."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c11-v1'))
from common11 import *
from common10 import named_sha,CLASSES,OUT
C11_RUN=FUTURE
HERE12=Path(__file__).resolve().parent
OUT12=ROOT/'data/open-images/preparation/phase10c12-v1'
REPORT12=ROOT/'reports/phase10c12-v1'
FUTURE=ROOT/'data/open-images/experiments/phase10c12-projection-preservation-v1'
PROJECTION_KEYS=('class_head.dense0.weight','class_head.dense0.bias')
PROJECTION_SHAPES=((512,768),(512,))


def verify_history():
    public=verify_seal(ROOT/'reports/phase10c11-v1/SHA256SUMS')
    rec=read(ROOT/'reports/phase10c11-v1/artifact-seal.json');private=ROOT/rec['private_seal']
    if sha(private)!=rec['private_seal_sha256']:raise ValueError('C11 private seal changed')
    count=verify_seal(private);prepared,train,dev=verify_inputs()
    done=read(C11_RUN/'complete.json');p=C11_RUN/done['final_checkpoint']['file']
    if sha(p)!=done['final_checkpoint']['sha256'] or done['updates']!=75 or not done['calibration_unchanged']:raise ValueError('C11 control checkpoint changed')
    return dict(C11_public_files=public,C11_private_files=count,original_checkpoint_sha256=sha(BASE/'model.safetensors'),C11_final_checkpoint_sha256=sha(p)),prepared,train,dev


def verify_preparation():
    prepared,train,dev=verify_inputs()
    if not (OUT12/'input-contract.json').exists():raise FileNotFoundError('Completed C12 CPU preparation required')
    for relative,expected in read(OUT12/'input-contract.json')['files'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('C12 input changed: '+relative)
    return prepared,train,dev
