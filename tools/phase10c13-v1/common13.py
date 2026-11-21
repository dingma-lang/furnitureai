"""C13 uses the sealed C12 implementation and its previously prepared destination."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c12-v1'))
from common12 import *
HERE13=Path(__file__).resolve().parent
OUT13=ROOT/'data/open-images/preparation/phase10c13-v1'
REPORT13=ROOT/'reports/phase10c13-v1'
AUTHORIZED_DECIMAL=78.92715856


def verify_inputs13():
    from config12 import check_spec
    check_spec(read(ROOT/'reports/phase10c12-v1/experiment-disabled.json'))
    prepared,train,dev=verify_preparation()
    if not all(r['historical_partition']=='train' and r['internal_partition']=='optimization' for r in train):raise ValueError('Only exact optimization photos permitted')
    for key in ('image_id','sha256','identity_group'):
        if {r[key] for r in train}&{r[key] for r in dev}:raise ValueError('Training/internal-development leakage')
    return prepared,train,dev
