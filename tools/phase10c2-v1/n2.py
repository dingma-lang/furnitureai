"""Offline preparation paths. No model/optimizer worker exists in this phase."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/open-images/preparation/phase10c2-v1'
REPORT=ROOT/'reports/phase10c2-v1'
sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
from c10 import read,save,sha,manifest,CLASSES

def digest(value):
    import hashlib,json
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def optimizer_member(row):
    return row.get('historical_partition')=='train' and row.get('internal_partition')=='optimization'
