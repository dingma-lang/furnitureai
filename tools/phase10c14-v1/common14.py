"""Training-only cached-query audit. No inference or training entry point."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/phase10c13-v1'))
from common13 import *
RUN13=FUTURE
HERE14=Path(__file__).resolve().parent
RAW14=ROOT/'data/open-images/diagnostics/phase10c14-v1'
REPORT14=ROOT/'reports/phase10c14-v1'
NATIVE=ROOT/'data/open-images/preparation/phase10c5-v1/native-object-inventory.json'
REGISTRY=ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json'
STATES=[dict(key='original',label='Original pretrained · epoch 0',root=C11_RUN,epoch=0,corrected=True)]
for key,label,root,corrected in [('c8_control','C8 control',C8/'A-control',False),('c8_corrected','C8 corrected',C8/'B-corrected',True),('c11','C11 calibration frozen',C11_RUN,True),('c13','C13 L2-SP',RUN13,True)]:
    for epoch in (1,3,5):STATES.append(dict(key=key+'_e'+str(epoch),label=label+' · epoch '+str(epoch),root=root,epoch=epoch,corrected=corrected))


def training_context():
    prepared,train,dev=verify_inputs13();byid={r['image_id']:r for r in train}
    native=read(NATIVE);registry=read(REGISTRY)
    # Filter identities before inspecting any native labels or query evidence.
    photos={p['image_id']:p for p in native['photos'] if p['image_id'] in byid};reg={p['image_id']:p for p in registry['photos'] if p['image_id'] in byid and p['role']=='optimization'}
    if set(photos)!=set(byid) or set(reg)!=set(byid):raise ValueError('Complete canonical optimization-only source context required')
    for ident,row in byid.items():
        for p in (photos[ident],reg[ident]):
            if any(p[k]!=row[k] for k in ('image_id','sha256','identity_group','width','height','path')):raise ValueError('Native/registry source frame or identity mismatch')
    return prepared,train,dev,photos,reg


def require_training_row(row,train,dev):
    index={r['image_id']:r for r in train}
    if row!=index.get(row['image_id']) or row['historical_partition']!='train' or row['internal_partition']!='optimization':raise PermissionError('Only exact optimization-training photos may be audited or served')
    for key in ('image_id','sha256','identity_group'):
        if row[key] in {r[key] for r in dev}:raise PermissionError('Development/evaluation leakage')
    return True


def source_image(row):
    _,train,dev=verify_inputs13();require_training_row(row,train,dev);p=(ROOT/row['path']).resolve()
    if not p.is_relative_to(ROOT/'data') or sha(p)!=row['sha256']:raise ValueError('Training photo path/hash changed')
    return p


def state_records():
    prepared,train,dev,photos,reg=training_context();frozen_ids=[r['image_id'] for r in read(SOURCE/'proposed-smoke-photos.json')['photos']]
    if len(frozen_ids)!=24 or len(set(frozen_ids))!=24 or not set(frozen_ids)<={r['image_id'] for r in train}:raise ValueError('Exact24 optimization probe identities required')
    records=[]
    for state in STATES:
        for i,ident in enumerate(frozen_ids,1):
            folder=state['root']/'probes'/('epoch-'+str(state['epoch']));p=folder/(str(i).zfill(2)+'.npz');receipt=read(p.with_suffix('.json'))
            if receipt['image_id']!=ident:raise ValueError('Cached probe identity mismatch')
            if 'query_file_sha256' in receipt and sha(p)!=receipt['query_file_sha256']:raise ValueError('Cached query hash mismatch')
            records.append(dict(state=state['key'],epoch=state['epoch'],corrected=state['corrected'],image_id=ident,file=str(p.relative_to(ROOT)),sha256=sha(p),receipt_file=str(p.with_suffix('.json').relative_to(ROOT)),receipt_sha256=sha(p.with_suffix('.json'))))
    return records
