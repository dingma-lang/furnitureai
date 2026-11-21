"""Compare future initial outputs with C11 originals without extra forwards."""
from common12 import *
from observe11 import compare_array_sets


def verify_initial_outputs12():
    records=[]
    for i in range(1,25):
        relative=Path('probes/epoch-0')/(str(i).zfill(2)+'.npz');a=read((FUTURE/relative).with_suffix('.json'));b=read((C11_RUN/relative).with_suffix('.json'))
        if a['image_id']!=b['image_id']:raise ValueError('Initial training probe identity mismatch')
        records.append(dict(role='optimization_probe',image_id=a['image_id'],arrays=compare_array_sets(load_npz(FUTURE/relative),load_npz(C11_RUN/relative),True)))
    newer=FUTURE/'evaluation/epoch-0';older=C11_RUN/'evaluation/epoch-0';new=read(newer/'raw-complete.json');old=read(older/'raw-complete.json')
    if new['rows_sha256']!=old['rows_sha256'] or len(new['records'])!=19:raise ValueError('Initial diagnostic partition mismatch')
    for a,b in zip(new['records'],old['records']):
        if a['image_id']!=b['image_id'] or a['source_sha256']!=b['source_sha256'] or sha(older/b['file'])!=b['sha256']:raise ValueError('Initial reference changed')
        records.append(dict(role='internal_development',image_id=a['image_id'],arrays=compare_array_sets(load_npz(newer/a['file']),load_npz(older/b['file']))))
    save(FUTURE/'initial-comparability.json',dict(status='PASS',records=records,extra_forwards=0,updates_before_check=0,regularizer_at_original_initialization=0.0))
