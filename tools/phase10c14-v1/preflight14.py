"""Verify historical claims without selecting training evidence from dev errors."""
import torch
from common14 import *


def run():
    if RAW14.exists() or REPORT14.exists():raise FileExistsError('Fresh C14 audit version required')
    public=verify_seal(ROOT/'reports/phase10c13-v1/SHA256SUMS');record=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');seal=ROOT/record['private_seal']
    if sha(seal)!=record['private_seal_sha256']:raise ValueError('C13 private seal changed')
    private=verify_seal(seal);old=verify_history()[0];c8=verify_seal(C8/'ARTIFACT-SHA256SUMS');prepared,train,dev,photos,reg=training_context()
    originals=original_heads();distances={};checkpoints={}
    for label,root in [('c8_corrected',C8/'B-corrected'),('c11',C11_RUN),('c13',RUN13)]:
        done=read(root/'complete.json');p=(C8 if label=='c8_corrected' else root)/done['final_checkpoint']['file']
        if sha(p)!=done['final_checkpoint']['sha256']:raise ValueError('Historical checkpoint changed')
        data=torch.load(p,map_location='cpu',weights_only=True)
        if data['updates']!=75 or any(int(v['step'])!=75 for v in data['optimizer']['state'].values()):raise ValueError('Historical update count mismatch')
        if label!='c8_corrected' and any(not torch.equal(data['weights'][n],originals[n]) for n in CALIBRATION_KEYS):raise ValueError('Calibration freeze mismatch')
        distances[label]=sum(float((data['weights'][n].double()-originals[n].double()).square().sum()) for n in PROJECTION_KEYS)**.5;checkpoints[label]=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),updates=75)
    # Historical development results are read for verification only, never passed
    # to the training-query/candidate pipeline.
    result=read(ROOT/'reports/phase10c13-v1/results.json');a=result['comparison']['frozen_calibration']['headline'];b=result['comparison']['projection_regularized']['headline'];reduction=1-distances['c13']/distances['c11']
    if abs(reduction-.8447750272787125)>1e-9 or not (b['fp']<a['fp'] and b['recall']<a['recall'] and b['ap75']<a['ap75']):raise ValueError('Historical claims do not match saved evidence')
    matched=result['comparison']['projection_regularized']['precision_at_matched_recall']['original']['precision'];control=result['comparison']['frozen_calibration']['precision_at_matched_recall']['original']['precision']
    if matched>control:raise ValueError('Historical matched-recall claim mismatch')
    for path,expected in read(NATIVE)['metadata_sources'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Existing native metadata source changed')
    RAW14.mkdir(parents=True);REPORT14.mkdir(parents=True);records=state_records()
    save(RAW14/'cache-index.json',dict(records=records,canonical_training_rows=train,photo_keys={r['image_id']:digest(r['image_id'])[:16] for r in train},native_inventory_sha256=sha(NATIVE),registry_sha256=sha(REGISTRY)))
    save(REPORT14/'historical-verification.json',dict(status='PASS',C13_public_files=public,C13_private_files=private,C8_private_files=c8,C11=old,checkpoints=checkpoints,joint_projection_displacements=distances,C13_reduction_vs_C11=reduction,FP_decreased_but_recall_AP75_regressed=True,matched_recall_precision_did_not_improve=True,calibration_frozen_C11_C13=True,
        historical_dev_results_scope='Verification of completed claims only;no dev images/errors supplied to candidate selection or objective calibration',training_photos=len(train),known_training_groups=len({r['identity_group'] for r in train}),cached_training_photos=24,uncached_training_photos=35,unique_cached_states=len(STATES),saved_probe_array_files=len(records),new_model_forwards=0,training_updates=0))
    print('C14 historical verification PASS;24/59 training photos have13 cached states;no inference',flush=True)


if __name__=='__main__':run()
