"""Full bookkeeping rehearsal using scalar checkpoint fixtures. No optimizer or model."""
from p10 import *
from protocol import FIT_IDS
from state import Ledger
from worker import grouped_epoch

def run():
    root=RAW/'dry-run-v1'
    if root.exists():raise FileExistsError('Preserve earlier rehearsal')
    m=read(RAW/'eligibility-manifest.json');rows=[r for r in m['rows'] if r['internal_partition']=='optimization']
    counts={};bindings=dict(fixture_only=True,dataset_sha256=sha(RAW/'eligibility-manifest.json'))
    for fit in FIT_IDS:
        ledger=Ledger(root,bindings,clock=lambda:0);ledger.start(fit);ledger.resume();cursor=0;seen=[]
        for epoch in range(5):
            for group,batch in enumerate(grouped_epoch(rows,20261118,epoch)):
                assert all(r['historical_partition']=='train' and r['internal_partition']=='optimization' for r in batch)
                ledger.reserve(cursor);cursor+=1
                receipt=ledger.checkpoint(dict(cursor=cursor,epoch=epoch,next_group=group+1,
                    fixture_only=True,model_weights_updated=False,images=[r['image_id'] for r in batch]))
                ledger.committed(receipt);seen.extend(r['image_id'] for r in batch)
                if cursor==7:
                    ledger=Ledger(root,bindings,clock=lambda:0);ledger.start(fit);ledger.resume()
        assert cursor==75 and len(seen)==59*5
        counts[fit]=dict(simulated_calls=cursor,actual_optimizer_calls=0,model_parameter_changes=0,
            unique_optimization_photos=len(set(seen)),training_image_uses=len(seen),interrupted_and_resumed=True)
    save(RAW/'dry-run-result.json',dict(status='PASS',fits=counts,actual_optimizer_calls=0,GPU_model_forwards=0,
         no_weights_updated=True,scope='Bookkeeping and source-membership rehearsal only; CUDA behavior untested'))
    print('DRY RUN PASS:2x75 simulated updates,59imagesx5epochs each, recovery exercised; zero model/optimizer calls',flush=True)

if __name__=='__main__':run()
