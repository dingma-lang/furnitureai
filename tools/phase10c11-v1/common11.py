"""One authorized C11 arm; preserve the prepared C10 implementation and outputs."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[2];HERE11=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase10c10-v1'))
from common10 import read,save,sha,digest,verify_seal,verify_inputs,C8,BASE,SOURCE,FUTURE,TRAIN_KEYS,CALIBRATION_KEYS,TRAIN_COUNT,HEAD_KEYS,subsets,load_npz,original_heads,tensor_sha,metric_module,quantiles
OUT11=ROOT/'data/open-images/preparation/phase10c11-v1'
REPORT11=ROOT/'reports/phase10c11-v1'

def verify_preflight():
    import subprocess
    if OUT11.exists() or REPORT11.exists() or FUTURE.exists():raise FileExistsError('Fresh C11 preparation and prepared experiment destination required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='experiment/phase10c11-frozen-calibration-ablation':raise ValueError('Dedicated branch required')
    public=verify_seal(ROOT/'reports/phase10c10-v1/SHA256SUMS');old=read(ROOT/'reports/phase10c10-v1/artifact-seal.json');p=ROOT/old['private_seal']
    if sha(p)!=old['private_seal_sha256']:raise ValueError('C10 private seal changed')
    private=verify_seal(p);prepared,train,dev=verify_inputs()
    OUT11.mkdir(parents=True);REPORT11.mkdir(parents=True)
    save(REPORT11/'input-preflight.json',dict(status='PASS',parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),C10_public_sealed_files=public,C10_private_sealed_files=private,
        original_checkpoint_sha256=sha(BASE/'model.safetensors'),training_photos=len(train),known_training_groups=len({r['identity_group'] for r in train}),positive_boxes=sum(not a['ignore'] for r in train for a in r['annotations']),development_photos=len(dev),
        original_data_manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),historical_weights_modified=False,future_destination_was_unused=True,
        only_intended_optimization_change='Freeze4 original class-head calibration tensors; remaining data/loss/order/optimizer identical to C8B',
        diagnostic_execution_differences=['Standalone initialization rechecks19 FP32 development photos;467 total forward ceiling versus448 for C8B which reused A reference','Detached logging adds Chair-only positive/negative loss components; original loss tensor/parts/gradients unchanged','CPU initial raw-output comparison against saved C8 original occurs before first optimizer update'],
        limits=dict(arms=1,lr=1e-5,max_epochs=5,max_updates=75,max_GPU_seconds_including_inference=600,max_VRAM_bytes=int(6.5*2**30)),Phase10D_authorized=False))
    print('C11 INPUT PREFLIGHT PASS59train/19diagnostic;C10seals preserved;fresh destination',flush=True)

if __name__=='__main__':verify_preflight()
