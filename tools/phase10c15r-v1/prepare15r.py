"""Fresh CPU-only binding;no historical output is modified."""
import subprocess
from common15r import *


def run():
    if RUNR.exists() or REPORTR.exists():raise FileExistsError('Fresh recovery destination required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT_R or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH_R:raise ValueError('Expected parent commit and experimental branch')
    prior,reused,history=verify_prior();prepared,train,dev,_,reg=training_context();byid={r['image_id']:r for r in train};indices=list(range(5,25));validate_indices(indices)
    from common6 import candidates
    from masks6 import validate_bundle
    frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,obs)
    if len(bundle['records'])!=41 or len(prior['photo_ids'])!=24 or len({byid[i]['identity_group'] for i in prior['photo_ids']})!=24:raise ValueError('24/41/photo-group contract changed')
    for i in prior['photo_ids']:
        row=byid[i];require_training_row(row,train,dev)
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Photo bytes changed')
    files={rel:h for rel,h in prior['files'].items() if not (ROOT/rel).is_relative_to(HERE15)}
    for p in [*HERE15.glob('*.py'),*HERER.glob('*.py'),PRIOR_RUN/'authorized-config.json',PRIOR_RUN/'initialization.json',PRIOR_REPORT/'recovery-proposal.json',ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/modeling_owlv2.py']:
        files[str(p.relative_to(ROOT))]=sha(p)
    for r in reused:
        files[r['file']]=r['sha256'];files[r['raw_file']]=r['raw_sha256']
    for rel,h in files.items():
        if sha(ROOT/rel)!=h:raise ValueError('Pinned data/environment mismatch before GPU')
    RUNR.mkdir(parents=True);REPORTR.mkdir(parents=True)
    cfg=dict(schema='phase10c15r-one-recovery-v1',recovery_authorized=True,training_authorized=False,Phase10D_authorized=False,parent_commit=PARENT_R,branch=BRANCH_R,approval='Current user Phase10C.15R explicitly authorizes20additional forwards,25cumulative;one attempt,zero updates',files=files,photo_ids=prior['photo_ids'],photo_indices=indices,reused_records=reused,prior_forwards=5,limits=dict(new_forwards=20,cumulative_forwards=25,seconds=180,VRAM_bytes=MAX_VRAM,output_bytes=MAX_BYTES,optimizer_updates=0),seed=6106,gradient_scale=128.,head=prior['head'],head_binding_sha256=prior['head_binding_sha256'],original_checkpoint_sha256=prior['original_checkpoint_sha256'],initial_model_sha256=read(PRIOR_RUN/'initialization.json')['model_state']['sha256'],trainable_keys=prior['trainable_keys'],calibration_keys=prior['calibration_keys'],loss_implementation='Pinned repaired tools/phase10c15-v1/loss15.py;no variants or coefficient changes',automatic_retry=False,evaluation_authorized=False)
    emit_r(RUNR/'authorized-config.json',cfg);emit_r(REPORTR/'preflight.json',dict(status='PASS',parent_commit=PARENT_R,prior_seals=history,reused_records=reused,photo_indices=indices,known_groups=24,source_regions=41,configuration_sha256=sha(RUNR/'authorized-config.json'),original_checkpoint_sha256=cfg['original_checkpoint_sha256'],head=cfg['head'],first_attempt_peak_VRAM=None,first_attempt_terminal_model_hash=None,new_GPU_forwards=0,training_authorized=False))
    print('C15R CPU input preflight PASS;four reused,indices5–24 only;no GPU initialized',flush=True)


if __name__=='__main__':run()
