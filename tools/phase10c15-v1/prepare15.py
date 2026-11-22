"""CPU preflight and exact approval binding; never initializes a GPU/model."""
import subprocess
from common15 import *


def run():
    if RUN15.exists() or REPORT15.exists():raise FileExistsError('Fresh approved C15 destination required;no overwrite')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:raise ValueError('Expected parent and dedicated experimental branch')
    public=verify_seal(REPORT14/'SHA256SUMS');seal_record=read(REPORT14/'artifact-seal.json');seal=ROOT/seal_record['private_seal']
    if sha(seal)!=seal_record['private_seal_sha256']:raise ValueError('C14 private seal changed')
    private=verify_seal(seal);prepared,train,dev,native,reg=training_context();proposal=read(REPORT14/'next-experiment-proposal.json');old=read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json')
    from masks6 import validate_bundle
    from common6 import candidates
    frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,obs)
    if len(bundle['records'])!=41:raise ValueError('Exact41 source regions required')
    photo_manifest=SOURCE/'proposed-smoke-photos.json';ids=[r['image_id'] for r in read(photo_manifest)['photos']];rows=[{r['image_id']:r for r in train}[i] for i in ids]
    if len(ids)!=24 or len(set(ids))!=24 or {e['image_id'] for e in bundle['records']}!=set(ids):raise ValueError('Exact24 source-region photos required')
    for row in rows:
        require_training_row(row,train,dev)
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Optimization photo changed')
    expected=proposal['next_approval'];head=expected['head'];checkpoint=ROOT/head['path']
    if sha(BASE/'model.safetensors')!=expected['base_sha256'] or sha(checkpoint)!=head['sha256'] or sha(photo_manifest)!=expected['photo_manifest_sha256'] or sha(SOURCE/'source-negative-evidence.json')!=proposal['training_data']['evidence_sha256']:raise ValueError('Approved model/photo/evidence fingerprint mismatch')
    files=dict(read(ROOT/'data/open-images/preparation/phase10c10-v1/input-contract.json')['files'])
    for path in [checkpoint,C11_RUN/'authorized-config.json',SOURCE/'source-negative-evidence.json',photo_manifest,ROOT/proposal['training_data']['manifest'],REGISTRY,NATIVE,ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/modeling_owlv2.py',ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/image_processing_owlv2.py',*HERE15.glob('*.py')]:files[str(path.relative_to(ROOT))]=sha(path)
    for row in rows:files[row['path']]=row['sha256']
    for r in frozen:files[str((SOURCE/'observations'/(r['candidate_id']+'.json')).relative_to(ROOT))]=sha(SOURCE/'observations'/(r['candidate_id']+'.json'))
    RUN15.mkdir(parents=True);REPORT15.mkdir(parents=True)
    cfg=dict(schema='phase10c15-one-zero-update-gradient-smoke-v1',smoke_authorized=True,training_authorized=False,Phase10D_authorized=False,approval='Current user Phase10C.15 authorization;exactly one smoke,not training',parent_commit=PARENT,branch=BRANCH,files=files,photo_ids=ids,photo_rows_sha256=digest(rows),source_evidence_sha256=digest(bundle),head=head,head_binding_sha256=sha(C11_RUN/'authorized-config.json'),original_checkpoint_sha256=expected['base_sha256'],trainable_keys=sorted(TRAIN_KEYS),calibration_keys=sorted(CALIBRATION_KEYS),gradient_scale=128.,seed=6106,limits=dict(forwards=24,seconds=180,VRAM_bytes=MAX_VRAM,output_bytes=MAX_BYTES,optimizer_updates=0),initialization='Original OWLv2 base plus exact saved C11 final head,as proposed in C14',variants=['Existing corrected source-negative focal','BCE-with-logits on exactly the same added Chair-negative mask;alpha_negative=.75/nGT'],category_order=list(CLASSES),evaluation_authorized=False,automatic_retry=False)
    emit(RUN15/'authorized-config.json',cfg)
    emit(REPORT15/'preflight.json',dict(status='PASS',C14_public_files=public,C14_private_files=private,training_photos=24,training_source_groups=len({r['identity_group'] for r in rows}),active_positive_boxes=sum(not a['ignore'] for r in rows for a in r['annotations']),trusted_negative_regions=41,original_checkpoint_sha256=cfg['original_checkpoint_sha256'],head=head,configuration_sha256=sha(RUN15/'authorized-config.json'),scoring='Owlv2ClassPredictionHead returns pre-sigmoid shifted/scaled cosine logits;BCE-with-logits appropriate',normalization='Same per-image active GT count and negative alpha0.75;positive/native Focal gamma2 unchanged',GPU_forwards=0,optimizer_updates=0))
    print('C15 CPU PRECHECK PASS;24optimization photos/41regions;C11 head;0forwards',flush=True)


if __name__=='__main__':run()
