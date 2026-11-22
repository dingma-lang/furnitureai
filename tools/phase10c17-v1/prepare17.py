"""CPU-only inputs,original initialization and full-sequence contract."""
import subprocess
from common17 import *
from balanced16 import catalog,CLASS_KEYS
from common8 import grouped_epoch


def run():
    if RUN17.exists() or REPORT17.exists():raise FileExistsError('Fresh prepared smoke destination required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT17 or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH17:raise ValueError('Expected parent and dedicated branch')
    public=verify_seal(REPORT16/'SHA256SUMS');meta=read(REPORT16/'artifact-seal.json');private=ROOT/meta['private_manifest']
    if sha(private)!=meta['private_manifest_sha256']:raise ValueError('C16 private manifest changed')
    sealed=read(private)
    for rel,h in sealed['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C16 private artifact changed')
    for rel,h in meta['code_sha256'].items():
        if sha(ROOT/rel)!=h:raise ValueError('C16 code changed')
    if read(REPORT16/'tests-v2.json')['CPU_passed']!=150:raise ValueError('Completed C16 CPU validation required')
    proposed=read(REPORT16/'next-approval-request.json');prepared,train,dev,_,reg=training_context();groups=grouped_epoch(train,0);photos=read(DATA16/'proposed-smoke-photos.json')
    if sha(DATA16/'proposed-smoke-photos.json')!=proposed['photo_manifest_sha256'] or [r['image_id'] for r in photos['photos']]!=[r['image_id'] for g in groups for r in g] or photos['group_sizes']!=[4]*14+[3]:raise ValueError('Approved sequence differs')
    for i,g in enumerate(groups,1):
        if [r['image_id'] for r in g]!=read(C8/'B-corrected'/('update-'+str(i).zfill(3)+'.json'))['source_order']:raise ValueError('C8 first-epoch ordering mismatch')
    from common6 import candidates
    from masks6 import validate_bundle
    frozen=candidates();obs={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen};bundle=read(SOURCE/'source-negative-evidence.json');validate_bundle(bundle,prepared['rows'],frozen,obs)
    if digest(catalog(bundle['records'],train))!=digest(read(DATA16/'catalog.json')) or len(bundle['records'])!=41:raise ValueError('Source catalog changed')
    for row in train:
        require_training_row(row,train,dev)
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Training photo changed')
        for a in row['annotations']:
            box=a['box']
            if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('Annotation frame invalid')
    files=dict(read(ROOT/'data/open-images/preparation/phase10c10-v1/input-contract.json')['files'])
    for p in [*HERE16.iterdir(),*HERE17.glob('*.py'),REPORT16/'future-experiment-disabled.json',DATA16/'catalog.json',DATA16/'proposed-smoke-photos.json']:
        if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    for row in train:files[row['path']]=row['sha256']
    for r in frozen:files[str((SOURCE/'observations'/(r['candidate_id']+'.json')).relative_to(ROOT))]=sha(SOURCE/'observations'/(r['candidate_id']+'.json'))
    expected=read(C8/'binding.json')['configs']['B-corrected']
    if sha(BASE/'model.safetensors')!=expected['base_sha256']:raise ValueError('Original model fingerprint mismatch')
    RUN17.mkdir(parents=True);REPORT17.mkdir(parents=True)
    for name in ('photos','groups','telemetry'):(RUN17/name).mkdir()
    cfg=dict(schema='phase10c17-one-full-sequence-zero-update-smoke-v1',smoke_authorized=True,training_authorized=False,Phase10D_authorized=False,approval='Current user C17:one59forward/295gradient/180second smoke only',parent=PARENT17,files=files,limits=dict(forwards=59,gradient_calculations=295,seconds=180,VRAM_bytes=MAX_VRAM17,output_bytes=MAX_OUTPUT17,optimizer_updates=0),group_order=[[r['image_id'] for r in g] for g in groups],group_sizes=[len(g) for g in groups],training_rows_sha256=digest(train),seed=6106,scale=128.,initialization='Original pretrained OWLv2 only;no adapted checkpoint',original_checkpoint_sha256=expected['base_sha256'],fine_tuned_checkpoint=None,optimizer_settings_unused=expected,trainable_keys=sorted(HEAD_KEYS),classification_keys=sorted(CLASS_KEYS),trainable_scalars=1579526,calibration_trainable=True,source_catalog_sha256=sha(DATA16/'catalog.json'),automatic_retry=False)
    emit17(RUN17/'authorized-config.json',cfg);emit17(REPORT17/'preflight.json',dict(status='PASS',C16_public_files=public,C16_private_files=len(sealed['files']),training_photos=59,known_source_groups=56,positive_boxes=182,source_regions=41,accumulation_groups=15,group_sizes=cfg['group_sizes'],original_checkpoint_sha256=expected['base_sha256'],original_initialization_only=True,calibration_trainable=True,training_authorized=False,model_forwards=0))
    print('C17 CPU input preflight PASS;original base,59photos/182positives/41regions;15C8groups',flush=True)


if __name__=='__main__':run()
