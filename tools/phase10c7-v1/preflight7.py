"""Read-only source/model verification; fresh versioned smoke output only."""
import subprocess,importlib.metadata
from common7 import *

def run():
    if RUN.exists() or REPORT.exists():raise FileExistsError('Fresh destination required; never repeat/overwrite prior smoke')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='experiment/phase10c7-query-negative-smoke':raise ValueError('Dedicated smoke branch required')
    proposal=read(ROOT/'reports/phase10c6-v1/training-proposal.json');original=manifest();photos=read(SOURCE/'proposed-smoke-photos.json')['photos']
    if sha(SOURCE/'proposed-smoke-photos.json')!=proposal['next_approval']['photo_manifest_sha256'] or sha(SOURCE/'source-negative-evidence.json')!=proposal['source_evidence_sha256'] or sha(SOURCE/'training-evidence-manifest.json')!=proposal['training_manifest_sha256']:raise ValueError('Prepared exact source scope changed')
    bundle=read(SOURCE/'source-negative-evidence.json');frozen=candidates();observations={r['candidate_id']:read(SOURCE/'observations'/(r['candidate_id']+'.json')) for r in frozen}
    from masks6 import validate_bundle
    validate_bundle(bundle,original['rows'],frozen,observations)
    if len(bundle['records'])!=41 or {e['image_id'] for e in bundle['records']}!={p['image_id'] for p in photos}:raise ValueError('Not exact41 regions /24 photos')
    if any(read(p).get('decision') in ('exclude','needs_recheck') for p in (SOURCE/'corrections').glob('*.json')):raise ValueError('Optional correction requires new evidence version first')
    model_files=read(ROOT/'reports/phase10a-v1/input-contract.json')['model_files'];files={str((BASE/name).relative_to(ROOT)):h for name,h in model_files.items()}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')})
    for p in (ROOT/'tools/phase10c6-v1').glob('*.py'):files[str(p.relative_to(ROOT))]=sha(p)
    for path in (SOURCE/'source-negative-evidence.json',SOURCE/'training-evidence-manifest.json',SOURCE/'proposed-smoke-photos.json',ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json',ROOT/'data/open-images/preparation/phase10c5-v1/review-registry.json',ROOT/'tools/phase10a-v1/expanded.py'):files[str(path.relative_to(ROOT))]=sha(path)
    for p in SOURCE.glob('INITIAL-SHA256SUMS'):
        for line in p.read_text().splitlines():
            h,rel=line.split('  ',1)
            if sha(ROOT/rel)!=h:raise ValueError('Phase10C.6 seal changed: '+rel)
    binding=dict(schema='phase10c7-zero-update-query-smoke-v1',GPU_smoke_authorized=True,approval_reference='User Phase10C.7 explicit24-forward zero-update GPU authorization',
        training_authorized=False,backward_authorized=False,optimizer_authorized=False,max_forward_passes=24,max_seconds=600,max_vram_bytes=MAX_VRAM,
        photos=photos,negative_regions=41,policy_sha256=digest(ADMISSION),prompts=PROMPTS,precision='float32 original weights; CUDA fp16 autocast as Phase10B optimization',
        preprocessing='Pinned Owlv2Processor; bottom/right gray pad to max(originalH,W), then960square; unclipped inverse coordinates',
        seed=6106,original_checkpoint_sha256=model_files['model.safetensors'],files=files,
        original_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','Pillow','scipy')})
    rows=validate_binding(binding);protected={r['image_id']:r for r in original['rows'] if not optimizer_member(r)}
    for key in ('image_id','sha256','identity_group'):
        if {r[key] for r in rows}&{r[key] for r in protected.values()}:raise ValueError('Training/validation leakage')
    # Reverify every rights record and additional seating/ignored geometry via
    # the immutable source registry; no validation photo is decoded.
    reg={p['image_id']:p for p in registry()['photos'] if p['role']=='optimization'}
    for r in rows:
        p=reg[r['image_id']]
        if p['existing_positive_annotation_digest']!=digest(r['annotations']):raise ValueError('Positive/ignored annotations changed')
        for key,hkey in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
            if sha(ROOT/p['rights'][key])!=p['rights'][hkey]:raise ValueError('Rights evidence changed')
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True);save(RUN/'binding.json',binding)
    save(REPORT/'preflight.json',dict(status='PASS',photos=24,groups=24,regions=41,original_checkpoint_sha256=binding['original_checkpoint_sha256'],files_verified=len(files),positive_ignored_geometry_preserved=True,validation_images_accessed=False,training_authorized=False))
    print('PREFLIGHT PASS exact24photos/24groups/41regions;original weights and processor verified;no GPU forwards yet')

if __name__=='__main__':run()
