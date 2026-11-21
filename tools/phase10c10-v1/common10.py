"""C10 CPU attribution and disabled training preflight. No implicit model execution."""
from pathlib import Path
import sys,os,json,hashlib,math
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/phase10c9-v1'))
from common9 import read,save,sha,digest,verify_seal,C8,MODELS,subsets,manifest,CLASSES,SOURCE,load_npz,metric_module,quantiles
sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
from expanded import HEAD_KEYS,HEAD_PARAMETERS
BASE=ROOT/'models/owlv2-original-baseline-20261103'
OUT=ROOT/'data/open-images/preparation/phase10c10-v1'
REPORT=ROOT/'reports/phase10c10-v1'
FUTURE=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1'
CALIBRATION_KEYS={f'class_head.{module}.{suffix}' for module in ('logit_shift','logit_scale') for suffix in ('weight','bias')}
TRAIN_KEYS=HEAD_KEYS-CALIBRATION_KEYS
TRAIN_COUNT=1_577_988

def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def named_sha(model,names):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if n in names:h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def original_heads():
    from safetensors import safe_open
    with safe_open(BASE/'model.safetensors',framework='pt',device='cpu') as f:return {n:f.get_tensor(n) for n in sorted(HEAD_KEYS)}

def verify_inputs():
    from common8 import verify_binding,image_path
    contract=read(OUT/'input-contract.json')
    for path,expected in contract['files'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Pinned original/evidence changed: '+path)
    prepared=verify_binding(read(C8/'binding.json'));train,dev=subsets(prepared)
    for row in train+dev:image_path(row,row['internal_partition'],prepared)
    return prepared,train,dev

def preflight():
    import subprocess
    if OUT.exists() or REPORT.exists():raise FileExistsError('Fresh C10 version required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='experiment/phase10c10-calibration-preflight':raise ValueError('Dedicated branch required')
    c9public=verify_seal(ROOT/'reports/phase10c9-v1/SHA256SUMS');s=read(ROOT/'reports/phase10c9-v1/artifact-seal.json');seal=ROOT/s['private_seal_path']
    if sha(seal)!=s['private_seal_sha256']:raise ValueError('C9 private seal changed')
    c9private=verify_seal(seal);c8public=verify_seal(ROOT/'reports/phase10c8-v2/SHA256SUMS');c8private=verify_seal(C8/'ARTIFACT-SHA256SUMS')
    files=dict(read(C8/'binding.json')['files'])
    for arm in ('A-control','B-corrected'):
        p=C8/arm/'complete.json';files[str(p.relative_to(ROOT))]=sha(p);rec=read(p)['final_checkpoint'];files[str((C8/rec['file']).relative_to(ROOT))]=rec['sha256']
    for p in (ROOT/'reports/phase10c9-v1/prediction-summary.json',ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/modeling_owlv2.py',ROOT/'.venv/lib/python3.14/site-packages/transformers/models/owlv2/image_processing_owlv2.py',ROOT/'tools/phase10b-v1/worker.py'):
        files[str(p.relative_to(ROOT))]=sha(p)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    save(OUT/'input-contract.json',dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=files,C9_public=c9public,C9_private=c9private,C8_public=c8public,C8_private=c8private,real_model_forwards=0,training_authorized=False))
    prepared,train,dev=verify_inputs()
    save(REPORT/'input-verification.json',dict(status='PASS',C9_public=c9public,C9_private=c9private,C8_public=c8public,C8_private=c8private,training_photos=len(train),development_diagnostic_photos=len(dev),positive_boxes=sum(not a['ignore'] for r in train for a in r['annotations']),original_checkpoint_sha256=files[str((BASE/'model.safetensors').relative_to(ROOT))],training_authorized=False,Phase10D_authorized=False))
    print('C10 input preservation PASS;CPU analysis only',flush=True)

if __name__=='__main__':preflight()
