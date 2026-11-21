"""Fresh C9 output and read-only verification of C8, photographs and model files."""
import subprocess
from common9 import *

def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('Fresh investigation version required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='experiment/phase10c9-confidence-investigation':raise ValueError('Separate branch required')
    pub=verify_seal(ROOT/'reports/phase10c8-v2/SHA256SUMS');seal=read(ROOT/'reports/phase10c8-v2/artifact-seal.json');p=ROOT/seal['private_seal_path']
    if sha(p)!=seal['private_seal_sha256']:raise ValueError('C8 seal changed')
    private=verify_seal(p)
    from common8 import verify_binding
    binding=read(C8/'binding.json');verify_binding(binding);train,dev=subsets()
    for r in train+dev:source_path(r)
    entry=next(e for e in read(C8/'metric-results.json')['entries'] if e['arm']=='B-corrected' and e['epoch']==5)
    if (entry['metrics']['tp'],entry['metrics']['fp'],entry['metrics']['fn'],entry['metrics']['categories']['chair']['fp'])!=(58,788,3,636):raise ValueError('C8 headline mismatch')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    save(REPORT/'preflight.json',dict(status='PASS',prior_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),C8_public_sealed_files=pub,C8_private_sealed_files=private,train_photos=len(train),internal_dev_photos=len(dev),eligible_photo_hashes_verified=78,
        original_checkpoint_sha256=binding['configs']['A-control']['base_sha256'],model_forwards_authorized=0,optimizer_updates_authorized=0,downloads_authorized=False,Phase10D_authorized=False))
    print('C9 PREFLIGHT PASS',pub,private,'sealed artifacts;78image hashes;CPU only',flush=True)

if __name__=='__main__':run()
