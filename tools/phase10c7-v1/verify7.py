"""Post-smoke hashes/receipts/resources, without model forwards or backward."""
import importlib.util,subprocess
from common7 import *

def run():
    binding=inputs();rows=validate_binding(binding);done=read(RUN/'raw-complete.json');result=read(REPORT/'results.json')
    intents=sorted(RUN.glob('forward-*.intent.json'));receipts=sorted(RUN.glob('queries-*.receipt.json'));raw=sorted(RUN.glob('queries-*.npz'))
    if len(intents)!=24 or len(receipts)!=24 or len(raw)!=24 or done['forward_calls']!=24 or len(done['records'])!=24:raise ValueError('Forward/reservation/receipt cap mismatch')
    for row,rec in zip(rows,done['records']):
        p=RUN/rec['raw_file'];load_arrays(p)
        if sha(p)!=rec['raw_sha256'] or rec['source_sha256']!=row['sha256']:raise ValueError('Raw cache/source changed')
    if done['seconds']>600 or max(done['peak_allocated_bytes'],done['peak_reserved_bytes'],done['peak_device_used_bytes'])>MAX_VRAM or done['initial_model_state_sha256']!=done['final_model_state_sha256']:raise ValueError('Runtime, memory or model-preservation failure')
    if done['backward_calls'] or done['optimizer_steps'] or result['positive_or_protected_overlap_targets'] or result['coordinate_membership_discrepancies']:raise ValueError('Zero-update/mask safety failure')
    private=[read(RUN/'cpu-analysis-v1'/(str(i).zfill(2)+'.json')) for i in range(1,25)]
    if sum(p['additional_Chair_negative_queries'] for p in private)!=result['actual_new_Chair_negative_queries']:raise ValueError('Per-image masks do not sum')
    sys.path.insert(0,str(ROOT/'tools/phase10c-v1'))
    spec=importlib.util.spec_from_file_location('smoke7_prior_verify',ROOT/'tools/phase10c-v1/verify.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);preserved=m.preservation()
    for rel,seal in (('data/open-images/experiments/phase10c-recovery-v1','SHA256SUMS'),('reports/chair-fp-investigation-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1','SHA256SUMS'),('data/open-images/preparation/phase10c2-v1/submissions/manual-v1','SHA256SUMS'),('data/open-images/preparation/phase10c3-v1','SHA256SUMS'),('data/open-images/preparation/phase10c4-v1','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c5-v1','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c5-v2','INITIAL-SHA256SUMS'),('data/open-images/preparation/phase10c6-v1','INITIAL-SHA256SUMS')):
        folder=ROOT/rel;count=0
        for line in (folder/seal).read_text().splitlines():
            h,path=line.split('  ',1);p=(ROOT/path).resolve() if rel.endswith(('phase10c5-v2','phase10c6-v1')) else (folder/path).resolve()
            allowed=p.is_relative_to(folder) or (rel.endswith('phase10c5-v1') and p.is_relative_to(ROOT/'reports/phase10c5-v1')) or (rel.endswith('phase10c5-v2') and p.is_relative_to(ROOT/'reports/phase10c5-v2')) or (rel.endswith('phase10c6-v1') and p.is_relative_to(ROOT/'reports/phase10c6-v1'))
            if not allowed or sha(p)!=h:raise ValueError('Historical seal failed: '+str(p))
            count+=1
        preserved[rel+'_sealed_files']=count
    subprocess.run(['git','diff','--exit-code','b3b5126','--','tools/phase10c6-v1','reports/phase10c6-v1'],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    save(REPORT/'preservation.json',dict(status='PASS',historical=preserved,exact_forward_receipts=24,original_model_state_unchanged=True,raw_query_files_verified=24,
        runtime_limit_pass=True,VRAM_limit_pass=True,zero_backward_and_optimizer=True,protected_overlap_failures=0,coordinate_failures=0,CPU_analysis_completed_cleanly=True,
        original_checkpoint_sha256=binding['original_checkpoint_sha256'],new_data_bytes=sum(p.stat().st_blocks*512 for p in RUN.rglob('*') if p.is_file())))
    print('PRESERVATION PASS',preserved['historical_data_model_files'],'historical files;',preserved['phase10b_snapshots'],'snapshots;24atomic query records;0updates')

if __name__=='__main__':run()
