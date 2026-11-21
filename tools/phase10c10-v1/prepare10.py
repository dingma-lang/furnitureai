"""Write one disabled, immutable, deterministic candidate and diagnostic contract."""
from common10 import *
from config10 import specification,check_spec

def run(revision=1):
    if FUTURE.exists():raise FileExistsError('Future training destination already exists; do not modify it')
    verify_inputs();config=specification()
    config['code_hashes']={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    check_spec(config)
    suffix='' if revision==1 else '-v'+str(revision)
    config_path=REPORT/('experiment-disabled'+suffix+'.json')
    save(config_path,config)
    save(REPORT/('diagnostic-contract'+suffix+'.json'),dict(training_authorized=False,Phase10D_authorized=False,reference='Original/C8 A/C8 B, same frozen19-photo exposed internal protocol; not independent acceptance',
        frozen_original_calibration=dict(keys=sorted(CALIBRATION_KEYS),scalar_count=1538,hashes={n:tensor_sha(t) for n,t in original_heads().items() if n in CALIBRATION_KEYS}),
        persistent_guard='FrozenScope used at every mode transition and before/after every step; no legacy configure_heads call after installation; frozen gradients must be None and excluded from AdamW including weight decay',
        metrics=['Shared seven-category logit drift on same query indices','Chair raw/post-NMS score distributions and FP counts at.15/.50/.80','Precision/recall at.15; matched-original-recall PR envelope; AP50/AP75; TP/FP/FN','Per-category Sofa/Bed recall, AP75 and matched IoU regressions','Duplicate/fragment/mixed crop diagnostics with annotation uncertainty','Per-parameter gradient norms and byte hashes, calibration gradients None','Exact actual training-query mask counts, scores, source-region IDs and protected-positive violations'],
        cache='Pre-head/text activations from19 already-planned initial FP32 diagnostic forwards, no extra encoder forwards; saved atomically and private. None exist during this preflight.',
        attribution='Validate reconstructed original/A/B/candidate endpoints before all8 CPU projection/shift/scale swaps and six-order functional Shapley. Abort attribution if reconstruction fails; never invent activations.',
        reporting='Primary endpoint epoch5 fixed; milestones descriptive, no threshold/hyperparameter changes or independent/frozen scoring; preserve raw predictions before separate CPU metrics',
        success_proposal=read(ROOT/'reports/phase10c9-v1/next-experiment-proposal.json')['predeclared_evidence_criteria'],
        stop='Nonfinite loss/gradient; frozen scope/bytes drift; model/source/config mismatch; unsafe resource/cursor/checkpoint; no automatic retries or extra variants'))
    save(REPORT/('readiness'+suffix+'.json'),dict(status='CPU_PREFLIGHT_ONLY',configuration_file=str(config_path.relative_to(ROOT)),config_file_sha256=sha(config_path),config_semantic_sha256=digest(config),
        sole_training_change='Freeze4 calibration tensors (1538scalars); train8 tensors (1577988scalars)',loss_and_data_match_C8_corrected=True,real_model_forwards=0,optimizer_updates=0,future_experiment_directory_created=FUTURE.exists(),explicit_approval_needed=True))
    print('ONE CANDIDATE PREPARED;training_authorized=False;GPU directory absent',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--revision',type=int,default=1);run(p.parse_args().revision)
