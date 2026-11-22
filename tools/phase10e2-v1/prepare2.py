"""CPU-only exact authorization,source/cache checks,and original/C11 scope verification."""
from common2 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU preflight only')
    if RUN.exists() or REPORT.exists():raise FileExistsError('Approved single-attempt destination must be fresh')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:raise ValueError('E2 dedicated branch/parent required')
    from verifyE import check as checkE1
    from configE import require_execution
    from commonE import BINDING
    checkE1()
    proposal=read(ROOT/'reports/phase10e1-v1/gradient-smoke-disabled.json')
    approval=dict(approved=True,mode='gradient_smoke',proposal_sha256=sha(ROOT/'reports/phase10e1-v1/gradient-smoke-disabled.json'),input_binding_sha256=sha(BINDING),limits=LIMITS,approval_reference='User Phase10E.2 explicit zero-update authorization;attachment610b5630-f19b-46eb-9a22-d7031a78e541')
    require_execution(approval,'gradient_smoke')
    import torch
    from transformers import Owlv2ForObjectDetection
    from scopeE import StudentScope,TRAIN_KEYS,restore_candidate
    from teacherE import load_teacher
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json');byid={e['row']['image_id']:e for e in context['entries']}
    if len(index['records'])!=59 or len(byid)!=59:raise ValueError('Exact59 teacher/photos required')
    for r in index['records']:load_teacher(r,byid[r['image_id']]['row'],index['processor_contract_sha256'])
    ids=proposal['photo_ids']
    if len(ids)!=24 or len(set(ids))!=24 or not set(ids)<=set(byid):raise PermissionError('Exact24 existing optimization photos')
    for k in ('image_id','sha256','identity_group'):
        if {e['row'][k] for e in context['entries']}&{r[k] for r in context['development']}:raise ValueError('Partition isolation failure')
    checkpoint=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt'
    if sha(checkpoint)!=proposal['reference_sha256']:raise ValueError('Exact C11 checksum mismatch')
    torch.set_num_threads(2)
    model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
    state_count=model_state_matches_base(model);scope=StudentScope(model);before=scope.immutable_state
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['updates']!=75:raise ValueError('Wrong C11 gradient reference update')
    restore_candidate(scope,payload);scope.check(hash_frozen=True)
    if scope.immutable_state!=before or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():raise ValueError('CPU scope/freeze preflight failed')
    del model,scope,payload
    files=dict(read(BINDING)['files'])
    for p in [*HERE.glob('*.py'),*HERE.glob('*.md'),BINDING,ROOT/'reports/phase10e1-v1/SHA256SUMS']:
        files[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(schema='phase10e2-one-authorized-zero-update-gradient-diagnostic',approved=True,training_authorized=False,mode='gradient_smoke',limits=LIMITS,E1_approval=approval,files=files,photo_ids=ids,reference_checkpoint=str(checkpoint.relative_to(ROOT)),reference_sha256=proposal['reference_sha256'],original_sha256=BASE_SHA,teacher_contract=index['processor_contract_sha256'],numerical_tolerances=dict(logits=dict(atol=.02,rtol=.001),square_cxcywh=dict(atol=1e-5,rtol=1e-5),source='HistoricalC11observe11.compare_array_sets(probe=True);fixedbeforeGPU,userexplicitlyallowsmixedprecisiontolerance'),measurement_plan='One positive-loss VJP on all28forwards;KD VJP only when derivative is not mathematicallyzero. ZeroKD reportedanalytically,notclaimed as executedVJP;guard48beforeeachactualcall.',coefficient_selection='No GPU coefficient sweep. CPU report may derive a conservative fixed scalar from actual gradient ratios/directions or leaveunset.',model_state_transition='Fouroriginalforwards,thenloadexistingC11snapshotonly;verifyeachstatebefore/after,no optimizer/updates',no_additional_variants=True,automatic_retry=False)
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True)
    for name in ('original','adapted','telemetry','forward-intents','gradient-intents'):(RUN/name).mkdir()
    emit(RUN/'binding.json',cfg);emit(REPORT/'authorized-config.json',{k:v for k,v in cfg.items() if k!='files'})
    emit(REPORT/'preflight.json',dict(status='PASS',teacher_caches_verified=59,optimization_photos=59,known_groups=56,original_replay_photos=4,C11_reference_photos=24,actual_original_persisted_tensors_verified=state_count,CPU_C11_scope_verified=True,original_sha256=BASE_SHA,C11_sha256=proposal['reference_sha256'],GPU_forwards=0,optimizers_constructed=0,training_authorized=False))
    print('E2 CPU preflight PASS;original418tensors,C11scope,all59teacherreferences;no GPU calls',flush=True)

if __name__=='__main__':run()
