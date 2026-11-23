"""Fresh-root CPU preflight; selects all historical nonzero KD coverage, no tuning."""
from common4 import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only preflight')
    if RUN.exists() or (REPORT/'preflight.json').exists():raise FileExistsError('Fresh E4 destination required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:
        raise ValueError('Prepared parent and dedicated branch required')
    sys.path.insert(0,str(ROOT/'tools/phase10e3-v1'))
    from verify3 import check
    from objectiveE import selective_mask
    from teacherE import load_teacher
    from common8 import load_npz
    from scopeE import StudentScope,restore_candidate
    from transformers import Owlv2ForObjectDetection
    import torch
    torch.set_num_threads(2);check()
    proposal=read(ROOT/'reports/phase10e3-v1/recovery-disabled.json')
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json')
    byid={e['row']['image_id']:e for e in context['entries']};refs={r['image_id']:r for r in index['records']}
    files=dict(read(BINDING)['files'])
    for r in index['records']:load_teacher(r,byid[r['image_id']]['row'],index['processor_contract_sha256'])
    active=[];excluded=[];records=[];cachemap={};groups=set();terms=0
    for number,ident in enumerate(proposal['adapted_photo_ids'],1):
        entry=byid[ident];row=entry['row'];teacher=load_teacher(refs[ident],row,index['processor_contract_sha256']);tz,tb=teacher.tensors()
        path=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/probes/epoch-5'/f'{number:02}.npz'
        values=load_npz(path);z=torch.tensor(values['logits']);b=torch.tensor(values['square_cxcywh'])
        mask,info=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected'])
        record=dict(image_id=ident,source_group=row['identity_group'],terms=info['terms'],mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),cache=str(path.relative_to(ROOT)),cache_sha256=sha(path),historical_index=number)
        files[str(path.relative_to(ROOT))]=sha(path);cachemap[ident]=record
        if info['terms']:
            active.append(ident);terms+=info['terms'];groups.add(row['identity_group']);records.append(record)
        else:excluded.append(record|dict(reason='Saved immutable C11 outputs give zero selective terms;no GPU replay or invented gradients'))
    if (len(active),len(groups),terms)!=(15,15,267):raise ValueError('Expected unchanged15groups/267adapted terms')
    for key in ('image_id','sha256','identity_group'):
        if {e['row'][key] for e in context['entries']}&{r[key] for r in context['development']}:raise ValueError('Isolation failed')
    checkpoint=ROOT/proposal['reference_checkpoint']
    if sha(BASE/'model.safetensors')!=BASE_SHA or sha(checkpoint)!=proposal['reference_sha256']:raise ValueError('Checkpoint changed')
    model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    count=model_state_matches_base(model);scope=StudentScope(model)
    restore_candidate(scope,torch.load(checkpoint,map_location='cpu',weights_only=True));scope.check(hash_frozen=True)
    if torch.cuda.is_initialized() or any(p.grad is not None for p in model.parameters()):raise RuntimeError('CPU scope preflight failure')
    del model,scope
    for path in (BINDING,PREP/'training-context.json',PREP/'teacher-index.json',ROOT/'reports/phase10e3-v1/SHA256SUMS-v2',checkpoint,
                 ROOT/'tools/phase10e3-v1/stable3.py',ROOT/'tools/phase10e3-v1/components3.py',ROOT/'tools/phase10e3-v1/telemetry3.py'):
        files[str(path.relative_to(ROOT))]=sha(path)
    for p in HERE.glob('*'):
        if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(approved=True,training_authorized=False,limits=LIMITS,files=files,
             approval='User attachment141ca7af-af10-4a49-bc3b-b44499885d52 Phase10E.4',
             proposal_sha256=sha(ROOT/'reports/phase10e3-v1/recovery-disabled.json'),
             permitted_photo_ids=proposal['adapted_photo_ids'],active_photo_ids=active,
             original_photo_ids=proposal['original_replay_photo_ids'],reference_checkpoint=proposal['reference_checkpoint'],
             reference_sha256=proposal['reference_sha256'],cachemap=cachemap,expected_terms=267,
             expected_forwards=16,expected_contractions=16,expected_VJPs=31,measurement_scale=1.,
             box_probe_photo=active[0],coefficient=None,automatic_retry=False,
             precision='Unchanged FP16 forward;E3 observed-node FP32 adjoints for positive/KD;one native box VJP at scale1',
             subset_reason='Every saved nonzero adapted KD photograph/source group;9known-zero photographs excluded without substituting IDs or changing masks',
             original_correspondence='One fresh original check plus sealed E2 four original exact replays;no teacher editing')
    RUN.mkdir(parents=True)
    for name in ('raw','measurements','telemetry','forward-intents','contraction-intents','VJP-intents'):(RUN/name).mkdir()
    emit(RUN/'binding.json',cfg)
    emit(REPORT/'preflight.json',dict(status='PASS',original_tensors=count,teacher_caches=59,optimization_photos=59,
         positive_boxes=182,active_photos=15,active_groups=15,adapted_terms=267,original_terms=278,
         skipped_zero_photos=excluded,active_records=records,limits=LIMITS,scope_verified=True,
         classification_adjoint='FP32 observed-node;scale1',GPU_calls=0,coefficient=None))
    print('E4 CPU preflight PASS:59teacher pairs;15active groups/267terms;16forwards/31VJPs planned',flush=True)


if __name__=='__main__':run()
