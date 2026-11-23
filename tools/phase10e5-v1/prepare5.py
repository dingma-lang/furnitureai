"""CPU-only exact-data/parameter/input validation;no GPU model execution."""
from common5 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only preparation')
    if RUN.exists() or (REPORT/'authorized-config.json').exists():raise FileExistsError('Fresh single-run destination required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:raise ValueError('Exact parent/branch required')
    sys.path.insert(0,str(ROOT/'tools/phase10e4-v1'))
    from verify4 import check
    from future4 import gate
    from scopeE import StudentScope,TRAIN_KEYS,CALIBRATION_KEYS,original_heads
    from teacherE import load_teacher
    from objectiveE import selective_mask
    from transformers import Owlv2ForObjectDetection
    import torch
    torch.set_num_threads(2);check()
    proposal=ROOT/'reports/phase10e4-v1/training-disabled.json';plan=read(proposal)
    approval=dict(approved=True,mode='teacher_distillation_training',proposal_sha256=sha(proposal),limits=LIMITS,coefficient=COEFFICIENT)
    gate(approval)
    context=read(PREP/'training-context.json');index=read(PREP/'teacher-index.json');byid={r['image_id']:r for r in index['records']}
    candidates=[];total=0;groups=set()
    for entry in context['entries']:
        row=entry['row'];teacher=load_teacher(byid[row['image_id']],row,index['processor_contract_sha256']);z,b=teacher.tensors()
        mask,info=selective_mask(z,b,z,b,row,entry['distill_regions'],entry['distill_protected']);total+=info['terms']
        if info['terms']:candidates.append(dict(image_id=row['image_id'],source_group=row['identity_group'],terms=info['terms'],positive_classes=sorted({a['label'] for a in row['annotations'] if not a['ignore']})));groups.add(row['identity_group'])
    gate_ids=['coco2017:000000002347','coco2017:000000192834','coco2017:000000373101']
    chosen=[next(r for r in candidates if r['image_id']==ident) for ident in gate_ids]
    if total!=278 or len(groups)!=15 or len({r['source_group'] for r in chosen})!=3 or set().union(*(set(r['positive_classes']) for r in chosen))!={'chair','sofa','bed'}:raise ValueError('Nonzero representative gates/masks changed')
    train=[e['row'] for e in context['entries']];dev=context['development']
    if len(train)!=59 or len(dev)!=19 or sum(not a['ignore'] for r in train for a in r['annotations'])!=182:raise ValueError('59/182 training pool changed')
    for field in ('image_id','identity_group','sha256'):
        if {r[field] for r in train}&{r[field] for r in dev}:raise ValueError('Internal development leakage')
    from common8 import grouped_epoch
    for epoch in range(5):
        if [[r['image_id'] for r in group] for group in grouped_epoch(train,epoch)]!=context['orders'][epoch]:raise ValueError('Seed/order changed')
    model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    count=model_state_matches_base(model);scope=StudentScope(model);scope.check(hash_frozen=True)
    scope.eval_mode();scope.train_mode();scope.check(hash_frozen=True)
    if torch.cuda.is_initialized() or set(n for n,p in model.named_parameters() if p.requires_grad)!=TRAIN_KEYS:raise ValueError('CPU scope gate failed')
    del model,scope
    files=dict(plan['files'])
    for p in [proposal,ROOT/'tools/phase10c-v1/metric_cpu.py',*HERE.glob('*')]:
        if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    cfg=dict(approved=True,configurations=1,coefficient=COEFFICIENT,limits=LIMITS,learning_rate=1e-5,seed=6106,
             prepared_proposal_approval=approval,files=files,gate_ids=gate_ids,gate_photos=chosen,
             authorization='User Phase10E.5 attachment bf313b70-35d6-4af6-b896-14caaceb7fcd',
             precision=plan['precision'],trainable_keys=sorted(TRAIN_KEYS),calibration_keys=sorted(CALIBRATION_KEYS),
             gate_extension='Three existing original correspondence forwards with nonzero coverage across all3 supported positive classes;no extra forwards/precision/loss changes',
             monitoring='Actual positive/KD parameter VJPs on training forwards;fixed coefficient,scale1;dominance flags do not retune objective',automatic_retry=False)
    RUN.mkdir(parents=True)
    for name in ('telemetry','initial','gate','updates','checkpoints','predictions','CPU-analysis'):(RUN/name).mkdir()
    emit(RUN/'binding.json',cfg);emit(REPORT/'authorized-config.json',{k:v for k,v in cfg.items() if k!='files'})
    emit(REPORT/'preflight.json',dict(status='PASS',original_tensors=count,teacher_caches=59,photos=59,positive_boxes=182,
          source_groups=56,original_KD_terms=278,KD_groups=15,gate_photos=chosen,trainable_tensors=8,
          trainable_parameters=1577988,calibration_frozen=True,source_and_partition_unchanged=True,GPU_calls=0))
    print('E5 CPU preflight PASS:59/182;3nonzero-coverage source-group gates;original initialization',flush=True)

if __name__=='__main__':run()
