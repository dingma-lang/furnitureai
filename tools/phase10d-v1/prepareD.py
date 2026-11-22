"""CPU-only architecture, provenance, objective and execution binding."""
from commonD import *


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU preflight only')
    if RUN.exists() or REPORT.exists():raise FileExistsError('Fresh Phase10D destinations required')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT:
        raise ValueError('Dedicated original-parent branch required')
    old,prepared,train,dev,reg=context()
    import verify21
    verify21.check_seal()
    proposal=read(ROOT/'reports/phase10c21-v1/box-only-disabled.json')
    if sha(BASE/'model.safetensors')!=BASE_SHA or proposal['starting_sha256']!=BASE_SHA or proposal['limits']['max_forwards']!=322:
        raise ValueError('Approved original/configuration changed')
    import torch,inspect,transformers
    from transformers import Owlv2ForObjectDetection,Owlv2Processor
    from scopeD import BoxScopeD,BOX_KEYS
    from telemetry19 import tensor_hash
    torch.set_num_threads(2)
    model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
    state_count=model_state_matches_base(model)
    processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
    scope=BoxScopeD(model);opt=scope.optimizer();scope.check(opt,check_hash=True)
    params=dict(model.named_parameters())
    if set(proposal['trainable_keys'])!=BOX_KEYS:raise ValueError('Actual box scope disagrees with approval')
    architecture=dict(persisted_tensors_exact_original=state_count,trainable={n:dict(shape=list(p.shape),scalars=p.numel(),sha256=tensor_hash(p)) for n,p in params.items() if n in BOX_KEYS},frozen={n:dict(shape=list(p.shape),scalars=p.numel(),sha256=tensor_hash(p)) for n,p in params.items() if n not in BOX_KEYS},buffers={n:dict(shape=list(b.shape),sha256=tensor_hash(b)) for n,b in model.named_buffers()},frozen_sha256=scope.expected_frozen,total_parameters=sum(p.numel() for p in params.values()),trainable_parameters=sum(params[n].numel() for n in BOX_KEYS),shared_representations_frozen=True,classifier_frozen=True,objectness_frozen=True,box_bias_frozen=True,actual_forward_source_sha256=sha(Path(inspect.getfile(Owlv2ForObjectDetection))),transformers_version=transformers.__version__,torch_version=torch.__version__,precision='FP32',processor_size=processor.image_processor.size)
    architecture['unregistered_tensor_sha256']=dict(scope.derived_hashes)
    architecture['box_bias_is_registered_buffer']=False
    if architecture['trainable_parameters']!=1184260 or processor.image_processor.size!={'height':960,'width':960} or torch.cuda.is_initialized():raise ValueError('Actual architecture/frame/CUDA preflight mismatch')
    del scope,opt,model,processor,params
    files=dict(old['files'])
    extras=[ROOT/'reports/phase10c20-v1/results.json',ROOT/'data/open-images/experiments/phase10c8-ab-v2/metric-results.json']
    for folder in (ROOT/'data/open-images/experiments/phase10c8-ab-v2/A-control/evaluation/epoch-0',ROOT/'data/open-images/experiments/phase10c19-recovery-v1/evaluation/epoch-5'):
        extras.extend(p for p in folder.iterdir() if p.is_file() and not p.name.startswith('.'))
    for p in [*HERE.glob('*.py'),*BASE.iterdir(),* (ROOT/'tools/phase10c21-v1').glob('*.py'),ROOT/'tools/phase10c-v1/box_only.py',ROOT/'tools/phase10c-v1/metric_cpu.py',ROOT/'reports/phase10c21-v1/box-only-disabled.json',ROOT/'reports/phase10c21-v1/SHA256SUMS-v2',*extras]:
        if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    smoke_ids=[ident for group in old['orders'][0] for ident in group][:4]
    cfg=dict(schema='phase10d-one-original-classifier-box-only-v1',training_authorized=True,Phase10D_authorized=True,arms=1,base_sha256=BASE_SHA,limits=LIMITS,learning_rate=1e-5,precision='FP32',seed=6106,trainable_keys=sorted(BOX_KEYS),training_rows_sha256=digest(train),development_rows_sha256=digest(dev),orders=old['orders'],evaluation=old['evaluation'],files=files,smoke_ids=smoke_ids,frozen_sha256=architecture['frozen_sha256'],unregistered_tensor_sha256=architecture['unregistered_tensor_sha256'],new_Chair_negative_regions_active=False,new_positive_annotations=0,matching_and_box_loss='unchanged tools/phase10c-v1/box_only.py:detached Hungarian,5L1+2GIoU',classification_loss=0,background_targets=0,comparison_baselines=['original','C20_E5'],fixed_endpoint=75,automatic_retry=False,destination=str(RUN.relative_to(ROOT)),differences_from_C21_proposal=['Explicit user Phase10D authorization now enabled in new binding only;original C21 disabled proposal unchanged','New tested executor uses six box tensors;legacy twelve-head workers never invoked','Actual OWLv2 box_bias is an unregistered derived tensor;extra hash guard covers it without changing computation'],expected_forward_counts=dict(smoke=4,training=295,frozen_logit_checks=4,internal_diagnostic=19))
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True)
    for name in ('checkpoints','updates','telemetry','forward-intents','smoke','logit-checks','evaluation','CPU-analysis'):(RUN/name).mkdir()
    emit(RUN/'binding.json',cfg);emit(REPORT/'authorized-config.json',{k:v for k,v in cfg.items() if k!='files'})
    emit(RUN/'training-rows.json',train);emit(RUN/'development-rows.json',dev)
    emit(REPORT/'architecture.json',architecture)
    emit(REPORT/'preflight.json',dict(status='PASS',CPU_only=True,original_persisted_tensors_verified=state_count,original_sha256=BASE_SHA,expected_frozen_sha256=cfg['frozen_sha256'],photos=59,groups=56,positive_boxes=182,category_counts={'chair':114,'sofa':40,'bed':28},new_negatives_inactive=5,classification_related_parameters_frozen=True,additional_training_sources=0,baseline_protocol_unchanged=True,Phase10D_explicitly_authorized=True))
    print('Phase10D CPU architecture/state preflight PASS',state_count,'original tensors;6 box-only tensors',flush=True)


if __name__=='__main__':run()
