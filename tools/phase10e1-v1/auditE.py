"""Exact cache/provenance audit and training-only cached output-gradient analysis."""
import importlib.metadata as metadata
import numpy as np
import torch
from commonE import *
from teacherE import load_teacher
from objectiveE import objective,selective_mask,logit_diagnostics

def quantiles(a):
    return {str(q):float(np.quantile(a,q)) if a else None for q in (0,.5,.9,1)}

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU audit only')
    if PREP.exists() or REPORT.exists():raise FileExistsError('Fresh E1 roots required')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:raise ValueError('Dedicated E1 branch/parent required')
    old,prepared,entries,dev,bundle=source_context()
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from sealD import check as checkD
    checkD()
    from common21 import RUN17,C8,SOURCE,DATA21
    from teacher21 import teacher_index
    from common8 import load_npz
    refs=teacher_index();byid={r['image_id']:r for r in refs};c17=read(RUN17/'authorized-config.json')
    contract_files={p:h for p,h in c17['files'].items() if p.startswith('models/owlv2-original-baseline-20261103/') or '.venv/' in p or p.endswith('/gpu17.py') or p.endswith('/common8.py')}
    for p,h in contract_files.items():
        if sha(ROOT/p)!=h:raise ValueError('Teacher processor/model/library contract changed: '+p)
    contract=dict(files=contract_files,classes=list(CLASSES),prompts=PROMPTS,pad='bottom/right to max original side before960square resize',orientation='EXIF1,RGBconversion',shape=[1,3,960,960],query_grid=[60,60],patch_order='row-major reshape of frozen image feature map',augmentation='none',precision='FP16forward/FP32arrays',historical_processed_pixel_hashes_saved=False,historical_token_ID_hashes_saved=False,live_numeric_replay_required_before_training=True)
    contract_hash=digest(contract)
    from transformers import Owlv2Processor
    from PIL import Image
    processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
    records=[];initial=[];index={};excluded=[]
    for entry in entries:
        row=entry['row'];ref=byid[row['image_id']]
        r=dict(ref,classes=list(CLASSES),prompts=PROMPTS,precision='FP16forward/FP32arrays',augmentation='none',historical_partition='train',internal_partition='optimization',processor_contract_sha256=contract_hash)
        teacher=load_teacher(r,row,contract_hash);z,b=teacher.tensors();mask,stats=selective_mask(z,b,z,b,row,entry['distill_regions'],entry['distill_protected'])
        with Image.open(ROOT/row['path']) as image:
            if image.size!=(row['width'],row['height']) or image.getexif().get(274,1)!=1:raise ValueError('Source frame changed')
            batch=processor(text=[PROMPTS],images=image.convert('RGB'),return_tensors='pt')
        r['reconstructed_processor_tensors']={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
        r['historical_tensor_hash_comparison_available']=False
        records.append(r);index[row['image_id']]=r
        initial.append(dict(image_id=row['image_id'],source_group=row['identity_group'],regions=len(entry['distill_regions']),terms=stats['terms'],region_query_counts=stats['region_query_counts'],teacher_conflicts_excluded=stats['teacher_conflicts_excluded']))
        excluded.extend(dict(image_id=row['image_id'],**e) for e in entry['excluded_distill_regions'])
    # Cached historical predictions are optimization-only;no development features.
    probe_ids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']]
    c11=ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1'
    oldrows={e['row']['image_id']:e for e in entries};analysis=[];identical=0
    for i,ident in enumerate(probe_ids,1):
        entry=oldrows[ident];row=entry['row'];teacher=load_teacher(index[ident],row,contract_hash);tz,tb=teacher.tensors()
        original=load_npz(c11/'probes/epoch-0'/f'{i:02}.npz')
        equal=np.array_equal(original['logits'],teacher.logits) and np.array_equal(original['square_cxcywh'],teacher.boxes)
        if not equal:raise ValueError('C11 eight-tensor original-start cache differs from original teacher')
        identical+=1
        a=load_npz(c11/'probes/epoch-5'/f'{i:02}.npz');z=torch.tensor(a['logits'],requires_grad=True);b=torch.tensor(a['square_cxcywh'],requires_grad=True)
        _,parts,stats=objective(z,b,row,tz,tb,entry,1.) # unit coefficient measures the term;not a proposed lambda.
        gp=torch.autograd.grad(parts['positive'],z,retain_graph=True)[0]
        gk=torch.autograd.grad(parts['distillation'],z)[0]
        mask,ms=selective_mask(z,b,tz,tb,row,entry['distill_regions'],entry['distill_protected'])
        analysis.append(dict(image_id=ident,source_group=row['identity_group'],positive_loss=float(parts['positive'].detach()),unit_distillation_loss=stats['distillation'],terms=ms['terms'],positive_output_gradient_L1=float(gp.abs().sum()),positive_output_gradient_L2=float(gp.double().norm()),distillation_output_gradient_L1=float(gk.abs().sum()),distillation_output_gradient_L2=float(gk.double().norm()),projection_gradient_measured=False,diagnostics=logit_diagnostics(z.detach(),tz,mask),source='Actual saved C11 E5 FP16 optimization-photo logits/boxes;CPU output derivatives,not model parameter derivatives'))
    PREP.mkdir(parents=True);REPORT.mkdir(parents=True)
    emit(PREP/'processor-contract.json',contract);emit(PREP/'teacher-index.json',dict(records=records,processor_contract_sha256=contract_hash))
    emit(PREP/'training-context.json',dict(entries=entries,development=dev,orders=old['orders'],training_rows_sha256=digest([e['row'] for e in entries]),development_rows_sha256=digest(dev),source_bundle_sha256=digest(bundle)))
    emit(PREP/'initial-mask-audit.json',dict(records=initial,excluded_regions=excluded))
    emit(PREP/'cached-gradient-audit.json',dict(records=analysis,proposed_coefficient=None,unit_coefficient_is_measurement_only=True,parameter_gradient_measurements_missing=True,holdout_used=False))
    summary=dict(teacher_photos=len(records),known_source_groups=len({r['source_group'] for r in records}),queries=59*3600,cached_logit_values=59*3600*7,positive_boxes=182,precision='FP16 forward,savedFP32 arrays',C11_initial_cache_pairs_bit_identical=identical,processor_model_tokenizer_files_verified=contract_files,feature_cache_available=False,development_prehead_cache_used=False,processed_tensor_hashes='Reconstructed today,not historically recorded',alignment='Structurally supported for frozen backbone/identical source+processor+prompts+precision,noaug;live initial numeric replay remains mandatory',legacy_source_regions=len(bundle['records']),distill_admitted_regions=sum(len(e['distill_regions']) for e in entries),conservative_overlap_excluded_regions=len(excluded),initial_selected_terms=sum(r['terms'] for r in initial),initial_selected_photos=sum(r['terms']>0 for r in initial),initial_selected_source_groups=len({r['source_group'] for r in initial if r['terms']}),initial_zero_coverage_regions=sum(n==0 for r in initial for n in r['region_query_counts'].values()),teacher_conflicts_excluded=sum(r['teacher_conflicts_excluded'] for r in initial),six_other_categories_distilled=False,new_C21_five_regions_activated=False,mean_unit_distill_loss_C11E5=float(np.mean([a['unit_distillation_loss'] for a in analysis])),positive_losses_C11E5=quantiles([a['positive_loss'] for a in analysis]),unit_distill_losses_C11E5=quantiles([a['unit_distillation_loss'] for a in analysis]),output_gradient_L1_positive=quantiles([a['positive_output_gradient_L1'] for a in analysis]),output_gradient_L1_distill=quantiles([a['distillation_output_gradient_L1'] for a in analysis]),projection_gradient_measured=False,coefficient=None,gradient_smoke_required=True,global_seven_class_inflation_guaranteed_controlled=False,real_model_forwards=0,GPU_calls=0,model_updates=0,package_versions={p:metadata.version(p) for p in ('torch','transformers','Pillow','numpy','tokenizers')})
    emit(REPORT/'teacher-and-mask-audit.json',summary)
    if torch.cuda.is_initialized():raise RuntimeError('CPU audit initialized CUDA')
    print('E1 teacher audit PASS',summary['teacher_photos'],'photos;',summary['initial_selected_terms'],'selective terms;',summary['distill_admitted_regions'],'admitted regions',flush=True)

if __name__=='__main__':run()
