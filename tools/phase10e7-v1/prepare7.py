"""CPU-only exact image/checkpoint/processor preflight;no photo inference."""
from common7 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU preflight required')
    if RUN.exists() or REPORT.exists():raise FileExistsError('New E7 destination required;do not replace or resume automatically')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PARENT or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Exact clean E6 parent required before E7 preflight')
    sys.path.insert(0,str(ROOT/'tools/phase10e6-v1'))
    import verifyE6
    verifyE6.check()
    import torch
    from PIL import Image
    from transformers import Owlv2Processor
    from teacherE import verify_alignment
    sys.path.insert(0,str(ROOT/'tools/phase10c10-v1'))
    from common10 import HEAD_KEYS,TRAIN_KEYS,named_sha
    request=read(ROOT/'reports/phase10e6-v1/inference-request-disabled.json')
    if request['checkpoint_sha256']!=CHECKPOINT_SHA or sha(CHECKPOINT)!=CHECKPOINT_SHA or sha(BASE/'model.safetensors')!=BASE_SHA:
        raise ValueError('Pinned checkpoint/base SHA mismatch')
    c,train,dev=context();teacher=read(PREP/'teacher-index.json');refs={r['image_id']:r for r in teacher['records']}
    if list(train)!=request['exact_photo_ids'] or sha(PREP/'training-context.json')!=request['source_manifest_sha256']:
        raise ValueError('Exact approved IDs/source context mismatch')
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    done=read(CHECKPOINT.parent.parent/'complete.json')
    if set(payload['weights'])!=HEAD_KEYS or payload['updates']!=75 or payload['trainable_keys']!=sorted(TRAIN_KEYS):raise ValueError('E5head schema/state mismatch')
    h=hashlib.sha256()
    for n in payload['optimizer']['param_groups'][0]['param_names']:
        h.update(n.encode());h.update(payload['weights'][n].contiguous().numpy().tobytes())
    if h.hexdigest()!=done['trainable_sha256']:raise ValueError('Saved endpoint is not the E5completed state')
    processor=Owlv2Processor.from_pretrained(str(BASE),local_files_only=True,use_fast=False)
    files={str(p.relative_to(ROOT)):sha(p) for p in [CHECKPOINT,BASE/'model.safetensors',PREP/'training-context.json',PREP/'teacher-index.json',ROOT/'reports/phase10e6-v1/inference-request-disabled.json']}
    files.update({str(p.relative_to(ROOT)):sha(p) for p in BASE.iterdir() if p.is_file() and p.suffix in ('.json','.txt')})
    native_path=ROOT/'data/open-images/preparation/phase10c21-v1/full-local-native-inventory.json'
    native=read(native_path);native_photos={p['image_id']:p for p in native['photos']}
    files[str(native_path.relative_to(ROOT))]=sha(native_path)
    for source in native['source_files']:
        if sha(ROOT/source['source'])!=source['sha256']:raise ValueError('Official annotation source changed')
        files[source['source']]=source['sha256']
    for rel in ('data/open-images/preparation/phase10c5-v1/review-registry.json','data/open-images/preparation/phase10c6-v1/source-negative-evidence.json','data/open-images/preparation/phase10c21-v1/hard-negative-manifest-v1.json','data/open-images/preparation/phase10e6-v1/candidate-evidence-v1.json','data/open-images/preparation/phase10e6-v1/query-index-v3.json'):
        files[rel]=sha(ROOT/rel)
    observations=[]
    for ident,row in train.items():
        require_train(ident,train);p=ROOT/row['path'];files[row['path']]=sha(p)
        if files[row['path']]!=row['sha256']:raise ValueError('Source photograph mismatch')
        native_photo=native_photos[ident]
        if any(native_photo[k]!=row[k] for k in ('path','sha256','width','height')) or native_photo['source_group']!=row['identity_group']:raise ValueError('Native/source group identity mismatch')
        for k,hk in (('evidence_file','evidence_sha256'),('rights_record','rights_record_sha256')):
            rel=native_photo['rights'][k]
            if sha(ROOT/rel)!=native_photo['rights'][hk]:raise ValueError('Recorded individual-photo rights evidence changed')
            files[rel]=native_photo['rights'][hk]
        ref=refs[ident];verify_alignment(ref,row,teacher['processor_contract_sha256'])
        if sha(ROOT/ref['cache'])!=ref['cache_sha256']:raise ValueError('Existing original-query cache changed')
        files[ref['cache']]=ref['cache_sha256']
        with Image.open(p) as im:
            if im.size!=(row['width'],row['height']) or im.getexif().get(274,1)!=1:raise ValueError('Source dimensions/orientation mismatch')
            im.load();batch=processor(text=[PROMPTS],images=im.convert('RGB'),return_tensors='pt')
        observed={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for k,v in batch.items()}
        if observed!=ref['reconstructed_processor_tensors']:raise ValueError('Exact preprocessing/prompt contract mismatch:'+ident)
        observations.append(dict(image_id=ident,source_sha256=row['sha256'],source_group=row['identity_group'],width=row['width'],height=row['height'],processor_tensors=observed))
    if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
    RUN.mkdir(parents=True);REPORT.mkdir(parents=True)
    emit(RUN/'binding.json',dict(approved=True,authorization='User Phase10E.7 exactly one inference-only cache completion',limits=LIMITS,
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=CHECKPOINT_SHA,original_base=str(BASE.relative_to(ROOT)),original_base_sha256=BASE_SHA,
        image_ids=list(train),rows_sha256=digest(list(train.values())),files=files,classes=list(CLASSES),prompts=PROMPTS,query_layout=[60,60],
        precision='FP16autocast_forward/FP32logits_boxes;FP64original-coordinate_mapping',
        precision_comparability='Matches existing optimization-teacher/training-forward contract;E5development inference used FP32 and is not rerun or compared as an equal-precision benchmark',
        preprocessing='Pinned existing Owlv2Processor,use_fast=False,RGB,no EXIF transpose or augmentation;all59tensor hashes equal existing teacher contracts',
        coordinate_mapping='Model normalized cxcywh on bottom/right padded square;xyxy*max(width,height),no centering offset;unclipped originals retained',
        optimizer_updates=0,backward_calls=0,additional_variants=0,automatic_retry=False))
    emit(RUN/'preprocessing-manifest.json',dict(records=observations,processor_contract_sha256=teacher['processor_contract_sha256']))
    emit(REPORT/'preflight.json',dict(status='PASS',photos=59,source_groups=56,positive_boxes=182,image_hashes_verified=59,exact_processor_tensor_contracts=59,
         E5_checkpoint_sha256=CHECKPOINT_SHA,original_sha256=BASE_SHA,live_E5_trainable_sha256=h.hexdigest(),development_or_reserved_images_opened=0,
         precision=read(RUN/'binding.json')['precision'],precision_comparability=read(RUN/'binding.json')['precision_comparability'],GPU_calls=0,
         binding_sha256=sha(RUN/'binding.json'),expected_uncompressed_array_bytes=59*(3600*4+3600*7*4*2+3600*4*4+3600*4*8*2)))
    print('CPU preflight PASS:59exact optimization photographs,56groups,E5/base hashes,59processor contracts;no CUDA initialization',flush=True)

if __name__=='__main__':run()
