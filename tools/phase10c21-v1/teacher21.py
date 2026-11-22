"""Index original cached training outputs,never infer or invent teacher labels."""
import numpy as np
from common21 import *


def query_arrays(path,expected_hash):
    if sha(path)!=expected_hash or path.stat().st_size>1500000:raise ValueError('Query cache hash/size mismatch')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())>20 or sum(i.file_size for i in z.infolist())>2000000:raise ValueError('Unsafe query cache')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in ('logits','square_cxcywh')}
    if arrays['logits'].shape!=(3600,7) or arrays['square_cxcywh'].shape!=(3600,4) or any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Query shape/dtype invalid')
    return arrays


def teacher_index():
    cfg,prepared,train,dev,reg,native,photos=context21();byid={r['image_id']:r for r in train};done=read(RUN17/'complete.json');init=read(RUN17/'initialization.json');records=[];oldids=[p['image_id'] for p in read(SOURCE/'proposed-smoke-photos.json')['photos']]
    if done['status']!='PASS' or done['photos']!=59 or done['model_updates'] or done['model_state_before']!=done['model_state_after'] or init['original_checkpoint_sha256']!=sha(BASE/'model.safetensors'):raise ValueError('Original teacher provenance incomplete')
    for ref in done['records']:
        p=RUN17/ref['file'];meta=read(p)
        if sha(p)!=ref['sha256'] or meta['image_id'] not in byid:raise ValueError('Teacher receipt identity/hash changed')
        row=byid[meta['image_id']];require_training_row(row,train,dev)
        if meta['source_sha256']!=row['sha256']:raise ValueError('Teacher source image changed')
        path=RUN17/meta['raw_file'];a=query_arrays(path,meta['raw_sha256']);old_equal=None
        if row['image_id'] in oldids:
            old=load_npz(C8/'B-corrected/probes/epoch-0'/(str(oldids.index(row['image_id'])+1).zfill(2)+'.npz'));old_equal=all(np.array_equal(old[k],a[k]) for k in a)
            if not old_equal:raise ValueError('Original24teacher cache differs')
        records.append(dict(image_id=row['image_id'],source_group=row['identity_group'],source_sha256=row['sha256'],width=row['width'],height=row['height'],cache=str(path.relative_to(ROOT)),cache_sha256=meta['raw_sha256'],receipt_sha256=ref['sha256'],matches_original24_cache=old_equal,precision='FP16 forward,FP32 output arrays',prompt_sha256=digest(cfg['evaluation']['prompts']),base_sha256=init['original_checkpoint_sha256'],query_layout=[60,60],query_indices_not_object_identities=True))
    if {r['image_id'] for r in records}!=set(byid):raise ValueError('Teacher coverage is not exact59optimization photos')
    return records


def run():
    records=teacher_index();emit21(DATA21/'teacher-cache-index.json',dict(records=records,model_inference=0,training_activated=False))
    emit21(REPORT21/'teacher-feasibility.json',dict(available_training_teacher_photos=len(records),historical_C14_reported24=True,coverage_expanded_by_existing_C17=True,original24_bit_identical=sum(r['matches_original24_cache'] is True for r in records),additional_original_cached_photos=sum(r['matches_original24_cache'] is None for r in records),uncached_optimization_photos=59-len(records),teacher_FP32_evaluation_cache_for59=False,cache_precision='FP16forward/FP32arrays;cannot certify FP32inference equality against these',fixed_patch_correspondence_requires=['Exact same image bytes,orientation,resize/padding,prompts andprecision','Frozen original vision/text/backbone;no augmentations/crops/flips','Original60x60query lattice;queryindex is patchidentity,not objectidentity','Exclude protected furniture,ignored/conflicting/native-positive regions from any consistency mask','Teacher disagreements remainunknown;teacher lowconfidence isnot Chair absence'],missing=['Cached pre-head embeddings andobjectness for59not certified by these logits-only records','An independently justified distillation mask/coefficient andtraining-time gradient-budget protocol','Evidence that preserving teacher mistakes ishelpful'],distillation_is_behavior_consistency_not_groundtruth=True,proposed_primary=False,GPU_authorized=False,training_authorized=False,new_inference=0))
    print('Verified original teacher caches for59optimization photos;24old+35C17',flush=True)


if __name__=='__main__':run()
