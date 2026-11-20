"""Prepared inference-only recovery; GPU execution requires separate approval.

Raw tensors are durably saved BEFORE any NMS or metric code. Scoring runs in
a separate CPU process with no Torch/model imports. No training/resume entry.
"""
import argparse
import os
from pathlib import Path
import time
import uuid
import numpy as np
from c10 import *

DEST=ROOT/'data/open-images/experiments/phase10c-recovery-v1'
MAX_VRAM=int(6.5*2**30)

def proposal():
    entry=read(RUN/'comparison-v1/results.json')['entries']['heads-original-lr3e-5']
    return dict(schema='phase10c-inference-only-v1',checkpoint=entry['final_recovery_checkpoint'],checkpoint_sha256=entry['final_recovery_sha256'],
        manifest_sha256=sha(PREP/'eligibility-manifest.json'),image_ids=sorted([r['image_id'] for r in internal_rows()]),
        source_hashes={r['image_id']:r['sha256'] for r in internal_rows()},destination=str(DEST.relative_to(ROOT)),
        models=1,max_forwards=19,max_seconds=600,max_allocated_vram_bytes=MAX_VRAM,max_new_storage_bytes=128*2**20,
        max_raw_bytes_per_image=512000,precision='FP32 identical to historical validation',optimizer_updates=0,
        confidence=.15,ap_score_floor=.001,nms_iou=.5,nms_max_boxes=300,matching_iou=[.5,.75],
        frozen_or_reserved_images_allowed=False,training_allowed=False,approved=False)

def authorize(path):
    a=read(path)
    if a.get('approved') is not True or a.get('inference_only') is not True or a.get('training_allowed') is not False or not a.get('approval_reference'):
        raise PermissionError('Separate explicit inference-only approval required; current task is CPU-only')
    p=proposal()
    if a.get('proposal')!=p or a.get('code_hashes')!=code_hashes():raise ValueError('Approval/code/data/scope binding mismatch')
    if sha(ROOT/p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    contract=read(ROOT/'reports/phase10a-v1/input-contract.json')
    for name,digest in contract['model_files'].items():
        if sha(ROOT/'models/owlv2-original-baseline-20261103'/name)!=digest:raise ValueError('Pinned original/preprocessing changed')
    return p,dict(proposal=p,code_hashes=code_hashes(),authorization_sha256=sha(path),model_files=contract['model_files'])

def arrays_valid(arrays):
    if set(arrays)!=set(('logits','square_cxcywh','boxes','scores','labels')):raise ValueError('Raw array keys')
    n=len(arrays['scores'])
    if not 1<=n<=3600 or arrays['logits'].shape!=(n,7) or arrays['boxes'].shape!=(n,4) or arrays['square_cxcywh'].shape!=(n,4) or arrays['labels'].shape!=(n,):raise ValueError('Raw array dimensions')
    for name,arr in arrays.items():
        if arr.dtype!=np.dtype('int64' if name=='labels' else 'float32') or not np.isfinite(arr).all():raise ValueError('Raw dtype/integrity')
    if np.any((arrays['scores']<0)|(arrays['scores']>1)) or np.any((arrays['labels']<0)|(arrays['labels']>6)) or np.any((arrays['boxes']<0)|(arrays['boxes']>1)):raise ValueError('Raw value range')

def load_arrays(path):
    import zipfile
    if path.stat().st_size>512000:raise ValueError('Oversized raw record')
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())!=5 or sum(i.file_size for i in archive.infolist())>512000:raise ValueError('Unsafe raw archive')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays_valid(arrays);return arrays

def write_arrays(path,arrays):
    arrays_valid(arrays);tmp=path.with_name('.'+path.name+'-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f:
            np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        if tmp.stat().st_size>512000:raise ValueError('Raw image record limit')
        os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def infer(authorization,resume=False):
    p,binding=authorize(authorization) # Before Torch/CUDA import, directory creation or images.
    if DEST.is_symlink() or DEST.resolve()!=DEST:raise ValueError('Unsafe output destination')
    if DEST.exists() and not resume:raise FileExistsError('Explicit --resume required')
    DEST.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (DEST/'recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bp=DEST/'binding.json'
        if bp.exists():
            if read(bp)!=binding:raise ValueError('Recovery binding changed')
        else:save(bp,binding)
        completed={};prior_seconds=0
        for cp in sorted(DEST.glob('charge-*.json')):prior_seconds+=read(cp)['seconds']
        for r in internal_rows():
            ident=r['image_id'];raw=DEST/(ident+'.npz');receipt=DEST/(ident+'.receipt.json');intent=DEST/(ident+'.intent.json')
            if receipt.exists():
                rec=read(receipt)
                if rec['sha256']!=sha(raw) or rec['source_sha256']!=r['sha256']:raise ValueError('Completed raw cache changed')
                load_arrays(raw);completed[ident]=rec
            elif intent.exists() or raw.exists():raise RuntimeError('Uncertain prior forward/write boundary: stop; no automatic repeat')
        if len(completed)==19:
            complete=DEST/'raw-complete.json'
            if not complete.exists():save(complete,dict(binding=binding,images=19,actual_forwards_this_process=0,records=completed,completion_recovered_from_receipts=True))
            print('All19 cached; no forward repeated');return
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        import sys
        sys.path.insert(0,str(ROOT/'tools/phase10a-v1'))
        from expanded import HEAD_KEYS,from_square,cxcywh_to_xyxy
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback or automatic retry')
        torch.cuda.set_per_process_memory_fraction(.8,0)
        started=time.monotonic();charged=0
        def guard():
            nonlocal charged
            required=int((time.monotonic()-started)//10+1)*10
            while charged<required:
                save(DEST/('charge-'+uuid.uuid4().hex+'.json'),dict(seconds=10));charged+=10
            if prior_seconds+charged>600:raise RuntimeError('600-second cumulative inference budget')
            if torch.cuda.max_memory_allocated(0)>MAX_VRAM:raise RuntimeError('6.5GiB VRAM stop target')
            used=sum(f.stat().st_blocks*512 for f in DEST.rglob('*') if f.is_file())
            if used+2**20>128*2**20:raise RuntimeError('128MiB recovery storage budget')
            if (DEST/'STOP').exists():raise RuntimeError('Requested stop')
        guard();base=ROOT/'models/owlv2-original-baseline-20261103'
        model=Owlv2ForObjectDetection.from_pretrained(str(base),local_files_only=True,use_safetensors=True)
        state=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=True)
        if state['cursor']!=75 or state['binding']['dataset_sha256']!=p['manifest_sha256'] or set(state['weights'])!=HEAD_KEYS:raise ValueError('Wrong final checkpoint')
        parameters=dict(model.named_parameters())
        with torch.no_grad():
            for name,value in state['weights'].items():
                if value.shape!=parameters[name].shape or not torch.isfinite(value).all():raise ValueError('Checkpoint tensor mismatch')
                parameters[name].copy_(value)
        del state
        model.requires_grad_(False);model.eval();model.to('cuda')
        processor=Owlv2Processor.from_pretrained(str(base),local_files_only=True)
        forward_calls=0
        for row in sorted(internal_rows(),key=lambda r:r['image_id']):
            ident=row['image_id']
            if ident in completed:continue
            guard();path=source_path(row)
            with Image.open(path) as im:
                if im.size!=(row['width'],row['height']):raise ValueError('Image coordinate frame changed')
                batch=processor(text=[['a photo of a '+c for c in CLASSES]],images=im.convert('RGB'),return_tensors='pt')
            save(DEST/(ident+'.intent.json'),dict(source_sha256=row['sha256'],forward_reserved=True))
            with torch.inference_mode():out=model(**{k:v.to('cuda') for k,v in batch.items()})
            forward_calls+=1
            scores,labels=out.logits[0].float().sigmoid().max(-1)
            arrays=dict(logits=out.logits[0].float().cpu().numpy(),square_cxcywh=out.pred_boxes[0].float().cpu().numpy(),
                boxes=from_square(cxcywh_to_xyxy(out.pred_boxes[0].float()),row['width'],row['height']).clamp(0,1).cpu().numpy(),
                scores=scores.cpu().numpy(),labels=labels.cpu().numpy())
            raw=DEST/(ident+'.npz');write_arrays(raw,arrays)
            rec=dict(sha256=sha(raw),source_sha256=row['sha256'],checkpoint_sha256=p['checkpoint_sha256'],queries=len(scores),bytes=raw.stat().st_size)
            save(DEST/(ident+'.receipt.json'),rec);completed[ident]=rec
            del out,batch,arrays;torch.cuda.synchronize(0);guard()
        if sha(ROOT/p['checkpoint'])!=p['checkpoint_sha256']:raise ValueError('Checkpoint changed during inference')
        save(DEST/'raw-complete.json',dict(binding=binding,images=len(completed),actual_forwards_this_process=forward_calls,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),seconds_this_process=time.monotonic()-started,records=completed))

def score():
    # Independent CPU process; import of this module never imports Torch.
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('Set CUDA_VISIBLE_DEVICES empty for CPU scoring')
    import metric_cpu as metrics
    done=read(DEST/'raw-complete.json');p=proposal()
    if done['binding']['proposal']!=p or done['images']!=19 or done['binding']['code_hashes']!=code_hashes():raise ValueError('Raw completion scope/code changed')
    preds={}
    for row in internal_rows():
        ident=row['image_id'];raw=DEST/(ident+'.npz');rec=read(DEST/(ident+'.receipt.json'))
        if rec!=done['records'][ident] or sha(raw)!=rec['sha256'] or rec['source_sha256']!=row['sha256'] or rec['checkpoint_sha256']!=p['checkpoint_sha256']:raise ValueError('Prediction provenance changed')
        arrays=load_arrays(raw);ps=[]
        for b,s,c in zip(arrays['boxes'].tolist(),arrays['scores'].tolist(),arrays['labels'].tolist()):
            if s>=.001 and b[2]>b[0] and b[3]>b[1]:ps.append(dict(box=b,score=s,label=CLASSES[c]))
        preds[ident]=metrics.nms(ps,.5,300)
    # Checkpointed predictions FIRST; isolated metrics can be replayed CPU-only.
    target=DEST/'post-nms-predictions.json'
    if target.exists():
        if read(target)!=preds:raise ValueError('Saved predictions differ')
    else:save(target,preds)
    rows=[metrics.scopes(r) for r in internal_rows()];result=metrics.evaluate(rows,preds)
    save(DEST/('metrics-cpu-'+uuid.uuid4().hex+'.json'),dict(metrics=result,predictions_sha256=sha(target),protocol='unchanged Phase10B; equivalence-tested CPU control flow',GPU=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['infer','score']);parser.add_argument('--authorization',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='infer':
        if args.authorization is None:parser.error('--authorization required')
        infer(args.authorization,args.resume)
    else:score()
