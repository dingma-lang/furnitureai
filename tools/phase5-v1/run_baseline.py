"""No-download inference on a future approved public detection pilot only.

Explicit --resume validates inputs/code/settings and immutable completed stages.
CUDA/model failures propagate; finished stages survive for recovery, never empty predictions.
"""
import argparse
import hashlib
import re
from pathlib import Path
from PIL import Image, ImageOps
import detection_prep as d


def infer(bundle_path,out,checkpoint,expected_sha,device='cuda',resume=False):
    bundle_path=Path(bundle_path);out=Path(out);checkpoint=Path(checkpoint)
    if out.exists() and not resume: raise FileExistsError('Fresh run required, or explicit --resume')
    bundle=d.validate_public_bundle(bundle_path)
    if bundle.get('source')!='Open Images' or bundle.get('schema')!='phase5-public-boxes-v1': raise ValueError('Require audited public pilot bundle')
    d.assert_isolation(bundle['images'])
    frozen=[r for r in bundle['images'] if r['split']=='frozen_evaluation']
    if not frozen or len(bundle['images'])>150: raise ValueError('Bounded pilot requires 1-150 images')
    paths={}
    for r in frozen:
        if not all(r.get(k) is True for k in ('license_verified','annotations_verified','orientation_verified')) or r.get('annotation_provenance')!='published_open_images' or r.get('identity_uncertain'): raise ValueError('Unaudited example')
        p=(bundle_path.parent/r['file']).resolve()
        if not p.is_relative_to(bundle_path.parent.resolve()) or p.is_relative_to(d.ROOT/'data/abo') or d.sha(p)!=r['sha256']: raise ValueError('Unsafe/reserved/changed image')
        paths[r['image_id']]=p
    detector=d.detector_loader(checkpoint,expected_sha,device)
    # Bind processor/tokenizer too; model weights alone do not define preprocessing.
    model_files={str(p.relative_to(checkpoint)):d.sha(p) for p in checkpoint.rglob('*') if p.is_file()}
    binding=dict(manifest_sha256=d.sha(bundle_path),checkpoint_sha256=expected_sha,
                 checkpoint=str(checkpoint.resolve()),model_files=model_files,device=device,threshold=.15,nms_iou=.5,
                 categories=list(d.CLASSES),runner_sha256=d.sha(__file__),library_sha256=d.sha(d.__file__))
    if out.exists():
        if d.read(out/'input-lock.json')!=binding: raise ValueError('Resume binding mismatch')
    else:
        out.mkdir(parents=True);d.write_new(out/'input-lock.json',binding)
    stages={}
    for r in frozen:
        stage=out/(r['image_id']+'.json');stage_sum=out/(r['image_id']+'.sha256')
        if stage.exists():
            if not stage_sum.is_file() or stage_sum.read_text().strip()!=d.sha(stage): raise ValueError('Incomplete or corrupted stage; preserve and use a fresh output directory')
            record=d.read(stage)
            if record['image_id']!=r['image_id'] or record['image_sha256']!=r['sha256']: raise ValueError('Stage identity mismatch')
            stages[r['image_id']]=record;continue
        if stage_sum.exists(): raise ValueError('Orphan stage checksum')
        with Image.open(paths[r['image_id']]) as src: im=ImageOps.exif_transpose(src).convert('RGB')
        # No broad exception handler: CUDA failures propagate, leaving finished stages intact.
        detections,seconds=detector.detect(im)
        preds=[dict(label=p['label'],score=p['score'],box=[p['box'][0]/im.width,p['box'][1]/im.height,p['box'][2]/im.width,p['box'][3]/im.height]) for p in detections]
        record=dict(image_id=r['image_id'],image_sha256=r['sha256'],predictions=preds,seconds=seconds)
        d.write_new(stage,record)
        with stage_sum.open('x') as f:f.write(d.sha(stage)+'\n')
        stages[r['image_id']]=record
    if any(d.sha(checkpoint/p)!=h for p,h in model_files.items()): raise ValueError('Model or processor changed')
    result=dict(**binding,images=[stages[r['image_id']] for r in frozen])
    if (out/'predictions.json').exists():
        if d.read(out/'predictions.json')!=result: raise ValueError('Completed run differs')
    else:d.write_new(out/'predictions.json',result)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--checkpoint',type=Path,default=d.ROOT/'models/owlv2-original-baseline-20261103');ap.add_argument('--sha256',default=d.OWL_SHA);ap.add_argument('--device',default='cuda');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();infer(a.bundle,a.out,a.checkpoint,a.sha256,a.device,a.resume)
