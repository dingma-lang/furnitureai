"""Full-resolution authorization boundary, raw retention, diagnostics and review crops."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from .backends import restore,BackendUnavailable


def mask_array(mask,size):
    a=np.asarray(mask)
    if a.shape!=(size[1],size[0]) or not np.isfinite(a).all():raise ValueError('Masks must be finite single-channel arrays at original image dimensions')
    return a>0


def guard(original,candidate,allowed):
    a=np.asarray(candidate)
    if a.shape!=original.shape or a.dtype!=np.uint8:raise ValueError('Candidate must be uint8 RGB at the requested resolution')
    result=original.copy();result[allowed]=a[allowed];return result


def seam_metric(image,allowed):
    # Adjacent RGB difference across permission boundary; not a realism score.
    samples=[]
    for axis in (0,1):
        edge=np.diff(allowed.astype(np.int8),axis=axis)!=0
        diff=np.abs(np.diff(image.astype(np.float32),axis=axis)).mean(axis=2)
        if edge.any():samples.append(diff[edge])
    v=np.concatenate(samples) if samples else np.array([])
    return {'boundary_neighbor_pairs':len(v),'mean_rgb_jump_0_255':float(v.mean()) if len(v) else None,'p95_rgb_jump_0_255':float(np.percentile(v,95)) if len(v) else None,'interpretation':'Boundary contrast diagnostic, not a perceptual or geometric accuracy score'}


def evaluate(original_image,removal_mask,protected_mask,configuration,output_dir,backend_function=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();config=dict(configuration)
    report={'status':'failed','backend':config.get('backend','classical'),'configuration':config,'neural_inference_executed':False}
    try:
        image=original_image.convert('RGB');original=np.array(image);original_hash=hashlib.sha256(original.tobytes()).hexdigest()
        mask=mask_array(removal_mask,image.size);protected=mask_array(protected_mask,image.size)
        crop=config.get('crop',[0,0,image.width,image.height])
        if not isinstance(crop,list) or len(crop)!=4 or any(type(v)!=int for v in crop):raise ValueError('Crop must be four integer original-image coordinates')
        x1,y1,x2,y2=crop
        if not 0<=x1<x2<=image.width or not 0<=y1<y2<=image.height:raise ValueError('Crop must lie inside original photograph')
        allowed=mask & ~protected;window=np.zeros_like(allowed);window[y1:y2,x1:x2]=True;allowed &= window
        if not allowed.any():raise ValueError('No authorized restoration pixels in this crop')
        Image.fromarray(allowed.astype('uint8')*255).save(out/'permitted-mask.png')
        image.crop(crop).save(out/'input-crop.png');Image.fromarray(allowed[y1:y2,x1:x2].astype('uint8')*255).save(out/'model-mask.png')
        raw,mapped,meta=(backend_function or restore)(image.crop(crop),Image.fromarray(allowed[y1:y2,x1:x2].astype('uint8')*255),Image.fromarray(protected[y1:y2,x1:x2].astype('uint8')*255),config)
        raw.save(out/'raw-native.png')  # Saved before compositing or mask enforcement.
        if mapped.size!=(x2-x1,y2-y1):raise ValueError('Mapped candidate size mismatch')
        candidate=original.copy();candidate[y1:y2,x1:x2]=np.array(mapped.convert('RGB'));Image.fromarray(candidate).save(out/'raw-mapped.png')
        final=guard(original,candidate,allowed);Image.fromarray(final).save(out/'background.png')
        changed=np.any(candidate!=original,axis=2);final_changed=np.any(final!=original,axis=2)
        report.update(status='complete',neural_inference_executed=config.get('backend') in ('lama','diffusion'),backend_diagnostics=meta,original_rgb_sha256=original_hash,output_resolution=image.size,raw_native_resolution=raw.size,crop=crop,permitted_pixels=int(allowed.sum()),raw_protected_pixel_differences=int(changed[protected].sum()),raw_outside_mask_pixel_differences=int(changed[~allowed].sum()),protected_pixel_differences=int(final_changed[protected].sum()),outside_mask_pixel_differences=int(final_changed[~allowed].sum()),seam=seam_metric(final,allowed),original_seam=seam_metric(original,allowed),original_unchanged=hashlib.sha256(np.array(original_image.convert('RGB')).tobytes()).hexdigest()==original_hash)
        assert report['protected_pixel_differences']==report['outside_mask_pixel_differences']==0
        regions=config.get('review_regions',{})
        for name,box in regions.items():
            if not name.replace('_','').isalnum():raise ValueError('Invalid review-region name')
            Image.fromarray(final).crop(box).save(out/('region-'+name+'.png'))
    except BackendUnavailable as exc:report.update(status='unavailable',error=str(exc))
    except Exception as exc:report.update(status='failed',error=type(exc).__name__+': '+str(exc))
    report.update(total_seconds=time.perf_counter()-start,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),rss_scope='Process lifetime high-water mark including imports; each CLI case uses a fresh subprocess')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report
