"""Classical, image-space restoration. Filled pixels remain unknown/estimated."""
from pathlib import Path
import hashlib
import json
import math
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(name + ' must be numeric')
    try:
        n = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ' must be numeric') from exc
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError(f'{name} must be finite and between {low} and {high}')
    return n


def restore_background(original, removal, protection, options, exclusion=None, override=None, cache_dir=None, coverage_alpha=None, override_mask=None):
    """Preserve observed pixels; exclude protected foreground from fill donors.

    Manual patches copy an explicitly selected observed rectangle at 1:1 scale.
    They are estimates of hidden appearance, never newly measured background.
    Full-mode cache excludes placement. Visible-mode cache includes actual target
    coverage; shadows never affect restoration. Scoped uploads cannot overwrite
    regions newly revealed after the upload.
    """
    start = perf_counter()
    h, w = original.shape[:2]
    method = options.get('repair_method', 'telea')
    if method not in ('telea', 'ns'):
        raise ValueError('Repair method must be telea or ns')
    resolution = int(number(options.get('repair_resolution', 1000), 'Repair resolution', 256, 4096))
    radius = number(options.get('inpaint_radius', 5), 'Fill radius', 1, 15)
    expansion = int(number(options.get('mask_expand', 0), 'Mask expansion', 0, 12))
    exclude_foreground = bool(options.get('exclude_foreground_donors', False))
    unknown = removal.copy()
    if expansion:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*expansion+1, 2*expansion+1))
        unknown = cv2.dilate(unknown.astype(np.uint8), kernel).astype(bool)
    unknown &= ~protection
    if exclusion is not None:
        unknown &= ~exclusion
    if unknown.mean() > .72:
        raise ValueError('Expanded removal mask covers too much of the photograph')
    scope = options.get('restoration_scope', 'full')
    if scope not in ('full', 'visible'):
        raise ValueError('Restoration scope must be full or visible')
    target = unknown
    visibility = None
    roi_info = {}
    if scope == 'visible':
        from .visibility import coverage_masks
        margin_value = number(options.get('visibility_margin', 2), 'Visibility margin', 0, 12)
        context_value = number(options.get('visibility_context', 64), 'Visibility context', 16, 256)
        if int(margin_value) != margin_value or int(context_value) != context_value:
            raise ValueError('Visibility margin and context must be whole pixels')
        if coverage_alpha is None:
            raise ValueError('Visibility restoration requires transformed replacement alpha')
        target, exposed, opaque = coverage_masks(unknown, protection, coverage_alpha, int(margin_value))
        visibility = dict(exposed_pixels=int(exposed.sum()), restoration_pixels=int(target.sum()),
                          original_unknown_pixels=int(unknown.sum()), opaque_coverage_pixels=int(opaque.sum()),
                          partially_transparent_pixels=int(((coverage_alpha>0)&(coverage_alpha<1)&unknown&~protection).sum()),
                          margin=int(margin_value), context=int(context_value),
                          hidden_skipped_pixels=int((unknown&~target).sum()),
                          background_is_partial=True)
    patches = options.get('background_patches', [])
    if not isinstance(patches, list) or len(patches) > 20:
        raise ValueError('At most 20 explicit background patches are supported')
    normalized = []
    for patch in patches:
        if not isinstance(patch, dict) or not isinstance(patch.get('source'), list) or len(patch['source']) != 4 or not isinstance(patch.get('target'), list) or len(patch['target']) != 2:
            raise ValueError('Background patch needs source [x,y,width,height] and target [x,y]')
        sx, sy, pw, ph = [int(number(v, 'Patch coordinate', 0, max(w,h))) for v in patch['source']]
        tx, ty = [int(number(v, 'Patch target', 0, max(w,h))) for v in patch['target']]
        if min(pw, ph) < 2 or sx+pw>w or sy+ph>h or tx+pw>w or ty+ph>h or pw*ph>w*h*.2:
            raise ValueError('Background patch is out of bounds or too large')
        if (unknown | protection)[sy:sy+ph,sx:sx+pw].any():
            raise ValueError('Patch donor must contain only observed, unprotected background pixels')
        feather = number(patch.get('feather', 3), 'Patch feather', 0, 12)
        normalized.append(dict(source=[sx,sy,pw,ph], target=[tx,ty], feather=feather))
    if override is not None and np.asarray(override).shape != original.shape:
        raise ValueError('Manual background must match original image dimensions')
    if override_mask is not None and (np.asarray(override_mask).shape != unknown.shape or not np.isfinite(override_mask).all()):
        raise ValueError('Background correction scope must match original dimensions')
    override_target = target if override_mask is None else target & (np.asarray(override_mask)>0)
    parameters = dict(version=2,method=method,resolution=resolution,radius=radius,expansion=expansion,
                      exclude_foreground=exclude_foreground,patches=normalized)
    if visibility is not None: parameters.update(version=3,scope=scope,visibility=visibility)
    digest = hashlib.sha256(json.dumps(parameters,sort_keys=True).encode())
    for a in (original,unknown,protection):digest.update(np.ascontiguousarray(a).tobytes())
    if visibility is not None: digest.update(np.ascontiguousarray(target).tobytes())
    if override is not None:digest.update(np.ascontiguousarray(override).tobytes())
    if override is not None and override_mask is not None: digest.update(np.ascontiguousarray(override_target).tobytes())
    key = digest.hexdigest()
    cache = Path(cache_dir)/('background-'+key+'.png') if cache_dir else None
    hit = False
    if cache and cache.exists():
        try:
            background = np.array(Image.open(cache).convert('RGB'))
            hit = background.shape == original.shape
        except (OSError,ValueError):pass
    if not hit:
        if visibility is not None:
            from .visibility import inpaint_regions
            background, roi_info = inpaint_regions(original, unknown, protection, target, method, radius, resolution, visibility['context'])
        else:
            # Protected objects are holes for the solver, then restored exactly below.
            work = unknown | protection if exclude_foreground else unknown
            if work.all():raise ValueError('No observed background donors remain')
            scale = min(1., resolution/max(w,h))
            size = (max(8,round(w*scale)),max(8,round(h*scale)))
            rgb = cv2.resize(original,size,interpolation=cv2.INTER_AREA) if scale<1 else original
            mask = cv2.resize(work.astype(np.uint8)*255,size,interpolation=cv2.INTER_NEAREST)
            if not (mask==0).any():raise ValueError('No background donors at this repair resolution')
            filled = cv2.inpaint(rgb,mask,max(1,radius*scale),cv2.INPAINT_NS if method=='ns' else cv2.INPAINT_TELEA) if mask.any() else rgb.copy()
            if scale<1:filled=cv2.resize(filled,(w,h),interpolation=cv2.INTER_LINEAR)
            background = np.where(unknown[:,:,None],filled,original).astype(np.uint8)
        for patch in normalized:
            sx,sy,pw,ph=patch['source'];tx,ty=patch['target'];feather=patch['feather']
            yy,xx=np.mgrid[:ph,:pw];distance=np.minimum.reduce([xx+1,yy+1,pw-xx,ph-yy])
            blend=np.minimum(1.,distance/max(1,feather)) * target[ty:ty+ph,tx:tx+pw]
            target=background[ty:ty+ph,tx:tx+pw];donor=original[sy:sy+ph,sx:sx+pw]
            target[:]=np.rint(donor*blend[:,:,None]+target*(1-blend[:,:,None])).astype(np.uint8)
        if override is not None:background[override_target]=np.asarray(override,dtype=np.uint8)[override_target]
        background[~target]=original[~target]
        if cache:
            cache.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(background).save(cache)
    # Cache is never authoritative over known pixels.
    background[~target]=original[~target]
    return background, unknown, dict(**parameters,cache_key=key,cache_hit=hit,seconds=perf_counter()-start,
                                    manual_override_applied_pixels=int(override_target.sum()) if override is not None else 0,roi=roi_info,unknown_pixels=int(unknown.sum()),manual_override=override is not None,
                                    geometry='Hidden regions remain unobserved; appearance is estimated')


def refine_alpha(asset, options):
    """Only narrow boundary RGB/alpha changes; preserve dimensions and opaque core."""
    data=np.array(asset.convert('RGBA'));alpha=data[:,:,3].astype(np.float32)/255
    feather=number(options.get('edge_feather',0), 'Edge feather',0,1.5)
    contract=number(options.get('edge_contract',0), 'Edge contraction',0,1.5)
    decontam=number(options.get('edge_decontam',0), 'Edge decontamination',0,1)
    if not (feather or contract or decontam):return asset.copy(),dict(changed_pixels=0)
    core=alpha>.98
    inside=cv2.distanceTransform(core.astype(np.uint8),cv2.DIST_L2,5)
    outside=cv2.distanceTransform((alpha<.01).astype(np.uint8),cv2.DIST_L2,5)
    band=(inside<=3)&(outside<=3)
    adjusted=alpha.copy()
    if contract:
        eroded=cv2.erode(alpha,np.ones((3,3),np.uint8));adjusted=alpha+(eroded-alpha)*min(1,contract)
        if contract>1:adjusted+= (cv2.erode(eroded,np.ones((3,3),np.uint8))-adjusted)*(contract-1)
    if feather:adjusted=cv2.GaussianBlur(adjusted,(0,0),feather)
    data[:,:,3][band]=np.rint(np.clip(adjusted[band],0,1)*255).astype(np.uint8)
    if decontam and (inside>2).any():
        from scipy.ndimage import distance_transform_edt
        _,indices=distance_transform_edt(inside<=2,return_indices=True)
        edge=band & (alpha>0) & (inside<=2)
        nearest=data[indices[0],indices[1],:3].astype(np.float32)
        data[:,:,:3][edge]=np.rint(data[:,:,:3][edge]*(1-decontam)+nearest[edge]*decontam).astype(np.uint8)
    original=np.array(asset.convert('RGBA'))
    return Image.fromarray(data),dict(changed_pixels=int(np.any(data!=original,axis=2).sum()),
                                    feather=feather,contraction=contract,decontamination=decontam,
                                    opaque_core_unchanged=bool(np.array_equal(data[inside>3],original[inside>3])))


def contact_shadow(alpha, placement, protection, options):
    h,w=alpha.shape
    opacity=number(options.get('shadow_opacity',.18),'Shadow opacity',0,.6)
    softness=number(options.get('shadow_softness',3),'Shadow softness',0,12)
    dx=number(options.get('shadow_offset_x',0),'Shadow X offset',-30,30)
    dy=number(options.get('shadow_offset_y',2),'Shadow Y offset',-30,30)
    coverage=number(options.get('shadow_coverage',.025),'Shadow contact width fraction',.005,.08)
    image=Image.new('L',(w,h));draw=ImageDraw.Draw(image);ys,xs=np.nonzero(alpha>.3);contacts=[]
    if len(xs):
        left,right=int(xs.min()),int(xs.max());third=max(1,(right-left)//3)
        for lo,hi in [(left,left+third),(right-third,right+1)]:
            yy,xx=np.nonzero(alpha[:,lo:hi]>.3)
            if len(xx):
                near=yy>=np.percentile(yy,99);cx=float(np.median(xx[near])+lo)+dx;cy=float(yy.max())+dy
                rx=max(2,placement['width']*coverage);ry=max(1,rx*.1)
                draw.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),fill=round(opacity*255));contacts.append([cx,cy])
    if softness:image=image.filter(ImageFilter.GaussianBlur(softness))
    layer=np.asarray(image).astype(np.float32)/255;layer[protection]=0
    return layer,dict(opacity=opacity,softness=softness,offset=[dx,dy],coverage=coverage,contacts=contacts,
                      lighting='Editable 2D appearance approximation, not physical lighting')
