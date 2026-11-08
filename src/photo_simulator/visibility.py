"""Conservative image-space visibility and bounded classical repair regions."""
import cv2
import numpy as np


def coverage_masks(unknown, protection, alpha, margin=2):
    """Only exactly opaque pixels can hide old furniture; all soft edges need fill.

    All arrays share original-photo coordinates. Shadows do not count as coverage.
    Margin extends repair under the opaque edge, never beyond unknown/protection.
    """
    a = np.asarray(alpha)
    if a.shape != unknown.shape or protection.shape != unknown.shape or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Coverage alpha must be finite [0,1] at original image dimensions')
    if type(margin) != int or not 0 <= margin <= 12:
        raise ValueError('Visibility margin must be an integer from 0 to 12 pixels')
    opaque = (a == 1) & ~protection
    exposed = unknown & ~opaque & ~protection
    safe = opaque
    if margin:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
        safe = cv2.erode(opaque.astype('uint8'), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    target = unknown & ~safe & ~protection
    return target, exposed, opaque


def _regions(target, context):
    h, w = target.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(target.astype('uint8'), 8)
    if count > 257:
        components = stats[1:]
        return [[max(0,int(components[:,0].min())-context),max(0,int(components[:,1].min())-context),
                 min(w,int((components[:,0]+components[:,2]).max())+context),min(h,int((components[:,1]+components[:,3]).max())+context)]]
    boxes = [[max(0,int(x)-context), max(0,int(y)-context), min(w,int(x+ww)+context), min(h,int(y+hh)+context)] for x,y,ww,hh,_ in stats[1:]]
    # Merge overlapping context windows to avoid order-dependent seams and duplicate work.
    merged = []
    for box in boxes:
        i = 0
        while i < len(merged):
            b = merged[i]
            if box[0] <= b[2] and b[0] <= box[2] and box[1] <= b[3] and b[1] <= box[3]:
                box = [min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3])]
                merged.pop(i); i = 0
            else: i += 1
        merged.append(box)
    if len(merged) > 32:
        return [[min(b[0] for b in merged),min(b[1] for b in merged),max(b[2] for b in merged),max(b[3] for b in merged)]]
    return merged


def inpaint_regions(original, unknown, protection, target, method, radius, resolution, context):
    """Reuse V2's Telea/NS solvers, exclude all old furniture as donors.

    Repair only context ROIs; copy only target pixels, with exact original guards.
    Intermediates may fill hidden pixels as solver holes, not authorized output.
    """
    h,w = target.shape
    background = original.copy()
    blocked = unknown | protection
    regions = _regions(target, context)
    if any(blocked[y1:y2,x1:x2].all() for x1,y1,x2,y2 in regions):
        regions = [[0,0,w,h]]
    scale = min(1., resolution/max(w,h))
    processed = 0; records = []
    for x1,y1,x2,y2 in regions:
        work = blocked[y1:y2,x1:x2]
        if work.all(): raise ValueError('No observed donors for exposed-region repair')
        rgb = original[y1:y2,x1:x2]
        size = (max(8,round((x2-x1)*scale)),max(8,round((y2-y1)*scale)))
        small = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA) if scale<1 else rgb
        mask = cv2.resize(work.astype('uint8')*255,size,interpolation=cv2.INTER_NEAREST)
        if not (mask==0).any(): raise ValueError('No observed donors at the requested repair resolution')
        filled = cv2.inpaint(small,mask,max(1,radius*scale),cv2.INPAINT_NS if method=='ns' else cv2.INPAINT_TELEA)
        if scale<1: filled=cv2.resize(filled,(x2-x1,y2-y1),interpolation=cv2.INTER_LINEAR)
        active=target[y1:y2,x1:x2]
        background[y1:y2,x1:x2][active]=filled[active]
        processed += size[0]*size[1]
        records.append([x1,y1,x2,y2])
    return background, {'roi_bounds':records,'solver_pixels':processed,'solver_regions':len(records),'donors':'All original furniture and protected foreground excluded'}


def visible_seam_metrics(background, target, alpha, protection):
    """Boundary contrast weighted by visible background contribution, not realism."""
    values=[];weights=[]
    transmittance=(1-alpha)*~protection
    for axis in (0,1):
        boundary=np.diff(target.astype(np.int8),axis=axis)!=0
        contrast=np.abs(np.diff(background.astype(np.float32),axis=axis)).mean(axis=2)
        a=transmittance[:-1,:] if axis==0 else transmittance[:,:-1]
        b=transmittance[1:,:] if axis==0 else transmittance[:,1:]
        weight=np.maximum(a,b);active=boundary&(weight>0)
        values.extend(contrast[active]);weights.extend(weight[active])
    v=np.asarray(values);weight=np.asarray(weights)
    return {'visible_boundary_pairs':len(v),
            'mean_background_jump_weighted_0_255':float(np.average(v,weights=weight)) if len(v) else None,
            'mean_contribution_jump_0_255':float((v*weight).mean()) if len(v) else None,
            'note':'Pairs hidden under fully opaque coverage excluded; contrast is not a perceptual accuracy score'}
