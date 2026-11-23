"""Continuous polygon/rectangle area diagnostics,never a relaxed admission mask."""
import math,sys
from common8q import ROOT,POLICY,area,overlap
sys.path.insert(0,str(ROOT/'tools/phase10c6-v1'))
from geometry6 import polygons,certify

def clipped_polygon(poly,rect):
    result=list(poly)
    for dim,edge,lower in ((0,rect[0],True),(0,rect[2],False),(1,rect[1],True),(1,rect[3],False)):
        previous=result;result=[]
        if not previous:break
        a=previous[-1];ain=(a[dim]>=edge if lower else a[dim]<=edge)
        for b in previous:
            bin=(b[dim]>=edge if lower else b[dim]<=edge)
            if ain!=bin:
                t=(edge-a[dim])/(b[dim]-a[dim]);result.append(tuple(a[j]+t*(b[j]-a[j]) for j in (0,1)))
            if bin:result.append(b)
            a=b;ain=bin
    return result

def polygon_area(poly):
    if len(poly)<3:return 0.
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2

def coverage(poly,rect):
    return min(1.,max(0.,polygon_area(clipped_polygon(poly,rect))/area(rect))) if area(rect)>0 else 0.

def admission(box_px,width,height,polys,protected):
    # Match the existing E6/C6 policy;do not allow partial mask intersection,
    # source bbox context or an expanded display crop to authorize a negative.
    rect=[math.floor(box_px[0]),math.floor(box_px[1]),math.ceil(box_px[2]),math.ceil(box_px[3])]
    norm=[rect[0]/width,rect[1]/height,rect[2]/width,rect[3]/height]
    bounds=0<=rect[0]<rect[2]<=width and 0<=rect[1]<rect[3]<=height
    sizes=min(rect[2]-rect[0],rect[3]-rect[1])>=POLICY['minimum_side_pixels']
    blocked=[p['key'] for p in protected if overlap(norm,p['box'])]
    full=[i for i,p in enumerate(polys) if bounds and certify(rect,p,margin=POLICY['boundary_margin_pixels'])]
    return dict(rectangle_pixels=rect,bounds_ok=bounds,minimum_side_ok=sizes,protected_overlaps=blocked,certified_polygon_indices=full,
                geometric_eligible=bounds and sizes and not blocked and bool(full))
