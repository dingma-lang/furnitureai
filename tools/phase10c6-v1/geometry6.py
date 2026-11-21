"""Raster proposals with exact continuous polygon containment certification.

Pillow rasterization is only a search aid, not asserted to equal COCO raster
decoding. Acceptance uses the original continuous simple polygon and margins.
"""
import math
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion,distance_transform_edt
from common6 import POLICY,overlap

def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def on(a,b,p):return abs(cross(a,b,p))<1e-8 and min(a[0],b[0])-1e-8<=p[0]<=max(a[0],b[0])+1e-8 and min(a[1],b[1])-1e-8<=p[1]<=max(a[1],b[1])+1e-8
def intersects(a,b,c,d):
    x,y,z,w=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    return (x*y<0 and z*w<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
def edges(poly):return list(zip(poly,poly[1:]+poly[:1]))
def point_inside(p,poly):
    if any(on(a,b,p) for a,b in edges(poly)):return False
    x,y=p;inside=False
    for a,b in edges(poly):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside
def distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];n=dx*dx+dy*dy
    if not n:return math.dist(p,a)
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n));return math.dist(p,(a[0]+t*dx,a[1]+t*dy))
def simple(poly):
    n=len(poly)
    if not 3<=n<=2000:return False
    es=edges(poly)
    for i,(a,b) in enumerate(es):
        if a==b:return False
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            if intersects(a,b,*es[j]):return False
    return True
def certify(rect,poly,margin=2):
    x1,y1,x2,y2=rect;corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    if not x1<x2 or not y1<y2 or not all(point_inside(c,poly) for c in corners):return False
    for a,b in edges(poly):
        for c,d in edges(corners):
            if intersects(a,b,c,d):return False
            if min(distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))<margin:return False
    return True
def polygons(annotation,w,h):
    seg=annotation.get('segmentation');out=[]
    if annotation.get('iscrowd',0) or not isinstance(seg,list):return out
    for raw in seg:
        if not isinstance(raw,list) or len(raw)<6 or len(raw)%2 or len(raw)>4000:continue
        if not all(type(x) in (int,float) and math.isfinite(x) for x in raw):continue
        poly=list(zip(raw[::2],raw[1::2]))
        if poly[0]==poly[-1]:poly.pop()
        # Remove consecutive duplicate vertices without moving coordinates.
        poly=[v for i,v in enumerate(poly) if i==0 or v!=poly[i-1]]
        if not all(0<=x<=w and 0<=y<=h for x,y in poly) or not simple(poly):continue
        out.append(poly)
    return out
def propose(annotation,w,h,protected):
    best=None
    for pi,poly in enumerate(polygons(annotation,w,h)):
        mask=Image.new('1',(w,h));ImageDraw.Draw(mask).polygon(poly,fill=1);array=binary_erosion(np.asarray(mask,dtype=bool),iterations=3,border_value=0)
        for p in protected:
            b=p['box'];x1=max(0,math.floor(b[0]*w)-2);y1=max(0,math.floor(b[1]*h)-2);x2=min(w,math.ceil(b[2]*w)+2);y2=min(h,math.ceil(b[3]*h)+2);array[y1:y2,x1:x2]=False
        if not array.any():continue
        dist=distance_transform_edt(array);cy,cx=np.unravel_index(np.argmax(dist),dist.shape);radius=int(dist[cy,cx]/math.sqrt(2))-2
        while radius>=6:
            b=[int(cx-radius),int(cy-radius),int(cx+radius),int(cy+radius)]
            if (b[2]-b[0])*(b[3]-b[1])>=POLICY['minimum_area_pixels'] and certify(b,poly):
                normalized=[b[0]/w,b[1]/h,b[2]/w,b[3]/h]
                if any(overlap(normalized,p['box']) for p in protected):break
                rec=dict(box_pixels=b,box=normalized,polygon_index=pi,polygon=[list(v) for v in poly],boundary_margin_pixels=2,source_polygon_interior_verified=True)
                if best is None or (b[2]-b[0])*(b[3]-b[1])>(best['box_pixels'][2]-best['box_pixels'][0])*(best['box_pixels'][3]-best['box_pixels'][1]):best=rec
                break
            radius-=1
    return best
