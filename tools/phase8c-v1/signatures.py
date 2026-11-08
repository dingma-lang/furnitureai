"""Model-free signatures compatible with historical Phase6D screening."""
from pilot_core import *

def signature(image):
 import numpy as np
 from PIL import ImageOps
 rgb=ImageOps.exif_transpose(image).convert('RGB');gray=np.asarray(rgb.convert('L'),dtype=float);small=np.asarray(rgb.convert('L').resize((9,8)),dtype=float)
 dh=sum(int(b)<<i for i,b in enumerate((small[:,1:]>small[:,:-1]).ravel()))
 a=np.asarray(rgb.convert('L').resize((32,32)),dtype=float);cos=np.cos(np.pi*(2*np.arange(32)[None,:]+1)*np.arange(8)[:,None]/64);coeff=(cos@a@cos.T).ravel()[1:];ph=sum(int(b)<<i for i,b in enumerate(coeff>np.median(coeff)))
 lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
 return dict(decoded_pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),dhash64=f'{dh:016x}',phash63=f'{ph:016x}',gray_std=float(gray.std()),laplacian_variance=float(lap.var()),white_fraction=float(np.mean(np.all(np.asarray(rgb)>245,axis=2))))

def duplicate(a,b):
 if a.get('sha256') and a['sha256']==b.get('sha256') or a.get('decoded_pixel_sha256') and a['decoded_pixel_sha256']==b.get('decoded_pixel_sha256'):return 'exact'
 if any(a.get(k) and b.get(k) and (int(a[k],16)^int(b[k],16)).bit_count()<=threshold for k,threshold in [('dhash64',4),('phash63',6)]):return 'suspected_perceptual'
 return None

def xyxy(a):
 x,y,w,h=a['bbox'];return [x,y,x+w,y+h]

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/union if union else 0.
