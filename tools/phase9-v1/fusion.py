"""Pure coordinate transforms and conservative confidence-weighted consensus.

This is our tested fusion implementation, not an external WBF package. Scores
remain uncalibrated, and transformed input views are not independent captures.
"""
import math
from core import CLASSES

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def iou(a,b):
 v=inter(a,b);u=area(a)+area(b)-v;return v/u if u else 0.
def valid(p):
 b=p['box']
 if p['label'] not in CLASSES or not math.isfinite(p['score']) or not 0<=p['score']<=1 or len(b)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in b) or area(b)<=0:raise ValueError('Malformed normalized prediction')
def nms(predictions,threshold=.5,limit=300):
 out=[]
 for p in sorted(predictions,key=lambda p:-p['score']):
  valid(p)
  if not any(p['label']==q['label'] and iou(p['box'],q['box'])>threshold for q in out):out.append(dict(p))
  if len(out)==limit:break
 return out
def map_box(box,region,flip=False):
 if len(box)!=4 or area(box)<=0 or not all(math.isfinite(x) for x in box):raise ValueError('Invalid source box')
 x1,y1,x2,y2=box
 if flip:x1,x2=1-x2,1-x1
 a,b,c,d=region
 if not 0<=a<c<=1 or not 0<=b<d<=1:raise ValueError('Invalid view region')
 return [max(0,min(1,a+x1*(c-a))),max(0,min(1,b+y1*(d-b))),max(0,min(1,a+x2*(c-a))),max(0,min(1,b+y2*(d-b)))]
def internal_edge(box,region,epsilon=.025):
 a,b,c,d=region
 return bool(a>0 and box[0]<epsilon or b>0 and box[1]<epsilon or c<1 and box[2]>1-epsilon or d<1 and box[3]>1-epsilon)
def views():
 out=[dict(name='full960',size=960,region=[0,0,1,1],flip=False),dict(name='full768',size=768,region=[0,0,1,1],flip=False),dict(name='full1152',size=1152,region=[0,0,1,1],flip=False),dict(name='flip960',size=960,region=[0,0,1,1],flip=True)]
 for y in (0.,.375):
  for x in (0.,.375):out.append(dict(name=f'tile{len(out)-4}',size=960,region=[x,y,x+.625,y+.625],flip=False))
 return out
def consensus(groups,minimum=2,weighted=False,threshold=.55):
 clusters=[];ambiguities=0
 for view,items in groups.items():
  for p in sorted(items,key=lambda p:-p['score']):
   valid(p);eligible=[]
   for k,cluster in enumerate(clusters):
    if cluster[0][1]['label']!=p['label'] or any(v==view for v,_ in cluster):continue
    scores=[iou(p['box'],q['box']) for _,q in cluster]
    if min(scores)>=threshold:eligible.append((min(scores),k))
   if len(eligible)>1:
    ambiguities+=1;clusters.append([(view,dict(p,ambiguous_cluster_assignment=True))]);continue
   if eligible:clusters[max(eligible)[1]].append((view,p))
   else:clusters.append([(view,p)])
 output=[]
 for cluster in clusters:
  if len(cluster)<minimum:continue
  points=[p for _,p in cluster];p=dict(max(points,key=lambda p:p['score']))
  if weighted:
   total=sum(max(q['score'],1e-8) for q in points);p['box']=[sum(q['box'][j]*max(q['score'],1e-8) for q in points)/total for j in range(4)]
   p['score']=sum(q['score'] for q in points)/len(points)
  else:p['score']=min(q['score'] for q in points)
  p.update(support_views=[v for v,_ in cluster],support_count=len(cluster),confidence_is_calibrated=False)
  output.append(p)
 return nms(output),dict(unmatched_or_single_view_clusters=sum(len(x)<minimum for x in clusters),ambiguous_cluster_assignments=ambiguities)
def method(name,cache):
 full=cache['full960']
 if name=='baseline':return nms(full),{}
 scales={v:cache[v] for v in ('full960','full768','full1152')} if name.startswith('scale_') else {}
 if name=='scale_nms':return nms([p for ps in scales.values() for p in ps]),{}
 if name=='scale_agreement':return consensus(scales)
 if name=='scale_weighted':return consensus(scales,weighted=True)
 if name=='flip_agreement':return consensus({v:cache[v] for v in ('full960','flip960')},weighted=True)
 if name=='tiled_guarded':
  additions=[];boundary=0;supported=0
  for v in ('tile0','tile1','tile2','tile3'):
   for p in cache[v]:
    if p['internal_tile_edge']:boundary+=1;continue
    if any(q['label']==p['label'] and iou(q['box'],p['box'])>=.3 for q in full):supported+=1;continue
    if p['score']>=.25 and area(p['box'])<=.2:additions.append(p)
  return nms(full+additions),dict(boundary_fragment_candidates_not_added=boundary,existing_full_frame_boxes_preserved=True,full_frame_supported_tile_proposals=supported)
 raise ValueError('Unknown recipe')
