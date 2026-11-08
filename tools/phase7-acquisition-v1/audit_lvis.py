"""Streaming native counts, federated scope and bounded SQLite furniture index."""
import collections as C
import json
import math
import re
import sqlite3
from core import *
from reader import stream

def connect(path):
 db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA cache_size=-32768');return db

def geometry(annotation,width,height):
 flags=[];bbox=annotation.get('bbox')
 if not isinstance(bbox,list) or len(bbox)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in bbox):raise ValueError('Malformed/nonfinite native box')
 x,y,w,h=bbox
 if w<=0 or h<=0:flags.append('degenerate_box')
 if x<-.02 or y<-.02 or x+w>width+.02 or y+h>height+.02:flags.append('box_outside_image')
 if w*h/(width*height)<.005 or min(w/width,h/height)<.04:flags.append('small_furniture_region')
 if min(x,y)<1 or x+w>width-1 or y+h>height-1:flags.append('border_contact_proxy')
 mask=annotation.get('segmentation');kind='missing'
 if isinstance(mask,list) and mask:
  kind='polygons'
  for polygon in mask:
   if not isinstance(polygon,list) or len(polygon)<6 or len(polygon)%2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in polygon):raise ValueError('Malformed/nonfinite polygon')
 elif isinstance(mask,dict):
  kind='rle'
  if mask.get('size')!=[height,width] or not isinstance(mask.get('counts'),(str,list)):raise ValueError('Malformed native RLE')
 else:flags.append('missing_mask')
 return kind,flags

def run():
 if read(OUT/'archive-integrity.json')['status']!='PASS':raise ValueError('Archive integrity must pass before processing')
 dbpath=OUT/'furniture-index.sqlite'
 if dbpath.exists():raise FileExistsError('Audit index already exists; use a fresh audit version, no overwrite')
 db=connect(dbpath)
 db.executescript('''
 CREATE TABLE images(split TEXT,id INTEGER,width INTEGER,height INTEGER,url TEXT,neg TEXT,nex TEXT,PRIMARY KEY(split,id));
 CREATE TABLE facts(split TEXT,image INTEGER,category INTEGER,instances INTEGER,masks INTEGER,polygons INTEGER,rles INTEGER,quality TEXT,PRIMARY KEY(split,image,category));
 CREATE TABLE coco(id INTEGER PRIMARY KEY,source_split TEXT,width INTEGER,height INTEGER,url TEXT,flickr TEXT,license INTEGER);
 ''')
 summaries={};metadata={};overlays=[]
 targets={r['coco_image_id']:r for r in read(ROOT/'reports/phase7-data-strategy-v1/lvis-overlap-targets-v2.json')['targets']}
 for split in ('train','val'):
  path=OUT/f'lvis_v1_{split}.json.zip';cats={};total_images=0;negative=C.Counter();nonex=C.Counter();url_issues=[]
  for kind,r in stream(path):
   if kind=='categories':
    if r['id'] in cats:raise ValueError('Duplicate native category ID')
    cats[r['id']]=r
   elif kind=='images':
    iid=r['id'];url=r.get('coco_url','');m=re.search(r'/(\d{12})\.jpg$',url)
    if not isinstance(iid,int) or not m or int(m[1])!=iid or r['width']<=0 or r['height']<=0:raise ValueError('LVIS/COCO identity/dimension schema mismatch')
    neg=r['neg_category_ids'];nex=r['not_exhaustive_category_ids']
    if not all(isinstance(c,int) for c in neg+nex) or len(set(neg))!=len(neg) or len(set(nex))!=len(nex) or set(neg)&set(nex):raise ValueError('Conflicting native federated lists')
    selected_neg=sorted(set(neg)&RELEVANT);selected_nex=sorted(set(nex)&RELEVANT)
    negative.update(selected_neg);nonex.update(selected_nex)
    db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?)',(split,iid,r['width'],r['height'],url,json.dumps(selected_neg),json.dumps(selected_nex)))
    total_images+=1
    if total_images%10000==0:db.commit();space()
  db.commit()
  if len(cats)!=1203 or any(cats[c]['name']!=name for c,name in CORE.items()):raise ValueError('Native LVIS taxonomy changed')
  metadata[split]=cats
  counter=C.Counter();mask_counts=C.Counter();polygons=C.Counter();rles=C.Counter();quality=C.Counter();seen=set();total_annotations=0
  for kind,a in stream(path):
   if kind!='annotations':continue
   if a['id'] in seen:raise ValueError('Duplicate native annotation ID')
   seen.add(a['id']);total_annotations+=1;cid=a['category_id']
   if cid not in cats:raise ValueError('Unknown native category')
   if cid not in RELEVANT:continue
   im=db.execute('SELECT width,height,neg,nex FROM images WHERE split=? AND id=?',(split,a['image_id'])).fetchone()
   if im is None:raise ValueError('Dangling furniture annotation')
   if cid in json.loads(im[2]):raise ValueError('Positive annotation contradicts native negative list')
   kindmask,flags=geometry(a,im[0],im[1]);counter[cid]+=1;mask_counts[cid]+=kindmask!='missing';polygons[cid]+=kindmask=='polygons';rles[cid]+=kindmask=='rle'
   for flag in flags:quality[cid,flag]+=1
   old=db.execute('SELECT instances,masks,polygons,rles,quality FROM facts WHERE split=? AND image=? AND category=?',(split,a['image_id'],cid)).fetchone()
   if old:
    union=sorted(set(json.loads(old[4]))|set(flags));db.execute('UPDATE facts SET instances=?,masks=?,polygons=?,rles=?,quality=? WHERE split=? AND image=? AND category=?',(old[0]+1,old[1]+int(kindmask!='missing'),old[2]+int(kindmask=='polygons'),old[3]+int(kindmask=='rle'),json.dumps(union),split,a['image_id'],cid))
   else:db.execute('INSERT INTO facts VALUES(?,?,?,?,?,?,?,?)',(split,a['image_id'],cid,1,int(kindmask!='missing'),int(kindmask=='polygons'),int(kindmask=='rle'),json.dumps(flags)))
   if a['image_id'] in targets:overlays.append(dict(lvis_split=split,lvis_annotation=a,existing=targets[a['image_id']]))
   if total_annotations%10000==0:db.commit();space()
  db.commit();rows=[]
  for cid in sorted(RELEVANT):
   image_count=db.execute('SELECT COUNT(*) FROM facts WHERE split=? AND category=?',(split,cid)).fetchone()[0]
   not_exhaustive_positive=sum(cid in json.loads(nex) for nex, in db.execute('SELECT i.nex FROM images i JOIN facts f ON i.split=f.split AND i.id=f.image WHERE f.split=? AND f.category=?',(split,cid)))
   declared=cats[cid]
   if image_count!=declared['image_count'] or counter[cid]!=declared['instance_count']:raise ValueError('Computed/declared native count mismatch '+str((split,cid,image_count,counter[cid],declared)))
   rows.append(dict(native_category_id=cid,native_name=cats[cid]['name'],definition=cats[cid]['def'],synonyms=cats[cid]['synonyms'],frequency=declared['frequency'],
    core_project_group=CORE.get(cid),mapping='core_native' if cid in CORE else 'subtype_kept_separate_not_automatically_mapped',
    images=image_count,instances=counter[cid],boxes=counter[cid],segmented_instances=mask_counts[cid],polygon_instances=polygons[cid],rle_instances=rles[cid],
    explicit_negative_images=negative[cid],not_exhaustive_flag_images=nonex[cid],positive_not_exhaustive_images=not_exhaustive_positive,
    unknown_presence_images=total_images-image_count-negative[cid],quality_flags={k[1]:v for k,v in quality.items() if k[0]==cid},
    source_declared_counts_match=True))
  summaries[split]=dict(images=total_images,all_native_annotation_instances=total_annotations,all_categories=len(cats),furniture_categories=rows)
  print('AUDITED',split,total_images,'images',total_annotations,'all annotations',sum(counter.values()),'relevant furniture/subtype instances',flush=True)
 if set(r[0] for r in db.execute("SELECT id FROM images WHERE split='train'"))&set(r[0] for r in db.execute("SELECT id FROM images WHERE split='val'")):raise ValueError('LVIS official split image leakage')
 for cid in metadata['train']:
  for key in ('name','synonyms','synset','def'):
   if metadata['train'][cid][key]!=metadata['val'][cid][key]:raise ValueError('Source split taxonomy inconsistency')
 commit(OUT/'native-category-definitions.json',dict(categories=list(metadata['train'].values()),annotation_license='https://creativecommons.org/licenses/by/4.0/'))
 commit(OUT/'furniture-counts.json',dict(splits=summaries,core_categories=CORE,subtypes_kept_separate=SUBTYPES,missing_annotations_are_unknown=True,full_json_extracted=False))
 commit(OUT/'existing-image-lvis-overlays.json',dict(source_targets=77,records=overlays,photos_downloaded=0,partitions_modified=0,provisional_only=True))
 db.close();space()

if __name__=='__main__':run()
