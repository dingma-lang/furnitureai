"""Detect descendant-only source regions without relabeling them as parents."""
import collections as C
import csv
from core import *

def overlap(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]))
 intersection=x*y;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
 return intersection/union if union else 0

def descendant_parents(tree):
 result=C.defaultdict(set);reverse={v:k for k,v in MIDS.items()}
 def walk(node,ancestors):
  mid=node['LabelName']
  for parent in ancestors:
   if parent in reverse and mid not in reverse:result[mid].add(reverse[parent])
  for child in node.get('Subcategory',[]):walk(child,ancestors+[mid])
 walk(tree,[]);return result

def conflicts(source_annotations,native_annotations,parents):
 records=[]
 for raw in source_annotations:
  if raw['LabelName'] not in parents:continue
  xy=box([float(raw[k]) for k in ('XMin','YMin','XMax','YMax')])
  for parent in sorted(parents[raw['LabelName']]):
   covered=any(g['label']==parent and overlap(g['box'],xy)>=.9 for g in native_annotations)
   if not covered:records.append(dict(source_mid=raw['LabelName'],source_box=xy,affected_parent=parent,reason='Descendant source region has no comparable native parent box; mask parent optimization coverage, do not invent parent GT'))
 return records

def run():
 m=read(MANIFEST);ids={r['image_id'] for r in m['images']};raw=C.defaultdict(list)
 with (META/'validation-annotations-bbox.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   if r['ImageID'] in ids:raw[r['ImageID']].append(r)
 parents=descendant_parents(read(META/'bbox_labels_600_hierarchy.json'));records=[]
 for r in m['images']:
  found=conflicts(raw[r['image_id']],r['annotations'],parents)
  records.append(dict(image_id=r['image_id'],original_source_rows=raw[r['image_id']],uncovered_descendants=found,optimization_classes_to_mask=sorted({g['affected_parent'] for g in found}|{'desk','cabinet'}),primary_native_scoring_unchanged=True))
 write_new(OUT/'annotation-scope-audit.json',dict(records=records,native_gt_unchanged=True,manifest_sha256=sha(MANIFEST),source_annotations_sha256=sha(META/'validation-annotations-bbox.csv')))
 print('Scope audit:',sum(bool(r['uncovered_descendants']) for r in records),'photos with additional conservative optimization masks')

if __name__=='__main__':run()
