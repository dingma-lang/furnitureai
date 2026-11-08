"""Read-only verification of acquired metadata and immutable historical artifacts."""
import base64
import collections as C
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import prepare_pilot as p
from metadata_dictionary import read_dictionary


def verify(meta):
    dictionary=read_dictionary(meta/'oidv7-class-descriptions-boxable.csv')
    tree=p.d.read(meta/'bbox_labels_600_hierarchy.json');hierarchy=set()
    def visit(node):
        hierarchy.add(node['LabelName'])
        for child in node.get('Subcategory',[]):visit(child)
    visit(tree)
    result=dict(dictionary_classes=len(dictionary),hierarchy_ids=len(hierarchy),dictionary_missing_from_hierarchy=sorted(set(dictionary)-hierarchy),partitions={})
    all_ids=set();furniture=p.d.read(meta/'audit-v1/furniture-candidates.json')
    for part in ('validation','test'):
        info={};warnings=C.Counter();box_rows=label_rows=0
        with (meta/(part+'-images-with-rotation.csv')).open(newline='') as f:
            for row in csv.DictReader(f):
                ident=row['ImageID']
                if not re.fullmatch('[0-9a-f]{16}',ident) or ident in info or ident in all_ids:raise ValueError('Repeated/invalid source identity')
                if row['Subset']!=part:raise ValueError('Image partition mismatch')
                info[ident]=row;all_ids.add(ident)
                for key in ('OriginalMD5','OriginalSize','OriginalURL','OriginalLandingURL','License','AuthorProfileURL','Rotation'):
                    if not row[key]:warnings['missing_'+key]+=1
                if row['OriginalMD5'] and len(base64.b64decode(row['OriginalMD5'],validate=True))!=16:raise ValueError('Malformed original MD5')
        for suffix,kind in [('-annotations-bbox.csv','bbox'),('-annotations-human-imagelabels-boxable.csv','human_labels')]:
            with (meta/(part+suffix)).open(newline='') as f:
                for row in csv.DictReader(f):
                    if row['ImageID'] not in info or row['LabelName'] not in dictionary:raise ValueError('Annotation references absent source identity/class')
                    confidence=float(row['Confidence'])
                    if confidence not in (0,1) or (kind=='bbox' and confidence!=1):raise ValueError('Nonhuman/bad annotation confidence')
                    if kind=='bbox':box_rows+=1
                    else:label_rows+=1
        counts=C.Counter(row['image_id'] for row in furniture[part]['annotations'])
        result['partitions'][part]=dict(source_images=len(info),bbox_rows=box_rows,human_image_label_rows=label_rows,
              furniture_images=len(counts),furniture_bbox_rows=sum(counts.values()),optional_metadata_missing=dict(warnings))
    return result


def historical(meta):
    root=p.ROOT;checked=stat_count=0;bad=[];identities=set()
    snapshots=[root/'reports/phase4-room-comparison-v1/preservation-before.json',root/'reports/phase5-detection-preparation-v1/local-diagnostics-v1/preservation-before.json']
    for snap in snapshots:
        for rel,rec in p.d.read(snap).items():
            if rel in identities:continue
            identities.add(rel);path=root/rel;checked+=1
            if not path.is_file():bad.append(rel);continue
            if isinstance(rec,str):expected=rec;actual=p.d.sha(path)
            elif 'stat_only' in rec:
                s=path.stat();expected=rec['stat_only'];actual=[s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino];stat_count+=1
            else:expected=rec['sha256'];actual=p.d.sha(path)
            if expected!=actual:bad.append(rel)
    # Also verify the complete Phase 5 preparation source and result checksums.
    for base,name in [(root/'tools/phase5-v1','implementation-SHA256SUMS'),(root/'reports/phase5-detection-preparation-v1','artifact-SHA256SUMS')]:
        for line in (base/name).read_text().splitlines():
            expected,rel=line.split('  ',1);path=base/rel;checked+=1
            if p.d.sha(path)!=expected:bad.append(str(path.relative_to(root)))
    result=dict(checked_records=checked,historical_unique_snapshot_paths=len(identities),official_holdout_stat_only=stat_count,mismatches=bad,official_holdout_photographs_or_features_opened=False,model_evaluation_run=False)
    if bad:raise ValueError('Historical preservation failure '+str(bad[:5]))
    return result

if __name__=='__main__':
    meta=p.META
    out=meta/'metadata-integrity-verification-v2.json'
    p.d.write_new(out,verify(meta))
    preserved=historical(meta)
    if p.d.read(meta/'historical-preservation.json')!=preserved:raise ValueError('Preservation outcome changed')
    print(json.dumps(p.d.read(out),indent=2))
    print(json.dumps(preserved,indent=2))
