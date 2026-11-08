"""Versioned source-aware detection contracts, independent of production code."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/phase7-data-strategy-v1'
CATEGORIES = ('sofa', 'chair', 'table', 'bed', 'desk', 'cabinet', 'lamp')
OI = {'/m/02crq1': 'sofa', '/m/01mzpv': 'chair', '/m/04bcr3': 'table',
      '/m/03ssj5': 'bed', '/m/01y9k5': 'desk', '/m/01s105': 'cabinet', '/m/0dtln': 'lamp'}
LVIS = {982: 'sofa', 232: 'chair', 1050: 'table', 77: 'bed', 361: 'desk', 181: 'cabinet', 626: 'lamp'}
ALLOWED_LICENSES = {'https://creativecommons.org/licenses/by/2.0/',
                    'https://creativecommons.org/licenses/by/4.0/',
                    'https://creativecommons.org/publicdomain/zero/1.0/'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, obj):
    path = Path(path)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def box(value, width, height):
    if len(value) != 4 or not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in value):
        raise ValueError('Nonfinite/invalid xywh box')
    x, y, w, h = value
    if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0 and x+w <= width+1e-5 and y+h <= height+1e-5):
        raise ValueError('Box outside image')
    return list(value)

def mask_valid(mask, width, height, kind):
    if kind=='coco_polygons':
        if not isinstance(mask,list) or not mask:raise ValueError('Missing polygons')
        for polygon in mask:
            if not isinstance(polygon,list) or len(polygon)<6 or len(polygon)%2:
                raise ValueError('Invalid polygon vertices')
            for i,v in enumerate(polygon):
                if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=(width if i%2==0 else height):
                    raise ValueError('Invalid polygon coordinate')
    elif kind=='coco_uncompressed_rle':
        if not isinstance(mask,dict) or mask.get('size') != [height,width] or not isinstance(mask.get('counts'),list):
            raise ValueError('Invalid RLE dimensions')
        counts=mask['counts']
        if not counts or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in counts) or sum(counts)!=width*height:
            raise ValueError('Invalid RLE runs')
    else:
        raise ValueError('Mask codec not validated; preserve raw and quarantine until a tested decoder exists')

def validate(record):
    """Only eligible records may cross this boundary; pending audits use a different schema."""
    required = {'schema','source','source_image_id','domain','partition','width','height','image_sha256',
                'source_groups','photo_rights','annotation_rights','annotations','coverage','native_coverage','provenance'}
    if not required <= record.keys() or record['schema'] != 'furniture-detection-v1':
        raise ValueError('Missing unified provenance/schema')
    if record['domain'] not in ('real','synthetic') or record['partition'] not in ('train','development','frozen_evaluation'):
        raise ValueError('Unknown domain/partition')
    if record['domain'] == 'synthetic' and record['partition'] != 'train':
        raise ValueError('Initial synthetic pilot is training-only; real evaluation stays real')
    if not record['source_image_id'] or not record['source_groups'] or not record['provenance']:
        raise ValueError('Unknown identity/provenance')
    if any(not isinstance(record[k],int) or isinstance(record[k],bool) or record[k] <= 0 for k in ('width','height')):
        raise ValueError('Invalid dimensions')
    for rights in (record['photo_rights'],record['annotation_rights']):
        if rights.get('status') != 'verified' or rights.get('license_url') not in ALLOWED_LICENSES:
            raise ValueError('Unverified or incompatible rights')
        for key in ('creator','source_url','evidence_sha256','verified_utc','attribution'):
            if not rights.get(key):
                raise ValueError('Missing rights evidence '+key)
    if len(record['image_sha256']) != 64 or any(c not in '0123456789abcdef' for c in record['image_sha256']):
        raise ValueError('Missing image SHA-256')
    if set(record['coverage']) != set(CATEGORIES) or any(x not in ('unknown','positive_nonexhaustive','positive_exhaustive','explicit_negative') for x in record['coverage'].values()):
        raise ValueError('Incomplete annotation-scope policy')
    if not isinstance(record['native_coverage'],dict) or not record['native_coverage']:
        raise ValueError('Original source-native scope must be preserved independently of broad category groups')
    ids = set()
    for a in record['annotations']:
        if a['source_annotation_id'] in ids:
            raise ValueError('Duplicate annotation ID')
        ids.add(a['source_annotation_id'])
        if a['category'] not in CATEGORIES or not a.get('original') or not a.get('mapping_rule'):
            raise ValueError('Unsupported category or missing original annotation/mapping')
        box(a['bbox_xywh'],record['width'],record['height'])
        if record['coverage'][a['category']] in ('unknown','explicit_negative'):
            raise ValueError('Contradictory annotation scope')
        if a.get('segmentation') is not None and not a.get('segmentation_format'):
            raise ValueError('Unknown mask format')
        if a.get('segmentation') is not None:
            mask_valid(a['segmentation'],record['width'],record['height'],a['segmentation_format'])
    return record

def assert_isolated(records):
    """Transitive identity bridges fail closed; never silently merge/reassign products."""
    parent = {}
    def root(x):
        parent.setdefault(x,x)
        if parent[x] != x:
            parent[x] = root(parent[x])
        return parent[x]
    seen_ids = set()
    for i, r in enumerate(records):
        identity = (r['source'],str(r['source_image_id']))
        if identity in seen_ids:
            raise ValueError('Duplicate source image')
        seen_ids.add(identity)
        tokens = [('row',i),('source',identity)]
        tokens += [('hash',r[k]) for k in ('image_sha256','decoded_pixel_sha256') if r.get(k)]
        tokens += [('group',x) for x in r.get('source_groups',[])]
        tokens += [('identity',x) for x in r.get('identity_groups',[])]
        for t in tokens[1:]:
            parent[root(t)] = root(tokens[0])
    parts = {}
    for i, r in enumerate(records):
        g = root(('row',i))
        parts.setdefault(g,set()).add((r['domain'],r['partition']))
        if len(parts[g]) > 1:
            raise ValueError('Cross-partition/domain identity or duplicate leakage')
    return len(parts)

def lvis_scope(image, native_category_id, has_positive):
    negatives = set(image.get('neg_category_ids',[]))
    incomplete = set(image.get('not_exhaustive_category_ids',[]))
    if native_category_id in negatives and (has_positive or native_category_id in incomplete):
        raise ValueError('Conflicting LVIS federated scope')
    if native_category_id in negatives:
        return 'explicit_negative'
    if has_positive:
        return 'positive_nonexhaustive' if native_category_id in incomplete else 'positive_exhaustive'
    return 'unknown'
