"""Conservative project-head targets, native completeness and deterministic grouping."""
import collections
import hashlib
import math
from common import CLASSES

CORE = {'OpenImages': {'/m/02crq1':'sofa','/m/01mzpv':'chair','/m/04bcr3':'table','/m/03ssj5':'bed','/m/01y9k5':'desk','/m/01s105':'cabinet','/m/0dtln':'lamp'},
        'COCO': {62:'chair',63:'sofa',65:'bed',67:'table'},
        'LVIS': {982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}}
SUPPORTED = ('sofa', 'chair', 'bed')
SEED = 'phase10a-inner-20261118-v1'

class Union:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)

def positive_allowed(target):
    a = target['original_annotation']
    return (target['comparison_group'] in SUPPORTED
            and CORE.get(target['native_dataset'], {}).get(target['native_id']) == target['comparison_group']
            and target['mapping_scope'] == 'source_positive_comparison_group'
            and target['native_scope'] in ('positive_source_coverage','positive_source_exhaustive','positive_completeness_unverified','positive_nonexhaustive')
            and not target.get('group_of') and not a.get('ignore') and not a.get('iscrowd')
            and not a.get('flags', {}).get('IsDepiction') and not a.get('flags', {}).get('IsInside'))

def normalized_box(dataset, annotation, width, height):
    if min(width, height) <= 0: raise ValueError('Invalid source dimensions')
    if dataset == 'OpenImages': b = list(annotation['box'])
    elif dataset in ('COCO','LVIS'):
        x,y,w,h = annotation['bbox']; b = [x/width,y/height,(x+w)/width,(y+h)/height]
    else: raise ValueError('Unknown coordinate namespace')
    if len(b) != 4 or any(not math.isfinite(x) for x in b) or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
        raise ValueError('Invalid/out-of-frame native box; never silently clip')
    return b

def explicit_negatives(source):
    scope = source['source_native_scope']; out = {c:False for c in CLASSES}
    for native, value in scope.get('native_image_level_labels', {}).items():
        cat = CORE['OpenImages'].get(native)
        if cat in SUPPORTED and value == 0: out[cat] = True
    # Only exact safe core categories. No parent/alias negative expansion.
    for native in scope.get('neg_category_ids', []):
        cat = CORE['LVIS'].get(native)
        if cat in SUPPORTED: out[cat] = True
    return out

def summarize(rows):
    result = {}
    for c in CLASSES:
        selected = [r for r in rows if any(a['label']==c for a in r['annotations'] if not a['ignore'])]
        aa = [a for r in selected for a in r['annotations'] if a['label']==c and not a['ignore']]
        result[c] = dict(images=len(selected),boxes=len(aa),masks=sum(a['has_mask'] for a in aa),
                         groups=len({r['identity_group'] for r in selected}))
    return result

def inner_split(rows):
    groups = collections.defaultdict(list)
    for r in rows: groups[r['identity_group']].append(r)
    counts = {c:sum(any(a['label']==c and not a['ignore'] for a in r['annotations']) for r in rows) for c in SUPPORTED}
    targets = {c:max(1,round(v*.2)) for c,v in counts.items()}; selected = set(); current = collections.Counter()
    order = sorted(groups, key=lambda g: hashlib.sha256((SEED+g).encode()).hexdigest())
    # Greedy coverage, never split a source component; all inputs come from historical train.
    for group in order:
        contribution = {c:sum(any(a['label']==c and not a['ignore'] for a in r['annotations']) for r in groups[group]) for c in SUPPORTED}
        if sum(current.values()) and all(current[c]>=targets[c] for c in SUPPORTED): break
        before = sum(abs(current[c]-targets[c]) for c in SUPPORTED)
        after = sum(abs(current[c]+contribution[c]-targets[c]) for c in SUPPORTED)
        if after < before and all(counts[c]-current[c]-contribution[c]>=1 for c in SUPPORTED):
            selected.add(group); current.update(contribution)
    result = [dict(r,internal_partition='inner_development' if r['identity_group'] in selected else 'optimization') for r in sorted(rows,key=lambda r:r['image_id'])]
    if not selected or len(selected)==len(groups): raise ValueError('No valid grouped internal split')
    return result
