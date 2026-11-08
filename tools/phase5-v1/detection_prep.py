"""Isolated Phase 5 diagnostics and subset detection evaluation; no training/downloads.

All crop flags are unfitted heuristics, never human annotations or correctness scores.
The evaluator is a flat, partial-label-aware pilot metric, NOT the official OID metric.
"""
import argparse
import collections
import copy
import csv
import hashlib
import functools
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('sofa', 'chair', 'table', 'bed', 'desk', 'cabinet', 'lamp')
# Candidate exact display names. Resolve against the approved dictionary; never guess MIDs.
NAMES = {'Couch': 'sofa', 'Chair': 'chair', 'Table': 'table', 'Bed': 'bed',
         'Desk': 'desk', 'Cabinetry': 'cabinet', 'Lamp': 'lamp'}
OWL_SHA = 'e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7'
THRESHOLDS = dict(min_side=32, border_px=8, extreme_aspect=4,
                  low_detector_score=.20, low_margin=.10, containment=.80,
                  overlap_iou=.30, blank_std=3, low_detail=4)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def box(value):
    if len(value) != 4: raise ValueError('xyxy requires four coordinates')
    b = list(map(float, value))
    if not all(map(math.isfinite, b)) or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError('Nonfinite or degenerate box')
    return b


def area(b): return (b[2] - b[0]) * (b[3] - b[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    a, b = box(a), box(b)
    v = intersection(a, b)
    return v / (area(a) + area(b) - v)


def quality(row, peers, size, pixels=None):
    b = box(row['original_box']); w, h = size
    if min(w, h) <= 0 or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
        raise ValueError('Box outside oriented image')
    bw, bh = b[2] - b[0], b[3] - b[1]
    margins = [b[0], b[1], w-b[2], h-b[3]]
    measurements = dict(width=bw, height=bh, area_fraction=area(b)/(w*h),
                        aspect_ratio=max(bw/bh, bh/bw), border_distances_px=margins,
                        detector_score=row['detector_score'], overlaps=[])
    flags = []
    if min(bw, bh) < THRESHOLDS['min_side']: flags.append('small_crop')
    if min(margins) < THRESHOLDS['border_px']: flags.append('border_truncation_risk')
    if measurements['aspect_ratio'] > THRESHOLDS['extreme_aspect']: flags.append('extreme_aspect_fragment_risk')
    if row['detector_score'] < THRESHOLDS['low_detector_score']: flags.append('low_detector_score')
    for peer in peers:
        if peer['detection_id'] == row['detection_id']: continue
        pb = box(peer['original_box']); v = intersection(b, pb)
        ov = dict(id=peer['detection_id'], iou=iou(b, pb),
                  crop_coverage=v/area(b), peer_coverage=v/area(pb),
                  same_detector_category=row['detector_label'] == peer['detector_label'])
        if v: measurements['overlaps'].append(ov)
        if v/min(area(b), area(pb)) >= THRESHOLDS['containment']:
            flags.append('nested_same_category_risk' if ov['same_detector_category'] else 'mixed_object_overlap_risk')
        elif ov['iou'] >= THRESHOLDS['overlap_iou']:
            flags.append('overlapping_same_category_risk' if ov['same_detector_category'] else 'mixed_object_overlap_risk')
    # Model disagreement is recorded as a signal, never used as a quality exclusion.
    measurements['classifiers_disagree'] = row['baseline']['category'] != row['expanded']['category']
    measurements['expanded_softmax7_margin_uncalibrated'] = row['expanded']['softmax7_margin']
    if row['expanded']['softmax7_margin'] < THRESHOLDS['low_margin']: flags.append('low_classifier_margin')
    if pixels is not None:
        import numpy as np
        gray = np.asarray(pixels.convert('L'), dtype=float)
        measurements['gray_std'] = float(gray.std())
        # Resolution-dependent high-frequency content, NOT a calibrated blur detector.
        dy = abs(gray[1:] - gray[:-1]).mean() if gray.shape[0] > 1 else 0.0
        dx = abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
        detail = float((dx + dy)/2)
        measurements['mean_adjacent_gray_difference'] = detail
        if gray.std() < THRESHOLDS['blank_std']: flags.append('blank_low_contrast_risk')
        if detail < THRESHOLDS['low_detail']: flags.append('low_detail_or_blur_risk')
    flags = sorted(set(flags))
    status = 'unsuitable_geometry' if 'small_crop' in flags else ('unresolved_risk' if flags else 'candidate_unverified')
    return dict(detection_id=row['detection_id'], original_box=copy.deepcopy(row['original_box']),
                measurements=measurements, heuristic_flags=flags, status=status,
                advisory_category=row['expanded']['category'] if status == 'candidate_unverified' else None,
                human_verified=False, suitable_ground_truth=None)


def class_mapping(dictionary):
    names = {}
    with open(dictionary, newline='') as stream:
        for mid, name in csv.reader(stream):
            if mid in names: raise ValueError('Repeated class ID')
            names[mid] = name
    mapping = {mid: NAMES[name] for mid, name in names.items() if name in NAMES}
    missing = sorted(set(NAMES) - set(names.values()))
    return dict(mapping=mapping, missing_exact_names=missing,
                dictionary_sha256=sha(dictionary), policy='exact display names only; no automatic hierarchy propagation')


def parse_annotations(path, mapping):
    rows = []; seen = set()
    required = {'ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax', 'Confidence',
                'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'}
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []): raise ValueError('Unexpected OID bbox schema')
        for r in reader:
            # Validate every row before filtering to avoid concealing malformed metadata.
            b = box([r['XMin'], r['YMin'], r['XMax'], r['YMax']])
            if not all(0 <= v <= 1 for v in b): raise ValueError('Unnormalized annotation')
            if not r['ImageID'] or not r['LabelName']: raise ValueError('Missing identity')
            confidence = float(r['Confidence'])
            if not math.isfinite(confidence) or confidence != 1: raise ValueError('Unexpected bbox confidence')
            flags = {k: int(r[k]) for k in required if k.startswith('Is')}
            if any(v not in (-1, 0, 1) for v in flags.values()): raise ValueError('Unexpected attribute value')
            key = (r['ImageID'], r['LabelName'], *b)
            if key in seen: raise ValueError('Repeated annotation row')
            seen.add(key)
            if r['LabelName'] in mapping:
                rows.append(dict(image_id=r['ImageID'], source_mid=r['LabelName'],
                                 label=mapping[r['LabelName']], box=b, flags=flags,
                                 ignore=flags['IsGroupOf'] != 0 or flags['IsDepiction'] != 0))
    return rows


def parse_image_labels(path, mapping):
    """Retain human positive AND negative class coverage; unknown classes stay unscored."""
    labels = {}
    with open(path, newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'ImageID','LabelName','Confidence'} <= set(reader.fieldnames or []): raise ValueError('Unexpected image-label schema')
        for row in reader:
            if row['LabelName'] not in mapping: continue
            c=mapping[row['LabelName']]; value=float(row['Confidence'])
            if value not in (0,1): raise ValueError('Expected verified binary image label')
            key=(row['ImageID'], c)
            if key in labels and labels[key]!=value: raise ValueError('Conflicting image-level labels')
            labels[key]=value
    return labels


@functools.lru_cache(maxsize=1)
def reserved_identities():
    # Metadata only: never open official heldout photographs, features or decisions.
    inventory=read(ROOT/'data/abo/training-prep/phase1d-v1/proposal-v2/eligibility-inventory.json')['images']
    panel=read(ROOT/'data/abo/experiments/phase3-preparation-v1/frozen-development-protocol.json')['development_rows']
    protected=[r for r in inventory if r.get('proposed_split') in ('validation','test')]+panel
    return {r[k] for r in protected for k in ('sha256','image_id','abo_product_id','group_id','family_group_id','human_audited_family_group_id','decoded_pixel_sha256') if r.get(k)}


def assert_isolation(rows):
    """Reject exact/source/family reuse across any split, including frozen eval."""
    required = ('image_id', 'sha256', 'source_identity', 'identity_group')
    owners = {k: {} for k in required}
    seen = set()
    protected=reserved_identities()
    for row in rows:
        if any(row.get(k) in protected for k in (*required,'decoded_pixel_sha256')): raise ValueError('Reserved historical holdout identity/hash')
        if row.get('identity_uncertain') is not False: raise ValueError('Unresolved or unspecified identity status')
        if row.get('split') not in ('train', 'development', 'frozen_evaluation'): raise ValueError('Unknown partition')
        if row.get('official_phase2b') or row.get('abo_frozen_panel'): raise ValueError('Reserved prior holdout')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', row.get('image_id', '')): raise ValueError('Unsafe image ID')
        if not re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', '')): raise ValueError('Invalid image hash')
        if row['image_id'] in seen: raise ValueError('Duplicate image manifest row')
        seen.add(row['image_id'])
        for key in required:
            value = row.get(key)
            if not value: raise ValueError('Missing isolation identity: '+key)
            if value in owners[key] and owners[key][value] != row['split']: raise ValueError('Partition leakage: '+key)
            owners[key][value] = row['split']


def select_pilot(candidates, seed='phase5-pilot-v1'):
    """Future local metadata selection. No URLs fetched. Group first, then deterministic assignment.

    Candidates must already be license-, indoor- and annotation-eligible; unknown scene
    suitability cannot be converted to true by this function. Frozen test sourced from
    official test; train/dev from official validation. One photo per known family.
    """
    eligible = []
    for r in candidates:
        if not all(r.get(k) is True for k in ('license_verified', 'indoor_verified', 'annotations_verified')): continue
        if r.get('identity_uncertain') is not False or r.get('official_phase2b') or r.get('abo_frozen_panel'): continue
        if not r.get('box_counts') or set(r['box_counts']) - set(CLASSES): continue
        eligible.append(copy.deepcopy(r))
    key = lambda r: hashlib.sha256((seed+'|'+r['image_id']).encode()).hexdigest()
    eligible.sort(key=key)
    # Deterministic rarity-first queue, cycling through seven categories. No model results.
    support = collections.Counter(c for r in eligible for c,n in r['box_counts'].items() if n > 0)
    queues = {c: [r for r in eligible if r['box_counts'].get(c, 0)>0] for c in sorted(CLASSES,key=lambda c:(support[c],c))}
    ranked=[]; ranked_ids=set()
    while any(queues.values()):
        for queue in queues.values():
            while queue and queue[0]['image_id'] in ranked_ids: queue.pop(0)
            if queue:
                r=queue.pop(0);ranked.append(r);ranked_ids.add(r['image_id'])
    eligible=ranked
    selected = []; used = {k: set() for k in ('image_id', 'sha256', 'source_identity', 'identity_group')}
    counts = collections.Counter()
    for r in eligible:
        if any(not r.get(k) or r[k] in v for k, v in used.items()): continue
        if r.get('source_partition') == 'test': part, limit = 'frozen_evaluation', 60
        elif r.get('source_partition') == 'validation':
            part = 'train' if counts['train'] < 60 else 'development'; limit = 60 if part == 'train' else 30
        else: continue
        if counts[part] >= limit: continue
        r['split'] = part; selected.append(r); counts[part] += 1
        for k, v in used.items(): v.add(r[k])
    assert_isolation(selected)
    return dict(seed=seed, targets=dict(train=60, development=30, frozen_evaluation=60),
                actual=dict(counts), rows=selected, complete=len(selected)==150,
                per_partition_box_counts={p: dict(sum((collections.Counter(r['box_counts']) for r in selected if r['split']==p), collections.Counter())) for p in counts})


def evaluate_image(predictions, annotations, verified_classes, threshold=.5):
    """Single-image flat pilot scores. Unknown classes unscored, group/depiction ignored.

    No hierarchy AP or official group-of credit. Confidence-order greedy one-to-one
    matches. Duplicate FP and fragment candidates are distinct diagnostic quantities.
    All boxes are normalized xyxy. verified_classes comes from audited image labels.
    """
    if not 0 < threshold <= 1: raise ValueError('Invalid IoU threshold')
    verified = set(verified_classes)
    if verified - set(CLASSES): raise ValueError('Invalid verified category')
    gts = copy.deepcopy(annotations)
    for g in gts:
        b=box(g['box'])
        if not all(0 <= v <= 1 for v in b) or g['label'] not in verified: raise ValueError('GT label coverage/coordinates inconsistent')
    out = {c: dict(tp=0, fp=0, fn=0, matched_ious=[], duplicate_fp=0, fragment_candidates=0) for c in CLASSES}
    matched = set(); ignored = unknown = 0
    preds = sorted(copy.deepcopy(predictions), key=lambda p: -float(p['score']))
    for p in preds:
        if p['label'] not in CLASSES or not math.isfinite(float(p['score'])) or not 0 <= float(p['score']) <= 1: raise ValueError('Invalid prediction')
        pb=box(p['box'])
        if not all(0 <= v <= 1 for v in pb): raise ValueError('Unnormalized prediction')
        c = p['label']
        if c not in verified: unknown += 1; continue
        candidates = [(iou(pb, g['box']), j) for j, g in enumerate(gts) if g['label']==c and not g.get('ignore') and j not in matched]
        best, j = max(candidates, default=(0, -1))
        if j >= 0 and best >= threshold:
            matched.add(j); out[c]['tp'] += 1; out[c]['matched_ious'].append(best); continue
        # Only unmatched predictions can be ignored; valid positives get first chance.
        if any(g['label']==c and g.get('ignore') and intersection(pb,g['box'])/area(pb) >= .5 for g in gts):
            ignored += 1; continue
        out[c]['fp'] += 1
        if any(g['label']==c and iou(pb,g['box'])>=threshold for j,g in enumerate(gts) if j in matched): out[c]['duplicate_fp'] += 1
        if any(g['label']==c and not g.get('ignore') and intersection(pb,g['box'])/area(pb)>=.8 and area(pb)<.5*area(g['box']) for g in gts): out[c]['fragment_candidates'] += 1
    for j, g in enumerate(gts):
        if not g.get('ignore') and j not in matched: out[g['label']]['fn'] += 1
    return dict(categories=out, unknown_class_predictions=unknown, ignored_predictions=ignored)


def summarize(scores):
    result = {}
    for c in CLASSES:
        s = {k: sum(r['categories'][c][k] for r in scores) for k in ('tp','fp','fn','duplicate_fp','fragment_candidates')}
        values = [v for r in scores for v in r['categories'][c]['matched_ious']]
        s.update(precision=s['tp']/(s['tp']+s['fp']) if s['tp']+s['fp'] else None,
                 recall=s['tp']/(s['tp']+s['fn']) if s['tp']+s['fn'] else None,
                 mean_matched_iou=sum(values)/len(values) if values else None)
        result[c] = s
    return dict(categories=result, images=len(scores), unknown_class_predictions=sum(s['unknown_class_predictions'] for s in scores),
                ignored_predictions=sum(s['ignored_predictions'] for s in scores), metric='flat partial-label pilot; NOT official Open Images mAP')


def detector_loader(checkpoint, expected_sha, device='cpu', threshold=.15, nms_iou=.5):
    p = Path(checkpoint).resolve()
    if not p.is_dir() or not (p/'model.safetensors').is_file(): raise ValueError('Local safetensors model directory required')
    if len(expected_sha) != 64 or sha(p/'model.safetensors') != expected_sha: raise ValueError('Detector hash mismatch')
    # Read-only loader, pinned weights plus local processor. No remote code or downloads.
    sys.path.insert(0, str(ROOT/'src'))
    from furniture_bench.detection.owlv2 import Owlv2Detector
    return Owlv2Detector(model_id=str(p), device=device, threshold=threshold, nms_iou=nms_iou,
                         categories=CLASSES, allow_download=False)


def validate_public_bundle(bundle_path):
    """Reconstruct published GT from hashed CSVs, rather than trusting manifest booleans."""
    bundle_path=Path(bundle_path); bundle=read(bundle_path)
    if bundle.get('schema')!='phase5-public-boxes-v1' or bundle.get('source')!='Open Images': raise ValueError('Only audited public annotation bundles supported')
    records=bundle.get('annotation_files',[])
    dictionary=bundle.get('class_dictionary')
    if not records or not dictionary: raise ValueError('Missing source provenance')
    def source_path(record):
        if not record.get('url','').startswith('https://storage.googleapis.com/openimages/'): raise ValueError('Unapproved annotation source')
        p=(bundle_path.parent/record['file']).resolve()
        if not p.is_relative_to(bundle_path.parent.resolve()) or sha(p)!=record['sha256']: raise ValueError('Annotation source path/hash mismatch')
        return p
    mapping=class_mapping(source_path(dictionary))['mapping']
    sources={}
    for record in records:
        key=(record['partition'],record['kind'])
        if key in sources: raise ValueError('Duplicate source partition/kind')
        p=source_path(record)
        if record['kind']=='boxes': sources[key]=parse_annotations(p,mapping)
        elif record['kind']=='human_labels': sources[key]=parse_image_labels(p,mapping)
        else: raise ValueError('Unsupported source kind')
    rows=bundle['images']; assert_isolation(rows)
    for row in rows:
        part=row.get('source_partition')
        if part not in ('validation','test'): raise ValueError('Unexpected public source partition')
        if (row['split']=='frozen_evaluation')!=(part=='test'): raise ValueError('Frozen/source partition separation differs')
        if (part,'boxes') not in sources or (part,'human_labels') not in sources: raise ValueError('Missing class coverage or bbox CSV')
        published=[r for r in sources[(part,'boxes')] if r['image_id']==row['image_id']]
        coverage=sources[(part,'human_labels')]
        verified=sorted(c for (ident,c),v in coverage.items() if ident==row['image_id'])
        canonical=lambda v: sorted(json.dumps(r,sort_keys=True) for r in v)
        if canonical(published)!=canonical(row['annotations']) or sorted(row['verified_classes'])!=verified: raise ValueError('Bundle GT/coverage differs from published metadata')
        if any(coverage.get((row['image_id'],g['label']))!=1 for g in published): raise ValueError('BBox contradicts image-level positive label')
        if not published: raise ValueError('No mapped furniture annotations')
    return bundle


def evaluate_bundle(bundle_path, predictions_path, output):
    """Explicitly requires audited public annotation provenance; no Phase2B loader."""
    bundle=validate_public_bundle(bundle_path)
    rows=bundle['images']
    pred = read(predictions_path)
    if pred.get('manifest_sha256') != sha(bundle_path): raise ValueError('Prediction/bundle mismatch')
    if len({r['image_id'] for r in pred['images']}) != len(pred['images']): raise ValueError('Repeated prediction identity')
    byid = {r['image_id']:r for r in pred['images']}
    chosen = [r for r in rows if r['split']=='frozen_evaluation']
    if set(byid) != {r['image_id'] for r in chosen}: raise ValueError('Require exactly the frozen panel predictions')
    scores = []
    for r in chosen:
        if not all(r.get(k) is True for k in ('license_verified','annotations_verified','orientation_verified')): raise ValueError('Unaudited public labels/license')
        if r.get('annotation_provenance') != 'published_open_images': raise ValueError('Unsupported label provenance')
        # Uncertainty is excluded, never silently resolved; expected eligibility audited upstream.
        if r.get('identity_uncertain'): raise ValueError('Unresolved identity in frozen evaluation')
        if byid[r['image_id']]['image_sha256'] != r['sha256']: raise ValueError('Prediction source hash differs')
        scores.append(evaluate_image(byid[r['image_id']]['predictions'], r['annotations'], r['verified_classes']))
    if not scores: raise ValueError('No eligible frozen evaluation images')
    result = summarize(scores); result['bundle_sha256'] = sha(bundle_path)
    result['prediction_sha256'] = sha(predictions_path)
    write_new(output, result)


def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit-metadata'); audit.add_argument('--dictionary',required=True); audit.add_argument('--boxes',required=True); audit.add_argument('--out',required=True)
    ev=sub.add_parser('evaluate'); ev.add_argument('--bundle',required=True); ev.add_argument('--predictions',required=True); ev.add_argument('--out',required=True)
    sel=sub.add_parser('select-pilot'); sel.add_argument('--candidates',required=True); sel.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.command=='evaluate': evaluate_bundle(a.bundle,a.predictions,a.out)
    elif a.command=='select-pilot': write_new(a.out,select_pilot(read(a.candidates)['images']))
    else:
        mapping=class_mapping(a.dictionary); rows=parse_annotations(a.boxes,mapping['mapping'])
        counts={c:dict(boxes=sum(r['label']==c for r in rows),images=len({r['image_id'] for r in rows if r['label']==c}),ordinary_boxes=sum(r['label']==c and not r['ignore'] for r in rows)) for c in CLASSES}
        write_new(a.out,dict(**mapping,counts=counts,bbox_sha256=sha(a.boxes),annotation_rows=rows))

if __name__=='__main__': main()
