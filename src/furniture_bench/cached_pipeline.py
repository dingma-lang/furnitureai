"""Opt-in detection-only v2 adapter; legacy runners and cache files are unchanged.

Raw here means pre-NMS candidates at an explicit acquisition score floor, not
necessarily every model query. The existing detector may discard lower scores.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import __version__ as pillow_version

from .cache import (canonical, evaluation_fingerprint, file_identity, fingerprint,
                    make_contract, read_prediction_cache, write_prediction_cache)
from .data import CATEGORIES, digest, load_image_with_digest, load_manifest
from .evaluation import evaluate
from .geometry import validate_box
from .persistence import artifact_lock, atomic_json
from .postprocessing import FilterConfig, filter_detections

SCHEMA = 2
FRAME = 'oriented-xyxy-pixels'
MASK_POLICY = 'positive-matches-first; incomplete-image-unmatched-ignored-v1'
ARTIFACTS = {'input.json', 'predictions.json', 'postprocessed.json', 'supervision.json', 'evaluation.json'}


class CacheConflictError(ValueError):
    """An immutable artifact at this identity has different or damaged contents."""


def _publish(path, value):
    try:
        atomic_json(path, value, overwrite=False)
    except FileExistsError:
        if canonical(json.loads(Path(path).read_bytes())) != canonical(value):
            raise CacheConflictError('Existing artifact differs; preserved: '+str(path))


def stage_keys(contract, policy, protocol):
    """Declare dependencies separately for raw, postprocessed, evaluation and masks."""
    source = Path(__file__).parent
    post = fingerprint(dict(raw=contract['inference_key'], policy=policy.to_dict(),
                            implementation=file_identity({n:source/(n+'.py') for n in ('postprocessing','geometry')})['sha256']))
    evaluation = evaluation_fingerprint(contract, dict(postprocessing_key=post, protocol=protocol,
        implementation=file_identity({n:source/(n+'.py') for n in ('evaluation','data','geometry')})['sha256'],
        completeness_policy=MASK_POLICY))
    # A geometry-only supervision descriptor; not a trained-query target tensor.
    supervision = fingerprint(dict(schema_version=SCHEMA, image_id=contract['inference']['image_id'],
        image_sha256=contract['inference']['image_sha256'],
        coordinates=contract['inference']['preprocessing'], context_key=contract['context_key'], policy=MASK_POLICY))
    return dict(raw=contract['inference_key'], postprocessed=post, evaluation=evaluation, supervision=supervision)


def _check_metadata(detector, spec):
    meta = detector.metadata()
    expected = dict(model_id=spec['model']['model_id'], categories=spec['categories'],
                    query_texts=spec['prompts'], query_categories=spec['query_categories'])
    if any(meta.get(k) != v for k,v in expected.items()):
        raise ValueError('Detector metadata differs from declared model/query configuration')
    if not isinstance(meta.get('filter_config'), dict) or not {'threshold','nms_iou'} <= meta['filter_config'].keys():
        raise ValueError('Detector must declare its candidate acquisition filter settings')
    if FilterConfig.from_dict(meta['filter_config']).candidate_floor != spec['settings']['candidate_floor']:
        raise ValueError('Detector acquisition score floor differs from cache contract')


class CandidateCache:
    """Fresh adapter over detect()/last_candidates with process-safe identity reuse."""
    def __init__(self, root, detector, spec, model_files, processor_files):
        self.root, self.detector, self.spec = Path(root), detector, copy.deepcopy(spec)
        self.model_files, self.processor_files = model_files, processor_files
        self.verify()

    def verify(self):
        if file_identity(self.model_files)['sha256'] != self.spec['model']['checkpoint_sha256']:
            raise ValueError('Local checkpoint identity differs from declared cache identity')
        if file_identity(self.processor_files)['sha256'] != self.spec['model']['processor_sha256']:
            raise ValueError('Local processor identity differs from declared cache identity')
        _check_metadata(self.detector, self.spec)

    def read(self, contract):
        return read_prediction_cache(self.root/'raw'/(contract['inference_key']+'.json'), contract)[0]

    def predict(self, image, contract):
        key = contract['inference_key']
        with artifact_lock(self.root/'.locks'/('raw-'+key+'.lock')):
            path = self.root/'raw'/(key+'.json')
            if path.exists():
                return self.read(contract)
            _check_metadata(self.detector, self.spec)
            self.detector.detect(image)
            candidates = getattr(self.detector, 'last_candidates', None)
            if candidates is None:
                raise ValueError('Detector exposes no pre-NMS candidates; legacy filtered predictions cannot be trusted as raw')
            payload = dict(image_id=contract['inference']['image_id'], box_frame=FRAME,
                           capture='pre-NMS-candidates-at-declared-floor', detections=copy.deepcopy(candidates))
            for p in payload['detections']:
                validate_box(p['box'])
                if not (0 <= p['box'][0] < p['box'][2] <= image.width and 0 <= p['box'][1] < p['box'][3] <= image.height):
                    raise ValueError('Candidate coordinates outside oriented image')
                if p['score'] < self.spec['settings']['candidate_floor']:
                    raise ValueError('Candidate violates declared acquisition floor')
                if 'logits' in p:
                    logits = p['logits']
                    if (not isinstance(logits,list) or len(logits) != len(self.spec['prompts'])
                            or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in logits)):
                        raise ValueError('Candidate logits do not match ordered text queries')
                    best = max(range(len(logits)),key=lambda i:logits[i])
                    v = logits[best]
                    score = 1/(1+math.exp(-v)) if v >= 0 else math.exp(v)/(1+math.exp(v))
                    if self.spec['query_categories'][best] != p['label'] or not math.isclose(score,p['score'],abs_tol=1e-6):
                        raise ValueError('Candidate label/score disagrees with ordered query logits')
            try:
                write_prediction_cache(path, contract, payload)
            except FileExistsError:
                if canonical(self.read(contract)) != canonical(payload):
                    raise CacheConflictError('Conflicting raw predictions for one identity')
            return payload


def _prepare(manifest, spec):
    original_sha = digest(manifest)
    rows = load_manifest(manifest, 'rooms')
    if digest(manifest) != original_sha:
        raise ValueError('Input manifest changed during validation')
    preprocessing = copy.deepcopy(spec['preprocessing'])
    if (set(preprocessing) != {'box_frame','orientation','color'}
            or preprocessing.get('box_frame') != FRAME or preprocessing.get('orientation') != 'EXIF-transpose'
            or preprocessing.get('color') != 'RGB'):
        raise ValueError('Adapter requires explicit EXIF-transposed RGB pixel coordinates')
    preprocessing.update(pillow_version=pillow_version, loader_sha256=digest(Path(__file__).with_name('data.py')))
    if set(spec['categories']) != set(CATEGORIES):
        raise ValueError('This adapter requires an explicit seven-category ordering')
    groups, hashes, prepared = {}, set(), []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('source_group_id','partition')):
            raise ValueError('Explicit source group and partition required')
        group, partition = row['source_group_id'], row['partition']
        families = row.get('identity_family_group_ids', [])
        if not isinstance(families, list) or any(not isinstance(g,str) or not g.strip() for g in families):
            raise ValueError('Known identity-family groups must be explicit strings')
        for identity in [('source',group)]+[('family',g) for g in families]:
            if identity in groups and groups[identity] != partition:
                raise ValueError('Source group spans partitions (including known identity families)')
            groups[identity] = partition
        image, sha = load_image_with_digest(row['resolved_file'], row.get('image_sha256'))
        if sha in hashes:
            raise ValueError('Duplicate image content; refusing double counting')
        hashes.add(sha)
        completeness = row.get('completeness', {})
        if set(completeness) != set(CATEGORIES) or any(v not in ('complete','partial','unknown') for v in completeness.values()):
            raise ValueError('Explicit per-category completeness required')
        protected = row.get('protected_regions', [])
        if not isinstance(protected, list):
            raise ValueError('Protected regions must be a list')
        for obj in (row.get('objects') or [])+row.get('ignore_regions', [])+protected:
            box = validate_box(obj['box'])
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError('Annotation outside oriented image')
        supervision = dict(annotations=row.get('objects'), ignore_regions=row.get('ignore_regions', []),
            protected_regions=protected, taxonomy_version=spec['taxonomy_version'], completeness=completeness,
            source_group_id=group, partition=partition, identity_family_group_ids=families)
        # Unannotated rows have no known positives, not an empty verified annotation.
        supervision['annotations'] = supervision['annotations'] or []
        supervision['annotations_available'] = row.get('objects') is not None
        contract = make_contract(image_id=row['image_id'], image_sha256=sha, model=spec['model'],
            preprocessing=preprocessing, categories=spec['categories'], prompts=spec['prompts'],
            query_categories=spec['query_categories'], inference_settings=spec['settings'], supervision=supervision)
        prepared.append((row, contract, image.size))
    return original_sha, prepared


def _bind_partitions(root, prepared):
    """Immutable assignments across runs in this root; no automatic reassignment.

    A crash may leave a restrictive prefix of bindings, never permissive defaults.
    Unknown external identities still require the caller's historical registry.
    """
    bindings = {}
    for row,contract,_ in prepared:
        identities = [('image-id',row['image_id']),('image-sha256',contract['inference']['image_sha256']),
                      ('source-group',row['source_group_id'])]
        identities += [('family-group',g) for g in row.get('identity_family_group_ids', [])]
        for role,value in identities:
            key = fingerprint(dict(role=role,value=value))
            bindings[key] = dict(schema_version=SCHEMA,role=role,value=value,partition=row['partition'])
    with artifact_lock(root/'.locks/partition-bindings.lock'):
        for key,value in bindings.items():
            path = root/'partition-bindings'/(key+'.json')
            if path.exists() and canonical(json.loads(path.read_bytes())) != canonical(value):
                raise ValueError('Immutable partition assignment conflicts with a previous run: '+value['role'])
        for key,value in bindings.items():
            _publish(root/'partition-bindings'/(key+'.json'),value)


def load_completed_run(folder, *, expected_request=None):
    """Only a verified completion manifest makes a multi-file result readable."""
    folder = Path(folder)
    value = json.loads((folder/'completed.json').read_bytes())
    if (type(value.get('schema_version')) is not int or value['schema_version'] != SCHEMA
            or value.get('status') != 'complete' or set(value.get('artifacts', {})) != ARTIFACTS):
        raise ValueError('Incomplete or unsupported run commit')
    if fingerprint(value.get('request')) != value.get('run_key'):
        raise ValueError('Stale or damaged run manifest')
    if expected_request is not None and canonical(value['request']) != canonical(expected_request):
        raise ValueError('Run manifest does not match current request')
    documents = {}
    for name, record in value['artifacts'].items():
        path = folder/name
        if path.is_symlink():
            raise ValueError('Committed artifacts cannot be symlinks')
        payload = path.read_bytes()
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Committed artifact integrity failure: '+name)
        documents[name] = json.loads(payload)
    return dict(folder=str(folder), run_key=value['run_key'], artifacts=documents)


def run_cached_detection(manifest, output, detector, *, spec, model_files, processor_files,
                         policy=None, iou_threshold=.5):
    """Resume compatible raw work; publish derived artifacts then a completion marker.

    No retrieval, segmentation, source-annotation mutation or implicit legacy cache
    migration. Callers must explicitly enumerate actual local model/processor files.
    """
    # Own the dependency snapshot before invoking any caller-controlled callbacks.
    spec = copy.deepcopy(spec)
    policy = FilterConfig.from_dict((policy or FilterConfig()).to_dict())
    if policy.candidate_floor < spec['settings']['candidate_floor']:
        raise ValueError('Requested threshold is below the cached candidate acquisition floor')
    if not 0 < iou_threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with artifact_lock(root/'.locks/root.lock'):
        marker = root/'adapter.json'
        def initial_control_orphan(p):
            return (p.name.startswith('.adapter.json.') and p.name.endswith('.pending')
                    and p.is_file() and not p.is_symlink())
        if not marker.exists() and any(p.name != '.locks' and not initial_control_orphan(p) for p in root.iterdir()):
            raise ValueError('Existing legacy/unmarked output cannot be reused; choose a fresh v2 root')
        _publish(marker, dict(schema_version=SCHEMA, kind='detection-only-v2'))
    adapter = CandidateCache(root, detector, spec, model_files, processor_files)
    input_sha, prepared = _prepare(manifest, spec)
    _bind_partitions(root, prepared)
    protocol = dict(iou_threshold=iou_threshold, matching='confidence-ordered, one-to-one, same-category',
                    require_human_review=False, completeness_policy=MASK_POLICY)
    identities = [dict(image_id=c['inference']['image_id'], keys=stage_keys(c,policy,protocol)) for _,c,_ in prepared]
    request = dict(schema_version=SCHEMA, input_sha256=input_sha, manifest_location=str(Path(manifest).resolve()),
                   images=identities, protocol=protocol, filter=policy.to_dict(), adapter_sha256=digest(__file__))
    run_key = fingerprint(request)
    folder = root/'runs'/run_key
    with artifact_lock(root/'.locks'/('run-'+run_key+'.lock')):
        if (folder/'completed.json').exists():
            for _, contract, _ in prepared:
                adapter.read(contract)  # Also reject corruption of reusable raw artifacts.
            return load_completed_run(folder, expected_request=request)
        folder.mkdir(parents=True, exist_ok=True)
        images, inputs, postprocessed, supervision = [], [], [], []
        for row, contract, size in prepared:
            image, _ = load_image_with_digest(row['resolved_file'], contract['inference']['image_sha256'])
            raw = adapter.predict(image, contract)
            predictions, audit = filter_detections(raw['detections'], policy)
            image_id = row['image_id']
            images.append(dict(image_id=image_id, image_sha256=contract['inference']['image_sha256'],
                width=size[0], height=size[1], status='ok', detection_completed=True,
                errors=[], timing_seconds={}, detections=predictions))
            postprocessed.append(dict(image_id=image_id, audit=audit,
                                     raw_payload_sha256=fingerprint(raw), keys=stage_keys(contract,policy,protocol)))
            snapshot = {k:copy.deepcopy(v) for k,v in row.items() if k != 'resolved_file'}
            snapshot.update(file=row['resolved_file'], image_sha256=contract['inference']['image_sha256'])
            snapshot['ignore_regions'] = copy.deepcopy(row.get('ignore_regions', [])+row.get('protected_regions', []))
            if any(v != 'complete' for v in row['completeness'].values()):
                snapshot['ignore_regions'].append(dict(box=[0,0,*size], reason=MASK_POLICY))
            inputs.append(snapshot)
            supervision.append(dict(image_id=image_id, context=contract['supervision'],
                                    supervision_key=stage_keys(contract,policy,protocol)['supervision']))
        adapter.verify()
        if digest(manifest) != input_sha:
            raise ValueError('Source manifest changed during processing; completion withheld')
        _publish(folder/'input.json', dict(schema_version=1, images=inputs))
        _publish(folder/'predictions.json', dict(schema_version=1, kind='cached_detection_experiment',
            room_manifest_sha256=digest(folder/'input.json'), detector=spec, images=images, retrieval_enabled=False))
        _publish(folder/'postprocessed.json', dict(schema_version=SCHEMA, images=postprocessed))
        _publish(folder/'supervision.json', dict(schema_version=SCHEMA, policy=MASK_POLICY, images=supervision))
        metrics = evaluate(folder/'input.json', folder/'predictions.json', iou_threshold=iou_threshold)
        _publish(folder/'evaluation.json', metrics)
        entries = {name:dict(bytes=(folder/name).stat().st_size, sha256=digest(folder/name)) for name in sorted(ARTIFACTS)}
        _publish(folder/'completed.json', dict(schema_version=SCHEMA, status='complete', run_key=run_key,
                                             request=request, artifacts=entries))
        return load_completed_run(folder, expected_request=request)
