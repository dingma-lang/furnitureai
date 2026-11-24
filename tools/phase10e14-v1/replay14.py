"""Auditable synthetic-only replay, dependency matrix and completion evidence."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

from verify14 import ROOT, OUT, emit
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from furniture_bench.cache import make_contract, validate_contract
from furniture_bench.cached_pipeline import _prepare, stage_keys
from furniture_bench.data import read_json, write_json
from furniture_bench.pipeline import run_cached_detection
from furniture_bench.postprocessing import FilterConfig
from synthetic_pipeline_fixture import create_fixture, SyntheticDetector, semantic_result


def main():
    with tempfile.TemporaryDirectory(prefix='E14-isolated-synthetic-replay-') as tmp:
        root=Path(tmp);manifest,queries,spec,weights,processor=create_fixture(root/'inputs')
        detector=SyntheticDetector(queries,spec)
        def run(policy=None):
            return run_cached_detection(manifest,root/'cache',detector,spec=spec,model_files=weights,
                                        processor_files=processor,policy=policy)
        initial=run();reloaded=run()
        before=read_json(OUT/'baseline-before.json')
        semantic=semantic_result(initial['artifacts']['predictions.json'],initial['artifacts']['evaluation.json'])
        if semantic != before['semantic'] or initial != reloaded or detector.calls != 3:
            raise AssertionError('Pre-change semantics or cache reload differs')
        emit(OUT/'synthetic-equivalence.json',dict(status='EXACT_MATCH',real_model_forwards=0,
            generated_images=3,oriented_dimensions=[[r['width'],r['height']] for r in semantic['images']],
            synthetic_raw_candidates=sum(map(len,queries.values())),raw_logit_values=sum(len(p['logits']) for ps in queries.values() for p in ps),
            source_groups=3,synthetic_partitions={'training':2,'development':1},synthetic_detector_calls=3,
            reload_extra_calls=0,semantic=semantic))
        _,prepared=_prepare(manifest,spec);c=prepared[0][1];p=FilterConfig();protocol={'iou_threshold':.5}
        base=stage_keys(c,p,protocol);matrix=[]
        def record(name,contract=c,policy=p,rules=protocol):
            keys=stage_keys(contract,policy,rules)
            matrix.append(dict(change=name,invalidates=[k for k in keys if keys[k]!=base[k]],action='versioned-recompute'))
        changes=[('image bytes','inference','image_sha256','e'*64),
            ('orientation/preprocessing','inference','preprocessing',dict(c['inference']['preprocessing'],orientation='different-protocol')),
            ('checkpoint','inference','model',dict(spec['model'],checkpoint_sha256='e'*64)),
            ('processor','inference','model',dict(spec['model'],processor_sha256='e'*64)),
            ('prompt order','inference','prompts',list(reversed(spec['prompts']))),
            ('category order','inference','categories',list(reversed(spec['categories']))),
            ('taxonomy description only','supervision','taxonomy_version','v2'),
            ('ground truth boxes','supervision','annotations',[]),
            ('annotation completeness','supervision','completeness',{v:'unknown' for v in spec['categories']}),
            ('ignored regions','supervision','ignore_regions',[]),
            ('protected regions','supervision','protected_regions',[]),
            ('source group same partition','supervision','source_group_id','different-source-group')]
        for name,section,key,value in changes:
            x=copy.deepcopy(c);x[section][key]=value;i=x['inference']
            contract=make_contract(image_id=i['image_id'],image_sha256=i['image_sha256'],model=i['model'],
                preprocessing=i['preprocessing'],categories=i['categories'],prompts=i['prompts'],
                query_categories=i['query_categories'],inference_settings=i['settings'],supervision=x['supervision'])
            record(name,contract)
        record('NMS',policy=FilterConfig(nms_iou=1.));record('confidence',policy=FilterConfig(threshold=.95))
        record('matching IoU',rules={'iou_threshold':.75})
        x=copy.deepcopy(c);x['schema_version']=3
        try:validate_contract(x)
        except ValueError:matrix.append(dict(change='unsupported schema',invalidates=['raw','postprocessed','evaluation','supervision'],action='reject, never migrate implicitly'))
        else:raise AssertionError('Unknown schema accepted')
        emit(OUT/'cache-invalidation-matrix.json',dict(rows=matrix,
            raw='pre-NMS candidates at explicit acquisition floor; NMS/confidence above floor are separate',
            supervision='geometry descriptor, not model-query target tensor',
            limitations=['Unsupported preprocessing protocols are rejected by this adapter.',
                         'Category/prompt meaning changes require raw recomputation; description-only taxonomy changes do not.',
                         'Partition reassignment is rejected, not merely invalidated.']))
        value=read_json(manifest);value['images'][0]['objects'][0]['box']=[2,25,16,39]
        write_json(manifest,value);changed=run()
        nms=run(FilterConfig(nms_iou=1.))
        if detector.calls != 3:raise AssertionError('Derived-only edits reran synthetic inference')
        emit(OUT/'synthetic-derived-results.json',dict(synthetic_only=True,real_model_forwards=0,
            runs_have_distinct_keys=len({initial['run_key'],changed['run_key'],nms['run_key']})==3,
            original=initial['artifacts']['evaluation.json']['detection'],
            edited_synthetic_box=changed['artifacts']['evaluation.json']['detection'],
            NMS_synthetic_test_only=nms['artifacts']['evaluation.json']['detection'],
            total_synthetic_detector_calls=detector.calls,raw_recomputed_for_label_or_NMS_change=False,
            source_annotations_changed='temporary generated fixture only; zero real annotations changed'))
        print('Synthetic exact equivalence, reload, derived recomputation and matrix PASS')

if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise PermissionError('Offline CPU-only execution required')
    main()
