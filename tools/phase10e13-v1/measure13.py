"""CPU saved-candidate filtering benchmark, not a detector benchmark or tuning run."""
import ast
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from verify13 import ROOT, OUT, emit
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools/phase10e12-v1')]
import furniture_bench.postprocessing as post


def original_function(path, name):
    source = subprocess.check_output(['git','show','5818ab333dfd239d581129e02af93238bc85e191:'+path],text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope = {'Sequence':list}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'preserved-original-'+name,'exec'),scope)
    return scope[name]


def main():
    source = ROOT/'reports/gpu-owlv2-rtx4070-test01/predictions.json'
    captures = json.load(source.open())['images']
    configurations = [post.FilterConfig(),post.FilterConfig(containment_threshold=.8),post.FilterConfig(cross_class_iou=.7)]
    cases = [(r.get('detections',[]),cfg) for r in captures for cfg in configurations]
    versions = {'inline_original':(original_function('src/furniture_bench/geometry.py','iou'),original_function('src/furniture_bench/postprocessing.py','containment')),
                'shared_current':(post.iou,post.containment)}
    times = {key:[] for key in versions}
    outputs = {}
    for round_id in range(10):
        order = list(versions)
        if round_id%2:
            order.reverse()
        for name in order:
            iou,containment = versions[name]
            with patch.object(post,'iou',iou),patch.object(post,'containment',containment):
                output = [post.filter_detections(d,c) for d,c in cases]
                output_hash = hashlib.sha256(json.dumps(output,sort_keys=True,allow_nan=False).encode()).hexdigest()
                outputs[name] = output_hash
                start = time.perf_counter()
                for _ in range(200):
                    for candidates,cfg in cases:
                        post.filter_detections(candidates,cfg)
                times[name].append(time.perf_counter()-start)
    if len(set(outputs.values()))!=1:
        raise ValueError('Filtering behavior changed')
    medians = {key:statistics.median(value) for key,value in times.items()}
    emit(OUT/'representative-CPU-performance.json',dict(case_count=len(cases),source_images=len(captures),
         detections_per_source=[len(r.get('detections',[])) for r in captures],rounds=10,repeats_per_case_per_round=200,
         seconds=times,median_seconds=medians,ratio_shared_to_inline=medians['shared_current']/medians['inline_original'],
         output_sha256=outputs,thresholds_are_frozen_fixture_settings_not_selected=True,
         actual_model_inference=False,limitation='Warm saved-candidate CPU workload, only three rooms; no GPU, detector latency or end-to-end improvement inference',
         optimization_decision='No geometry optimization or threshold change; shared semantics retained.'))
    import checks12
    checks12.REPORT = OUT
    checks12.behavior('current')
    before=json.load(open(ROOT/'reports/phase10e12-v1/behavior-after.json'))
    current=json.load(open(OUT/'behavior-current.json'))
    if before!=current:
        raise ValueError('Legacy geometry/fingerprint/saved-output behavior changed')
    emit(OUT/'historical-output-equivalence.json',dict(status='EXACT_MATCH',geometry_cases=25000,saved_filter_cases=9,
         EXIF_and_colour_mode_cases=12,synthetic_matching_protocols=3,official_holdout_scoring=False))
    print('Median CPU seconds:',medians,'shared/inline:',medians['shared_current']/medians['inline_original'])

if __name__=='__main__':
    main()
