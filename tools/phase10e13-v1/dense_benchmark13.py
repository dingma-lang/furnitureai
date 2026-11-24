"""Saved exposed-development predictions only: no images, labels or scoring."""
import hashlib
import json
import statistics
import sys
import time
from unittest.mock import patch
from verify13 import ROOT, OUT, emit, sha
from measure13 import original_function
sys.path.insert(0,str(ROOT/'src'))
import furniture_bench.postprocessing as post


def main():
    path=ROOT/'data/open-images/experiments/phase10e-distillation-fp32-v1/CPU-analysis/predictions-E5.json'
    predictions=json.load(path.open())
    rows=json.load(open(ROOT/'data/open-images/preparation/phase10e1-v1/training-context.json'))['development']
    if set(predictions)!={r['image_id'] for r in rows} or len(rows)!=19 or any(r['internal_partition']!='inner_development' for r in rows):
        raise ValueError('Exact exposed19-image internal protocol required')
    current,old=post.iou,original_function('src/furniture_bench/geometry.py','iou')
    times={name:[] for name in ('inline_original','shared_current')};checksums={};cfg=post.FilterConfig()
    for round_id in range(8):
        order=[('inline_original',old),('shared_current',current)]
        if round_id%2:order.reverse()
        for name,fn in order:
            with patch.object(post,'iou',fn):
                results=[post.filter_detections(predictions[k],cfg) for k in sorted(predictions)]
                checksums[name]=hashlib.sha256(json.dumps(results,sort_keys=True,allow_nan=False).encode()).hexdigest()
                start=time.perf_counter()
                for _ in range(3):
                    for key in sorted(predictions):post.filter_detections(predictions[key],cfg)
                times[name].append(time.perf_counter()-start)
    if len(set(checksums.values()))!=1:raise ValueError('Changed filtering output')
    medians={name:statistics.median(values) for name,values in times.items()}
    emit(OUT/'dense-saved-CPU-performance.json',dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),images=19,
         saved_predictions=sum(map(len,predictions.values())),maximum_per_image=max(map(len,predictions.values())),
         rounds=8,repeats=3,seconds=times,median_seconds=medians,ratio_shared_to_inline=medians['shared_current']/medians['inline_original'],
         exact_output_sha256=checksums,photographs_decoded=0,ground_truth_used_for_filtering_or_scoring=False,partition_metadata_read=True,model_forward_calls=0,heldout_scoring=False,
         threshold=.15,nms_iou=.5,settings_not_tuned=True,geometry_optimization_applied=False,
         limitations='Saved already-filtered/capped outputs, not raw3600queries or full detector latency; exposed internal data only.'))
    print(medians,'ratio',medians['shared_current']/medians['inline_original'])

if __name__=='__main__':main()
