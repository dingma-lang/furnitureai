"""Capture pre-change active pipeline behavior using generated synthetic pixels only."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from furniture_bench.pipeline import run_experiment
from furniture_bench.evaluation import evaluate
from synthetic_pipeline_fixture import create_fixture, SyntheticDetector, semantic_result

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU only')
    with tempfile.TemporaryDirectory(prefix='E14-synthetic-baseline-') as tmp:
        root=Path(tmp);manifest,queries,spec,_,_=create_fixture(root/'inputs')
        detector=SyntheticDetector(queries,spec)
        run=run_experiment(manifest,root/'baseline',detector)
        metrics=evaluate(manifest,root/'baseline/predictions.json')
        result=dict(source_commit='743c39180868162cb009740a95d8e528b8513537',
                    synthetic_only=True,real_model_forwards=0,synthetic_calls=detector.calls,
                    semantic=semantic_result(run,metrics))
        path=ROOT/'reports/phase10e14-v1/baseline-before.json'
        with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print('Baseline recorded:',result['semantic']['detection'])
