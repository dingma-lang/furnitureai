"""File-based, offline test runner; real model-forward tests remain excluded."""
import argparse
import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from verify14 import ROOT, OUT, emit, sha
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]

def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item

def run(label, mode):
    log_path = OUT/('CPU-'+label+'.log')
    if log_path.exists() or (OUT/('CPU-'+label+'.json')).exists():
        raise FileExistsError('Choose a fresh report label')
    if mode == 'eleven':
        ids = json.load(open(ROOT/'reports/phase10e12-v1/preexisting-fixture-errors.json'))['test_ids']
        suite = unittest.defaultTestLoader.loadTestsFromNames(ids)
    else:
        pattern = 'test_pipeline_reliability.py' if mode == 'targeted' else 'test_*.py'
        suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern)
    tests = list(flatten(suite))
    excluded = [t.id() for t in tests if 'TinyTransformersTests.' in t.id()]
    tests = [t for t in tests if t.id() not in excluded]
    ids = {id(t): t.id() for t in tests}
    log = io.StringIO()
    with patch('socket.create_connection', side_effect=AssertionError('No source network requests allowed')), contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(tests))
    log_path.write_text(log.getvalue())
    skipped = [{'test': ids[id(t)], 'reason': reason} for t,reason in result.skipped]
    errors = [{'test': ids[id(t)], 'traceback': trace} for t,trace in result.errors]
    failures = [{'test': ids[id(t)], 'traceback': trace} for t,trace in result.failures]
    value = dict(status='PASS' if result.wasSuccessful() else 'FAIL', ran=result.testsRun,
                 passed=result.testsRun-len(skipped)-len(errors)-len(failures), skipped=skipped, errors=errors,
                 failures=failures, excluded_actual_model_forwards=excluded, log_sha256=sha(log_path),
                 model_forwards=0, GPU_calls=0, downloads=0)
    emit(OUT/('CPU-'+label+'.json'), value)
    print(json.dumps({k:v for k,v in value.items() if k!='traceback'}, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('HF_HUB_OFFLINE')!='1':
        raise PermissionError('Offline CPU flags required')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['eleven','targeted','broad'])
    parser.add_argument('label')
    args=parser.parse_args()
    if not args.label.replace('-','').isalnum():
        raise ValueError('Safe report label required')
    run(args.label,args.mode)
