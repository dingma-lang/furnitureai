"""CPU tests, historical preservation and immutable new-artifact sealing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from audit3 import ROOT, HERE, REPORT, PREP3, emit, sha, read
from telemetry3 import atomic_json


def tests():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU-only test command required')
    suites = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    plans = [
        ('E3', ['discover', '-s', str(HERE), '-p', 'test_*3.py'], ROOT),
        ('E1-mask-cache', ['test_distillationE.MaskLossTests', 'test_distillationE.CacheTests'], ROOT/'tools/phase10e1-v1'),
        ('C11-scope', ['test_preflight10.ScopeTests'], ROOT/'tools/phase10c10-v1'),
        ('E2-legacy', ['test_gradient2', 'test_failure2'], ROOT/'tools/phase10e2-v1'),
    ]
    for name, args, cwd in plans:
        log = PREP3 / ('tests-' + name + '-' + uuid.uuid4().hex[:8] + '.log')
        with log.open('x') as f:
            r = subprocess.run([sys.executable, '-B', '-m', 'unittest', *args, '-v'],
                               cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
        text = log.read_text()
        match = re.search(r'Ran (\d+) tests?', text)
        if r.returncode or not match or '\nOK' not in text or 'skipped=' in text:
            raise RuntimeError('CPU suite failed: ' + str(log))
        suites.append(dict(suite=name, passed=int(match[1]), log=str(log.relative_to(ROOT)), sha256=sha(log)))
        print('CPU PASS', name, match[1], flush=True)
    # Test reruns are allowed, without replacing any previous result.
    result = dict(status='PASS', passed=sum(s['passed'] for s in suites), suites=suites,
                  GPU_model_forwards=0, model_weight_updates=0,
                  note='Scope regression tests use isolated synthetic optimizer fixtures, never real checkpoint updates.')
    target = REPORT / ('tests.json' if not (REPORT/'tests.json').exists() else 'tests-' + uuid.uuid4().hex[:8] + '.json')
    atomic_json(target, result)


def preservation():
    from verifyE import check as check1
    sys.path.insert(0, str(ROOT/'tools/phase10e2-v1'))
    from verify2 import check as check2
    sys.path.insert(0, str(ROOT/'tools/phase10d-v1'))
    from sealD import check as checkD
    sys.path.insert(0, str(ROOT/'tools/phase10c21-v1'))
    from verify21 import check_seal
    from common21 import verify20_seals, verify19_seals, verify18_seals, verify17_seals, C8, verify_seal
    for check in (check1, check2, checkD, check_seal):
        check()
    spec = importlib.util.spec_from_file_location('_historical_integrity3', ROOT/'tools/phase10c-v1/verify.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    result = old.preservation()
    extra = {}
    for phase in ('phase10c11-v1', 'phase10c13-v1'):
        folder = ROOT/'reports'/phase
        seal = read(folder/'artifact-seal.json')
        private = ROOT/seal['private_seal']
        if sha(private) != seal['private_seal_sha256']:
            raise ValueError('Historical private seal changed')
        extra[phase] = dict(public=verify_seal(folder/'SHA256SUMS'), private=verify_seal(private))
    result.update(C11_C13=extra, C20=verify20_seals(), C19=verify19_seals(), C18=verify18_seals(),
                  C17=verify17_seals(), C8_private_files=verify_seal(C8/'ARTIFACT-SHA256SUMS'),
                  D_verified=True, E1_verified=True, E2_verified=True, C21_verified=True,
                  main_commit=subprocess.check_output(['git','rev-parse','main'], text=True).strip(),
                  new_GPU_forwards=0, new_optimizer_updates=0, new_training=False,
                  preserved_files=read(REPORT/'objective-preservation.json')['file_fingerprints'])
    if result['main_commit'] != 'e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6':
        raise ValueError('main changed')
    for path, h in result['preserved_files'].items():
        if sha(ROOT/path) != h:
            raise ValueError('Immutable input changed: ' + path)
    emit('preservation.json', result)
    print('Historical preservation PASS', result['historical_data_model_files'], flush=True)


def dtype_evidence():
    import torch
    from collections import Counter
    from safetensors import safe_open
    from commonE import BASE
    plan = read(REPORT/'recovery-disabled.json')
    with safe_open(BASE/'model.safetensors', framework='pt', device='cpu') as f:
        counts = Counter(f.get_slice(k).get_dtype() for k in f.keys())
    checkpoint = torch.load(ROOT/plan['reference_checkpoint'], map_location='cpu', weights_only=True)
    dtypes = {k:str(v.dtype) for k,v in checkpoint['weights'].items()}
    if dict(counts) != {'F32':418} or set(dtypes.values()) != {'torch.float32'} or torch.cuda.is_initialized():
        raise ValueError('Unexpected original/C11 dtype or CUDA use')
    emit('checkpoint-dtypes.json', dict(original_persisted_tensors=dict(counts),
         C11_head_tensor_dtypes=dtypes, device='cpu', CUDA_calls=0, checkpoint_updates=0,
         method='Read original safetensors slice headers and C11 weights_only CPU payload; no model construction/inference'))


def references():
    import torch
    from test_numerics3 import fixture, reference
    from stable3 import widened_vjp
    torch.set_num_threads(2)
    records = []
    for dtype in (None, torch.float16, torch.bfloat16):
        for magnitude in (1e-4, 1., 1e5):
            h, _, _, _, snapshot = fixture(dtype)
            g = torch.randn_like(snapshot.logits) * magnitude
            actual = widened_vjp(snapshot, g, h.dense0.weight)
            ref = reference(h, snapshot, g)
            a = torch.cat([v.flatten().double() for v in actual])
            b = torch.cat([v.flatten() for v in ref])
            torch.testing.assert_close(a, b, rtol=2e-5, atol=max(1e-6,magnitude*2e-6))
            records.append(dict(forward_precision=str(dtype) if dtype else 'torch.float32',
                terminal_gradient_magnitude=magnitude, maximum_absolute_error=float((a-b).abs().max()),
                relative_L2_error=float((a-b).norm()/b.norm()), finite=bool(torch.isfinite(a).all()),
                reference='Independent FP64 autograd evaluated at observed rounded nodes'))
    emit('CPU-reference-results.json', dict(status='PASS', comparisons=records,
         CUDA_validation=False, exact_E2_failure_reproduced=False, coefficient=None))


def public_files():
    return [p for p in sorted(REPORT.iterdir()) if p.is_file() and not p.name.startswith('.')]


def seal(revision=1):
    if read(REPORT/'tests.json')['status'] != 'PASS' or read(REPORT/'preservation.json')['status'] != 'PASS':
        raise ValueError('Passing tests and preservation required')
    if not (REPORT/'report.md').exists():
        raise ValueError('Report required')
    paths = [p for folder in (HERE, PREP3) for p in sorted(folder.rglob('*')) if p.is_file()]
    latest_tests = max(REPORT.glob('tests*.json'), key=lambda p:p.stat().st_mtime_ns)
    suffix = '' if revision == 1 else '-v2'
    emit('final-verification' + suffix + '.json', dict(status='PASS', CPU_tests=read(latest_tests)['passed'],
         test_record=str(latest_tests.relative_to(ROOT)), test_record_sha256=sha(latest_tests),
         GPU_forwards=0, coefficient=None, training_authorized=False))
    emit('artifact-seal' + suffix + '.json', dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},
          GPU_forwards=0, training=False, original_and_historical_parameters_unchanged=True,
          supersedes='Initial E3 seal retained;v2 omits private atomic staging hardlinks from the public checksum index' if revision == 2 else None))
    files = [p for p in public_files() if p.name != 'SHA256SUMS' + suffix]
    # Atomic text seal with the same no-overwrite publication discipline.
    path = REPORT/('SHA256SUMS' + suffix)
    raw = ''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in files)
    with path.open('x') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print('E3 sealed', len(paths), 'code/private files;', len(files), 'public files', flush=True)


def check():
    suffix = '-v2' if (REPORT/'artifact-seal-v2.json').exists() else ''
    for rel, h in read(REPORT/('artifact-seal' + suffix + '.json'))['files'].items():
        if sha(ROOT/rel) != h:
            raise ValueError('Sealed E3 artifact changed: ' + rel)
    for line in (REPORT/('SHA256SUMS' + suffix)).read_text().splitlines():
        h, rel = line.split('  ',1)
        if sha(ROOT/rel) != h:
            raise ValueError('Sealed public E3 report changed: ' + rel)
    print('E3 final seal PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tests', action='store_true')
    p.add_argument('--preservation', action='store_true')
    p.add_argument('--dtype-evidence', action='store_true')
    p.add_argument('--references', action='store_true')
    p.add_argument('--seal', action='store_true')
    p.add_argument('--seal-v2', action='store_true')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU-only verification required')
    if a.tests:
        tests()
    elif a.preservation:
        preservation()
    elif a.dtype_evidence:
        dtype_evidence()
    elif a.references:
        references()
    elif a.seal:
        seal()
    elif a.seal_v2:
        seal(2)
    elif a.check:
        check()
    else:
        p.error('Specify a CPU verification action')
