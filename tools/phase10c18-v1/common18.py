"""One authorized C18 arm; immutable files and exact historical inputs."""
import sys, io, signal, threading, subprocess, copy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/phase10c17-v1'))
from common17 import *
HERE18 = Path(__file__).resolve().parent
RUN18 = ROOT / 'data/open-images/experiments/phase10c18-source-balanced-v1'
REPORT18 = ROOT / 'reports/phase10c18-v1'
BRANCH18 = 'experiment/phase10c18-source-balanced-training'
PARENT18 = '00b372072f52fbab8cebaa308b96a395c6f3e6f8'
MAX_OUTPUT18 = 4 * 2**30
LOCK18 = threading.RLock()


def allocation18():
    return sum(p.stat().st_blocks * 512 for folder in (RUN18, REPORT18) if folder.exists() for p in folder.rglob('*'))


def atomic18(path, data):
    with LOCK18:
        path = Path(path)
        if not any(path.resolve().is_relative_to(base) for base in (RUN18, REPORT18)):
            raise PermissionError('Outside C18 output roots')
        if path.exists():
            raise FileExistsError('Immutable C18 artifact exists')
        if allocation18() + 2 * ((len(data) + 4095) // 4096) * 4096 + 8192 > MAX_OUTPUT18:
            raise RuntimeError('Conservative 4GiB output bound')
        pending = path.with_name('.' + path.name + '.pending-' + uuid.uuid4().hex[:8])
        with pending.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(pending, path)  # Retain, and count, recovery links.


def emit18(path, value):
    atomic18(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def verify17_seals():
    count = verify_seal(REPORT17 / 'SHA256SUMS')
    seal = read(REPORT17 / 'artifact-seal.json')
    private = ROOT / seal['private_manifest']
    if sha(private) != seal['private_manifest_sha256']:
        raise ValueError('C17 private seal changed')
    for rel, h in {**read(private)['files'], **seal['code_sha256']}.items():
        if sha(ROOT / rel) != h:
            raise ValueError('C17 historical artifact changed: ' + rel)
    return dict(public_files=count, private_files=seal['private_files'])


def inputs18():
    cfg = read(RUN18 / 'binding.json')
    if not cfg.get('training_authorized') or cfg['arms'] != 1 or cfg['limits'] != dict(epochs=5, updates=75, GPU_seconds=600, VRAM_bytes=MAX_VRAM17, output_bytes=MAX_OUTPUT18, forwards=448, training_VJPs=885, probe_VJPs=288):
        raise PermissionError('Exact single-arm C18 authorization required')
    for rel, h in cfg['files'].items():
        if sha(ROOT / rel) != h:
            raise ValueError('Bound input changed: ' + rel)
    prepared, train, dev, _, registry = training_context()
    if digest(train) != cfg['training_rows_sha256'] or digest(dev) != cfg['development_rows_sha256']:
        raise ValueError('Partitions changed')
    old = read(C8 / 'binding.json')['configs']['B-corrected']
    if cfg['unchanged_C8_settings'] != old or cfg['evaluation'] != read(C8 / 'binding.json')['evaluation_protocol']:
        raise ValueError('C8 configuration/protocol drift')
    return cfg, prepared, train, dev, registry


def write_npz(path, arrays):
    import numpy as np
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite raw outputs')
    b = io.BytesIO(); np.savez_compressed(b, **arrays)
    atomic18(path, b.getvalue())


def prepare():
    if RUN18.exists() or REPORT18.exists():
        raise FileExistsError('Fresh C18 destination required')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != PARENT18 or subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != BRANCH18:
        raise ValueError('Dedicated C18 branch and original parent required')
    seal = verify17_seals()
    previous, prepared, groups, registry = inputs17()
    future = read(REPORT17 / 'future-training-disabled.json')
    from spec16 import validate_spec
    validate_spec(future)
    done = read(RUN17 / 'complete.json')
    if done['status'] != 'PASS' or done['forwards'] != 59 or done['model_updates'] or done['model_state_before'] != done['model_state_after']:
        raise ValueError('Successful unchanged-model C17 smoke required')
    _, train, dev, _, _ = training_context()
    from common8 import grouped_epoch
    orders = [[[r['image_id'] for r in group] for group in grouped_epoch(train, epoch)] for epoch in range(5)]
    for epoch in range(5):
        for index, order in enumerate(orders[epoch], 1):
            if read(C8 / 'B-corrected' / ('update-' + str(epoch * 15 + index).zfill(3) + '.json'))['source_order'] != order:
                raise ValueError('C8 order mismatch')
    files = dict(previous['files'])
    for p in [*HERE18.glob('*.py'), REPORT17/'results.json', REPORT17/'future-training-disabled.json', C8/'binding.json', C8/'B-corrected/complete.json', C8/'metric-results.json']:
        files[str(p.relative_to(ROOT))] = sha(p)
    cfg = dict(schema='phase10c18-single-source-balanced-training-v1', training_authorized=True, arms=1, Phase10D_authorized=False, parent=PARENT18, branch=BRANCH18, files=files, training_rows_sha256=digest(train), development_rows_sha256=digest(dev), orders=orders, unchanged_C8_settings=future['unchanged_C8_settings'], evaluation=future['evaluation'], trainable_keys=future['trainable_keys'], method=dict(future['method'], enabled=True), initialization=future['initialization'], limits=dict(epochs=5, updates=75, GPU_seconds=600, VRAM_bytes=MAX_VRAM17, output_bytes=MAX_OUTPUT18, forwards=448, training_VJPs=885, probe_VJPs=288), comparison_notes=['Sole intentional objective change: trusted Chair focal to immutable C16 source-normalized BCE plus accumulated 25% class-parameter cap.', 'Three isolated scaled VJPs compose base/source; upstream AMP rounding may differ from the historical single-backward summed objective, as disclosed in C17.', 'Original 12 heads including calibration; no freezing, projection regularization, new labels or changed evaluation.', 'Fresh C18 output path; retain all75 snapshots. No automatic retry. Endpoint epoch5 fixed.', '24 training-only probes at0/1/3/5,FP16;19historical diagnostic predictions at1/3/5,FP32. Original reference reuses saved predictions.'], automatic_retry=False)
    RUN18.mkdir(parents=True); REPORT18.mkdir(parents=True)
    for name in ('updates','checkpoints','probes','evaluation','telemetry'):
        (RUN18/name).mkdir()
    emit18(RUN18/'binding.json', cfg)
    emit18(REPORT18/'authorized-config.json', {k:v for k,v in cfg.items() if k!='files'})
    emit18(REPORT18/'preflight.json', dict(status='PASS',C17_seal=seal,photos=59,source_groups=56,positive_boxes=182,source_regions=41,original_checkpoint_sha256=cfg['initialization']['SHA256'],same_order_all75_updates=True,comparison_notes=cfg['comparison_notes'],training_not_yet_executed=True))
    print('C18 input preflight PASS; one fresh original-initialized arm,75updates maximum', flush=True)


if __name__ == '__main__':
    prepare()
