"""CPU-only proposal validation. No GPU execution authorized by Phase10E.3."""
import argparse
import json
from pathlib import Path


def validate(plan):
    if any(plan[k] is not False for k in ('approved', 'enabled', 'training_authorized')):
        raise PermissionError('This preparation cannot activate GPU execution')
    if plan['coefficient'] is not None or plan['limits']['optimizer_updates'] != 0:
        raise ValueError('Coefficient must stay unset and updates forbidden')
    ids = plan['adapted_photo_ids']
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError('Exact existing 24-photo evidence required')
    if plan['limits'] != dict(detector_forwards=25, head_verification_contractions=25,
                             VJPs=48, seconds=240, VRAM_bytes=6979321856,
                             output_bytes=134217728, optimizer_updates=0):
        raise ValueError('Recovery limits changed')
    return dict(status='DISABLED_READY_FOR_SEPARATE_AUTHORIZATION', GPU_calls=0,
                adapted_photos=24, original_replay_photos=1, coefficient=None)


def execute(*args, **kwargs):
    raise PermissionError('No GPU execution in Phase10E.3. Separate recovery authorization required.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(validate(json.loads((root / 'reports/phase10e3-v1/recovery-disabled.json').read_text())), indent=2))
