"""Byte-level backup inventory only; never decodes images or deserializes models."""
import datetime
import subprocess
from common import ROOT, OUT, sha, save, read

def run():
    if (OUT / 'backup-inventory.json').exists():
        raise FileExistsError('Preserve existing inventory; use verify()')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    records = []
    for folder in ('data', 'models'):
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_symlink():
                raise ValueError('Inventory requires explicit symlink policy: ' + str(p))
            if not p.is_file():
                continue
            before = p.stat()
            digest = sha(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('File changed during hash: ' + str(p))
            relative = str(p.relative_to(ROOT))
            records.append(dict(path=relative, bytes=after.st_size, allocated_bytes=after.st_blocks*512,
                                sha256=digest, mtime_ns=after.st_mtime_ns,
                                git_tracked=relative in tracked,
                                checkpoint_candidate=p.suffix in ('.pt', '.pth', '.safetensors', '.bin', '.onnx'),
                                independent_backup_verified=False))
    save(OUT / 'backup-inventory.json', dict(schema='phase10a-byte-inventory-v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=records,
        files=len(records), bytes=sum(r['bytes'] for r in records),
        checkpoint_candidates=sum(r['checkpoint_candidate'] for r in records),
        scope=['data/**', 'models/**'], photo_decoding=False, network_or_copy=False,
        backup_status='No separately verified backup destination was available. Hashes are integrity evidence, not backups.'))
    with (OUT / 'backup-paths.txt').open('x') as f:
        f.write(''.join(r['path']+'\n' for r in records))
    print('INVENTORY', len(records), 'files', sum(r['bytes'] for r in records), 'bytes', flush=True)

def verify():
    data = read(OUT / 'backup-inventory.json')
    for r in data['records']:
        p = ROOT / r['path']
        if p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Historical bytes changed: ' + r['path'])
    return dict(status='PASS', files=len(data['records']), byte_level_verified=True,
                frozen_photo_access='Checksums only; no decoding, inference or scoring')

if __name__ == '__main__':
    run()
