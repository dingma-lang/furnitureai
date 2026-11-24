"""Read-only preservation and CPU test evidence for Phase10E.13."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/phase10e13-v1'
PARENT = '5e9c53d446c5659c2ef82bb17a91372bc88cda4d'
MAIN = 'e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def emit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')

def snapshot(stage):
    records = []
    inode_hashes = {}
    for root in ('data','models','reports','src','scripts','tests','tests_photo_simulator','tests_inpainting_eval','tools','simulator-assets'):
        for folder, dirs, names in os.walk(ROOT / root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not (Path(folder)/d).is_symlink())
            for name in sorted(names):
                p = Path(folder)/name
                rel = p.relative_to(ROOT).as_posix()
                if p.is_symlink() or name.startswith('._') or rel.startswith(('reports/phase10e13-v1/','tools/phase10e13-v1/')):
                    continue
                stat = p.stat()
                key = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
                if key not in inode_hashes:
                    inode_hashes[key] = sha(p)
                records.append(dict(path=rel,bytes=stat.st_size,sha256=inode_hashes[key]))
    emit(OUT/'private'/('protected-'+stage+'.json'),dict(records=records))
    print('Protected snapshot',stage,len(records),'files',flush=True)


def seal():
    """Freeze this phase's evidence without rebinding any older source seal."""
    code = [ROOT/'.gitignore']
    code += [ROOT/'src/furniture_bench'/(name+'.py') for name in (
        'data','pipeline','retrieval','catalog_expansion','cache','persistence')]
    code += [ROOT/'tests'/(name+'.py') for name in (
        'test_home_transfer','test_owned_inventory','test_photo_projects',
        'test_replacements','test_room_viewer','synthetic_support','test_reliability')]
    code += sorted((ROOT/'tools/phase10e13-v1').glob('*.py'))
    code += [ROOT/'tools/phase10e13-v1/README.md']
    reports = sorted(p for p in OUT.iterdir() if p.is_file()
                     and p.name not in ('artifact-seal.json','SHA256SUMS'))
    private = [OUT/'private/protected-before.json', OUT/'private/protected-after.json']
    paths = code + reports + private
    fingerprints = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    emit(OUT/'artifact-seal.json', dict(parent=PARENT, main=MAIN, files=fingerprints))
    # Public checksum list deliberately excludes private file names and contents.
    checks = code + reports + [OUT/'artifact-seal.json']
    with (OUT/'SHA256SUMS').open('x', encoding='utf-8') as stream:
        for p in checks:
            stream.write(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n')
    print('Sealed', len(fingerprints), 'files; private inventories remain local')


def check(full=False):
    value = json.loads((OUT/'artifact-seal.json').read_text())
    if value['parent'] != PARENT or value['main'] != MAIN:
        raise ValueError('Unexpected source ancestry')
    for rel, expected in value['files'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Sealed artifact changed: '+rel)
    if subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip() != MAIN:
        raise ValueError('main changed')
    subprocess.run(['git','merge-base','--is-ancestor',PARENT,'HEAD'],cwd=ROOT,check=True)
    if full:
        records=json.loads((OUT/'private/protected-after.json').read_text())['records']
        for record in records:
            if sha(ROOT/record['path']) != record['sha256']:
                raise ValueError('Protected artifact changed: '+record['path'])
        print('Historical content verified:',len(records),'files')
    print('Phase10E.13 artifact seal PASS; main unchanged')

if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU-only environment required')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['before','after','seal','check'])
    parser.add_argument('--full',action='store_true',help='Rehash every protected after-snapshot file')
    args = parser.parse_args()
    if args.action in ('before','after'):
        snapshot(args.action)
    elif args.action == 'seal':
        seal()
    else:
        check(args.full)
