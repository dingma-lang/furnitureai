"""Targeted integrity/sealing of this closed pilot; no model or network work."""
import subprocess
from common16 import *
sys.path.insert(0,str(ROOT/'tools/phase10e15-v1'))
from verify15 import preserve as preserve15

PARENT='03534497404bf3a8dc6c1bf49c108f336b1b28f4'
MAIN='e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6'


def preservation():
    previous=preserve15()
    seal=read(ROOT/'reports/phase10e15-v1/artifact-seal.json')
    for rel,expected in seal['files'].items():
        payload=subprocess.check_output(['git','show',PARENT+':'+rel],cwd=ROOT)
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Parent source/evidence unavailable')
        if rel!='.gitignore' and sha(ROOT/rel)!=expected:raise ValueError('E15 historical artifact changed')
    closed=read(OUT/'acquisition-closed.json')
    if closed['stop'] or tuple(r['source_id'] for r in closed['records'])!=IDS:raise ValueError('Pilot completeness differs')
    for r in closed['records']:
        p=ROOT/r['file']
        if sha(p)!=r['sha256'] or sha(p,'sha1')!=r['sha1']:raise ValueError('Original acquired photo changed')
    if allocated()>DISK:raise RuntimeError('128MiB allocation exceeded')
    return dict(status='PASS',previous_historical_photos_verified=previous['historical_photos_verified'],
        original_pilot_photos_unchanged=3,E15_artifacts_unchanged=True,core_code_unchanged=True,
        main=MAIN,new_dataset_partitions_or_annotations=False,
        verification_scope='E15 input/source/photo preservation plus parent code/report seals and pilot hashes; no exhaustive new code audit')


def seal():
    preserved=preservation();emit(REPORT/'historical-preservation.json',preserved)
    closed=read(OUT/'acquisition-closed.json');browser=read(REPORT/'browser-smoke-v2.json')
    local=sum(json.loads(l)['body_bytes'] for l in (OUT/'browser-local-http.jsonl').read_text().splitlines())
    # Source stream charge includes HTTP/TLS and a per-request allowance. Local
    # browser bodies/allowances are recorded separately and stay below60M combined.
    local_requests=len((OUT/'browser-local-http.jsonl').read_text().splitlines())
    combined=closed['network']['conservative_transfer_charge']+local+local_requests*65536
    if combined>CAP:raise RuntimeError('Combined source/local browser conservative allowance exceeded')
    emit(REPORT/'summary.json',dict(parent=PARENT,acquired=3,image_body_bytes=21251887,
        network=closed['network'],local_browser_response_body_bytes=local,
        source_plus_local_browser_conservative_charge=combined,
        suitability={'A':1,'B':1,'C':1},active_training_images=0,active_boxes=0,active_masks=0,
        independent_evaluation_images=0,new_verified_hard_negatives=0,
        intake_content_verified=3,intake_metadata_unqualified_pending_real_confirmation=3,
        browser_end_to_end_verified=False,browser_final_status=browser['status'],browser_attempts=2,
        E16_focused_CPU_tests_passed=16,E15_intake_regressions_passed=12,
        real_model_calls=0,GPU_calls=0,training_updates=0,
        E15_backup='origin/experiment/phase10e15-modern-room-intake matches0353449',
        gallery=str((OUT/'gallery.html').relative_to(ROOT)),allocated_bytes_before_seal=allocated(),storage_cap=DISK,
        next_action='Prepare exact grants/IDs for eight targeted creator-consented scenes; no automatic outreach/acquisition',
        main=MAIN))
    paths=[ROOT/'.gitignore',*sorted(HERE.glob('*')),*sorted(REPORT.glob('*'))]
    paths=[p for p in paths if p.is_file() and p.name not in ('artifact-seal.json','SHA256SUMS')]
    # Only approved pilot data is sealed locally. Public seal contains hashes/paths, not binaries.
    data=[p for p in OUT.rglob('*') if p.is_file() and not any(part.startswith('phase10e16-profile-') for part in p.relative_to(OUT).parts)]
    emit(REPORT/'artifact-seal.json',dict(parent=PARENT,code_and_reports={str(p.relative_to(ROOT)):sha(p) for p in paths},
        pilot_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(data)},
        isolated_browser_profiles_excluded_from_replay_seal=True))
    with (REPORT/'SHA256SUMS').open('x') as f:
        for p in paths+[REPORT/'artifact-seal.json']:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print('E16 seal PASS; allocated bytes:',allocated())


def check():
    value=read(REPORT/'artifact-seal.json')
    for group in ('code_and_reports','pilot_files'):
        for rel,expected in value[group].items():
            if sha(ROOT/rel)!=expected:raise ValueError('Sealed file changed: '+rel)
    preservation();print('E16 closed-pilot and historical preservation PASS; allocated bytes:',allocated())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['seal','check']);a=p.parse_args()
    seal() if a.action=='seal' else check()
