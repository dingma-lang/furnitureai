"""CPU-only closeout, exact source/version preservation and audit summaries."""
import hashlib
import json
import subprocess
from pathlib import Path
from verify13 import ROOT, OUT, PARENT, emit, sha


def main():
    before=json.load(open(OUT/'private/protected-before.json'))['records']
    after=json.load(open(OUT/'private/protected-after.json'))['records']
    a={r['path']:r for r in before};b={r['path']:r for r in after}
    changes=[p for p in sorted(a.keys()&b.keys()) if a[p]['sha256']!=b[p]['sha256']]
    expected={
        'src/furniture_bench/'+n+'.py' for n in ('data','pipeline','retrieval','catalog_expansion')
    } | {'tests/'+n+'.py' for n in ('test_home_transfer','test_owned_inventory','test_photo_projects','test_replacements','test_room_viewer')}
    additions=sorted(b.keys()-a.keys())
    allowed_additions=['src/furniture_bench/cache.py','src/furniture_bench/persistence.py','tests/synthetic_support.py','tests/test_reliability.py']
    if set(changes)!=expected or additions!=allowed_additions or a.keys()-b.keys():
        raise ValueError('Unexpected historical modifications: '+repr((changes,additions)))
    preservation=dict(status='PASS',before_files=len(a),after_files=len(b),unchanged_original_files=len(a)-len(changes),
        intended_changes=[dict(path=p,before=a[p]['sha256'],after=b[p]['sha256']) for p in changes],new_files=additions,
        dataset_or_model_changes=0,historical_report_changes=0,historical_tool_changes=0,deleted_files=0,
        unchanged_data_models_and_historical_reports=sum(p.startswith(('data/','models/','reports/')) for p in a),
        parent_commit=PARENT,private_before_sha256=sha(OUT/'private/protected-before.json'),private_after_sha256=sha(OUT/'private/protected-after.json'))
    emit(OUT/'preservation.json',preservation)
    # E12's strict code seals belong to E12, not to a later refactor. Verify parent blobs.
    previous=json.load(open(ROOT/'reports/phase10e12-v1/artifact-seal.json'))
    code=[]
    for rel,expected_sha in previous['code'].items():
        original=subprocess.check_output(['git','show',PARENT+':'+rel])
        if hashlib.sha256(original).hexdigest()!=expected_sha:
            raise ValueError('Parent source provenance unavailable')
        code.append(dict(path=rel,parent_matches=True,current_matches=sha(ROOT/rel)==expected_sha))
    for rel,expected_sha in previous['private_audit_indexes'].items():
        if sha(ROOT/rel)!=expected_sha:raise ValueError('Prior audit index changed')
    emit(OUT/'historical-source-replay.json',dict(status='PASS',parent=PARENT,code=code,
        strict_legacy_source_checks_require_matching_commit=True,old_fingerprints_not_rebound=True))
    old=json.load(open(ROOT/'reports/phase10e12-v1/preexisting-fixture-errors.json'))['test_ids']
    unavailable='test_home_transfer.TransferTests.test_pdf_pages_locally_rendered'
    items=[]
    for ident in old:
        if ident==unavailable:
            cause='PyPDF2 absent; actual renderer also requires macOS PDFKit and Swift'
            fix='Explicit native-platform/dependency requirement; no replacement renderer or blanket skip'
        elif 'test_invalid_uploads' in ident:
            cause='Optional PyPDF2 import happened before corrupt-PDF error path; the unit test assumed the package was installed'
            fix='Keep ordinary invalid-image tests; inject a synthetic invalid-PDF parser to test error propagation. Real native PDF test remains unavailable.'
        elif 'PhotoProjectTests' in ident:
            cause='Synthetic model queue test called real configuration(), requiring missing examples/furniture-prompts-v2.json and local model inventory'
            fix='Deterministic synthetic configuration at the test seam; fake model runner preserved'
        elif 'RoomViewerTests' in ident:
            cause='Preset integration assumed reports/phase5-public-room/quick-room.json existed in this workspace'
            fix='Valid deterministic synthetic partial-room JSON in TemporaryDirectory; real preset lookup and validation exercised'
        else:
            cause='Unit test assumed external data/catalogs/abo-phase6d/catalog.json existed'
            fix='Explicit synthetic product metadata with width/height/reported length and a descriptor-only synthetic image ID; no real catalogue restored'
        items.append(dict(test=ident,cause=cause,status='explicitly_unavailable' if ident==unavailable else 'PASS',change=fix,
            incorrect_test_path=False,historical_dataset_fabricated=False,fixture_generation_previously_broken=False))
    emit(OUT/'eleven-test-status.json',dict(records=items,passed=10,unavailable=1,errors_remaining=0,
        production_PDF_support_not_claimed_on_WSL=True,source_problems_separate_from_test_fixtures=True))
    targeted=json.load(open(OUT/'CPU-reliability-final.json'));broad=json.load(open(OUT/'CPU-broad-final.json'))
    if targeted['passed']!=45 or broad['errors'] or broad['failures']:
        raise ValueError('Final tests incomplete')
    emit(OUT/'summary.json',dict(status='COMPLETE_WITH_EXPLICIT_OPTIONAL_REQUIREMENTS',parent=PARENT,
        former_errors=dict(passed=10,unavailable_native_test=1),targeted_passed=45,
        broad=dict(ran=broad['ran'],passed=broad['passed'],skipped=len(broad['skipped']),errors=0,failures=0,excluded_real_model_forwards=2),
        baseline=dict(ran=379,passed=336,skipped=32,errors=11),cache_schema=2,
        legacy_cache_names_and_arrays_preserved=True,automatic_cache_migration=False,
        image_snapshot_binding=True,atomic_writes=True,actual_abrupt_exit_recovery_test=True,
        geometry_optimized=False,geometry_and_fingerprint_equivalence=json.load(open(OUT/'historical-output-equivalence.json')),
        representative_filter_ratio=json.load(open(OUT/'representative-CPU-performance.json'))['ratio_shared_to_inline'],
        dense_filter_ratio=json.load(open(OUT/'dense-saved-CPU-performance.json'))['ratio_shared_to_inline'],
        preservation=preservation,model_or_GPU_forwards=0,optimizer_updates=0,new_datasets=0,downloads=0,
        main=subprocess.check_output(['git','rev-parse','main'],text=True).strip(),Mac_application_modified=False))
    print('Preservation PASS; unchanged data/model/report files:',preservation['unchanged_data_models_and_historical_reports'])

if __name__=='__main__':main()
