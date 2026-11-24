"""Exact historical preservation and synthetic CPU verification closeout."""
import hashlib
import json
import subprocess
from verify14 import ROOT, OUT, PARENT, emit, sha


def main():
    before=json.loads((OUT/'private/protected-before.json').read_text())['records']
    after=json.loads((OUT/'private/protected-after-final.json').read_text())['records']
    a={r['path']:r for r in before};b={r['path']:r for r in after}
    changes=[p for p in a.keys()&b.keys() if a[p]['sha256'] != b[p]['sha256']]
    additions=sorted(b.keys()-a.keys())
    expected={'src/furniture_bench/pipeline.py','src/furniture_bench/persistence.py'}
    if set(changes)!=expected or a.keys()-b.keys() or additions != ['src/furniture_bench/cached_pipeline.py','tests/test_pipeline_reliability.py']:
        raise ValueError('Unexpected protected-file delta: '+repr((changes,additions)))
    record=dict(status='PASS',before_files=len(a),after_files=len(b),
        modified=[dict(path=p,before=a[p]['sha256'],after=b[p]['sha256']) for p in sorted(changes)],new_files=additions,
        unchanged_original_files=len(a)-len(changes),
        unchanged_data_models_and_historical_reports=sum(p.startswith(('data/','models/','reports/')) for p in a),
        deleted_files=0,dataset_changes=0,checkpoint_changes=0,historical_report_changes=0,historical_tool_changes=0,
        private_before_sha256=sha(OUT/'private/protected-before.json'),private_after_sha256=sha(OUT/'private/protected-after-final.json'),
        note='Synthetic fixture was created and included before baseline capture and preservation snapshot; new phase tools/reports are sealed separately.')
    emit(OUT/'preservation-final.json',record)
    old=json.loads((ROOT/'reports/phase10e13-v1/artifact-seal.json').read_text())
    replay=[]
    for rel,expected_sha in old['files'].items():
        if '/private/' in rel:
            if sha(ROOT/rel) != expected_sha:raise ValueError('Prior private index changed')
        else:
            payload=subprocess.check_output(['git','show',PARENT+':'+rel],cwd=ROOT)
            if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ValueError('Prior source/evidence unavailable')
            replay.append(dict(path=rel,parent_matches=True,current_matches=sha(ROOT/rel)==expected_sha))
    emit(OUT/'historical-source-replay-final.json',dict(status='PASS',parent=PARENT,records=replay,
        prior_source_checks_require_recorded_commit=True,old_seals_rebound=False))
    tests=json.loads((OUT/'CPU-targeted-final2.json').read_text())
    broad=json.loads((OUT/'CPU-broad-final2.json').read_text())
    if tests['passed']!=34 or tests['errors'] or tests['failures'] or broad['errors'] or broad['failures']:
        raise ValueError('CPU verification incomplete')
    emit(OUT/'summary-final.json',dict(status='COMPLETE',parent=PARENT,targeted_passed=tests['passed'],
        broad=dict(ran=broad['ran'],passed=broad['passed'],skipped=len(broad['skipped']),errors=0,failures=0,real_forward_tests_excluded=2),
        synthetic_equivalence='EXACT_MATCH',invalidation_matrix_rows=16,zero_update=True,
        real_model_forwards=0,GPU_calls=0,downloads=0,real_annotation_changes=0,
        multi_file_commit='validated immutable artifact list + atomic completed.json',
        concurrency='cooperative POSIX/WSL locks + atomic no-clobber + immutable partition bindings',
        historical_preservation=record,main=subprocess.check_output(['git','rev-parse','main'],text=True,cwd=ROOT).strip(),
        decision='READY for separately authorized dataset acquisition and bounded model experiments using versioned adapters and existing preflights; no further routine cleanup needed',
        critical_CPU_reliability_blockers=[],
        limitations=['No live model execution was tested.',
            'Legacy UI caches are not converted or endorsed as v2.',
            'Cross-root unknown identity relationships need authoritative dataset registry checks.',
            'Native PDF integration remains unavailable on WSL.',
            'Supervision output is provenance/geometry, not generated model-query training targets.']))
    print('Preservation PASS:',len(a)-len(changes),'original files unchanged')


if __name__=='__main__':main()
