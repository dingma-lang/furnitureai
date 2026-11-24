"""Confirm missing-fixture errors against unchanged parent code in memory only."""
import contextlib,io,json,os,re,subprocess,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
from audit12 import emit,OUT
PARENT='5818ab333dfd239d581129e02af93238bc85e191'
def main():
    if (OUT/'CPU-parent-missing-fixtures.log').exists() or (OUT/'preexisting-fixture-errors.json').exists():raise FileExistsError('Existing audit records are immutable')
    import furniture_bench
    for short in ('geometry','data','postprocessing','evaluation','dataset_quality'):
        name='furniture_bench.'+short;rel='src/furniture_bench/'+short+'.py'
        source=subprocess.check_output(['git','show',PARENT+':'+rel],text=True)
        m=types.ModuleType(name);m.__file__=str(ROOT/rel);m.__package__='furniture_bench';sys.modules[name]=m
        exec(compile(source,m.__file__,'exec'),m.__dict__);setattr(furniture_bench,short,m)
    text=(OUT/'CPU-extended-v2.log').read_text();ids=re.findall(r'^ERROR: .*? \(([^)]+)\)$',text,re.M)
    if len(ids)!=11:raise ValueError('Unexpected error list; review first')
    log=io.StringIO()
    with patch('socket.create_connection',side_effect=AssertionError('No network')),contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(ids))
    (OUT/'CPU-parent-missing-fixtures.log').write_text(log.getvalue())
    oldpaths=sorted(re.findall("No such file or directory: '([^']+)'",log.getvalue()));newpaths=sorted(re.findall("No such file or directory: '([^']+)'",text))
    value=dict(parent=PARENT,source_loaded_from_git_into_memory_only=True,test_ids=ids,ran=result.testsRun,errors=len(result.errors),failures=len(result.failures),
        same_missing_fixture_errors=len(result.errors)==11 and not result.failures and oldpaths==newpaths,
        missing_paths=sorted(set(oldpaths)),production_or_historical_files_changed=False,models_run=0)
    emit(OUT/'preexisting-fixture-errors.json',value);print(json.dumps(value,indent=2))
    if not value['same_missing_fixture_errors']:raise SystemExit(1)
if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    main()
