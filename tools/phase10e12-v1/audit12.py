"""Read-only inventory/AST audit; no module imports, model loads or data writes."""
import argparse,ast,collections,hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/phase10e12-v1'
CODE_ROOTS=('src','scripts','tests','tests_inpainting_eval','tests_photo_simulator','tools','data')
EXCLUDE_PARTS={'.git','.venv','.runtime','.cache','.local-simulations','__pycache__'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def emit(p,value):
 p=Path(p)
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def files(roots):
 for root in roots:
  base=ROOT/root
  if not base.exists():continue
  for parent,dirs,names in os.walk(base,followlinks=False):
   dirs[:]=sorted(d for d in dirs if d not in EXCLUDE_PARTS and not (Path(parent)/d).is_symlink())
   for name in sorted(names):
    p=Path(parent)/name
    if p.is_symlink() or p.name.startswith('._'):continue
    yield p

def snapshot(stage):
 start=time.perf_counter();records=[];cache={};stats=collections.defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0));seen=set()
 roots=['data','models','reports','src','scripts','tests','tests_inpainting_eval','tests_photo_simulator','tools','simulator-assets']
 for p in files(roots):
  rel=p.relative_to(ROOT).as_posix()
  if rel.startswith(('reports/phase10e12-v1/','tools/phase10e12-v1/')):continue
  s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if key not in cache:cache[key]=digest(p)
  records.append(dict(path=rel,size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=cache[key]))
  group=stats[rel.split('/')[0]];group['files']+=1;group['logical_bytes']+=s.st_size
  if key not in seen:group['allocated_bytes']+=s.st_blocks*512;seen.add(key)
 emit(OUT/'private'/('protected-'+stage+'.json'),dict(records=records,stats=dict(stats),elapsed_seconds=time.perf_counter()-start,root=str(ROOT)))
 print('Snapshot',stage,'files',len(records),'unique inodes',len(cache),'seconds',round(time.perf_counter()-start,3),flush=True)
 return records

def decisions(fn):
 count=1
 def visit(node):
  nonlocal count
  if node is not fn and isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)):return
  if isinstance(node,(ast.If,ast.For,ast.AsyncFor,ast.While,ast.ExceptHandler,ast.IfExp)):count+=1
  elif isinstance(node,ast.BoolOp):count+=len(node.values)-1
  elif isinstance(node,ast.comprehension):count+=1+len(node.ifs)
  elif isinstance(node,ast.Match):count+=max(0,len(node.cases)-1)
  for child in ast.iter_child_nodes(node):visit(child)
 visit(fn);return count

def static(stage):
 modules=[];functions=[];clones=collections.defaultdict(list);parse_errors=[];signals=collections.Counter();risks=[];import_names=collections.defaultdict(list)
 for p in files(CODE_ROOTS):
  rel=p.relative_to(ROOT).as_posix()
  if p.suffix!='.py' or rel.startswith('tools/phase10e12-v1/'):continue
  text=p.read_text(encoding='utf-8-sig');lines=text.splitlines()
  try:tree=ast.parse(text,filename=rel)
  except SyntaxError as e:parse_errors.append(dict(path=rel,line=e.lineno,message=e.msg));continue
  fns=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
  modules.append(dict(path=rel,lines=len(lines),bytes=p.stat().st_size,functions=len(fns),classes=sum(isinstance(n,ast.ClassDef) for n in ast.walk(tree))))
  import_names[p.stem].append(rel)
  for fn in fns:
   row=dict(path=rel,name=fn.name,line=fn.lineno,lines=fn.end_lineno-fn.lineno+1,decision_proxy=decisions(fn));functions.append(row)
   body=fn.body[1:] if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str) else fn.body
   key=ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)+'|'+ast.dump(fn.args,include_attributes=False)
   if row['lines']>=4:clones[hashlib.sha256(key.encode()).hexdigest()].append(row)
  for node in ast.walk(tree):
   if isinstance(node,ast.ExceptHandler) and (node.type is None or ast.unparse(node.type) in ('Exception','BaseException')):
    signals['broad_exception_handlers']+=1
    if all(isinstance(n,ast.Pass) for n in node.body):risks.append(dict(path=rel,line=node.lineno,kind='broad_exception_pass',needs_context_review=True))
   if isinstance(node,ast.Call):
    name=ast.unparse(node.func)
    if name.endswith('sys.path.insert'):signals['sys_path_insert_calls']+=1
    if name in ('eval','exec'):signals['dynamic_execution_calls']+=1
    if name.endswith('.load') or name.endswith('.from_pretrained'):signals['load_or_pretrained_calls']+=1
    for kw in node.keywords:
     if kw.arg=='verify' and isinstance(kw.value,ast.Constant) and kw.value.value is False:risks.append(dict(path=rel,line=node.lineno,kind='TLS_verify_false'))
  if '/home/' in text:signals['modules_with_absolute_home_paths']+=1
 duplicate=[group for group in clones.values() if len(group)>1];duplicate.sort(key=lambda g:sum(r['lines'] for r in g),reverse=True)
 by_root=collections.defaultdict(lambda:dict(modules=0,lines=0,functions=0))
 for m in modules:
  group=by_root[m['path'].split('/')[0]]
  for name in ('lines','functions'):group[name]+=m[name]
  group['modules']+=1
 result=dict(stage=stage,modules=modules,functions=functions,parse_errors=parse_errors,totals=dict(modules=len(modules),lines=sum(x['lines'] for x in modules),functions=len(functions)),
   by_root=dict(by_root),exact_AST_duplicate_groups=duplicate,duplicate_group_count=len(duplicate),signals=dict(signals),static_risk_signals=risks,
   colliding_module_basenames={k:v for k,v in import_names.items() if len(v)>1},
   complexity_definition='decision proxy=1+if/for/while/except/conditional-expr/boolean edges/comprehensions/match branches; nested functions excluded. Not external-tool McCabe.',
   semantic_unused_code_claims=False)
 emit(OUT/'private'/('AST-'+stage+'.json'),result)
 emit(OUT/('static-summary-'+stage+'.json'),{k:result[k] for k in ('stage','totals','by_root','parse_errors','duplicate_group_count','signals','complexity_definition')})
 print('AST',stage,result['totals'],'duplicate groups',len(duplicate),'syntax errors',len(parse_errors),flush=True)
 print('Active-package duplicate groups',json.dumps([g for g in duplicate if any(r['path'].startswith('src/furniture_bench/') for r in g)][:12],indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['snapshot','static']);p.add_argument('stage',choices=['before','after']);a=p.parse_args();globals()[a.action](a.stage)
