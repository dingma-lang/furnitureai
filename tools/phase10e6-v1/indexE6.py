"""Index source-bound optimization query caches;metadata first,never evaluation mining."""
from commonE6 import *

def run():
    import numpy as np,zipfile
    c,train,_=context();records={};unbound=[]
    def add(ident,path,h,state,receipt):
        row=require_train(ident,train);p=(ROOT/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not p.is_relative_to(ROOT/'data') or any(x in p.parts for x in ('evaluation','evaluations','predictions','prehead-cache')):raise PermissionError('Nontraining/ambiguous cache path')
        if sha(p)!=h:raise ValueError('Query cache hash mismatch:'+str(p))
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Image hash mismatch')
        key=str(p.relative_to(ROOT));value=records.setdefault(key,dict(image_id=ident,path=key,sha256=h,states=[],receipts=[],source_sha256=row['sha256'],source_group=row['identity_group']))
        if value['image_id']!=ident or value['sha256']!=h:raise ValueError('Cache bound to contradictory source')
        value['states']=sorted(set(value['states']+[state]));value['receipts']=sorted(set(value['receipts']+[str(receipt)]))
    summary=read(ROOT/'data/open-images/diagnostics/phase10c14-v1/query-summaries.json')
    for r in summary['records']:add(r['image_id'],r['query_file'],r['query_sha256'],r['state'],'C14 sealed query summary')
    for r in read(PREP/'teacher-index.json')['records']:add(r['image_id'],r['cache'],r['cache_sha256'],'original_C17',str(PREP/'teacher-index.json'))
    sys.path.insert(0,str(ROOT/'tools/phase10c21-v1'))
    from negative21 import query_caches
    for r in query_caches():add(r['image_id'],r['path'],r['sha256'],r['state'],'C21 source-bound cache registry')
    c13seal=read(ROOT/'reports/phase10c13-v1/artifact-seal.json');c13private=ROOT/c13seal['private_seal']
    if sha(c13private)!=c13seal['private_seal_sha256']:raise ValueError('C13 private seal changed')
    c13hashes={line.split('  ',1)[1]:line.split('  ',1)[0] for line in c13private.read_text().splitlines()}
    # Initial epoch-zero caches were absent from the older C14 state list.
    for folder in (ROOT/'data/open-images/experiments').glob('*/**/probes/epoch-0'):
        marker=read(folder/'complete.json') if (folder/'complete.json').exists() else {}
        receipt_map={r.get('image_id'):r for r in marker.get('records',[])} if isinstance(marker.get('records'),list) else {}
        for p in folder.glob('[0-9][0-9].npz'):
            receipt=p.with_suffix('.json')
            if not receipt.exists():continue
            r=read(receipt);ident=r.get('image_id')
            if ident not in train:continue
            h=r.get('query_file_sha256') or r.get('sha256') or receipt_map.get(ident,{}).get('sha256') or c13hashes.get(str(p.relative_to(ROOT)))
            if h:add(ident,str(p.relative_to(ROOT)),h,folder.parents[1].name+'-original-epoch0',str(receipt.relative_to(ROOT)))
    # Adjacent receipts are accepted only with explicit optimization identity
    # and a raw hash. This includes C15 recovery aliases without trusting names.
    for folder in (ROOT/'data/open-images/experiments').iterdir():
        if not folder.is_dir():continue
        for receipt in folder.glob('[0-9][0-9].json'):
            r=read(receipt)
            if not isinstance(r,dict) or r.get('image_id') not in train or not r.get('raw_file') or not r.get('raw_sha256'):continue
            if r.get('source_sha256',train[r['image_id']]['sha256'])!=train[r['image_id']]['sha256']:raise ValueError('Source receipt hash mismatch')
            add(r['image_id'],str((folder/r['raw_file']).relative_to(ROOT)),r['raw_sha256'],folder.name,str(receipt.relative_to(ROOT)))
    c7=ROOT/'data/open-images/experiments/phase10c7-query-smoke-v1'
    for r in read(c7/'raw-complete.json')['records']:
        if r['source_sha256']!=require_train(r['image_id'],train)['sha256']:raise ValueError('C7 source mismatch')
        add(r['image_id'],str((c7/r['raw_file']).relative_to(ROOT)),r['raw_sha256'],'C7-original',str((c7/'raw-complete.json').relative_to(ROOT)))
    box=ROOT/'data/open-images/experiments/phase10c21-box-only-v1'
    for r in read(box/'logit-checks/complete.json')['records']:
        add(r['image_id'],str((box/r['file']).relative_to(ROOT)),r['sha256'],'10D-box-only-FP32',str((box/'logit-checks/complete.json').relative_to(ROOT)))
    for r in read(box/'smoke-pass.json')['records']:
        add(r['image_id'],str((box/r['file']).relative_to(ROOT)),r['sha256'],'10D-original-smoke',str((box/'smoke-pass.json').relative_to(ROOT)))
    # E2 and E4 raw files have per-record references and explicit source IDs.
    e2=ROOT/'data/open-images/experiments/phase10e-gradient-smoke-v1'
    for role in ('original','adapted'):
        for p in (e2/role).glob('*.json'):
            r=read(p)
            if 'raw' in r:add(r['image_id'],str((e2/r['raw']['file']).relative_to(ROOT)),r['raw']['sha256'],'E2-'+role,str(p.relative_to(ROOT)))
    # Failed E2 second adapted forward has no finished measurement receipt;
    # its immutable failure/postmortem identifies the exact photograph.
    failure=read(e2/'failure.json');p=e2/'adapted/02.npz'
    add(failure['state']['photo'],str(p.relative_to(ROOT)),sha(p),'E2-adapted-partial','E2 sealed failure and postmortem')
    e4=ROOT/'data/open-images/experiments/phase10e3-recovery-v1'
    done=read(e4/'complete.json')
    for r in done['records']:add(r['image_id'],str((e4/r['raw']['file']).relative_to(ROOT)),r['raw']['sha256'],'E4-C11',str((e4/'complete.json').relative_to(ROOT)))
    orig=read(e4/'original-correspondence.json');ident=read(e4/'binding.json')['original_photo_ids'][0]
    add(ident,str((e4/orig['raw']['file']).relative_to(ROOT)),orig['raw']['sha256'],'E4-original',str((e4/'original-correspondence.json').relative_to(ROOT)))
    # Any remaining top-level raw receipt is catalogued explicitly;unknown
    # layouts/identities remain excluded instead of numerical identity guesses.
    groups={};invalid=[]
    for record in sorted(records.values(),key=lambda r:r['path']):
        p=ROOT/record['path']
        with zipfile.ZipFile(p) as z:
            if len(z.infolist())>20 or sum(v.file_size for v in z.infolist())>2_000_000:raise ValueError('Unsafe raw NPZ')
        with np.load(p,allow_pickle=False) as z:
            if not {'logits','square_cxcywh'}<=set(z.files):invalid.append(record);continue
            a=z['logits'];b=z['square_cxcywh']
        if a.shape!=(3600,7) or b.shape!=(3600,4) or a.dtype!=np.float32 or b.dtype!=np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Raw schema mismatch')
        key=record['image_id']+':'+hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()
        g=groups.setdefault(key,dict(snapshot_id='Q-'+digest(key)[:20],image_id=record['image_id'],source_sha256=record['source_sha256'],source_group=record['source_group'],canonical=record,aliases=[]))
        g['aliases'].append(record)
    for p in (ROOT/'data/open-images/experiments').rglob('*.npz'):
        rel=str(p.relative_to(ROOT))
        if rel in records or any(x in p.parts for x in ('evaluation','evaluations','predictions','prehead-cache')):continue
        unbound.append(dict(path=rel,reason='Not a source-bound raw prediction in approved indexes;may be derived masks,features or metadata-only aliases;not mined'))
    suffix='-v3' if (DATA/'query-index-v2.json').exists() else '-v2' if (DATA/'query-index.json').exists() else ''
    emit(DATA/('query-index'+suffix+'.json'),dict(records=list(groups.values()),raw_source_bound_files=len(records),unique_image_array_snapshots=len(groups),invalid_layout_records=invalid,unindexed_non_evaluation_NPZ=unbound,
         identical_array_aliases_collapsed=len(records)-len(groups),deduplication='Exact source image+logits/boxes bytes;state/receipt aliases preserved,not product identities merged',evaluation_cache_bodies_mined=0))
    emit(REPORT/('cache-coverage'+suffix+'.json'),dict(raw_source_bound_files=len(records),unique_snapshots=len(groups),optimization_images_with_any_cache=len({r['image_id'] for r in groups.values()}),
         nonoriginal_snapshot_images=len({r['image_id'] for r in groups.values() if any('original' not in s and s!='original' for a in r['aliases'] for s in a['states'])}),
         E5_full_training_raw_caches=0,E5_training_presentations_with_summary_only=295,E5_raw_development_caches_excluded=19,
         unindexed_non_evaluation_files=len(unbound),unindexed_files_not_mined=True))
    print('Indexed',len(records),'source-bound caches;',len(groups),'unique snapshots',flush=True)
if __name__=='__main__':run()
