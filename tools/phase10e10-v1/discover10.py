"""No-auth first-party image metadata; eight fixed strata, no pagination or image GET."""
from common10 import *
from net10 import Client
from wellcome10 import adapt,query_url,THEMES
from decisions10 import assert_unique

def decoded_receipt(r):
    import gzip
    b=(ROOT/r['file']).read_bytes()
    if sha(ROOT/r['file'])!=r['sha256']:raise ValueError('Evidence hash changed')
    if any(k.lower()=='content-encoding' and v=='gzip' for k,v in r['headers'].items()):b=gzip.decompress(b)
    return json.loads(b)
def run():
    c=Client();rows={};done=set();work_records=20 # Request22 returned20 works; count these conservatively within200.
    # Reuse the already completed first image query; never repeat successful requests.
    first=read(OUT/'requests/023.json');value=decoded_receipt(first)
    for item in value['results']:
        r=adapt(item,first,'bedroom');rows[r['candidate_id']]=r
    for p in sorted((OUT/'discovery-batches').glob('*.json')):
        b=read(p);done.add(b['theme'])
        for r in b['records']:
            if r['candidate_id'] not in rows:rows[r['candidate_id']]=r
    for i,theme in enumerate(THEMES):
        if theme in done:continue
        if len(rows)+work_records+20>MAX_CANDIDATES:raise RuntimeError('Conservative200record reservation exceeded')
        receipt,text=c.fetch(query_url(theme))
        if text is None:print('Provider stop',receipt['status'],flush=True);break
        value=json.loads(text)
        if value.get('type')!='ResultList' or 'results' not in value:raise ValueError('Unexpected catalogue schema; stop')
        if len(value['results'])>20:raise ValueError('Per-query record ceiling')
        batch=[adapt(item,receipt,theme) for item in value['results']]
        for r in batch:
            if r['candidate_id'] not in rows:rows[r['candidate_id']]=r
            elif theme not in rows[r['candidate_id']]['discovery_themes']:rows[r['candidate_id']]['discovery_themes'].append(theme)
        assert_unique(list(rows.values()))
        emit(OUT/'discovery-batches'/f'{i+1:02}.json',dict(theme=theme,records=batch,total_results_not_acquired=value.get('totalResults'),next_page_not_followed=value.get('nextPage'),unique_images=len(rows)))
        print(theme,'returned',len(batch),'unique',len(rows),'total source matches',value.get('totalResults'),flush=True)
    output=OUT/('discovered-candidates-v2.json' if (OUT/'discovered-candidates.json').exists() else 'discovered-candidates.json')
    emit(output,dict(records=sorted(rows.values(),key=lambda r:r['candidate_id']),unique_image_candidates=len(rows),additional_initial_work_records=work_records,
        conservative_candidate_record_count=len(rows)+work_records,max_candidates=200,network=c.summary()))
if __name__=='__main__':run()
