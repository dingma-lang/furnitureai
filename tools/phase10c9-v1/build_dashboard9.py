"""Assemble read-only dashboard data; all predictions come from saved arrays."""
import numpy as np
from common9 import *

def run():
    data=read(OUT/'dashboard-data-base.json');train,dev=subsets();rows={r['image_id']:r for r in train};coverage=read(OUT/'training-coverage.json')['photos'];m=metric_module();observations=read(OUT/'assistant-observations.json')['records'];lookup={p['prediction_id']:p for p in observations if p['prediction_id']}
    for photo in data['photos']:
        photo['photo_key']=digest(photo['image_id'])[:16]
        for ps in photo['models'].values():
            for p in ps:
                if p['prediction_id'] in lookup:p['assistant_observation']=lookup[p['prediction_id']]
    for index,c in enumerate(coverage,1):
        row=rows[c['image_id']];assert_mining_scope(row,train,dev);photo=dict(row,photo_key=digest(row['image_id'])[:16],role='optimization_training_only',native=[],models={},old_negative_regions=c['old_regions'],coverage_warning='Saved training probes are FP16; no development/acceptance metrics. All unmatched queries remain unverified. No training labels or approval are created.')
        for model,(arm,epoch) in MODELS.items():
            folder=C8/arm/'probes'/('epoch-'+str(epoch));p=folder/(str(index).zfill(2)+'.npz');receipt=read(p.with_suffix('.json'))
            if sha(p)!=receipt['query_file_sha256'] or receipt['image_id']!=row['image_id']:raise ValueError('Training-probe receipt mismatch')
            a=load_npz(p);scores=1/(1+np.exp(-a['logits'].astype(float)));lab=a['logits'].argmax(1);boxes=registered_box(a['square_cxcywh'],row['width'],row['height']);ps=[]
            for q in range(3600):
                b=boxes[q].tolist();score=float(scores[q,lab[q]])
                if score>=.001 and b[2]>b[0] and b[3]>b[1]:ps.append(dict(box=b,score=score,label=CLASSES[int(lab[q])],query_index=q))
            kept=m.nms(ps,.5,300);photo['models'][model]=[dict(p,prediction_id=f'training:{model}:{row["image_id"]}:q{p["query_index"]}',status='unscored_training_query',taxonomy_primary=None,evidence_tier='saved_query_not_a_label',geometry=geometry(p['box'],row['width'],row['height'])) for p in kept if p['score']>=.15]
        data['photos'].append(photo)
    data['training_summary']=read(REPORT/'hard-negative-summary.json');data['visual_summary']=read(REPORT/'visual-findings.json');data['training_samples']=read(OUT/'training-review-sample.json')['records'];data['observations']=observations
    data['schema']='phase10c9-read-only-dashboard-v1';data['training_enabled']=False;data['GPU_forwards']=0
    # Full source annotations/prediction evidence remain in the sealed analysis
    # records; the viewer transfers only fields needed for inspection.
    compact=[]
    for p in data['photos']:
        q={k:p[k] for k in ('image_id','photo_key','role','width','height','path','sha256','identity_group','coverage_warning')}
        q['annotations']=[{k:a[k] for k in ('box','label','ignore')} for a in p['annotations']]
        q['native']=[{k:a[k] for k in ('box','native_label','annotation_key')} for a in p['native']]
        q['old_negative_regions']=p.get('old_negative_regions',[])
        keys=('box','score','label','query_index','prediction_id','status','geometry','taxonomy_primary','evidence_tier','assistant_observation','pixel_box','fp_coverage')
        q['models']={model:[{k:v for k,v in a.items() if k in keys} for a in ps] for model,ps in p['models'].items()};compact.append(q)
    data['photos']=compact
    save(OUT/'dashboard-live.json',data);print('Dashboard prepared',len(data['photos']),'photos;3 saved model views;no forward passes')

if __name__=='__main__':run()
