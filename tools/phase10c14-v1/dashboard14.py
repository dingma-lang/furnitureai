"""Build/serve data from optimization-only caches; no model is loaded."""
import math
import numpy as np
from common14 import *
from mining14 import read_derived
from common9 import registered_box
from audit14 import overlap_arrays


def build():
    _,train,dev,native,reg=training_context();summaries=read(RAW14/'query-summaries.json')['records'];byid={r['image_id']:[] for r in train}
    for r in summaries:byid[r['image_id']].append(r['state'])
    old=read(SOURCE/'source-negative-evidence.json')['records'];photos=[];logs={r['image_id']:{} for r in train}
    for name,root,applied in [('c8_control',C8/'A-control',False),('c8_corrected',C8/'B-corrected',True),('c11',C11_RUN,True),('c13',RUN13,True)]:
        entries=[r for i in range(1,76) for r in read(root/('update-'+str(i).zfill(3)+'.json'))['images']]
        for row in train:
            part=[r for r in entries if r['image_id']==row['image_id']]
            if len(part)!=5:raise ValueError('Expected five actual training presentations per photo')
            logs[row['image_id']][name]=dict(presentations=5,mean_classification_loss=float(np.mean([r['classification_loss'] for r in part])),mean_positive_classification_loss=float(np.mean([r['positive_classification'] for r in part])),mean_native_negative_loss=float(np.mean([r['legacy_negative_classification'] for r in part])),mean_source_negative_loss=float(np.mean([r['added_Chair_negative_loss'] for r in part])),mean_box_loss=float(np.mean([r['box_loss'] for r in part])),source_negative_applied=applied,scope='Actual five training presentations;no per-query arrays were saved for these presentations')
    for row in train:
        require_training_row(row,train,dev)
        photos.append(dict(image_id=row['image_id'],photo_key=digest(row['image_id'])[:16],role='optimization',width=row['width'],height=row['height'],sha256=row['sha256'],identity_group=row['identity_group'],path=row['path'],available_states=byid[row['image_id']],
            annotations=[{k:a[k] for k in ('box','label','ignore')} for a in row['annotations']],native=[{k:a[k] for k in ('box','native_label','annotation_key','mapped_auxiliary_category','dataset','has_instance_mask')} for a in native[row['image_id']]['annotations']],
            protected=reg[row['image_id']]['protected'],training_log_means=logs[row['image_id']],source_regions=[dict(candidate_id=e['candidate_id'],box=e['box'],source_annotation_key=e['source_annotation_key']) for e in old if e['image_id']==row['image_id']]))
    data=dict(schema='phase10c14-training-query-audit-v1',photos=sorted(photos,key=lambda p:p['image_id']),states=[{k:r[k] for k in ('key','label','epoch','corrected')} for r in STATES],classes=list(CLASSES),summary=read(REPORT14/'query-assignment-summary.json'),hard_negative=read(REPORT14/'hard-negative-audit.json'),teacher=read(REPORT14/'teacher-feasibility.json'),
        candidate_regions=[{k:r[k] for k in ('candidate_key','image_id','native_label','box','evidence_tier','training_eligible','query_observations')} for r in read(RAW14/'new-hard-negative-candidates.json')['candidates']],inspection_samples=read(RAW14/'training-inspection-sample.json')['records'],visual_findings=read(REPORT14/'visual-findings.json'),read_only=True,model_forwards=0,training_enabled=False)
    save(RAW14/'dashboard-v3.json',data);print('Read-only training dashboard built:59photos,24cached;no inference',flush=True)


class ViewStore:
    def __init__(self,data=None):
        self.data=read(RAW14/'dashboard-v3.json') if data is None else data
        self.photos={p['photo_key']:p for p in self.data['photos']}
        summaries=read(RAW14/'query-summaries.json')['records']
        self.views={(r['photo_key'],r['state']):r for r in summaries}

    def load(self,key,state):
        if key not in self.photos or (key,state) not in self.views:raise KeyError('No cached state;no inference fallback')
        photo=self.photos[key];r=self.views[(key,state)]
        if r['image_id']!=photo['image_id'] or sha(ROOT/r['query_file'])!=r['query_sha256'] or sha(ROOT/r['derived_file'])!=r['derived_sha256']:raise ValueError('Cached query identity/hash mismatch')
        a=load_npz(ROOT/r['query_file']);d=read_derived(ROOT/r['derived_file']);box=registered_box(a['square_cxcywh'],photo['width'],photo['height'],clip=False)
        return photo,r,a,d,box

    def view(self,key,state):
        photo,r,a,d,box=self.load(key,state);score=1/(1+np.exp(-a['logits'].astype(float)));bits=2**np.arange(7,dtype=np.uint16)
        return dict(photo_key=key,state=state,boxes=box.tolist(),scores=score.tolist(),positive_bits=((d['target']*d['used'])*bits).sum(1).tolist(),valid_bits=(d['used']*bits).sum(1).tolist(),source_available=d['added_available'][:,1].tolist(),assignment=d['assignment'].tolist(),padding_anchor=d['padding_anchor'].tolist(),known_Chair_IoU05=d['known_Chair_IoU05'].tolist(),fragment_heuristic=d['fragment_heuristic'].tolist(),summary=r,
            caution='Actual cached probe outputs;CPU-reconstructed assignments,not per-update saved matches. Ignored does not mean background. No training-label writes.')

    def detail(self,key,state,q):
        photo,r,a,d,box=self.load(key,state)
        if type(q) is not int or not 0<=q<len(box):raise KeyError('Invalid cached query index')
        native=photo['native'];i,frac,_=overlap_arrays(box[q:q+1],[g['box'] for g in native]);context=[dict(native_label=g['native_label'],annotation_key=g['annotation_key'],dataset=g['dataset'],box=g['box'],IoU=float(i[0,j]),fraction_query_in_native_bbox=float(frac[0,j]),Chair_absence_proven=False) for j,g in enumerate(native) if frac[0,j]>=.1];context=sorted(context,key=lambda x:-x['fraction_query_in_native_bbox'])[:10]
        predicted=box[q].tolist();regions=[]
        for e in photo['source_regions']:
            b=e['box'];inter=max(0,min(b[2],predicted[2])-max(b[0],predicted[0]))*max(0,min(b[3],predicted[3])-max(b[1],predicted[1]))
            if inter:regions.append(dict(candidate_id=e['candidate_id'],fully_contains_query=bool(predicted[0]>=b[0] and predicted[1]>=b[1] and predicted[2]<=b[2] and predicted[3]<=b[3]),box=b))
        return dict(image_id=photo['image_id'],identity_group=photo['identity_group'],state=state,query_index=q,dense_patch_row=q//60,dense_patch_column=q%60,original_image_dimensions=[photo['width'],photo['height']],box_original_normalized_unclipped=predicted,box_original_pixels_unclipped=(box[q]*np.array([photo['width'],photo['height']]*2)).tolist(),
            logits=a['logits'][q].tolist(),sigmoid_scores=(1/(1+np.exp(-a['logits'][q].astype(float)))).tolist(),classes=list(CLASSES),positive_target=d['target'][q].tolist(),active_loss_mask=d['used'][q].tolist(),source_negative_available=d['added_available'][q].tolist(),reconstructed_positive_annotation_index=int(d['assignment'][q]),
            output_logit_gradients={k:d[k][q].tolist() for k in ('grad_positive','grad_native','grad_added','grad_classification')},output_box_gradient=d['grad_box'][q].tolist(),native_bbox_context=context,existing_Chair_negative_regions=regions,
            note='Model predictions and native bbox overlaps are not new labels. Gradients here are CPU-reconstructed output derivatives;head norms are separate photo-level recorded diagnostics.')


if __name__=='__main__':build()
