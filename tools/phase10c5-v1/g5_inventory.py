"""Join already-local licensed training photos to existing full native labels."""
import csv
import importlib.util
import io
import stat
import zipfile
from collections import Counter,defaultdict
from g5_common import *

def json_entries(path,zip_member=None):
    spec=importlib.util.spec_from_file_location('g5_pinned_stream',ROOT/'tools/phase6b-v1/streaming.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if zip_member is None:
        yield from mod.entries(path);return
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        if len(infos)!=1:raise ValueError('Unexpected LVIS archive')
        info=infos[0]
        if info.filename!=zip_member or info.flag_bits&1 or stat.S_ISLNK(info.external_attr>>16) or info.file_size>4*2**30 or info.file_size>max(1,info.compress_size)*100:raise ValueError('Unsafe ZIP entry')
        with archive.open(info) as binary:
            mod.open=lambda *_a,**_k:io.TextIOWrapper(binary,encoding='utf-8')
            yield from mod.entries(path)

def prepare():
    if RAW.exists() or REPORT.exists():raise FileExistsError('New dataset/review version required')
    RAW.mkdir(parents=True);REPORT.mkdir(parents=True)
    ready_path=ROOT/'data/lvis/image-pilots/phase8b-v1/combined-training-readiness.json';ready=read(ready_path)['source_rows']
    prepared=manifest();active={r['image_id']:r for r in prepared['rows']};rights={r['image_id']:r for r in read(ROOT/'reports/phase10a-v1/rights-replay-ledger.json')['records']}
    protected_alias=set()
    for s in ready:
        r=active.get(s['image_id'])
        if s['split']!='train' or (r and not optimizer_member(r)):
            protected_alias.update(v for v in (s['image_id'],s['sha256'],s['known_source_component'],s.get('original_source_group'),s['source_photo_id'],r and r['identity_group']) if v)
    for s in read(ROOT/'data/open-images/metadata/phase5-v1/pilot-proposal-v4/proposed-image-acquisition.json')['images']:
        if s['proposed_split']!='train':protected_alias.update((s['image_id'],s['source_identity'],s['identity_group']))
    source_layout={
        'legacy_openimages':('data/open-images/image-pilots/phase5-v1','data/open-images/experiments/phase5b-v1/dataset-lock.json','rows'),
        'phase6a_openimages':('data/open-images/image-pilots/phase6a-v1','data/open-images/image-pilots/phase6a-v1/screening-v3.json','images'),
        'phase6d_coco':('data/coco/image-pilots/phase6d-v1','data/coco/image-pilots/phase6d-v1/screening-v2.json','images'),
        'phase7_LVIS':('data/lvis/image-pilots/phase7-v1','data/lvis/image-pilots/phase7-v1/screening-v2.json','images'),
        'phase8c_LVIS':('data/lvis/image-pilots/phase8b-v1','data/lvis/image-pilots/phase8b-v1/screening-v2.json','images')}
    caches={k:{r['image_id']:r for r in read(ROOT/p)[key]} for k,(_,p,key) in source_layout.items()}
    photos={};excluded=[]
    for s in ready:
        r=active.get(s['image_id'])
        if s['split']!='train' or (r and not optimizer_member(r)):
            excluded.append(dict(image_id=s['image_id'],reason='Historical development/internal validation; no training evidence extraction'));continue
        grant=rights[s['image_id']];base=ROOT/source_layout[s['source']][0];original=caches[s['source']][s['image_id']]
        path=ROOT/r['path'] if r else base/original['file']
        aliases={s['image_id'],s['sha256'],s['known_source_component'],s.get('original_source_group'),s['source_photo_id'],grant['creator_group']}
        if aliases&protected_alias:
            excluded.append(dict(image_id=s['image_id'],reason='Known evaluation/source-group overlap'));continue
        if sha(path)!=s['sha256'] or sha(ROOT/grant['evidence_file'])!=grant['evidence_sha256'] or sha(ROOT/grant['rights_record'])!=grant['rights_record_sha256']:raise ValueError('Photograph/rights evidence changed')
        if grant['license_url']!='https://creativecommons.org/licenses/by/2.0/':raise ValueError('Unexpected photo-rights policy; quarantine/review required')
        photos[s['image_id']]=dict(image_id=s['image_id'],source_photo_id=s['source_photo_id'],path=str(path.relative_to(ROOT)),sha256=s['sha256'],source=s['source'],
            width=original['width'],height=original['height'],identity_group=r['identity_group'] if r else grant['creator_group'],
            cohort='optimization59' if r else 'inactive_training_pool_expansion_only',historical_partition='train',internal_partition='optimization' if r else 'inactive_not_assigned',
            eligible_for_primary_AB=bool(r),licence=grant,annotations=[],native_source_scopes={},
            existing_annotations=r['annotations'] if r else [],source_native_scope=s['source_native_scope'],human_complete=False)
    save(RAW/'photos-before-native-join.json',dict(photos=list(photos.values()),excluded=excluded))
    sources={};coco={int(k.split(':')[1]):k for k in photos if k.startswith('coco2017:')};oi={k:k for k in photos if not k.startswith('coco2017:')}
    cats=read(ROOT/'data/coco/metadata/phase6b-v1/category-and-license-definitions.json')['categories'];catco={c['id']:c for c in cats}
    catlv={c['id']:c for c in read(ROOT/'data/lvis/metadata/phase7-v1/native-category-definitions.json')['categories']}
    with (ROOT/'data/open-images/metadata/phase5-v1/oidv7-class-descriptions-boxable.csv').open() as f:catoi=dict(csv.reader(f))
    def add(ident,dataset,ann,categories,source,source_sha):
        p=photos[ident];native=ann['category_id'] if dataset!='OpenImages' else ann['LabelName'];label=categories[native]['name'] if dataset!='OpenImages' else categories[native];group=category_group(label)
        if group is None:return
        if dataset=='OpenImages':
            box=[float(ann[k]) for k in ('XMin','YMin','XMax','YMax')];bad=any(ann[k]=='1' for k in ('IsGroupOf','IsDepiction','IsInside'))
            if not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('Open Images native coordinate drift')
            source_id=digest(ann)[:20];mask=False
        else:
            try:box=normalized_box(ann['bbox'],p['width'],p['height'])
            except ValueError:
                p.setdefault('annotation_issues',[]).append(dict(dataset=dataset,annotation_id=ann['id'],reason='Native box outside preserved frame; quarantined'));return
            bad=bool(ann.get('iscrowd') or ann.get('ignore'));source_id=str(ann['id']);mask=bool(ann.get('segmentation'))
        furniture=[a for a in p['existing_annotations'] if a['label'] in CLASSES or a['ignore']]
        overlap_furniture=any(overlap(box,a['box']) for a in furniture)
        scopes=p['native_source_scopes'].get(dataset,{})
        native_complete=dataset=='LVIS' and native not in scopes.get('not_exhaustive_category_ids',[]) if scopes else False
        record=dict(annotation_key=dataset+':'+str(source_id),dataset=dataset,native_category_id=native,native_label=label,mapped_auxiliary_category=group,box=box,
            has_instance_mask=mask,original_annotation=ann,annotation_source=str(source.relative_to(ROOT)),annotation_source_sha256=source_sha,
            annotation_license='https://creativecommons.org/licenses/by/4.0/',source_native_ignored=bad,
            overlaps_existing_furniture_or_ignored_region=overlap_furniture,native_exhaustiveness_claim=bool(native_complete),
            missing_furniture_annotation_risk=True,Chair_absence_proven_by_object_label=False,
            metadata_candidate_eligible=not bad and group!='protected_furniture',human_reviewed=False,approved_for_new_training=False)
        p['annotations'].append(record)
    co_root=ROOT/'data/coco/metadata/phase6b-v1'
    for info in read(co_root/'extraction-complete.json')['members']:
        path=co_root/info['file'];h=sha(path)
        if h!=info['sha256']:raise ValueError('COCO annotations changed')
        sources[str(path.relative_to(ROOT))]=h;seen=set();temporary=[]
        for key,a in json_entries(path):
            if key=='images' and a['id'] in coco:
                p=photos[coco[a['id']]]
                if (a['width'],a['height'])!=(p['width'],p['height']):raise ValueError('COCO image registration differs')
                seen.add(a['id']);p['native_source_scopes']['COCO']=dict(completeness='Unknown for all seven project concepts; native instances positive only',native_image_id=a['id'])
            elif key=='annotations' and a['image_id'] in coco:temporary.append(a)
        for a in temporary:
            if a['image_id'] not in seen:raise ValueError('COCO annotation without exact image identity')
            add(coco[a['image_id']],'COCO',a,catco,path,h)
        print('JOINED COCO',path.name,'photos',len(seen),'annotations',len(temporary),flush=True)
    lv_root=ROOT/'data/lvis/metadata/phase7-v1'
    for info in read(lv_root/'archive-integrity.json')['archives']:
        path=lv_root/info['file'];h=sha(path)
        if h!=info['sha256']:raise ValueError('LVIS annotations changed')
        sources[str(path.relative_to(ROOT))]=h;seen=set();temporary=[]
        for key,a in json_entries(path,info['member']):
            if key=='images' and a['id'] in coco:
                p=photos[coco[a['id']]]
                if (a['width'],a['height'])!=(p['width'],p['height']):raise ValueError('LVIS coordinate registration differs')
                seen.add(a['id']);p['native_source_scopes']['LVIS']=dict(native_image_id=a['id'],neg_category_ids=a.get('neg_category_ids',[]),not_exhaustive_category_ids=a.get('not_exhaustive_category_ids',[]),missing_category_is_not_negative=True)
            elif key=='annotations' and a['image_id'] in coco:temporary.append(a)
        for a in temporary:
            if a['image_id'] not in seen:raise ValueError('LVIS annotation without exact image registration')
            add(coco[a['image_id']],'LVIS',a,catlv,path,h)
        print('JOINED LVIS',path.name,'photos',len(seen),'annotations',len(temporary),flush=True)
    oi_root=ROOT/'data/open-images/metadata/phase5-v1';expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (oi_root/'SHA256SUMS').read_text().splitlines()}
    for filename in ('validation-annotations-bbox.csv','test-annotations-bbox.csv'):
        path=oi_root/filename;h=sha(path)
        if h!=expected[filename]:raise ValueError('Open Images source annotation changed')
        sources[str(path.relative_to(ROOT))]=h;matched=0
        with path.open(newline='') as f:
            for a in csv.DictReader(f):
                if a['ImageID'] in oi:
                    add(a['ImageID'],'OpenImages',a,catoi,path,h);matched+=1
        print('JOINED OpenImages',filename,'native rows',matched,flush=True)
    # Complete-link duplicate hypotheses never merge photograph/listing identities.
    count=[];groups=[]
    for p in photos.values():
        clusters=[]
        for a in sorted(p['annotations'],key=lambda a:(a['mapped_auxiliary_category'],a['annotation_key'])):
            matches=[c for c in clusters if c[0]['mapped_auxiliary_category']==a['mapped_auxiliary_category'] and all(iou(x['box'],a['box'])>=.95 for x in c)]
            if len(matches)==1:matches[0].append(a)
            else:clusters.append([a])
        for i,c in enumerate(clusters):
            key=p['image_id']+':object-hypothesis-'+str(i)
            for a in c:a['object_hypothesis_group']=key
            if len(c)>1:groups.append(dict(image_id=p['image_id'],group=key,annotation_keys=[a['annotation_key'] for a in c],automatic_label_merge=False))
    for group in GROUPS:
        anns=[(p,a) for p in photos.values() for a in p['annotations'] if a['mapped_auxiliary_category']==group and a['metadata_candidate_eligible']]
        primary=[(p,a) for p,a in anns if p['eligible_for_primary_AB']]
        count.append(dict(category=group,native_annotation_candidates=len(anns),object_hypothesis_clusters=len({a['object_hypothesis_group'] for p,a in anns}),
            local_photos=len({p['image_id'] for p,a in anns}),known_source_groups=len({p['identity_group'] for p,a in anns}),
            primary59_annotations=len(primary),primary59_photos=len({p['image_id'] for p,a in primary}),primary59_groups=len({p['identity_group'] for p,a in primary}),
            source_datasets=dict(Counter(a['dataset'] for p,a in anns)),human_confirmed_new_hard_negatives=0,
            annotation_counts_are_not_independent_physical_objects=True))
    save(RAW/'native-object-inventory.json',dict(schema='phase10c5-local-native-object-inventory-v1',photos=list(photos.values()),excluded_photos=excluded,metadata_sources=sources,
        new_downloads=0,source_annotations_modified=False,all_photo_licences_replayed=True,cross_dataset_annotation_duplicate_hypotheses=groups,counts=count))
    save(REPORT/'inventory-summary.json',dict(counts=count,primary_optimization_photos=sum(p['eligible_for_primary_AB'] for p in photos.values()),
        inactive_training_pool_photos=sum(not p['eligible_for_primary_AB'] for p in photos.values()),excluded_from_train=len(excluded),new_human_confirmed_negatives=0,
        independent_physical_identity_not_verified=True,annotation_licence_does_not_establish_photo_rights=True,
        private_inventory_sha256=sha(RAW/'native-object-inventory.json'),no_new_inference_or_downloads=True))
    print('COMPLETE INVENTORY',[(r['category'],r['native_annotation_candidates'],r['primary59_annotations']) for r in count],flush=True)

if __name__=='__main__':prepare()
