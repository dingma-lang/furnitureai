"""Read existing LVIS ZIP/index only;no extraction,rights upgrade or new labels."""
import collections,importlib.util,sqlite3,zipfile
from common9 import *
CORE={982:'sofa',232:'chair',1050:'table',77:'bed',361:'desk',181:'cabinet',626:'lamp'}
# The historical SQLite index intentionally stores federated flags only for
# these23selected core/subtype IDs,not all1203native categories.
INDEX_RELEVANT=set(CORE)|{19,170,285,301,346,358,367,438,464,558,610,627,628,731,889,1051}
def run():
    rows=read(OUT/'local-candidate-audit.json')['records'];ids={r['native_image']['id']:r for r in rows}
    base=ROOT/'data/lvis/metadata/phase7-v1';db=base/'furniture-index.sqlite'
    archives={'train':('lvis_v1_train.json.zip','334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba'),
              'val':('lvis_v1_val.json.zip','5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5')}
    for name,h in archives.values():
        if sha(base/name)!=h:raise ValueError('Existing LVIS archive changed')
    defs={r['id']:r for r in read(base/'native-category-definitions.json')['categories']}
    c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);placeholders=','.join('?'*len(ids))
    images=list(c.execute('SELECT split,id,width,height,url,neg,nex FROM images WHERE id IN ('+placeholders+')',list(ids)))
    facts=list(c.execute('SELECT split,image,category,instances,masks,polygons,rles,quality FROM facts WHERE image IN ('+placeholders+')',list(ids)));c.close()
    if len(images)!=64 or any(split not in archives or (w,h)!=(ids[i]['width'],ids[i]['height']) for split,i,w,h,*_ in images):raise ValueError('LVIS/COCOidentity dimensions/split mismatch')
    expected={(image,category):n for split,image,category,n,*_ in facts};seen=collections.Counter();annotations=[];actual_images={}
    spec=importlib.util.spec_from_file_location('_E9_local_LVIS_stream',ROOT/'tools/phase6b-v1/streaming.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cache=OUT/'LVIS-selected-stream-cache.json';proof=OUT/'LVIS-selected-stream-cache-proof.json'
    if cache.exists():
        if sha(cache)!=read(proof)['sha256']:raise ValueError('Selected LVIS stream cache changed')
        value=read(cache);annotations=value['annotations'];actual_images={int(i):v for i,v in value['images'].items()}
        seen=collections.Counter((a['original_annotation']['image_id'],a['original_annotation']['category_id']) for a in annotations)
    else:
        for native_split,(name,h) in archives.items():
            archive=base/name
            with zipfile.ZipFile(archive) as z:
                members=z.infolist()
                if len(members)!=1 or members[0].filename!='lvis_v1_'+native_split+'.json' or members[0].file_size>4*2**30:raise ValueError('Unexpected existing LVIS archive structure')
                m.open=lambda *_a,**_k:io.TextIOWrapper(z.open(members[0]),encoding='utf-8')
                for kind,a in m.entries(archive):
                    if kind=='annotations' and (a['image_id'],a['category_id']) in expected:
                        seen[(a['image_id'],a['category_id'])]+=1;annotations.append(dict(original_annotation=a,native_source_split=native_split,sha256=digest(a),native_label=defs[a['category_id']]['name'],training_activated=False,rights_supported=False))
                    elif kind=='images' and a['id'] in ids:actual_images[a['id']]=a
            print('Streamed existing LVIS',native_split,'metadata;no full extraction or downloads',flush=True)
        emit(cache,dict(annotations=annotations,images=actual_images,archive_hashes={name:h for name,h in archives.values()}))
        emit(proof,dict(sha256=sha(cache),streaming_only=True,all_other_photograph_metadata_discarded_in_memory=True))
    if dict(seen)!=expected or set(actual_images)!=set(ids):raise ValueError('Original LVIS instances/index join disagrees')
    per=[]
    for split,i,w,h,url,neg,nex in images:
        native=actual_images[i]
        if native['width']!=w or native['height']!=h or set(native.get('neg_category_ids',[]))&INDEX_RELEVANT!=set(json.loads(neg)) or set(native.get('not_exhaustive_category_ids',[]))&INDEX_RELEVANT!=set(json.loads(nex)):
            raise ValueError('LVIS federated category sets/dimensions disagree:'+str(i))
        anns=[a for a in annotations if a['original_annotation']['image_id']==i]
        per.append(dict(image_id=ids[i]['image_id'],source_photo_id=ids[i]['source_photo_id'],native_LVIS_image_id=i,source_partition=split,
             width=w,height=h,source_coco_url=url,neg_category_ids=native.get('neg_category_ids',[]),not_exhaustive_category_ids=native.get('not_exhaustive_category_ids',[]),
             index_flags_are_selected_category_projection=True,index_relevant_category_ids=sorted(INDEX_RELEVANT),
             native_instance_counts=dict(collections.Counter(a['native_label'] for a in anns)),
             native_annotation_IDs=[a['original_annotation']['id'] for a in anns],image_sha256_not_available=True,
             photographic_rights=STATUS_B,source_photograph_not_a_new_independent_image=True,
             project_Chair_absence_not_inferred=True,new_training_labels=0))
    emit(OUT/'LVIS-existing-annotation-audit.json',dict(records=per,annotations=annotations,archives=[dict(path=str((base/name).relative_to(ROOT)),sha256=h) for name,h in archives.values()],
         readonly_index=str(db.relative_to(ROOT)),index_sha256=sha(db),native_categories={str(i):defs[i] for i in sorted({a['original_annotation']['category_id'] for a in annotations})},
         all_original_federated_flags_preserved=True,missing_annotations_are_not_negative=True,rights_upgraded=0,photos_acquired=0))
    core={name:dict(native_category_id=cid,photos=len({a['original_annotation']['image_id'] for a in annotations if a['original_annotation']['category_id']==cid}),
                   instances=sum(a['original_annotation']['category_id']==cid for a in annotations),mask_annotations=sum(a['original_annotation']['category_id']==cid and bool(a['original_annotation'].get('segmentation')) for a in annotations)) for cid,name in CORE.items()}
    extra={defs[cid]['name']:dict(native_category_id=cid,instances=n) for cid,n in collections.Counter(a['original_annotation']['category_id'] for a in annotations if a['original_annotation']['category_id'] not in CORE).items()}
    emit(REPORT/'LVIS-existing-annotation-summary.json',dict(matched_same_COCO_photographs=64,native_split_photographs=dict(collections.Counter(r[0] for r in images)),
         native_split_is_not_project_partition=True,project_partition_assignments=0,core_categories=core,core_instances=sum(a['instances'] for a in core.values()),
         native_subcategory_instances=extra,total_native_instances=len(annotations),native_source_polygons_not_photo_rights=True,
         photo_rights_supported=0,rights_status_unchanged=True,COCO_LVIS_instances_not_additive_independent_objects=True,
         semantic_limits=['Cabinet native instances may be fronts/parts or built-in storage;whole-cabinet equivalence unverified','Table,coffee_table and Desk remain distinct native labels','Lamp,table_lamp and lampshade remain distinct;lampshades are not standalone project Lamps',
                          'Chair,armchair and rocking_chair are preserved separately;seating-stool project completeness still unverified','sofa_bed is not silently duplicated into Sofa and Bed targets'],
         no_current_acquisition_or_training_eligibility=True))
    print('Read-only LVISjoin verified64photos,',len(annotations),'native instances;',sum(a['instances'] for a in core.values()),'core7instances;0rights upgrades',flush=True)
if __name__=='__main__':run()
