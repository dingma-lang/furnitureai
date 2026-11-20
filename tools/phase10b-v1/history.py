"""Read-only lineage and storage audit. Missing provenance stays explicitly unresolved."""
import collections
import csv
from p10 import *

def run():
    inventory=read(OLD/'backup-inventory.json')['records'];critical=read(ROOT/'data/open-images/experiments/phase9-multiview-v1/model-inventory.json')['weights']
    keep={v['path'] for v in critical.values()};classes=collections.Counter();artifact_rows=[]
    for r in inventory:
        p=r['path'];parts=Path(p).parts
        if p in keep or p.startswith('models/owlv2-original-') or p.startswith('models/siglip2-'):status='essential';why='Immutable reference or pinned feature encoder'
        elif p.startswith('models/'):status='unresolved';why='Other local model/simulator-labelled assets; detector lineage not established, exclude from cleanup proposal'
        elif any(x.startswith(('contact-sheet','diagnostic-gallery')) or x in ('galleries','curves','__pycache__') for x in parts):status='reproducible_temporary';why='Derived visual/cache artifact; reproducible only with preserved inputs/code/attribution'
        elif r['checkpoint_candidate'] and 'checkpoints' in parts:status='historical';why='Per-update recovery/history snapshot; preserve selected and last committed snapshots before any separate cleanup proposal'
        elif any(x.endswith(('.pending','.tmp')) for x in parts):status='unresolved';why='Possible interrupted transaction; never infer safe deletion'
        else:status='essential';why='Source data, licensing evidence, features, experiment lineage or unresolved recovery dependency'
        classes[status]+=r['allocated_bytes'];artifact_rows.append(dict(r,classification=status,reason=why))
    save(RAW/'artifact-classification.json',dict(records=artifact_rows,allocated_by_class=dict(classes),
        deletion_or_move_performed=False,independent_backup_still_unverified=True,
        conditional_recovery_upper_bound=classes['reproducible_temporary']+classes['historical'],safe_cleanup_authorized=False))
    def size(prefix):return sum(r['allocated_bytes'] for r in inventory if r['path'].startswith(prefix+'/'))
    datasets=[]
    def ds(name,path,source,licence,images,annotations,used,roles,exclusions,useful):
        datasets.append(dict(dataset=name,path=path,source=source,licence=licence,images=images,annotations=annotations,
            trained_models=used,validation_evaluation_role=roles,exclusions=exclusions,future_utility=useful,allocated_bytes=size(path)))
    a=read(ROOT/'data/abo/image-pilots/phase1b-v1/acquisition.json');screen=read(ROOT/'data/abo/image-pilots/phase1b-v1/final-screening.json')
    ds('ABO Phase1 metadata','data/abo/metadata/phase1-v1','Official Amazon Berkeley Objects','Preserved ABO CC BY4.0 documentation; source attribution retained',0,'Listings/image metadata, not detection boxes',[], 'Candidate selection only','Accessories, category conflicts, shared identities screened','Metadata for classification/retrieval expansion; no room-detection negatives')
    ds('ABO original image pilot','data/abo/image-pilots/phase1b-v1','Official ABO small image objects','ABO saved CC BY4.0 records',len(a['objects']),dict(screen['image_status_counts']),['Phase2A','auto-training-v1','weak-training-v1'],
       '402-feature cache:275 weak train,59 validation,68 test; weak training later288 views/130 products; official reviewed evaluator remains gated',
       screen['sample_rejected_tag_counts'],'Frozen-feature catalogue classification/retrieval, not detector training')
    ds('ABO Phase3 expansion','data/abo/image-pilots/phase3-expansion-v1','Official ABO small image objects','ABO saved CC BY4.0 records',1484,'Weak product categories; no scene boxes/masks',['phase3-gpu-v1'],
       '350 newly eligible vectors combined with182 baseline training images;106-image45-product comparison panel excluded from updates',
       {'rejected':605,'quarantined':304,'unresolved':225,'provisionally_eligible':350},'Useful weak catalogue supervision, with identity/filtering bias limitations')
    scopes=[('Open Images Phase5','data/open-images/image-pilots/phase5-v1',118,'63eligible:27train/18dev/18reserved','Phase5B original baseline; Phase5C two head fits',
             '18development exposed;18reserved untouched','32rights slots not acquired;50 unresolved and5scene quarantines'),
            ('Open Images Phase6A','data/open-images/image-pilots/phase6a-v1',41,'52eligible unmasked boxes','No new fitting; included in prospective expanded pool','9prospective train/5freshdev now exposed','27quarantined'),
            ('COCO Phase6D','data/coco/image-pilots/phase6d-v1',72,'362native boxes/masks;280eligible boxes/masks','No detector fit yet','45prospective train/10freshdev now exposed','16quarantined/1rejected'),
            ('LVIS Phase7','data/lvis/image-pilots/phase7-v1',66,'81eligible core boxes/masks:63train/18dev','No detector fit yet','40prospective train/8freshdev now exposed','18quarantined;33existing-photo overlays stay inactive'),
            ('LVIS Phase8C','data/lvis/image-pilots/phase8b-v1',27,'99native;39provisional and60inactive','No detector fit yet','6prospective train/13freshdev now exposed','8quarantined; category/unit and visual-registration restrictions')]
    for name,path,images,anns,models,roles,excluded in scopes:
        ds(name,path,'Official Open Images / COCO source objects with native annotations','Photograph-specific saved CC BY2.0 evidence; annotation CC BY4.0, retained separately',images,anns,[models],roles,excluded,
           'Real-room native positives; partial scope and unit policy required; no absent-label negatives')
    for name,path in [('Open Images metadata','data/open-images/metadata/phase5-v1'),('COCO2017 metadata','data/coco/metadata/phase6b-v1'),('LVIS metadata','data/lvis/metadata/phase7-v1')]:
        ds(name,path,'Official preserved annotation archives/CSVs','Annotation CC BY4.0; does not grant photo rights',0,'Native source categories/boxes/masks; exact class counts preserved in local audit reports',[],
           'Metadata and candidate auditing only','Rights-unverified photos not eligible','Reusable native annotation inventory; no photograph downloads authorized')
    ds('Combined Phase10B eligibility',str(RAW.relative_to(ROOT)),'Derived immutable Phase10A pool','Saved rights and annotation provenance retained',78,'243boxes/187masks;130targets remain inactive',[],
       '59optimization/19training-derived inner validation; not trained','42photos without supported positives; four broad categories unsupported','Ready for bounded three-category diagnostics after separate GPU approval')
    save(RAW/'dataset-history.json',dict(records=datasets,cohort_counts_not_additive=True,COCO_LVIS_same_photo_overlay_not_new_photo=True))
    models=[]
    def model(name,architecture,base,trained,settings,improved,regressed,disposition,source):
        models.append(dict(model=name,architecture=architecture,base_checkpoint=base,training_dataset=trained,settings=settings,
            improvements=improved,regressions_or_limits=regressed,disposition=disposition,
            weights_needed='Retain selected/final baseline and last committed recovery state; no cleanup without backup/approval',source=source))
    model('Original OWLv2','google/owlv2-base-patch16-ensemble',critical['original'],'No local detector updates','Pinned original processor;960pixel baseline',
      'Immutable production/reference detector','Exposed18:P21.71 R68.29 AP50 72.16 AP75 51.91; partial native labels','essential immutable default','data/open-images/experiments/phase5b-v1/report.md')
    model('SigLIP2 encoder','google/siglip2-base-patch16-224',critical['siglip'],'Frozen in all local classifier experiments','224pixel preprocessing;768-dimensional normalized embeddings',
      'Reusable catalogue/crop features','Not independently validated as a real-room verifier','essential pinned encoder','data/abo/experiments/phase2a-v1/experiment-lock.json')
    lock=read(ROOT/'data/abo/experiments/phase2a-v1/experiment-lock.json')
    model('Phase2A classifier','Frozen SigLIP2 + linear head',critical['siglip'],'275images/130products',lock,
      'Completed50epochs; training agreement improved','97.82%image/99.23%product are weak training metrics, not verified accuracy','historical comparator','data/abo/experiments/phase2a-v1/report.md')
    for family in ('auto-training-v1','weak-training-v1'):
        folder=ROOT/'data/abo/experiments'/family
        meta=read(folder/('run-meta.json' if family=='auto-training-v1' else 'queue-config.json'))
        model(family,'Frozen SigLIP2 + linear/MLP/cosine classifiers',critical['siglip'],
          '275train views' if family=='auto-training-v1' else '288images/130products',meta,
          'Best internal macroF1 about83.43%' if family=='auto-training-v1' else 'Cosine mean product macroF1 91.44%,foldSD12.49pp',
          'Adaptive training-derived weak-label selection; no official holdout accuracy','historical comparator, superseded for selected catalogue head',str(folder.relative_to(ROOT)))
    phase3=read(ROOT/'data/abo/experiments/phase3-gpu-v1/results-report.json')
    model('Phase3 expanded classifier','Frozen SigLIP2 + cosine head',critical['classifier'],'532images/281listing IDs/280known groups',phase3['winner'],
      'Matched106-image45-product panel macroF1 79.13%→88.96%',phase3['observed_tradeoff']+' '+phase3['scope'],
      'selected catalogue classifier, retain baseline and final head','data/abo/experiments/phase3-gpu-v1/results-report.json')
    for fit in ('heads-lr1e-5','heads-lr3e-5'):
        done=read(ROOT/'data/open-images/experiments/phase5c-v1'/fit/'complete.json')
        model('Phase5C '+fit,'OWLv212-tensor class/box head delta',critical['original'],'27OpenImages train photographs',dict(epochs=10,updates=70,batch=1,accumulation=4,learning_rate=1e-5 if '1e-5' in fit else 3e-5),
          'See native18-photo per-category comparison; selected3e-5 AP50 79.56 P30.21 R70.73 AP75 53.60',
          'Table FP increase; Bed/Table localization regressions; Cabinetry annotation influence',
          'essential adapted comparator' if '3e-5' in fit else 'historical adaptation comparator',str((ROOT/'data/open-images/experiments/phase5c-v1'/fit).relative_to(ROOT)))
    model('Phase4/5/9 inference methods','Original/adapted OWLv2 plus crops, fusion and optional SigLIP','Immutable references','No model updates',
      'Phase4 roomcomparison; Phase5cropgate; Phase9fixed15arms8views/model/photo',
      'Some source-relative FP reduction','No reliable precision gain at comparable recall;36-photo comparison now exposed',
      'historical inference reports, no new weights','reports/phase9-multiview-v1/report.md')
    model('Phase10A/10B prepared heads','OWLv2 head-only','Original OWLv2','Prepared59optimization/19innerdev',
      'LR1e-5/3e-5,batch1,acc4,≤5epochs,≤150updates/fit,≤300total,≤1hour,6.5GiB',
      'No results: not trained','Four categories inactive; very limited FP-observable validation','pending separate authorization','reports/phase10a-v1/report.md')
    unresolved=[str(p.relative_to(ROOT)) for p in (ROOT/'reports').iterdir() if p.is_dir() and not p.name.startswith(('phase','abo'))]
    save(RAW/'model-history.json',dict(records=models,checkpoint_artifact_index='artifact-classification.json',
        extra_project_local_report_directories=unresolved,extra_report_usage='Unresolved/other application diagnostics; not assumed detector training or safe cleanup'))
    # Preserve every run's actual JSON metadata rather than flattening incomparable metrics.
    evidence=[]
    for parent in (ROOT/'data/abo/experiments',ROOT/'data/open-images/experiments'):
        for p in parent.rglob('*.json'):
            if p.name in ('complete.json','completion.json','config.json','metadata.json','result.json','results.json'):
                try:d=read(p)
                except (ValueError,UnicodeError):continue
                evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),metadata=d))
    save(RAW/'all-experiment-metadata.json',dict(records=evidence,metrics_not_recomputed=True))
    save(REPORT/'history-summary.json',dict(dataset_cohorts=datasets,models=models,
       artifact_allocated_by_class=dict(classes),conditional_recovery_upper_bound=classes['reproducible_temporary']+classes['historical'],
       extra_reports_unresolved=unresolved,all_experiment_metadata_records=len(evidence),nothing_deleted_or_moved=True))
    print('HISTORY',len(datasets),'dataset cohorts',len(models),'model families',len(evidence),'run metadata records; storage',dict(classes),flush=True)

if __name__=='__main__':run()
