"""Actual original-model CPU scope and immutable preparation/code binding."""
from commonE import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only scope verification')
    import torch
    from transformers import Owlv2ForObjectDetection
    from scopeE import StudentScope,TRAIN_KEYS
    sys.path.insert(0,str(ROOT/'tools/phase10d-v1'))
    from commonD import model_state_matches_base
    from telemetry19 import tensor_hash
    torch.set_num_threads(2)
    model=Owlv2ForObjectDetection.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True)
    count=model_state_matches_base(model);scope=StudentScope(model);opt=scope.optimizer();scope.check(opt,hash_frozen=True)
    emit(REPORT/'parameter-scope.json',dict(persisted_tensors_verified=count,trainable_parameters=1577988,trainable={n:dict(shape=list(p.shape),scalars=p.numel()) for n,p in model.named_parameters() if n in TRAIN_KEYS},frozen={n:dict(shape=list(p.shape),sha256=tensor_hash(p)) for n,p in model.named_parameters() if n not in TRAIN_KEYS},buffers={n:tensor_hash(v) for n,v in model.named_buffers()},box_bias_sha256=tensor_hash(model.box_bias),immutable_state=scope.immutable_state,teacher_optimizer_parameters=0,real_model_forwards=0))
    del scope,opt,model
    models={}
    c20=read(ROOT/'reports/phase10c20-v1/results.json')
    for name,key in [('original','original'),('C8','C8_B_E5'),('C20','C20_E5')]:models[name]=dict(headline=c20['comparison'][key]['headline'],source='reports/phase10c20-v1/results.json')
    c11=read(ROOT/'reports/phase10c11-v1/results.json');c13=read(ROOT/'reports/phase10c13-v1/results.json');d=read(ROOT/'reports/phase10d-v1/results.json')
    models['C11']=dict(headline=c11['comparison']['frozen_calibration']['headline'],mean_shared_logit_drift=c11['common_logit_drift']['frozen_calibration']['common_mean'],source='reports/phase10c11-v1/results.json')
    models['C13']=dict(headline=c13['comparison']['projection_regularized']['headline'],mean_shared_logit_drift=c13['logit_drift_vs_original']['projection_regularized']['common_mean'],source='reports/phase10c13-v1/results.json')
    models['10D']=dict(headline=d['comparison']['Phase10D']['headline'],mean_shared_logit_drift=d['logit_drift']['Phase10D']['mean_all_categories'],logits_bit_identical=d['logit_drift']['Phase10D']['raw_logits_bit_identical'],source='reports/phase10d-v1/results.json')
    emit(PREP/'historical-evidence.json',dict(models=models,scope='Previously exposed19photo partial native three-class diagnostics;not used to choose coefficient',D_localization_regression_percentage_points=100*(models['10D']['headline']['ap75']-models['original']['headline']['ap75'])))
    emit(REPORT/'independent-evaluation-plan.json',dict(actual_eligible_complete_independent_photos=0,complete_instances_per_category={c:0 for c in CLASSES},historical_local_readiness_photos=156,model_exposed_Phase4_rooms=3,these_existing_resources_are_not_independent_acceptance=True,existing_resources=['Existing nativebox/maskparsersandlicensingledgers','C21historicalsource/creator/hashregistryandstrictseven-classreviewvalidator','Reserved/frozenresources remainlocked;no automaticreuse'],proposed_real_acceptance_rooms=140,proposed_distinct_source_capture_groups=140,minimum_complete_instances_per_category=50,prioritise=['cabinet','lamp','bed','desk'],collection_plan=['Prefer newly authorized owner-captured room photos with specific permission/rights/attribution records','SelectsourceIDs andcapture/room/creatorgroupsbeforeviewingdetectoroutputs','Checkagainstallhistoricaltrain/development/frozenIDs,sourcegroups,hashesandnearduplicates;quarantineunknownidentity','Independentlyannotateallsevenwhole-objectcategories,occlusion/truncation,explicitignoreandChair+seatingstooltaxonomy','Do not inferabsencesfromnativeCOCO/LVIS/OI incompleteness;requireexplicitall-sevencompletenessreview','Blindannotationtomodelpredictions;second-reviewallcompletenessanddoubleboxreviewatleast20percent','Freezeimage/annotation/protocolhashesbeforeoneoriginalversuspreselectedmodelcomparison','Useaseparatesource-disjointcalibrationsetifanythresholdchoiceisneeded'],metrics=['Fixedthreshold0.15P/R,AP50/AP75,PRcurves,duplicates/fragmentation','Per-class,size/occlusionresultsand2000source-groupbootstrapreplicates','Joint80percentprecision/90percentrecallcriterion;notestselection'],estimate_only=dict(photo_bytes=[140*200000,140*3000000],image_storage_cap_proposed=2**30,manual_full_annotation_hours=[23,47]),sources_or_downloads_selected=False,installations_authorized=False,source_specific_approval_required=True))
    # Freeze code and existing teacher/diagnostic inputs after every implementation file exists.
    old,_,_,_,_=source_context();files=dict(old['files'])
    index=read(PREP/'teacher-index.json')
    for r in index['records']:
        files[r['cache']]=r['cache_sha256'];files[str(Path(r['cache']).with_suffix('.json'))]=r['receipt_sha256']
    for p in [*HERE.glob('*.py'),*HERE.glob('*.md'),*PREP.glob('*.json'),*REPORT.glob('*.json'),ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt']:
        files[str(p.relative_to(ROOT))]=sha(p)
    for phase in ('phase10c11-v1','phase10c13-v1','phase10c20-v1','phase10d-v1'):
        p=ROOT/'reports'/phase/'results.json';files[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'data/open-images/experiments/phase10c10-calibration-frozen-v1/probes').glob('epoch-*/*.npz'):files[str(p.relative_to(ROOT))]=sha(p)
    emit(BINDING,dict(files=files,base_sha256=BASE_SHA,mode='CPUpreflightonly',training_authorized=False,GPU_authorized=False,parent=PARENT,branch=BRANCH))
    if torch.cuda.is_initialized():raise RuntimeError('CPU architecture verification initialized CUDA')
    print('E1 actual CPU eight-tensor scope PASS;immutable bindings created;GPU workers disabled',flush=True)

if __name__=='__main__':run()
