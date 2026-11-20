"""Import known historical measurements without recomputing predictions or scoring."""
from p10 import *
from tracking import record

def run():
    comparison=read(ROOT/'data/open-images/experiments/phase5c-v1/comparison-v1/comparison.json')
    registry=read(ROOT/'data/open-images/experiments/phase9-multiview-v1/model-inventory.json')['weights']
    for name,metrics in comparison['primary_metrics'].items():
        if name=='original':checkpoint=registry['original']['sha256'];settings={'updates':0};resources={}
        else:
            complete=read(ROOT/'data/open-images/experiments/phase5c-v1'/name/'complete.json')
            checkpoint=complete['best']['sha256'];settings={'epochs':10,'updates':70,'lr':1e-5 if '1e-5' in name else 3e-5}
            resources={k:complete[k] for k in ('seconds','peak_allocated_bytes')}
        record(RAW/'experiment-history',dict(experiment_id='phase5c/'+name,dataset_version='phase5b-locked18development',
          dataset_sha256=sha(ROOT/'data/open-images/experiments/phase5b-v1/dataset-lock.json'),git_commit=None,
          checkpoint_sha256=checkpoint,settings=settings,metrics=metrics,resources=resources,decision='retain',
          next_experiment='Preserve comparator; training-derived expanded-data diagnostics after approval',evaluation_scope='exposed_development',
          source='data/open-images/experiments/phase5c-v1/comparison-v1/comparison.json',historical_git_commit_unavailable=True))
    for family in ('auto-training-v1','weak-training-v1','phase3-gpu-v1'):
        folder=ROOT/'data/abo/experiments'/family
        if family=='auto-training-v1':
            data=read(folder/'final-metadata.json');checkpoint=data['checkpoint_sha256'];metrics={'selection_metric':data['selection_metric']};settings=data['selected_config']
        elif family=='weak-training-v1':
            data=read(folder/'completion.json');checkpoint=read(folder/'checkpoint-registry.json')['current'][0]['sha256'];metrics=data['selection'];settings=data['winner']
        else:
            data=read(folder/'results-report.json');checkpoint=registry['classifier']['sha256'];metrics=data['comparison'];settings=data['winner']['config']
        record(RAW/'experiment-history',dict(experiment_id=family,dataset_version=family,dataset_sha256=None,git_commit=None,
          checkpoint_sha256=checkpoint,settings=settings,metrics=metrics,resources={},decision='retain',
          next_experiment='Catalogue-classifier comparator only; not detection training or independent room accuracy',
          evaluation_scope='provisional_weak_label_internal',missing_historical_fields_not_invented=True))
    data=read(ROOT/'data/abo/experiments/phase2a-v1/classifier-v1/diagnostics.json')
    record(RAW/'experiment-history',dict(experiment_id='phase2a-v1',dataset_version='phase1d-weak275',dataset_sha256=None,git_commit=None,
        checkpoint_sha256=data['checkpoint_sha256'],settings={'epochs':50,'lr':data['lr']},
        metrics={'image':data['image_training_metrics'],'product':data['product_training_metrics']},resources={'seconds':data['training_seconds']},decision='retain',
        next_experiment='Historical weak-training comparator; preserve independent review gates',evaluation_scope='weak_label_training_only'))
    # Metadata corpus preserves per-fold details for all104 saved completion/config records.
    save(RAW/'history-import-summary.json',dict(canonical_events=7,per_run_metadata_records=len(read(RAW/'all-experiment-metadata.json')['records']),
        predictions_recomputed=False,missing_dataset_or_git_hashes_preserved_as_null=True))
    print('Imported7 canonical historical events plus104 preserved run metadata records; no scoring',flush=True)

if __name__=='__main__':run()
