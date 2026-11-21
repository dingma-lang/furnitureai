"""Deterministic disabled configuration and explicit pre-CUDA execution gate."""
import copy
from common10 import *

def specification():
    binding=read(C8/'binding.json');old=copy.deepcopy(binding['configs']['B-corrected']);source=copy.deepcopy(old['source_configuration'])
    source.update(enabled=False,approved=False,approval_reference='Disabled: separate C10 user training approval required')
    return dict(schema='phase10c10-one-calibration-freeze-v1',training_authorized=False,Phase10D_authorized=False,
        experiment_count=1,initialization=dict(path=str(BASE.relative_to(ROOT)),sha256=sha(BASE/'model.safetensors'),type='original_full_pretrained_OWLv2'),
        reference=dict(arm='C8 B corrected epoch5',checkpoint=read(C8/'B-corrected/complete.json')['final_checkpoint'],binding_sha256=sha(C8/'binding.json')),
        optimization_settings={k:copy.deepcopy(old[k]) for k in ('learning_rate','optimizer','betas','eps','weight_decay','seed','batch_size','accumulation','last_group_normalization','gradient_clip_norm','precision','loss_coefficients','scaler_init','scaler_growth_interval','max_epochs','max_updates','probe_epochs')},
        trainable_keys=sorted(TRAIN_KEYS),trainable_parameters=TRAIN_COUNT,frozen_calibration_keys=sorted(CALIBRATION_KEYS),frozen_calibration_scalars=1538,
        intended_change='Only original class_head.logit_shift and class_head.logit_scale weights+biases are removed from the C8 corrected arm optimizer',
        source_configuration=source,data=dict(manifest=str((ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json').relative_to(ROOT)),manifest_sha256=sha(ROOT/'data/open-images/preparation/phase10b-v1/eligibility-manifest.json'),training_rows_sha256=binding['training_rows_sha256'],development_rows_sha256=binding['development_rows_sha256'],train_photos=59,known_source_groups=56,positive_boxes=182,source_regions=41,development_photos=19,new_annotations=0),
        limits=dict(max_epochs=5,max_optimizer_updates=75,max_gpu_seconds=600,max_vram_bytes=int(6.5*2**30),max_model_forwards=467,initial_output_storage_guard_bytes=4*2**30),
        evaluation=copy.deepcopy(binding['evaluation_protocol']),primary_endpoint='Final epoch5 only, fixed before execution; milestones0/1/3/5 are diagnostic, no adaptive selection',
        destination=str(FUTURE.relative_to(ROOT)),resume='Fresh run only. Unique per-update snapshots preserve cursor/RNG/optimizer/scaler. No automatic replay after failure or uncertain update.',
        activation_cache=dict(enabled=True,scope='Same19 planned initial FP32 internal-development forwards only',contents=['frozen pre-head feature_map flattened3600x768','text_embeds7x512'],files_max=19,bytes_per_file_max=12000000,not_available_now=True,extra_encoder_forwards=0,purpose='Later CPU endpoint-validated projection/shift/scale head swaps for original and historical A/B; never used in optimization'),
        independent_evaluation='Separate unconsumed protocol; no reserved/frozen/official scoring',required_approval='New versioned authorized config plus explicit approval record bound to its SHA256; disabled config must remain unchanged')

def check_spec(config):
    expected=specification()
    # These are approval metadata, not optimizer/model/data changes.
    ignored={'training_authorized','code_hashes'}
    for key in expected:
        if key in ignored:continue
        if key=='source_configuration':
            for k in ('mode','evidence_sha256','manifest_sha256','legacy_loss'):
                if config[key][k]!=expected[key][k]:raise PermissionError('Changed source-mask semantics')
        elif config.get(key)!=expected[key]:raise PermissionError('Changed single-experiment specification: '+key)
    for path,value in config.get('code_hashes',{}).items():
        if sha(ROOT/path)!=value:raise ValueError('Prepared code changed: '+path)
    return True

def require_training_approval(config,approval=None):
    # Must run before importing CUDA workers, making output directories or loading weights.
    if config.get('training_authorized') is not True:raise PermissionError('C10 training is disabled; CPU preflight does not authorize GPU execution')
    if not approval or approval.get('user_training_authorized') is not True or approval.get('config_semantic_sha256')!=digest(config) or approval.get('max_optimizer_updates')!=75 or not str(approval.get('user_reference','')).strip():raise PermissionError('Explicit user approval bound to this exact one-arm configuration required')
    check_spec(config)
    if not config.get('code_hashes'):raise PermissionError('Pinned implementation required')
    if config['source_configuration'].get('enabled') is not True or config['source_configuration'].get('approved') is not True:raise PermissionError('Source-supervision activation requires the same future approval')
    return True
