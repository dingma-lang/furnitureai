"""Exactly one disabled C12 proposal; no implicit activation from past approval."""
import copy
from common12 import *


def required_code_pins():
    pins=dict(read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json')['code_hashes'])
    pins.update({str(p.relative_to(ROOT)):sha(p) for p in HERE12.glob('*.py')})
    return pins


def specification():
    cfg=copy.deepcopy(read(ROOT/'reports/phase10c11-v1/authorized-config-v2.json'));cfg.pop('code_hashes')
    cfg.update(schema='phase10c12-projection-preservation-v1',training_authorized=False,destination=str(FUTURE.relative_to(ROOT)),intended_change='Add optional L2-SP on the original classification projection only; freeze calibration and retain C11 data/loss/optimizer/box head',
        comparison_baseline=dict(phase='10C.11',directory=str(C11_RUN.relative_to(ROOT)),checkpoint=read(C11_RUN/'complete.json')['final_checkpoint']),
        required_approval='Separate explicit user GPU authorization and new versioned authorized config/record bound to its file and semantic SHA256; this preparation is disabled',
        resume='Fresh launch only; tested checkpoint restore helper binds objective/reference/config/RNG/scaler/cursor. No automatic replay after uncertain updates; a separately approved recovery entry point must honor remaining cumulative time/update budgets.')
    cfg['source_configuration'].update(enabled=False,approved=False,approval_reference='Disabled pending separate C12 GPU approval')
    original=original_heads();strength=read(REPORT12/'lambda-proposal.json')
    cfg['projection_preservation']=dict(enabled=False,lambda_value=strength['lambda_value'],selected_keys=list(PROJECTION_KEYS),reference_hashes={n:tensor_sha(original[n]) for n in PROJECTION_KEYS},form='half_squared_L2_sum',normalization='Joint half-squared sum,393728 scalars; no per-image positive-count scaling; averaged once per optimizer group',strength_evidence='reports/phase10c12-v1/lambda-proposal.json',strength_evidence_sha256=sha(REPORT12/'lambda-proposal.json'))
    cfg['activation_cache']['not_available_now']=True
    return cfg


def check_spec(cfg):
    expected=specification()
    if set(cfg)-{'code_hashes'}!=set(expected):raise PermissionError('Unexpected/missing C12 configuration fields')
    a=copy.deepcopy(cfg);a.pop('code_hashes',None);b=copy.deepcopy(expected)
    for obj in (a,b):
        obj['training_authorized']=False;obj['projection_preservation']['enabled']=False
        obj['source_configuration'].update(enabled=False,approved=False,approval_reference='activation metadata')
    if a!=b:raise PermissionError('Changed prepared C12 experiment, strength, reference or evaluation semantics')
    if 'code_hashes' in cfg and set(cfg['code_hashes'])!=set(required_code_pins()):raise PermissionError('Complete C12/historical implementation fingerprint set required')
    for name,value in cfg.get('code_hashes',{}).items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT/'tools') or sha(p)!=value:raise ValueError('C12 code binding mismatch: '+name)
    return True


def require_training_approval(cfg,approval=None):
    if cfg.get('training_authorized') is not True:raise PermissionError('C12 GPU execution disabled; separate explicit user authorization required')
    if not approval or approval.get('user_training_authorized') is not True or approval.get('config_semantic_sha256')!=digest(cfg) or not str(approval.get('user_reference','')).strip():raise PermissionError('Explicit approval bound to this exact C12 configuration required')
    if any(approval.get(k)!=v for k,v in dict(max_optimizer_updates=75,max_epochs=5,max_GPU_seconds=600,max_VRAM_bytes=int(6.5*2**30)).items()):raise PermissionError('Approval must bind all single-arm resource ceilings')
    check_spec(cfg)
    if not cfg.get('code_hashes') or cfg['projection_preservation']['enabled'] is not True or cfg['source_configuration']['enabled'] is not True or cfg['source_configuration']['approved'] is not True:raise PermissionError('Explicit C12 loss/source activation and pinned code required')
    verify_preparation();return True
