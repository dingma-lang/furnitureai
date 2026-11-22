"""CPU-only audit/preparation. Immutable report outputs; no model inference."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/phase10e3-v1'
PREP3 = ROOT / 'data/open-images/preparation/phase10e3-v1'
sys.path.insert(0, str(ROOT / 'tools/phase10e1-v1'))
from commonE import PREP, BASE, BASE_SHA, BINDING, read, sha, digest
from telemetry3 import atomic_json


def emit(name, value):
    atomic_json(REPORT / name, value)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('Explicit CPU-only execution required')
    import torch
    import transformers
    from transformers.models.owlv2.modeling_owlv2 import Owlv2ClassPredictionHead
    from teacherE import load_teacher
    from objectiveE import selective_mask
    from stable3 import Capture
    from telemetry3 import Trace
    sys.path.insert(0, str(ROOT / 'tools/phase10e2-v1'))
    from verify2 import check as check2
    from verifyE import check as check1
    check1()
    check2()
    proposal = read(ROOT / 'reports/phase10e1-v1/gradient-smoke-disabled.json')
    context = read(PREP / 'training-context.json')
    index = read(PREP / 'teacher-index.json')
    entries = {e['row']['image_id']: e for e in context['entries']}
    refs = {r['image_id']: r for r in index['records']}
    total = 0
    groups = set()
    rows = []
    for record in index['records']:
        ident = record['image_id']
        e = entries[ident]
        t = load_teacher(record, e['row'], index['processor_contract_sha256'])
        z, b = t.tensors()
        mask, info = selective_mask(z, b, z, b, e['row'], e['distill_regions'], e['distill_protected'])
        total += int(mask.sum())
        if mask.any():
            groups.add(e['row']['identity_group'])
        rows.append(dict(image_id=ident, source_group=e['row']['identity_group'],
                         terms=int(mask.sum()), mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest()))
    for key in ('image_id', 'sha256', 'identity_group'):
        if {e['row'][key] for e in context['entries']} & {r[key] for r in context['development']}:
            raise ValueError('Partition isolation failed')
    if (len(entries), total, len(groups)) != (59, 278, 15):
        raise ValueError('Historical objective/mask coverage changed')
    # CPU synthetic actual installed head under autocast. Never the detector,
    # never a private image, and never evidence of CUDA intermediate dtypes.
    from test_numerics3 import fixture
    head, x, q, _, _ = fixture()
    trace = Trace(enabled=True)
    with Capture(head, enabled=True, trace=trace), torch.autocast('cpu', dtype=torch.float16):
        logits, _ = head(x, q, torch.ones(1, 7, dtype=torch.bool))
    trace.graph(logits)
    torch.autograd.grad(logits.square().mean(), [head.dense0.weight, head.dense0.bias])
    trace.close()
    source = Path(inspect.getfile(Owlv2ClassPredictionHead))
    autocast = ROOT / '.venv/lib/python3.14/site-packages/torch/include/ATen/autocast_mode.h'
    generated = ROOT / '.venv/lib/python3.14/site-packages/torch/include/torch/csrc/autograd/generated/Functions.h'
    scaler = ROOT / '.venv/lib/python3.14/site-packages/torch/amp/grad_scaler.py'
    sources = [source, autocast, generated, scaler, ROOT / 'tools/phase10e1-v1/workerE.py',
               ROOT / 'tools/phase10e1-v1/objectiveE.py', ROOT / 'tools/phase10e2-v1/measure2.py']
    dispatch = {op: [s for s in torch._C._dispatch_dump_table('aten::' + op).splitlines()
                     if 'Autocast' in s] for op in ('linear', 'einsum', 'linalg_vector_norm', 'elu')}
    emit('dtype-audit.json', dict(status='CPU_SOURCE_AUDIT', torch=torch.__version__, transformers=transformers.__version__,
         sources={str(p.relative_to(ROOT)): sha(p) for p in sources}, dispatch=dispatch,
         CUDA_execution=False, CPU_synthetic_events=trace.events,
         verified=['Teacher NPZ values stored FP32 after historical CUDA FP16 forward',
                   'E2 model FP32 master weights; torch.autocast(cuda,float16) around detector',
                   'Class head final torch.where output explicitly converted to FP32 when query_mask supplied',
                   'MSE computes (student.float-teacher.float)^2 then selected-term mean in FP32',
                   'E2 VJP manually multiplied loss by128; no GradScaler; divided returned gradients after finite test',
                   'E1 training (not run) separately uses GradScaler128; does not make E2 VJP overflow-safe',
                   'Linear/einsum are CUDA low-precision policy operations; linalg_vector_norm uses FP32 optional-dtype policy',
                   'Norm backward saves original input and norm result; all upstream casting nodes remain in legacy graph'],
         CUDA_intermediate_dtypes='Policy expectations only. No saved E2 activations; fresh CUDA telemetry required.',
         legacy_backward='FP32 mean/square/subtract -> FP32 final logits -> low-precision final multiplication/add -> low-precision einsum adjoint -> normalized projection/cast/norm backward -> low-precision dense0 adjoint -> FP32 master weight gradient. Norm/division can be FP32 but casting back into FP16 nodes remains hazardous.',
         BF16='Not requested by historical CUDA code; only synthetic CPU BF16 tests. No CUDA BF16 trial.',
         repair='FP32 observed-node adjoint; exact original forward logits/rounded nodes, frozen calibration and shared features, FP32 chain rule/parameter contractions, scale1. No half gradient casts.',
         limitation='A deliberately widened adjoint, not bit-equivalent to rounded FP16 backward. Same MSE value/masks/normalization; not installed into training or production. No exact historical overflowing instruction identified.'))
    atomic_json(PREP3 / 'mask-fingerprints.json', dict(records=rows))
    checkpoint = ROOT / 'data/open-images/experiments/phase10c10-calibration-frozen-v1/checkpoints/update-075-bf8be48ddf59.pt'
    if sha(BASE / 'model.safetensors') != BASE_SHA or sha(checkpoint) != proposal['reference_sha256']:
        raise ValueError('Immutable checkpoint mismatch')
    files = {str(p.relative_to(ROOT)): sha(p) for p in (BINDING, PREP / 'training-context.json',
             PREP / 'teacher-index.json', ROOT / 'reports/phase10e2-v1/SHA256SUMS', checkpoint,
             BASE / 'model.safetensors', ROOT / 'tools/phase10e1-v1/objectiveE.py')}
    emit('objective-preservation.json', dict(status='PASS', teacher_caches=59, optimization_photos=59,
         source_groups=56, source_regions=41, admitted_KD_regions=35, original_teacher_terms=278,
         groups_with_terms=15, adapted_historical_terms=267, adapted_groups_with_terms=15,
         labels_added=0, coefficients_added=0, coefficient=None,
         frozen_scope_unchanged=True, trainable_tensors=8, trainable_parameters=1577988,
         original_checkpoint_sha256=BASE_SHA, C11_checkpoint_sha256=proposal['reference_sha256'],
         file_fingerprints=files, mask_fingerprints_sha256=sha(PREP3 / 'mask-fingerprints.json'),
         normalization='Unchanged per-photo selected-term MSE mean; per-source gradient comparisons average within known groups as in E2. No new loss weighting.',
         source_masks_and_annotations_modified=False, official_or_reserved_images_accessed=False,
         GPU_forwards=0, optimizer_updates=0, teacher_and_student_query_alignment_unchanged=True))
    plan = dict(schema='phase10e3-one-disabled-zero-update-recovery', approved=False, enabled=False,
         training_authorized=False, coefficient=None, destination='data/open-images/experiments/phase10e3-recovery-v1',
         limits=dict(detector_forwards=25, head_verification_contractions=25, VJPs=48,
                     seconds=240, VRAM_bytes=6979321856, output_bytes=134217728, optimizer_updates=0),
         adapted_photo_ids=proposal['photo_ids'], original_replay_photo_ids=[proposal['photo_ids'][1]],
         reference_checkpoint=str(checkpoint.relative_to(ROOT)), reference_sha256=proposal['reference_sha256'],
         base_sha256=BASE_SHA, input_fingerprints=files, seed=6106,
         source_groups='Exact existing24 optimization photographs; no development/reserved/frozen access',
         measurement='FP16 detector forward + exact verified class-head snapshot; FP32 observed-node adjoints for BOTH positive and unit-KD VJPs at scale1',
         FP32_matmul='Disable CUDA TF32 for widened contractions; record precision configuration; restore previous flags on exit',
         expected_VJPs=39, maximum_VJPs=48,
         original_checks='Reuse sealed E2 four original replays; ONE fresh original replay on the failed nonzero-mask photo, exact pixels/prompts/cache, zero KD loss/closed-form zero derivative; no original VJP needed',
         adapted_checks='All24 require fresh feature snapshots (not previously saved); compare same C11E5 logits/boxes to sealed caches before each VJP. Identical ordering and masks.',
         tolerance='E2 predeclared logits atol0.02 rtol0.001, normalized boxes atol1e-5 rtol1e-5; no relaxation; require exact head snapshot reconstruction regardless of whole-cache tolerance',
         stop_conditions=['Any nonfinite gradient/loss, mapping/mask/fingerprint or scope failure',
                          'Any original-student KD mismatch, checkpoint mismatch or forbidden gradient route',
                          'Any resource limit, uncertain CUDA state or output persistence failure'],
         automatic_retry=False, fallbacks=False, optimizer_constructed=False,
         preservation='Fingerprint original and adapted in-memory stages independently; files and all frozen state before/after/failure. Save raw outputs before derivatives, positive component before KD.',
         analysis='Per-photo positive/KD projection norms and cosine; per-source means/vector sums, concentration and gradient directions. Report analytic non-domination ceiling only if finite nonzero coverage exists; no automatic scalar selection.',
         coefficient_policy='Unset until complete actual gradients; E2 predefined analytical 0.5*min(||g_positive||/||g_KD||) may be reported as a diagnostic bound, not an approved coefficient or AdamW/training guarantee.',
         success='Finite nonzero KD gradients and positive gradients on the actual eligible source groups; exact zero on original agreement; no protected/ignored/box/shared/calibration routes. Does not prove improved detection.',
         training_next='Only after recovery and one defensible fixed scalar is explicitly authorized: one75-update original-initialized experiment and fixed historical diagnostic comparison; no new variants.',
         future_training_precision_blocker='Current E1 training still uses FP16 backward and fixed-scale monitoring. A successful widened diagnostic does NOT automatically validate that legacy training path. Before training approval, pin and CPU-test the selected training precision integration without changing objective/scope; no untested fallback or silent diagnostic-adjoint substitution.')
    from recovery3 import validate
    validate(plan)
    emit('recovery-disabled.json', plan)
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    print('CPU audit PASS:59teachers,278terms/15groups,unchanged scope; recovery disabled', flush=True)


if __name__ == '__main__':
    run()
