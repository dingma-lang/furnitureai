"""Compact CPU-only final report and typed resource/state evidence."""
import csv
from common4 import *


def storage():
    seen=set();unique=0
    for root in (RUN,REPORT):
        for p in [root,*root.rglob('*')]:
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);unique+=st.st_blocks*512
    return dict(unique_inode_allocated_bytes=unique,conservative_path_allocated_bytes=allocated())


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU reporting only')
    f=read(REPORT/'gradient-findings.json');plan=read(REPORT/'training-disabled.json')
    pres=read(REPORT/'preservation.json');post=read(REPORT/'postrun-tests.json');pre=read(REPORT/'CPU-tests.json')
    log=REPORT/'CPU-future.log'
    if 'Ran 6 tests' not in log.read_text() or '\nOK' not in log.read_text():raise ValueError('Future gate tests required')
    emit(REPORT/'training-CPU-tests.json',dict(status='PASS',passed=6,log_sha256=sha(log),GPU_calls=0,training_executed=False))
    original_before=read(RUN/'original-before.json');original_after=read(RUN/'original-after.json')
    adapted_before=read(RUN/'adapted-before.json');adapted_after=read(RUN/'adapted-after.json')
    emit(REPORT/'model-integrity.json',dict(original_before=original_before['sha256'],original_after=original_after['sha256'],
         adapted_before=adapted_before['sha256'],adapted_after=adapted_after['sha256'],
         original_unchanged=original_before==original_after,adapted_unchanged=adapted_before==adapted_after,
         fingerprinted_tensors=len(original_before['tensors']),original_checkpoint_sha256=pres['original_sha256'],adapted_checkpoint_sha256=pres['adapted_sha256'],
         stage_transition='Existing C11 heads loaded only between independently fingerprinted stages;no optimizer updates in either stage'))
    # Read actual forward dtype telemetry,not CUDA policy inference.
    event=read(sorted((RUN/'telemetry/head-original').glob('event-*.json'))[-1])
    emit(REPORT/'precision-verification.json',dict(forward_names=['head_input','raw_projection','saved_norm','normalized_projection','normalized_query','contracted_query','cosine','scale','returned_logits'],
         forward_tensors=event['tensors'],backward_dtypes=f['measurement_dtypes'],loss_scale=1.,
         measurement_events=f['typed_measurement_telemetry_events'],earliest_nonfinite=None,
         numerical_values_checked=True,nonzero_parameter_gradients_checked=True,
         original_terminal_zero_was_analytic_not_executed_VJP=True,
         box_native_probe_photos=1,full_combined_GPU_training_smoke_run=False,
         training_precision_pinned_to=plan['precision']))
    emit(REPORT/'resource-accounting.json',dict(launch=read(RUN/'launch-result.json'),
         worker={k:read(RUN/'complete.json')[k] for k in ('forwards','completed_forwards','actual_detector_entries','contractions','completed_contractions','VJPs','completed_VJPs','seconds','peak_allocated_bytes','peak_reserved_bytes')},
         storage_before_final_report_and_seal=storage(),limits=LIMITS,optimizer_steps=0,training=False,downloads=0,automatic_retries=0))
    details=read(REPORT/'per-photo.json')['records']
    rows='\n'.join(f"| {r['image_id'].split(':')[-1]} | {r['terms']} | {r['positive_L2']:.5g} | {r['unit_KD_L2']:.6g} | {r['unit_ratio']:.6g} | {r['cosine']:.4f} | {r['weighted_ratio']:.5g} |" for r in details)
    text=f"""# Phase 10E.4 — stable distillation gradient recovery

One authorized zero-update GPU recovery completed. **No training,optimizer,
weight update,download,development scoring or reserved-set scoring occurred.**
Every active source group yielded finite nonzero classification-projection
gradients. One proposed fixed coefficient is **{f['coefficient']:.16g}**.
The75-update candidate remains disabled and requires separate approval.

| Resource | Actual | Approved maximum |
|---|---:|---:|
| Detector forwards,all completed | {f['forwards']} | 25 |
| Head-verification contractions,all completed | {f['contractions']} | 25 |
| VJPs,all completed | {f['VJPs']} | 48 |
| Supervised wall time | {f['wall_seconds']:.3f}s | 240s |
| Worker elapsed,including CPU overhead | {f['seconds']:.3f}s | 240s |
| Peak allocated GPU memory | {f['peak_allocated_bytes']/2**30:.4f}GiB | 6.5GiB |
| Peak reserved GPU memory | {f['peak_reserved_bytes']/2**30:.4f}GiB | Reported separately |
| Optimizer steps / learned weight updates | 0 / 0 | 0 / 0 |

All59 teacher/image identities,source groups,processor tensor hashes,original
checkpoint and adapted checkpoint were verified before execution. {pre['passed']}
CPU tests passed before GPU use. We used one original correspondence photograph
and all15 saved-C11 photographs with nonzero selective coverage,representing15
source groups and267 terms. The other9 permitted photographs were skipped because
their immutable historical outputs had zero coverage. No IDs were substituted,
partitions changed or masks relaxed. The original59-photo teacher masks still
have278 terms across15 groups;the adapted count267 reflects existing query-box
and matching differences,not a changed annotation policy.

The one fresh original forward and all15 adapted forwards were **bit-identical**
to their historical logits/boxes. The measured forward is still FP16 autocast;
the numerical repair changes the adjoint,not these forward values. Internal head
reconstruction was exact. Original/student MSE and its closed-form terminal
derivative were exactly zero at the original initialization. No original VJP was
executed or falsely counted;the four E2 original replays were reused as sealed
evidence. Teacher targets were never rewritten.

Live CUDA dtype telemetry confirms FP32 shared head input,FP16 raw projection,
FP32 saved norm/normalized image/query features,FP16 contraction inputs/cosine
and calibration scale,and FP32 returned logits. Positive and KD adjoint stages
and parameter gradients were FP32,unscaled(scale1),with TF32 disabled for these
contractions. There was no GradScaler or manual128 multiplier in this run.
All {f['typed_measurement_telemetry_events']} saved measurement telemetry events
were finite. The original E2 exact failing operation remains unknown: changing
both adjoint precision and scale does not isolate which original operation
overflowed or prove that lowering128 alone would have sufficed.

The31 VJPs comprise15 positive,15 selective-KD,and one native box-loss VJP. Both
classification components route to the two actual projection parameters;their
box-output/box-parameter gradients remain absent. The native box probe reached
all six box parameters with finite gradients and did not reach the projection.
Known protected/positive overlap violations:0. All uncertain/other-class output
terms remain ignored by KD. Scalar MSE and terminal gradients are reported
separately from these actual parameter-gradient measurements.

| Actual parameter-gradient measure | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Positive projection L2 | {f['positive_projection_L2']['minimum']:.6g} | {f['positive_projection_L2']['median']:.6g} | {f['positive_projection_L2']['maximum']:.6g} |
| Unit KD projection L2 | {f['unit_KD_projection_L2']['minimum']:.6g} | {f['unit_KD_projection_L2']['median']:.6g} | {f['unit_KD_projection_L2']['maximum']:.6g} |
| Unit KD / positive ratio | {f['ratios']['minimum']:.6g} | {f['ratios']['median']:.6g} | {f['ratios']['maximum']:.6g} |
| Positive/KD cosine | {f['cosines']['minimum']:.6g} | {f['cosines']['median']:.6g} | {f['cosines']['maximum']:.6g} |

| COCO image ID | Terms | Positive L2 | Unit KD L2 | Unit ratio | Cosine | Weighted ratio |
|---|---:|---:|---:|---:|---:|---:|
{rows}

All15 KD directions oppose positive classification gradients,so an unweighted
penalty is not justified. All15 KD gradients have positive cosine with the past
C11 projection displacement(range{f['KD_displacement_cosines']['minimum']:.4f}–{f['KD_displacement_cosines']['maximum']:.4f});
descent would push against that displacement. This is consistent with restoring
pretrained behavior but **does not demonstrate less seven-category confidence
inflation**. Other six classes have no direct teacher constraint.

Source groups each contribute one measured photograph. The largest contributes
{100*f['top1_share_individual_KD_norms']:.2f}% of the sum of individual KD norms;
the top three contribute{100*f['top3_share_individual_KD_norms']:.2f}%.
These are norm shares,not fractions of a vector sum. The cosine between actual
source-summed positive/KD gradients is{f['source_sum_cosine']:.5f}.

The single coefficient uses only these training measurements:

    lambda = 0.5 * min(min_photo(||g_positive|| / ||g_unit_KD||),
                       ||sum_source_mean(g_positive)|| / ||sum_source_mean(g_unit_KD)||)

No coefficient sweep,development labels or development scores were used. With
lambda={f['coefficient']:.16g},weighted per-photo KD/positive ratios range
{f['weighted_ratios']['minimum']:.6g}–0.5(median{f['weighted_ratios']['median']:.6g}).
The minimum measured positive-descent fraction in ordinary Euclidean gradient
geometry is{f['positive_descent_fractions']['minimum']:.6f};the analytic worst-case
bound is0.5. This is not an AdamW update bound,not valid for every future training
state,and not a guarantee of precision or recall. The weighted source-summed KD
norm is only{100*f['weighted_source_sum_ratio']:.4f}% of the positive sum:the
weakest-positive examples set a conservative,possibly globally weak penalty.
At original agreement,the penalty starts with exactly zero gradient and grows
only as eligible student/teacher logits diverge.

Only{f['selected_Chair_scores_ge015']} of267 selected adapted Chair scores exceed
0.15. Most terms anchor already-low scores. Known positives are protected,but
shared parameter effects may still suppress useful Chair scores outside these
regions. These15 source groups and source-supported evidence are insufficient
to establish general real-room performance or universal annotation completeness.

Both in-memory stages retained identical421-tensor fingerprints before/after:
original `{original_before['sha256']}`,adapted `{adapted_before['sha256']}`.
The authorized transition loaded existing C11 heads between stages;it was not
an optimizer update. Original OWLv2 SHA-256 remains `{pres['original_sha256']}`;
C11 SHA-256 remains `{pres['adapted_sha256']}`. All9,973 inventoried historical
files and C8/C11/C13/C17–C21/D/E1/E2/E3 seals passed. `main` is unchanged.

Post-run checks passed{post['passed']} tests plus6 new future-gate checks. One
E3 preparation-only assertion that the recovery directory must not exist is
explicitly inapplicable after authorized creation;it passed pre-GPU,and its
source was preserved. Every numerical,mask,routing and persistence check remains
active. Diagnostics,positive components,raw predictions and state hashes were
saved atomically;see `resource-accounting.json` and the final artifact seal.

The proposed training implementation is pinned in `precision4.py:combined` and
`future4.py`. It keeps the same FP16 forward and selective FP32 objective,uses
the recovered FP32 observed-node adjoint for **all** classification terms,and
the native box backward at scale1. It uses no GradScaler or legacy128 monitor.
Synthetic combined-loss CPU tests verify all8 parameter routes and unchanged
loss/gradient behavior at coefficient0 on the same numerical path. Actual CUDA
positive/KD components and one native box VJP passed. A full combined GPU VJP
has intentionally **not** been run;it is a mandatory zero-update check within
the future authorized queue,before optimizer construction. Finite behavior over
all future parameter states cannot be promised.

This is explicitly a different numerical backward path from C11's rounded FP16
adjoints/GradScaler128. Comparing against C11 is useful context,not a pure
loss-only ablation. No existing training worker or production defaults changed.

One disabled configuration is ready for separate approval:original OWLv2,
59 photographs/182 positives,2 projection+6 box tensors,original frozen
calibration/shared features,seed6106,AdamW LR1e-5,batch1/accumulation4(last3),
5 epochs/75 updates,600 seconds,6.5GiB,373 forwards(59 correspondence+295
training+19 fixed diagnostic inference),354 head-verification contractions,
2GiB output ceiling,and lambda={f['coefficient']:.16g}. Checkpoints are versioned
under `{plan['destination']}`;that directory does not exist and no training was
launched. Loss/gradient/integrity/resource failures stop without retry.

The future19-photo comparison must use identical FP32 inference,threshold0.15,
annotations/mapping/matching and AP50/AP75 definitions for original and existing
adapted baselines. Report matched-recall precision,TP/FP/FN,Chair errors,Sofa/Bed
regressions,localization/duplicates,shared logit drift and disagreement. Save
predictions before separate CPU metrics. Do not score frozen/official material
or tune the fixed coefficient from this exposed panel. The ≥60%precision/recall
above68.85% development objective and eventual independent80%precision/90%recall
have not been advanced by this gradient-only diagnostic.

Recommendation:one separately authorized75-update diagnostic is technically
prepared,with cautious expectations because the global penalty is small and
coverage is mostly easy negatives. No further diagnostic variant or training
cycle is authorized. Rollback is to leave the new entry point disabled and use
the preserved original branch/checkpoints;delete or overwrite nothing.
"""
    atomic(REPORT/'report.md',text.encode())
    print('E4 report written;one proposed coefficient;training remains disabled',flush=True)


if __name__=='__main__':run()
