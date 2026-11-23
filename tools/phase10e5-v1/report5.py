"""Final evidence-based report;no automatic follow-on experiment."""
from common5 import *

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU reporting only')
    r=read(REPORT/'results.json');integ=read(REPORT/'integrity.json');pres=read(REPORT/'preservation.json')
    done=read(RUN/'complete.json');launch=read(RUN/'launch-result.json');gate=read(RUN/'gate/complete.json')
    comparison=r['comparison'];new=comparison['E5'];old=comparison['original'];ep=r['training_epochs']
    table='\n'.join(f"| {n} | {100*comparison[n]['headline']['precision']:.2f}% | {100*comparison[n]['headline']['recall']:.2f}% | {comparison[n]['headline']['tp']}/{comparison[n]['headline']['fp']}/{comparison[n]['headline']['fn']} | {100*comparison[n]['headline']['ap50']:.2f}% | {100*comparison[n]['headline']['ap75']:.2f}% | {100*comparison[n]['precision_at_original_recall']['precision']:.2f}% |" for n in ('original','C8','C11','E5'))
    cats='\n'.join(f"| {c} | {old['categories'][c]['tp']}/{old['categories'][c]['fp']}/{old['categories'][c]['fn']} | {new['categories'][c]['tp']}/{new['categories'][c]['fp']}/{new['categories'][c]['fn']} | {100*new['categories'][c]['precision']:.2f}% | {100*new['categories'][c]['recall']:.2f}% | {100*old['categories'][c]['ap75']:.2f}% → {100*new['categories'][c]['ap75']:.2f}% |" for c in ('chair','sofa','bed'))
    losses='\n'.join(f"| {e['epoch']} | {e['losses']['positive']:.6f} | {e['losses']['box']:.6f} | {e['losses']['unit_KD']:.6f} | {e['losses']['weighted_KD']:.8f} | {e['weighted_KD_positive_ratio']['0.5']:.6g} | {e['weighted_KD_positive_ratio']['1']:.6g} | {e['KD_term_occurrences']} |" for e in ep)
    drift='\n'.join(f"| {c} | {r['logit_drift']['E5']['per_class_mean'][c]:+.6f} |" for c in CLASSES)
    summary=dict(outcome='C — Ineffective',training_complete=True,epochs=5,updates=75,coefficient=COEFFICIENT,
        immediate_target_met=False,independent_accuracy_established=False,
        fixed_threshold_results=new['headline'],checkpoint=done['final_checkpoint'],
        next_direction='Improve training-only trusted hard-negative coverage at actual high-confidence Chair-confusion locations,with source evidence and positive/stool protections,before another training run.',
        next_direction_executed=False,new_training_authorized=False,
        conclusions=dict(original_precision_preserved=False,recall_improved=True,broad_inflation_avoided=False,
             discrimination_improved_at_matched_recall=False,Chair_misses_decreased=True,major_Sofa_Bed_localization_regression=False,
             KD_nonzero_gradient_measured=True,KD_positive_gradient_conflicts=True,
             coefficient_assessment='Too weak in aggregate within this sparse mostly-easy-negative coverage to preserve confidence;larger coefficients are not proven safe or useful.',
             causal_limitation='Historical C11 comparison also differs in FP32 adjoints/unscaled box backward;no matched zero-KD numerical control was authorized.'))
    emit(REPORT/'final-summary.json',summary)
    emit(REPORT/'execution-provenance.json',dict(launch=launch,resources=done['resources'],gate=gate,
         precision=done['precision'],final_checkpoint=done['final_checkpoint'],
         initial_frozen_sha256=done['initial_frozen_sha256'],final_frozen_sha256=done['final_frozen_sha256'],
         pre_CPU_tests=read(REPORT/'CPU-tests.json')['passed'],post_CPU_tests=read(REPORT/'postrun-tests.json')['passed'],
         original_model_sha256=BASE_SHA,downloads=0,extra_variants=0,automatic_retries=0,production_changed=False))
    text=f"""# Phase 10E.5 — first selective teacher-distillation training

**Result C — Ineffective for the immediate precision/recall objective.** The one
authorized experiment completed safely,but severe confidence inflation and
scored false positives persist. No further experiment or deployment occurred.

| Model | Precision@0.15 | Recall@0.15 | TP/FP/FN | AP50 | AP75 | Precision at ≥original recall |
|---|---:|---:|---:|---:|---:|---:|
{table}

The last column is the unchanged PR-envelope maximum precision at recall at
least42/61(68.85%),not an adopted threshold. Original's envelope precision is
68.85%(42TP/19FP),compared with E5's56.76%(42TP/32FP). E5 therefore cannot
recover ≥60%precision while retaining original recall anywhere on this saved
curve. No threshold,coefficient or hyperparameter was changed using these data.

E5's threshold counts are identical to C11;AP50 differs by only about
-0.00098 percentage points,and AP75 is identical. This is no meaningful observed
improvement over that historical model. AP/localization gains versus original
do not compensate for25→573 scored FP. The original-model precision target
was lost despite recall improving68.85%→90.16%.

| Category | Original TP/FP/FN | E5 TP/FP/FN | E5 precision | E5 recall | Original→E5 AP75 |
|---|---:|---:|---:|---:|---:|
{cats}

Chair contributes486/573(84.82%)of E5's scored FP. Chair misses fell15→4,
but Chair FP rose19→486. Sofa and Bed gain one TP each,with26 and61 scored FP.
Sofa/Bed AP75 improves. On the *same* GT matches,Sofa mean IoU improves by
{r['localization']['sofa']['mean_delta_IoU']:+.6f},Chair by{r['localization']['chair']['mean_delta_IoU']:+.6f},and Bed changes
{r['localization']['bed']['mean_delta_IoU']:+.6f}. The tiny shared-Bed decrease is not a major regression.
Comparing all matched-box means without accounting for newly matched objects
would confound localization with differing match difficulty.

Scored duplicate FP:original1,E5{sum(new['categories'][c]['duplicate_fp'] for c in CLASSES)}.
Fragmentation candidates increase4→44,mixed-object candidates15→61,and category
disagreement candidates14→29. These geometric heuristics are not human-confirmed
failure labels. The FP explosion is not explained by scored duplicates alone.

The native completeness policy leaves{sum(new['unknown_unmatched_at_015'].values())}
post-NMS predictions unscored at0.15,including849 Chair predictions. They are
not confirmed errors or negatives. Per-category `unknown_unmatched` in the
historical metric payload uses the0.001 AP floor;the separate `at_015` field
reports operating-point unknowns. Ground truth supports39 Chair,13 Sofa and9 Bed
instances across19 exposed,partially annotated photographs. Table,Desk,Cabinet
and Lamp lack supported positive scoring here. Native Chair/Stool semantics
also retain the historical difference from the project's Chair-includes-seating-
stools policy. Nothing was silently relabelled or treated as verified background.

Broad inflation persists on identical68,400 query locations. Mean seven-class
logit drift is+{r['logit_drift']['E5']['mean_all_categories']:.6f},versus C11's
+{r['logit_drift']['C11']['mean_all_categories']:.6f}. E5 differs from C11 by only
{r['logit_drift']['E5']['mean_all_categories']-r['logit_drift']['C11']['mean_all_categories']:+.6f}in mean raw logit.

| Category | Mean E5 logit change vs original |
|---|---:|
{drift}

Raw Chair scores≥0.15 rise82→2,713(C11:2,714);scores≥0.5 rise3→440.
E5's raw category argmax disagrees with original at41,763/68,400 query locations.
These are aligned patch queries,not independently matched physical objects.
Score/logit quantiles for all seven categories are saved in `results.json`.

The safety gate verified all59 original-cache correspondences,then the required
three representative full combined-loss checks,covering99 nonzero-coverage
Chair-distillation queries across3 source groups and all3 positive categories.
The checks used one original-initialized student,original teacher caches,and
the approved FP16-forward/FP32 observed-node classification adjoint,with native
box backward and scale1. No GradScaler or manual128 scaling was used. Full/base/
KD gradient decomposition,positive projection route,all six box gradients,
known-positive/ignored protections,frozen scope and coefficient application
passed. At original agreement KD loss and gradient are *zero*,as expected;the
gate's nonzero coverage must not be misreported as a nonzero initial KD gradient.
Actual nonzero adapted KD propagation had already been recovered in E4 and was
then monitored during this training trajectory.

No optimizer existed during the gate,and no parameter changed. The first update
intent follows the passing gate marker. The59 original checks and gate consumed
{gate['resources']['seconds']:.3f}s of the same600s queue. There were no automatic
retries,alternate precision paths or extra models.

Actual execution:5 epochs,75 verified AdamW steps,295 training presentations,
373 detector forwards(59original checks+295training+19final FP32 inference),
354 head-verification contractions,{done['resources']['VJPs']} explicit diagnostic VJPs,
and295 training backward calls. Wall time was{launch['wall_seconds']:.3f}s;
worker elapsed{done['resources']['seconds']:.3f}s;peak allocated VRAM
{done['resources']['peak_allocated_bytes']/2**30:.4f}GiB(reserved
{done['resources']['peak_reserved_bytes']/2**30:.4f}GiB),within6.5GiB.
The pinned data,seed6106,order,batch1/accumulation4(last3),LR1e-5,AdamW
betas0.9/0.999,eps1e-8,weight decay0.01,clip norm1 and coefficient
{COEFFICIENT:.16g}were preserved. No backbone/calibration updates occurred.

| Epoch | Positive loss | Box loss | Unit KD | Weighted KD | Median KD/positive grad norm | Maximum ratio | KD term occurrences |
|---|---:|---:|---:|---:|---:|---:|---:|
{losses}

Losses are per-image diagnostic means,not norms of the accumulated optimizer
gradient. Every epoch retained15 source groups with KD coverage. Term counts
are occurrences during an evolving epoch,not final-checkpoint coverage snapshots;
the initial teacher mask had278 terms. The fixed containment/protection policy
was unchanged as predicted boxes and positive matches evolved.

There were73 actual nonzero KD VJPs,and all73 conflicted with positive-gradient
directions. No ratio exceeded0.5 and no KD-dominance flag fired. Epoch5 active-
presentation median weighted KD/positive norm ratio was0.003647(0.365%),with
maximum0.445817. Weighted KD loss was only about0.051%of mean classification
loss at epoch5. The term was mathematically active,but its observed global
effect was negligible:the coefficient was too weak in aggregate within this
limited,mostly-easy-negative coverage. That does not show that increasing it
would be safe or sufficient;verified-positive conflicts and uncovered queries
remain substantial limitations.

The projection changed from original(weight L2 displacement
{integ['projection_changes']['class_head.dense0.weight']['delta_L2']:.8f},bias
{integ['projection_changes']['class_head.dense0.bias']['delta_L2']:.8f}),while
remaining extremely close to C11(weight difference L2
{integ['projection_changes']['class_head.dense0.weight']['difference_from_C11_L2']:.8g}).
All five new head/optimizer/RNG checkpoints passed CPU loading,hash,parameter-
scope,coefficient and optimizer-counter checks. All4 calibration tensors stayed
bit-identical,and full frozen-state hashes match before/after. Historical
original OWLv2,SigLIP2,classifiers and detector checkpoints remain unchanged;
9,973 inventoried historical files and prior experiment seals passed.

Final checkpoint:`{str((RUN/done['final_checkpoint']['file']).relative_to(ROOT))}`

SHA-256:`{done['final_checkpoint']['sha256']}`

All19 prediction arrays and their source/hash manifest were saved atomically
before the isolated Torch-free CPU metrics process. Original,C8 and C11 metrics
were recomputed from saved outputs and exactly matched their historical headlines.
{read(REPORT/'CPU-tests.json')['passed']} CPU tests passed before training and
{read(REPORT/'postrun-tests.json')['passed']} afterward. Additional integrity
checks verified actual forward partition membership,exactly five presentations
per training photograph,no updates before the gate,constant coefficient/scaling,
and immutable checkpoint contents. No frozen/reserved images were scored.

This is not a clean causal distillation-only ablation against C11:the historical
control also differs in numerical adjoints and GradScaler use. C8 additionally
trained calibration parameters. The19-photo panel is exposed and partially
annotated,and three-category native weak-label agreement cannot establish
independent seven-category accuracy. No progress toward the simultaneous
≥60%precision/improved-recall milestone,or eventual independent80%precision/
90%recall requirement,is demonstrated.

**One next direction:** improve *training-only trusted hard-negative coverage*
at actual high-confidence Chair-confusion locations,using source-supported or
human-confirmed evidence and protecting positive Chairs/seating stools and all
uncertain regions,before another training run. Do not mine the19-photo evaluation
panel into training labels. No additional acquisition,annotation activation,
inference,coefficient sweep or training was performed or authorized here.

The candidate remains isolated and undeployed. Rollback means retaining the
immutable original/previous checkpoints and leaving production defaults alone;
no historical artifact should be deleted or overwritten.
"""
    atomic(REPORT/'report.md',text.encode());print('E5 final report: C — Ineffective;one next direction,not executed',flush=True)

if __name__=='__main__':run()
