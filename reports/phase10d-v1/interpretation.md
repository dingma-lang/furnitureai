# Phase 10D interpretation

**Outcome B: classification preserved; localization regressed.**

The original 62.69% precision / 68.85% recall operating point was preserved exactly, not improved. The immediate 60% precision milestone was already met by the original baseline. AP50 changed by -0.41 percentage points and AP75 by -4.62 points. Sofa AP75 fell 13.86 points, including one previously matched box crossing below IoU 0.75. Mean IoU declined on common matches in all three supported categories.

Chair scored false positives stayed at 19 and Chair misses at 15; there was no C20-style explosion. Duplicate FPs and fragmentation heuristics stayed unchanged; mixed-object candidates rose from 15 to 16. The original and Phase10D PR-envelope precision at original recall are both 68.85%; no threshold was adopted from these curves.

All 478,800 aligned development query-category logits (19 × 3,600 × 7) were bit-identical to original. The four repeated training-photo FP32 checks also matched exactly. The changes therefore arose from box geometry and downstream matching/NMS, not changed category scores. This shows the preservation mechanism worked, while the box adaptation failed to generalize on this exposed panel.

Mean training box loss declined from 0.418619 to 0.367611. Lower training loss plus worse development localization is consistent with overfitting or source/annotation mismatch; this one run does not isolate the cause. Intermediate epochs were not evaluated or selected after seeing this outcome.

Retain original OWLv2 for continued baseline use. The saved experimental endpoint is diagnostic only. Recommend a separately prepared original-teacher distillation experiment, with explicit evidence masks and precision/query correspondence checks before any request to train. Do not automatically launch it or resume loss-balancing sweeps. Independent all-seven-class acceptance remains unavailable, and the 80% precision / 90% recall target remains untested.

The prior CPU preparation import-wiring error was corrected before the successful actual architecture/data/test preflight. The GPU launch and smoke succeeded on their sole attempt; no GPU retry or model-update replay occurred.
