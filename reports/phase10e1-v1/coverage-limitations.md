# Coverage and gradient interpretation

The original teacher has 278 admissible distillation query terms on 15 photos /
15 known source groups. At saved C11 epoch 5, joint teacher/student containment
leaves 267 terms on the same 15 photos/groups. Only **two** of these student
Chair scores exceed 0.15. Most anchoring therefore concerns numerical drift on
already low-confidence examples, not demonstrated coverage of the high-confidence
Chair false-positive mechanism.

A smaller logit error on this subset cannot establish better discrimination,
preserved real-world recall or global seven-class confidence control. Uncovered
examples remain unknown. The unchanged base still has its original 182 positive
boxes and 41 class-specific source-negative grants; the new consistency mask
does not create labels for unknown objects or other classes.

In the diagnostic records, `selected_teacher_negative_scores` means **student
scores on the selected teacher/source-negative query mask**, not the teacher's
probability values. Teacher logits themselves remain in the immutable cache.
`selected_Chair_above_015` counts the corresponding student scores. The older
field name is clarified here to avoid interpreting those values as teacher
conflicts; initial teacher conflicts are separately counted as zero.

Unit-strength output derivatives are diagnostic measurements, not a proposed
coefficient. The real projection-gradient norm/direction and its coupling to
positive Chair gradients are missing. This is why only a zero-update diagnostic
is proposed next and why training remains blocked until strength evidence and
separate approval exist.
