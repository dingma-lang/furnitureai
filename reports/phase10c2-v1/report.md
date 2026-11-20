Furniture AI Phase10C.2 — negative-supervision repair preparation

The implementation and CPU validation are complete; corrected training is blocked by missing trusted Chair-negative evidence and the absence of an unexposed evaluation set. No GPU inference, optimizer updates, new downloads, annotation activation, checkpoint changes or Phase10D box-only training occurred. All corrected-mode activation flags remain false.

The confirmed difference is the classification target mask. Phase5C used native class coverage: proposals not matched to a GT object became negative targets for covered classes, except padding and ignored regions. Phase10A/10B instead initialized valid negative targets from explicit source-negative classes only, then enabled matched positive targets. Unmatched queries in positive-only or incomplete classes were masked. This conservatively preserved partial COCO/LVIS semantics, but removed all unmatched Chair-negative supervision from the expanded optimization pool.

The focal BCE coefficient stayed 1, gamma stayed 2, positive/negative alpha stayed 0.25/0.75, and L1/GIoU weights stayed 5/2. Hungarian matching still combines negative sigmoid class score, 5*L1 and 2*(1-GIoU). Objectness remained frozen and received no background loss. Inference still used maximum sigmoid category scores at 0.15, AP floor0.001 and NMS IoU0.5. No explicit recall-loss term was added. Selection changed from development macro AP50 toward matched-recall precision with reference-recall, per-category TP and AP75 guards; the old rank gate did not require actual precision improvement.

Classification sums valid focal terms and divides by max(native positive GT count,1), rather than valid-query count. Mask changes therefore change effective classification/background gradient magnitude even with unchanged coefficients. Chair supplies114/182 retained optimization boxes in42/59 photographs; there is no class-frequency reweighting. Those counts do not identify actual per-class loss contribution because normalization is per image. The absence of Chair negatives plausibly permits a shared logit shift/scale and class projection to favor Chair on fragments and correlated scene content. Historical component gradients/losses were not logged, and no causal training ablation was run. Positive-only supervision is a supported mechanism hypothesis, not a proven sole cause.

The private versioned inventory is `data/open-images/preparation/phase10c2-v1/negative-candidates-v1.json`. Its hash and counts are recorded in the committed candidate-manifest index. It inventories156 existing prospective source records without moving partitions or assuming missing labels imply absence:

| Inventory | Count |
|---|---:|
| Historical prospective training photographs | 120 |
| Existing optimization / internal-validation photographs | 59 / 19 |
| Other prospective train photographs excluded from the active pool | 42 |
| External development photographs | 36 |
| Explicit native Chair-negative photographs among the120 | 0 |
| Human-verified new Chair-negative/completeness records | 0 |
| Priority optimization-photo completeness reviews | 9 |
| Other optimization photographs without trusted Chair scope | 50 |

The nine candidates span nine known source groups: seven Open Images and two LVIS images. They have native claims sufficient for the old FP-observability rule, but not independently verified completeness. All nine remain ineligible for new negative supervision pending review. They are existing positive-chair photographs, not nine chair-absent photographs: confirmation would permit background negatives only outside protected positive/unknown regions. The inventory records source ID, photograph hash, provenance, group, native annotations/completeness, rights/quarantine status, historical/internal partition, possible missing Chairs and eligibility. Historical non-training acquisition IDs are explicitly protected; no reserved image body was opened.

A bounded contact-sheet inspection of these nine found foreground bar stools outside active Chair boxes in one image and partial/ambiguous furniture in another. These observations are stored privately as assistant concerns, not human labels. No image was declared complete or Chair-free. This is direct reason to avoid blindly restoring Phase5C covered-class background supervision across the expanded data.

The local interface is `data/open-images/preparation/phase10c2-v1/review/index.html`. It contains65 items: the existing48 targeted Chair FPs plus8 unscored controls, and9 training-only category-scope reviews. The six prediction decisions are Genuine Chair, Other furniture, Background/non-furniture, Partial/ambiguous, Missing ground truth and Uncertain/defer. Native boxes and the selected prediction are displayed with original aspect ratio. Every decision starts blank and requires reviewer, date, concrete evidence and an explicit confirmation action. The nine scope items separately ask whether visible Chair instances are completely boxed; missing/ambiguous/uncertain decisions keep scope inactive. The interface exports only explicitly confirmed records to a new timestamped JSON file and can restore exported progress.

The56 prediction examples come from already-exposed internal validation and cannot become training negatives regardless of the review outcome. The importer checks exact queue and item identity, photograph hash, source group, partition, valid decisions, reviewer/date/evidence and confirmation. Only the nine optimization-photo scope items can yield proposed trusted completeness records. Importing never approves training, changes annotations or enables the configuration. Existing prediction artifacts, the original review CSV and prior assistant notes remain unchanged.

To open the interface locally:

```bash
cd /home/tristary/furniture-ai-ml
.venv/bin/python -B -m http.server 8765 --bind 127.0.0.1 \
  --directory data/open-images/preparation/phase10c2-v1/review
```

Open `http://localhost:8765/` in the Windows browser. Start with the nine `scope-*` items if the immediate priority is reliable training evidence, then the high-confidence FP items. Inspect the full image before confirming completeness. Export progress regularly; no answers are sent over a network. All decisions are currently blank. The server exposes only the prepared review folder, not checkpoints or the entire project. A short-lived localhost-only index/image check passed and its test server was stopped. Browser click automation was unavailable; backend decision/export-schema validation is covered by CPU tests.

After exporting a review, validate into a new private submission directory:

```bash
.venv/bin/python -B tools/phase10c2-v1/review.py validate \
  --input /absolute/path/to/phase10c2-review-TIMESTAMP.json \
  --output data/open-images/preparation/phase10c2-v1/submissions/manual-v1
```

Use a fresh version for subsequent submissions. Partial submissions are allowed and pending counts remain explicit. No actual human submission was created during preparation. Test submissions exist only as synthetic in-memory fixtures.

The isolated `tools/phase10c2-v1/trusted_loss.py` adapter preserves original behavior as the default `phase10b` mode. Corrected mode `trusted_chair_negative_v1` requires validated/approved/enabled flags, an approval reference, immutable optimization-row bindings, evidence-bundle hashes and accepted human category-scope evidence. `validate_bundle` is a mandatory future-worker preflight: it rejects empty evidence, new images, validation membership, conflicting/redundant records, source/hash/group leakage and annotation-lineage drift. No existing worker or production setting invokes this adapter.

The corrected target mask preserves every legacy matched positive and explicit-negative target. New Chair negatives may be introduced only from confirmed whole-image absence, confirmed completeness with positives, or an explicitly reviewed negative region. No other category's negative mask is broadened. A proposed box must lie fully within the trusted image/region and valid unpadded frame. Any overlap with a known Chair, ignored native region or marked potentially unannotated Chair region blocks new negative supervision. Whole-image absence contradicting native Chair annotations and completeness claims with unresolved possible positives are rejected. Regional negatives may coexist with uncertainty elsewhere, with those potential regions protected. This is deliberately conservative: duplicates/fragments overlapping known Chairs are not newly declared negative.

Native classes, coordinates, positive boxes, loss coefficients, matching and GT-count normalization are preserved. New diagnostics expose classification/L1/GIoU values, added trusted-negative loss, valid positive/negative terms and per-class mask counts. Known false Chair queries acquire a positive logit gradient that lowers their scores under descent; unknown queries retain zero Chair-negative gradient. Class+box head gradients remain enabled for a future paired comparison, while backbone/objectness stay frozen. Tests use synthetic tensors and randomly initialized head fixtures, never the pretrained encoder or real-image inference. In-memory head snapshots remain compatible with the existing twelve-tensor delta interface.

The proposed controlled experiment is a paired two-arm comparison, not Phase10D box-only adaptation:

| Fixed setting | Both arms |
|---|---|
| Initialization | Immutable original OWLv2; same head initialization and seed20261118 |
| Data | Same59 optimization photos/56 groups/182 positive boxes; unchanged partitions |
| Scope | Same twelve class+box tensors; backbone and objectness frozen |
| Learning rate / decay / precision | 1e-5 / 0.01 / FP32 |
| Batch / accumulation | 1 / 4 |
| Maximum epochs / updates | 5 / 75 per arm;150 combined |
| Queue time / VRAM / new storage | 60minutes /6.5GiB allocated /4GiB |
| Safety | Separate zero-update GPU smoke; prediction cache before isolated CPU metrics; immutable versioned checkpoints and reserve/commit recovery |
| Arm A | Exact Phase10B classification loss/masks |
| Arm B | Same loss with only accepted trusted Chair-negative mask additions |

These are proposed ceilings, not authorization. A new guarded worker would be required; historical runs must not be resumed or modified. Both configs are explicitly disabled and unapproved. With the current empty trusted bundle, B is numerically identical to A and the preflight refuses a meaningless comparison. Completing scope reviews can validate existing native completeness without adding photographs or relabeling positive boxes. Freeze that exact evidence bundle in both arms before authorization. If review requires new negative-region annotations, new photographs or positive-box corrections, exclude those changes from the primary comparison or define a separate data-expansion experiment; do not attribute their effects solely to implementation changes.

Evaluation must report P/R@0.15, precision at matched recall, AP50/AP75, ChairFP/FN, Sofa/Bed regressions, score distributions, per-photograph outcomes, runtime and peak VRAM. The19 internal and other historical exposed panels must not select epochs, learning rates or thresholds. At most they can provide a separately approved, fixed terminal descriptive comparison. There is currently no genuinely unexposed, independently verified evaluation set authorized for this A/B experiment. Reserved/frozen material remains inaccessible for optimization and no scoring occurs in this phase. Hence preparation is not sufficient for an independent80%precision/90%recall claim.

Predeclared exploratory support would require B to reduce ChairFP by at least30% versus original and improve over A, without increased ChairFN, improve matched-recall precision by at least3points, and preserve Sofa/BedTP with no per-category AP75 loss greater than2points. Evaluation membership, reference results and criteria must be frozen before execution, using an approved non-tuning protocol. Zero additional trusted-negative terms, leakage, changed inputs between arms, nonfinite results, violated limits or failed gates reject the experiment. No causal conclusion is possible now; no training approval is requested until evidence and evaluation blockers are resolved.

All190 relevant CPU tests pass, including22 new mask, gradient, normalization, freezing, determinism, checkpoint and review-validation tests. Historical byte verification covers9,973 dataset/model files,160 Phase10B snapshots,24 sealed comparison artifacts and27 historical implementation files, plus prior Phase10C recovery and Chair-investigation seals. No historical artifacts or Mac simulation files were modified. Git contains only isolated source/tests, disabled configurations, the private-manifest hash index and small summary reports; private datasets, photograph copies, full candidate metadata and review submissions remain ignored and local.
