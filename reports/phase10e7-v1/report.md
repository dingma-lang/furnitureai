# Phase 10E.7 — E5 optimization-query cache completion

Completed November 22, 2026, on branch
`experiment/phase10e7-e5-training-query-cache`, parent
`8bbd17e338c82173629c3e2570858aebe56ac1e4`.

**The single authorized run succeeded: 59 photographs, exactly 59 detector
forwards, zero backward calls and zero optimizer steps.** All complete query data
was saved atomically before CPU analysis. No retry, downloads, additional models,
development/reserved/frozen scoring, training or production changes occurred.
The new evidence does not justify another training cycle yet.

## Authorized model, inputs and preprocessing

E5 checkpoint:
`data/open-images/experiments/phase10e-distillation-fp32-v1/checkpoints/epoch-5.pt`.
SHA-256: `416620a947a96579bdd20c85e7adeb8b60f65c8983e73d3b3943a58ceabef4f0`.

Original base: `models/owlv2-original-baseline-20261103/model.safetensors`.
SHA-256: `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`.

The CPU preflight checked the exact 59 approved optimization image IDs and
hashes, 56 known source groups, source dimensions/orientation, individual-photo
rights-evidence hashes, original native annotation sources, protected-region
inputs and both checkpoint hashes. The E5 endpoint's eight trained tensors
reconstruct its saved completion fingerprint. The worker loaded the original
base, checked every original persisted tensor, restored only the saved E5 head
payload, and verified the loaded endpoint fingerprint before inference. No
optimizer object was created or optimizer state changed.

Every one of the 59 reconstructed image/text processor tensor contracts matched
the existing source-bound teacher index exactly. Configuration: existing local
`Owlv2Processor`, `use_fast=False`, RGB, no augmentation or EXIF transpose,
original 960-pixel processor configuration, seven prompts in this order:
Sofa, Chair, Table, Bed, Desk, Cabinet, Lamp, each `a photo of a <category>`.
The model's 60x60 query grid retains row-major IDs 0–3599.

The prepared contract uses **FP16 autocast forward, FP32 stored logits/boxes**
with FP32 parameters. This matches the original optimization teacher/training
forward precision. Earlier E5 development inference used FP32; it was not
rerun, read for this analysis, or presented as an identical-precision benchmark.
All modules stayed in evaluation mode with `requires_grad=False` and no gradients.

Model boxes are normalized `cxcywh` on the processor's square padded on the
bottom/right. Conversion is `(center +/- size/2) * max(original_width,height)`;
there is no centered-letterbox offset. Both normalized original-image `xyxy`
and original pixel `xyxy` are retained in FP64 without clipping. Bounds clipping
is used only for the separately labelled NMS/display summaries; it cannot make
an out-of-frame query eligible as a negative.

## Actual captured data and resources

Destination:
`data/open-images/experiments/phase10e6-training-query-cache-v1/`.

- **59/59 complete photographs**, with 59 distinct source-photo IDs and SHA-256s.
- **212,400 raw queries and boxes**, each with a stable per-image query ID.
- **1,486,800 class logits and 1,486,800 corresponding scores**, all seven
  categories retained; confidence filtering never removes raw query data.
- **59 attempted = 59 actual = 59 completed forwards**, no warmup or extra passes.
- Recorded GPU-active interval **26.526 seconds**; worker wall time
  **31.785 seconds**, launcher wall time **32.447 seconds**, including startup,
  CPU loading, output persistence and shutdown. These are wall intervals rather
  than sums of GPU kernel-event timings; the whole process stayed below 180s.
- Peak allocated VRAM **1,661,684,224 bytes (1.548 GiB)**; peak reserved VRAM
  **2,325,741,568 bytes (2.166 GiB)**, below the 6.5 GiB allocated limit.
- Outputs use approximately 22 MB allocated, including raw arrays, telemetry,
  analysis and contact sheets. Final bounded storage measurements are recorded
  in `final-summary.json`; the verifier also enforces the 128 MiB ceiling.

The launcher had a 175-second graceful-stop boundary and 180-second kill
deadline, stricter than a GPU-only time limit. It permits no automatic retry.
Per-forward intent/completion receipts and heartbeat telemetry preserve counts
and memory measurements. Each NPZ was atomically published and reloaded for
exact array equality before its source-bound receipt was committed. A complete
59-photo marker was written before any analysis process started.

The before/after fingerprint covering **all 420 registered parameter/buffer
tensors** is identical:
`cb7a0b2f132083c50c64041c4981fbed9fdf5eed36022e01036788da0002a0c2`.
The unregistered original `box_bias` tensor was checked independently and also
remained identical. Checkpoint files and source photographs passed post-run hashes.

## Confidence-filtered summaries

The following are **prediction counts, not scored false positives or accuracy**.
The NMS view uses per-query argmax, minimum score 0.15, class-aware IoU 0.5,
maximum 300 predictions per photograph, followed by the indicated score filter.
The raw cache retains all other class scores and suppressed queries.

| Category | Post-NMS >=0.15 | Post-NMS >=0.50 | Post-NMS >=0.80 | Raw class terms >=0.50 | Raw class terms >=0.80 |
|---|---:|---:|---:|---:|---:|
| Sofa | 534 | 148 | 39 | 318 | 49 |
| Chair | 4,692 | 805 | 162 | 1,394 | 198 |
| Table | 243 | 84 | 6 | 144 | 8 |
| Bed | 431 | 83 | 27 | 190 | 31 |
| Desk | 215 | 64 | 10 | 120 | 12 |
| Cabinet | 303 | 39 | 0 | 42 | 0 |
| Lamp | 593 | 121 | 12 | 123 | 12 |
| Total | **7,011** | **1,344** | **256** | **2,331** | **310** |

`prediction-counts.csv` also includes raw >=0.15 counts. `query-summary.json`
separates raw class terms, raw argmax queries and post-NMS predictions for all
three thresholds. Multiple category terms can belong to one query. Scores are
sigmoid logit compatibility scores, not independently calibrated probabilities.

## Chair annotation overlap and evidence status

Chair score >=0.50 occurs for **1,394 queries on 54 photographs from 52 known
source groups**; 198 score >=0.80. At the same fixed patch/query ID, 1,198 have
original-teacher Chair score below 0.15. Changed boxes can correspond to different
objects, so this is a query-trajectory comparison, not an object identity claim.

Conservative primary relations, in precedence order:

| Relation | Queries |
|---|---:|
| Verified Chair-positive overlap, IoU >=0.5 | 285 |
| Potential real Chair fragment | 183 |
| Other protected/incompletely annotated furniture overlap | 856 |
| Outside original image bounds | 5 |
| Not a reported Chair detection | 15 |
| Unmatched source context or unknown; not confirmed false positives | 50 |

In nonexclusive measurements, 1,324 overlap at least one protected native,
positive, ambiguous or ignored region; 496 overlap another reported Chair query
at IoU >=0.5. These are overlap/possible-fragmentation measurements, not proof
of mistakes or duplicate physical objects. Sixty-six overlap old negative
rectangles, but **zero are fully contained**, and none establishes new trusted
high-confidence coverage of the existing 41 source-supported regions.

**Zero genuinely new trusted hard-negative labels were created or activated.**
The versioned inactive candidate manifest separates **21 automatic/unverified
footprints** from **1,373 uncertain/protected/rejected footprints**. The latter
include genuine furniture and must not be interpreted as confirmed errors.

The 21 unverified candidates span **10 photographs/10 source groups**, scores
0.5012–0.7638; none reaches 0.80. Their native bbox context is:

| Automatic context | Candidates |
|---|---:|
| Person/body | 10 |
| Appliance/screen | 7 |
| Plant/decoration | 2 |
| Other native objects | 2 |

Native labels include person, laptop/laptop computer, refrigerator, sink,
microwave, potted plant, mirror and toilet. These are spatial context labels,
not automatically assigned crop identities. All 21 have native mask data and
supported simple polygons, but **none of the selected source polygons fully
contains the query rectangle with the existing two-pixel margin**. Eleven have
an exact minimum side below 32 pixels; median width/height is 37.45/38.46 pixels.
The E6 source-region/size/protection policy was preserved without relaxation.

Three bounded contact sheets were generated and actually inspected by the
assistant: all 21 unverified candidates plus three additional new high-score
protected-context examples. Many unverified crops show small dark equipment,
reflections, kitchen details or person/background boundaries; a native person
bbox sometimes contains a background query. A toilet fragment is visually
consistent with its native label, but complete source-mask support still failed.
The three additional examples clearly show real Chairs, including a Chair in a
bedroom with broad native House context. They remain protected, as do seating
stools and incompletely annotated furniture.

These 24 observations are explicitly **assistant visual assessments**, not
fabricated human reviews, source-verified crop absence labels, or new annotations.
No manual rectangle drawing was required and no model prediction became a
negative training label. Existing source masks, boxes, native class IDs,
LVIS federated completeness rules, ignored regions and source licensing remain
unchanged. All unspecified background remains ignored.

## Deduplication and provenance

The E6 registry contains 427 unique image/array snapshots after collapsing
617 cache-file aliases. All 59 E7 numerical snapshots are new relative to that
registry: **486 combined distinct snapshots**, not 486 independent photographs.

Across model states, Chair high-score patch trajectories are deduplicated by
source-image hash and row-major query ID: 621 current hotspots were seen before,
773 were not, and the combined registry has 2,367 unique high-score Chair patch
hotspots. In exact original-pixel footprint comparison, 588 current Chair
footprints were seen before and 806 were not. Small coordinate changes can create
new exact footprints; none of these counts establishes new physical objects or
new mistakes. Full original state/score/coordinate lineage is retained.

All 59 photographs have distinct source-photo IDs and hashes, so no photograph
was counted twice. Known repeated creator groups are tracked without merging
their different photographs: 59 photographs correspond to 56 known source groups.
Capture/physical-room independence is not proven by distinct hashes or source IDs.
No perceptual-independence or seven-class-completeness claim was manufactured.

The source-bound `raw-complete.json` identifies each `queries/NN.npz` and receipt.
The binding hashes original model/processor files, source photographs, original
teacher caches, official native annotation sources, rights evidence and previous
protection/candidate inputs. Environment and exact configuration hashes are in
`environment.json`; no packages, drivers or settings were changed.

## Tests, preservation and next action

**70 CPU tests passed:** 62 before inference, eight after caching. Coverage
includes exact authorization, forward limits, partition rejection, processor
contracts, coordinate inversion, immutable outputs, raw-array round trips,
threshold monotonicity, candidate protection and cross-state deduplication.
Post-cache tests and analysis ran with CUDA hidden in separate CPU processes.

Historical checks passed for the 9,973 earlier protected data/model files and
the later C8/C11/C13/C17–C21/D/E1–E6 artifact seals. Original OWLv2, E5, SigLIP 2,
historical datasets/results and frozen partitions remain unchanged. The Mac
application was not accessed. `main` remains
`e14c4c253d4680e7d3c2a87d0d896c6d20e86aa6`.
Historical preservation includes read-only file-checksum verification; it does
not decode, infer on, mine or score evaluation photographs. The clarified
`preservation-v2.json` distinguishes that checksum access from model/data use.

**One recommended next action:** perform a targeted CPU/source-evidence
qualification of these **21 actual E5 query candidates on ten existing training
photographs**, using the native instance masks and explicit Chair/stool
completeness evidence. Resolve box-versus-visible-mask correspondence and small
fragment uncertainty without weakening the negative-label gates. This should
precede more GPU training, another loss variant or speculative acquisition.

The current 59-image pool still provides no newly admitted source-supported
high-confidence negatives. More diverse, source-complete licensed real rooms are
likely needed for the long-term seven-category goal, but immediate new-download
necessity is not proven while these 21 candidates and the 42 previously audited
inactive local photographs remain unqualified. The separate conditional 64-ID
rights route from E6 remains unapproved and was not executed. Independent fully
annotated, source-disjoint seven-category acceptance coverage remains **zero**.

**Stop here: cache completion authorizes no training, loss changes, extra variants,
new images, official holdout scoring or deployment.**
