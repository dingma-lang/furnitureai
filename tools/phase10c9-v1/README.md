# Phase 10C.9: saved-query Chair diagnosis

The read-only dashboard is at **http://localhost:8772** while `server9.py` runs.
Use `.venv/bin/python`, with `CUDA_VISIBLE_DEVICES=''`. No installation or
network acquisition is needed. There is no inference or training launcher.

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10c9-v1/server9.py --port 8772
```

Select the development or optimization-training pool, photograph, score
threshold and prediction status. The three panels use the same original-image
coordinate frame. Red is native-scored FP, orange is unknown, green is matched
TP. Click a box for its immutable model/photo/query identity and coordinates.
Additional native annotations are optional context, not activated GT. Evidence
cards link to the corresponding saved query when retained by NMS; missing
retained boxes are explicitly reported. No box drawing or human confirmation
is required. No POST endpoint can modify labels or launch a model.

Results are under `data/open-images/diagnostics/phase10c9-v1/`; reports under
`reports/phase10c9-v1/`. Photographs remain at their existing paths. Detailed
predictions, native annotations, contact sheets and screenshots stay private.
`dashboard-live.json` is a compact viewer projection; the initial full
`dashboard-data.json` and all analysis records remain preserved.

Preparation sequence (already completed; scripts refuse existing result paths):

1. `preflight9.py`: C8 report/private artifact seals, model/input/photo hashes.
2. `native9.py`: stream existing COCO/LVIS/Open Images metadata for development
   context. This inventory is explicitly ineligible for training.
3. `analyze9.py`: recover exact post-NMS query identities and native TP/FP/
   unknown states; thresholds .15/.50/.80, score/size/common-logit-shift analysis.
4. `mine9.py`: source-polygon checks for saved optimization-query witnesses only.
   It never treats unmatched queries or loose native boxes as absence evidence.
5. `plates9.py`: bounded source/context evidence plates. `visual9.py` contains
   assistant observations recorded after actually viewing those plates; they
   cannot set human flags or activate training labels.
6. `audit_objective9.py`, `query_audit9.py`: local installed architecture, CPU
   checkpoint-head differences, geometry and actual saved loss-target coverage.
7. `build_dashboard9.py`: three saved-model views; development is FP32 and the
   training probes are FP16. No training performance is scored.
8. `proposal9.py`: one unauthorized calibration-frozen experiment, no launcher.
9. `browser9.py`, `live_browser9.py`, `verify9.py`, `report9.py`: synthetic and
   real browser checks, CPU tests, historical preservation and small reports.

CPU regression command:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c9-v1 -p 'test*.py' -v
```

`server-process.json` records the background PID. Stop only that dashboard
process with `kill <recorded-pid>` when finished; no experiment/artifact cleanup
is needed. Restart the read-only server with the command above.

Two cautions: native Chair exhaustiveness does not prove completeness for the
broader Chair-plus-seating-stools project category; and a common-mode shift
across seven saved logits does not uniquely identify which head layer caused
it. No new encoder/head forward passes were used to fill missing intermediate
activations or objectness scores. All such missing evidence stays explicit.
