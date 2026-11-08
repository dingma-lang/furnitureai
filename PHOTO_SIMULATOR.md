# Local furniture photo replacement simulator

This is the working **single-photo** experimental compositor in `src/photo_simulator`, separate from production recognition, classifier training and frozen evaluations. It uses the existing OWLv2 detection class and room-layout model loading, with a simulator-only GPU adapter. Existing Python room-viewer modules were present on Ubuntu, but their frontend assets were absent; a small dependency-free browser frontend was added under this new module rather than copying the full consumer workflow.

## Start on Ubuntu

```bash
cd /home/tristary/furniture-ai-ml
bash scripts/run_photo_simulator.sh
```

Open `http://127.0.0.1:8787`. The server binds **only** to loopback. Stop it with Ctrl+C if started in a terminal. During implementation it was started detached; its process information and log are under `reports/photo-replacement-simulator-v1/server-process.json` and `server.log`. Avoid starting a second writer/server for the same store. No pip/install or model download is performed by the startup script.

On the Mac, keep an SSH tunnel running:

```bash
ssh -N -L 127.0.0.1:8787:127.0.0.1:8787 furniture-ai-rtx
```

Then open the same `http://127.0.0.1:8787` address on the Mac. The existing Windows-to-WSL localhost forwarding was verified for this setup. If the port is occupied, reuse the running tunnel/server. No public listener or cloud service is needed. The GPU computer must remain awake and connected.

Optional CPU mode: `SIMULATOR_DEVICE=cpu bash scripts/run_photo_simulator.sh`. CPU support is implemented but the full three-model CPU path was not benchmarked in this phase. Render-only image processing is CPU-based in all modes. Typical room uploads are normalized to a maximum 1600-pixel side; the exact original upload is preserved separately.

## Use it

1. Upload a JPG, PNG or WebP room photograph, up to 18 MB and 24 megapixels. Click **Detect furniture**. Actual local OWLv2, UPerNet semantic segmentation and Depth Anything Small inference run sequentially in an offline subprocess, using existing checkpoints.
2. Click a detected object, or **Draw selection** when detection is wrong or unavailable. A detector box is image-space evidence, not a physical dimension. Fragmented detections may require drawing one corrected full-object box.
3. **Preview mask** shows the selected region. Add or subtract polygons; use **Protect foreground** for objects that must remain in front of the replacement. Finish a polygon explicitly; Undo removes the last polygon. Semantic masks are category masks, not independent instance identities or SAM output.
4. Choose the authorized green-sofa asset or upload your own furniture photo. Transparent PNG/WebP alpha is preserved. Ordinary backgrounds use classical GrabCut isolation; warnings appear if separation is uncertain or the background must remain. Upload a cleaner transparent cutout when needed.
5. Click **Generate replacement**. The application removes the masked object using classical Telea background filling, then transforms and alpha-composites the replacement. Depth assistance is a heuristic foreground ordering; explicit protection polygons override removal/insertion. Optional brightness matching and a soft contact shadow are approximate image adjustments.
6. Drag the placement outline, edit width/height/position/rotation/perspective, or flip the replacement. Generate again to apply those edits. The outline is an editing aid; moving it does not claim an already-rendered result has changed.
7. Compare with the before/after slider, download the PNG, **Save settings**, and reopen through the project picker. **Show original** is reversible and keeps generated files. Each render has a unique filename; prior outputs remain in the project history.

**Visual preview — physical dimensions not verified.** All placement coordinates are image pixels. Optional reference measurements are recorded as user-supplied evidence; they do not calibrate the camera or room automatically. No scene here can claim verified physical fit. Monocular depth is ordinal, scale is unresolved, and hidden room surfaces remain unknown. A single photo is sufficient; multi-photo registration and metric 3D editing are separate existing components, not required for this workflow.

## Camera capture

Use camera asks the browser for permission, shows a preview, and captures one still frame through the same upload pipeline. It stops tracks on close/capture/page exit. Camera access requires localhost/a secure browser context and an available device. Browser-permission denial is handled. Automated tests used Chromium's synthetic camera; a physical webcam or phone camera has **not** been validated. This is not continuous tracking or AR.

## Storage, APIs and offline requirements

Projects are isolated under `.local-simulations/<id>/`: untouched `original-upload.bin`, normalized `original.png`, inference outputs, asset versions, mask/result PNGs and atomic `project.json`. Saved settings include the source-pixel selection, edits, placement, reference notes and option flags. Model jobs are serialized; conflicting writes to a working project are rejected. Interrupted-job logs remain available; no automatic crash-retry loop runs. Reopening a project after restart reads saved outputs; an old in-memory job ID cannot be polled after restart. Completed inference is reused for the unchanged project image.

Existing checkpoint paths:

- `models/owlv2-original-baseline-20261103` — original detector, not changed.
- `models/photo-simulator-existing/depth-5426e4f0` — copy of previously cached Mac Depth Anything V2 Small checkpoint (Apache-2.0 model card).
- `models/photo-simulator-existing/segmentation-876ffc56` — copy of previously cached Mac UPerNet ConvNeXt Tiny checkpoint (MIT model card).

The two copied model directories were transferred between the user's machines, not newly downloaded. Their hashes are checked against the cached sources. SigLIP2/classifier weights remain available but are not needed to composite a user-selected replacement, so they were not trained or modified. No compatible installed generative inpainting model was found; that comparison is explicitly unavailable.

OpenCV `4.10.0.84` is installed with `--no-deps` in `.runtime/photo-simulator`, preserving the existing environment package versions. It is compatible with the observed NumPy `2.5.4` runtime. The app uses existing Pillow/NumPy/PyTorch/Transformers; its HTTP server uses Python's standard library. All Hugging Face loads are local-only/offline. A new machine must provide these dependencies/checkpoints explicitly; no silent downloads occur.

## Tests and reproduction

```bash
cd /home/tristary/furniture-ai-ml
PYTHONPATH="$PWD/.runtime/photo-simulator:$PWD/src" .venv/bin/python -B -m unittest discover -s tests_photo_simulator -v
```

This runs deterministic image-processing/storage/failure tests without large models. The recorded real-browser experiment uses authorized exploratory Unsplash photos already in `data/detection-pilot` and `data/room-validation`; it never uses frozen holdouts. The browser verifier is saved in the report directory and was run from the Mac using its already installed Playwright/Chrome. Set `SIMULATOR_RESUME_ID` to reuse the saved real-inference project rather than repeat model work. Synthetic detector/mask failures and camera frames are clearly separated from the genuine model run.

See `reports/photo-replacement-simulator-v1/RESULTS.md`, `benchmark.json`, browser screenshots/video and preservation results for actual evidence. The included authorized asset is a derived photograph cutout, not a catalog-verified product or known-size model. Photographer/source attribution is retained in `simulator-assets/manifest.json` and the original `data/detection-pilot/ATTRIBUTION.md`.

## Current limits

Classical filling can smear curtains, floor texture and object boundaries; it cannot reveal real hidden structure. Semantic masks can leak or miss details; manual correction remains important. Relative-depth occlusion can misorder thin objects. Perspective and lighting controls are 2D approximations; no new viewing angle, physically correct shadow or 3D geometry is generated. A successful composite is not evidence of accurate furniture dimensions, identity, comfortable arrangement, or fit. Generative refinement, verified metric geometry and continuous tracking require separate work.
