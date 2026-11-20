Phase10C.2 is CPU-only negative-supervision preparation. There is no GPU worker
or optimizer-execution entry point. Phase10D remains unauthorized.

The technical report and disabled proposed A/B configs are under
`reports/phase10c2-v1/`. Full candidate inventory, an empty trusted-evidence
bundle, immutable row bindings and the65-item local interface are private under
`data/open-images/preparation/phase10c2-v1/`. The nine scope reviews are existing
optimization images. The56 prediction examples are internal validation and may
never become training negatives.

Run CPU fixtures with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m unittest discover \
  -s tools/phase10c2-v1 -p test_negative.py -v
```

`trusted_loss.loss` defaults to the unchanged Phase10B loss, including exact
gradients. Corrected mode is opt-in and blocked by enabled/validated/approved
flags plus exact row/evidence bindings. A future authorized worker must call
`load_configuration`, pin/reload the original data manifest, and run
`validate_bundle` before loading images or taking any update. Current evidence
is empty, so this preflight refuses corrected training. No code in this phase
toggles those flags or changes the existing worker.

Trusted added negatives are Chair-specific, fully contained in reviewed regions
and outside positive/ignored/potentially positive regions. Source missing-label
flags never imply absence. Original focal/box coefficients and GT normalization
are preserved, so the proposed primary A/B changes only the trusted mask.

The review UI starts every item blank. It exports a new timestamped JSON after
explicit reviewer/date/decision/evidence/confirmation. `review.py validate`
accepts partial submissions into a fresh versioned private directory; it never
enables a config. See the report for localhost serving and submission commands.

`prepare.py` and `review.py build` are exclusive preparation entry points; they
refuse populated output directories. Preserve historical versions rather than
deleting files to repeat preparation. Synthetic test confirmations are kept in
memory and never exported as real evidence. Source labels and reviewed claims
remain distinct.
