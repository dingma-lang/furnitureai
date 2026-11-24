# Phase 10E.12 — offline audit and safe refactoring

Full read-only inventory and AST inspection cover the workspace's first-party
Python in `src`, `scripts`, root test directories, `tools`, and archived code
under `data`. Environment packages, runtime/cache trees and the separate Mac
workspace are outside the audit. Detailed source/data/checkpoint fingerprints
remain local under `reports/phase10e12-v1/private/`, never uploaded to Git.

Read the phase report for actual findings and limits. Historical experiment
runners were not consolidated. Changes affect five reusable active modules,
one new regression-test module and the new audit tooling/report namespace.

Safe artifact verification, with no model or image loading:

```bash
cd /home/tristary/furniture-ai-ml
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e12-v1/verify12.py check
```

Optional full rehash of all29,082protected current files (about23GBlogical bytes,
with hardlinked snapshots), still without GPU/model/image decoding:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e12-v1/verify12.py check --full
```

The focused synthetic regression suite is independently runnable:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
.venv/bin/python -B -m unittest discover -s tests -p test_audit_regressions.py -v
```

Audit/measurement/before-after commands created immutable reports and must not
be rerun into the same filenames. `checks12.py` requires offline CPU flags,
excludes the two real tiny-transformer forward tests, uses a file-based main
for spawned FAISS workers and snapshots unittest IDs before fixtures can shadow
TestCase.id. Do not run an unrestricted repository-wide tools discovery: legacy
phase tests have hash-pinned sources, lifecycle assumptions and writable fixture
roots. Use a fresh report namespace for future audits. Missing viewer/catalogue
fixtures and absent optional dependencies are documented, not silently fabricated
or skipped to declare the whole suite green.
