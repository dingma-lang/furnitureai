# Phase 10A: expanded-data training preflight

Use `.venv/bin/python -B`. No entry point constructs a detector, calls a model,
invokes an optimizer, installs a package or downloads a dataset. The CPU tests use
synthetic tensor gradients, the existing local image processor on synthetic
pixels, and CPU loading of the preserved Phase 5C head delta. `photo_check.py`
decodes the exact120 historical training photos for integrity only.

New artifacts are under `reports/phase10a-v1/`. Raw manifests, licence evidence
joins, annotation coordinates and backup inventories remain ignored. Shared
reports contain aggregate counts only. Source annotations and partitions stay
unchanged; the two inner partitions are a new proposal within historical train.

The conservative policy prepares Sofa/Chair/Bed positives. Table/Desk/Cabinet/Lamp
remain inactive because source-native units and broad project labels are not
interchangeable. One additional OI Bed marked IsInside is inactive. Unknown,
filtered, group, crowd, unresolved and nonexhaustive targets are never generic
background. Native explicit negatives only apply to exactly bound safe classes;
no parent/alias negatives or objectness loss are enabled. Instance masks are
preserved as provenance, not trained by OWLv2's box head.

`expanded.py` supplies the checked training-only path loader, square-padding box
transforms, head-only configuration, strict delta loading and partial-label loss.
`recovery.py` supplies durable budget/resume bookkeeping, tested using temporary
CPU fixtures. A separately approved execution worker must bind these pieces to
real GPU updates, enforce cumulative runtime/VRAM and validate complete snapshots.
**The historical Phase 5C worker cannot be reused unchanged:** it hardcodes27
photos,18 development photos,70 updates and a different loss. No runnable expanded
trainer or permission to start one is supplied by Phase 10A.

Completed preparation entry points are exclusive writers. Do not delete outputs
or rerun them to force a fresh audit. Only tests and read-only verification may
be repeated on this version:

```bash
.venv/bin/python -B -m unittest discover -s tools/phase10a-v1 -p 'test_*.py' -v
.venv/bin/python -B -c "import sys; sys.path.insert(0,'tools/phase10a-v1'); from inventory import verify; print(verify())"
```

The inventory covers all regular files under `data/` and `models/`, including
320 checkpoint/tensor candidates. It hashes frozen files only for backup
integrity, without decoding or scoring them. Checksums are not independent
backups. No Mac application directory is accessed or modified.

Proposed backup, requiring separate destination approval: a mounted encrypted
external drive at `/mnt/e/FurnitureAI-backups/phase10a-20261118/`, with at least20
GiB free. No `/mnt/e` drive is mounted here; verify the mount first, never create
that path on the WSL root by mistake. Use `backup-paths.txt` with rsync's relative
file-list mode, no `--delete`, resumable partial transfers, restrictive access,
and verify every destination SHA256 against `backup-inventory.json`. Include a
Git bundle of all branches/tags and this local inventory. Record a backup as
verified only after destination-side checks. No copy/upload has been performed.

GitHub HTTPS authentication is absent, and neither `gh` nor Git Credential
Manager is installed. If the user chooses to install Ubuntu's GitHub CLI package
in their interactive terminal, browser-based authorization and push commands are:

```bash
# Optional user-approved CLI installation; not performed by this phase.
sudo apt-get update
sudo apt-get install gh
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
cd /home/tristary/furniture-ai-ml
git push -u origin experiment/phase9-multiview-detection
```

No token command or insecure-storage flag is used. Do not share credentials in
Codex. See https://cli.github.com/manual/gh_auth_login for the browser flow.

After committing work, rollback code using `git switch main` (original baseline)
or `git switch experiment/phase9-multiview-detection` (Phase9 implementation).
No weights/defaults changed, so no model rollback is required. Keep ignored
artifacts. Future recovery must verify config/data/code/processor/model bindings,
restore only a committed snapshot, and refuse uncertain reserved/applied update
boundaries; never delete budget records or replay an uncertain optimizer call.
