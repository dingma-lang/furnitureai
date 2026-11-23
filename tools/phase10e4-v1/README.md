Phase10E.4 completed one authorized GPU recovery. No training was performed.

The immutable runtime is under
`data/open-images/experiments/phase10e3-recovery-v1/`. Do not relaunch: the
write-once launch marker and existing destination intentionally reject retries.
Scientific E3 sources are reused unchanged. E4 wraps only operation accounting
and atomic publication to enforce global limits. A CPU cache audit selected all
15 nonzero adapted-source groups from the24 permitted photographs;9 zero cases
were omitted without substitutions. Original correspondence needed one forward.

Live measurement used FP16 forward and the E3 FP32 observed-node adjoint at
scale1. `worker4.py` contains no optimizer construction/step or `.backward()`.
The31 actual VJPs include30 classification measurements and one native box probe.

Read the report and disabled candidate:

    reports/phase10e4-v1/report.md
    reports/phase10e4-v1/training-disabled.json

Read-only CPU verification:

    CUDA_VISIBLE_DEVICES='' .venv/bin/python -B tools/phase10e4-v1/verify4.py --check

`future4.py` is a separate,gated training entry point. Importing it or invoking
it without an explicit separate training-authorization file cannot construct a
model,initialize CUDA,create outputs or train. The fixed candidate uses the
tested `precision4.py:combined` helper. Its mandatory original59-photo replay and
one combined zero-update VJP precede optimizer construction. It saves full head,
optimizer,RNG/cursor and precision provenance at each epoch;partial-run recovery
would require a separately bounded recovery instruction rather than automatic
relaunch,which refuses to overwrite the directory.

Only after explicit future approval,the prepared command shape is:

    .venv/bin/python -B tools/phase10e4-v1/future4.py --authorization APPROVED_JSON_PATH

There is no such training authorization supplied by E4. Do not execute this
command now. Background supervision and600s hard timeout are in the launcher.
Future monitoring would use the new console log and update/checkpoint JSON;
creating a STOP file requests graceful termination before further updates.

No metrics were computed on the exposed19-photo panel,official holdouts or frozen
material in E4. No private photographs,datasets,raw arrays or model weights are
included in Git. The coefficient derives from training-only actual gradients,
not a score sweep. A successful gradient test is not a detection improvement.
