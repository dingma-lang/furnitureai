# Historical source bindings — important replay clarification

The immutable Phase1B/1C/1D/2A/2Brights/preservation snapshots also record SHA256
values for the five active `src/furniture_bench` modules refactored in Phase10E.12.
All **25bindings across five snapshots** match the original code available in
parent commit `5818ab333dfd239d581129e02af93238bc85e191`.

Those old records are preserved byte-for-byte and were not rebound to accept new
code. A strict legacy checker that compares those source hashes against the live
checkout will therefore need the matching original source commit; it should not
be expected to accept this refactoring branch. This is distinct from the passing
later experiment seals and the full unchanged data/checkpoint/result comparison.

The refactor's valid-input CPU outputs matched exactly in the reported checks,
but exact source hashes intentionally differ. Historical results remain tied to
their original source, data, checkpoints and environment. GPU re-execution was
not performed. With a clean working tree, switching back to
`experiment/phase10e11-wellcome-suitability` restores the matching source without
changing datasets or weights; do not weaken old integrity gates or edit their
recorded fingerprints. `historical-source-replay.json` contains the verified
bindings and snapshot hashes.
