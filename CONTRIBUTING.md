# Contributing

This benchmark has a frozen 2026-09-23 result. Please keep its source files,
per-item outputs, and result JSON immutable. Corrections should be documented
with a new version and a changelog entry, not silently applied to old scores.

For a new model or dataset:

1. Open an issue describing the question, data rights and intended metric.
2. Freeze the evaluation set and protocol before viewing outputs. Record the
   selection rule, sample count, source revision, checksum and model API setup.
3. Save per-item outputs, validity failures, run manifest, prices and timing
   provenance. Separate provider failures from model answer failures.
4. Compare candidates on identical item IDs. Report denominators, confidence
   intervals and null results; label post-hoc analyses as exploratory.
5. Regenerate figures and run `python3 -m scripts.verify_release` plus tests.

The primary score is designed for these five public decision datasets. New
benchmarks should publish both their own task scores and a clearly versioned
overall definition. Do not retrofit them into the frozen 2026-09-23 score.

The project code is MIT licensed. Contributions should not add third-party
text without checking its redistribution terms; see
[DATA_PROVENANCE.md](DATA_PROVENANCE.md).
