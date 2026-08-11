# Pass 205 RDGC DGX Runtime Amendment

Date: 2026-08-11

Status: prospective and binding before any Pass 205 RDGC scientific process.

## Purpose

This amendment corrects a non-scientific execution-host defect in the reviewed
Pass 205 RDGC candidate and implementation. The candidate copied Python
`3.12.3` from the local CPU calibration environment even though its designated
DGX execution environment was already Python `3.13.9`. The two requirements in
the reviewed CLI were therefore mutually unsatisfiable: it required both the
repository `.venv` interpreter and Python `3.12.3`, while the registered DGX
repository `.venv` is Python `3.13.9`.

No Pass 205 scientific process, checkpoint load, model construction, artifact
read, GPU computation, output publication, or attempt occurred before this
defect was found. Attempt 1 remains unused.

## Observed chronology

The live preflight on `spark-2751` observed, before checkout creation or any
scientific input access:

- interpreter: `/home/riomus/group-learning/.venv/bin/python`;
- Python: `3.13.9`;
- PyTorch: `2.12.1+cu130`;
- NumPy: `2.5.0`;
- CUDA available: `true`.

Repository evidence predating RDGC independently records the same designated
DGX interpreter:

- `docs/pass201_pa_source_prelaunch_manifest.json` records Python `3.13.9`,
  PyTorch `2.12.1+cu130`, NumPy `2.5.0`, and working directory
  `/home/riomus/group-learning`;
- `docs/pass201_pa_source_v2_closeout_2026-08-09.md` records the bound
  interpreter as Python `3.13.9`.

The Python `3.12.3` value came from the different local CPU environment
documented by
`docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md`. That historical
environment used `/home/rb/worktrees/sfora-emafactorial/.venv`, and its
completed Pass 200 validation receipt genuinely records Python `3.12.3`.
Therefore this is an authorship error, not DGX environment drift.

## Exact prospective correction

Only Pass 205 RDGC's prospective Python self-description changes:

1. `scripts/diagnose_pass205_rdgc_stage_b.py` must require
   `(sys.version_info.major, sys.version_info.minor,
   sys.version_info.micro) == (3, 13, 9)` at the public CLI boundary.
2. Its full and reduced result validator must require the exact built-in string
   `pre_import.python_version == "3.13.9"`.
3. RDGC tests and synthetic RDGC fixtures must use `"3.13.9"` for those same
   prospective fields.

PyTorch and NumPy remain observed runtime values with their existing exact type
checks. This amendment does not add version pins for them and does not change
their code paths. The designated DGX `.venv` must not be rebuilt, upgraded,
downgraded, or replaced.

The retrospective Pass 200 verifier and its fixtures are frozen and must not be
changed. In particular,
`scripts/verify_pass200_rsta_scientific_artifact.py` must continue to require
Python `3.12.3` for the already-completed Pass 200 recovery receipt.

## Scientific invariants

This amendment changes no candidate definition, loss, operator, control,
direction, contributor count, receiver selection, seed, artifact, checkpoint,
transform, tensor, bootstrap, threshold, decision rule, graph schedule,
floating-point arithmetic, hash rule, result schema apart from the literal
prospective Python version value, or atomic publication rule.

The reviewed Pass 205 candidate remains
`30d533e532d0f22c8b1e474987001685a4aa3488`. The reviewed scientific source
before this repair remains
`328a70ad809c1adeae9ccd2aea28b87f3243018b`. The reviewed but unexecuted
manifest-only handoff remains
`3c9e6b9fe5494f8f6a98caab18ff4923b66734ec`, with manifest SHA-256
`4d595c204781ed0dbcf824c9b3da9c8cc19b3d50f6fed481052969a424b99d57`.
It must never be executed after this amendment and must not be reclassified as
an attempt.

## Required TDD and assurance

Before source implementation:

- add a RED test that the RDGC CLI rejects a non-`3.13.9` interpreter;
- add a RED persisted result round-trip test requiring built-in string
  `"3.13.9"` in both full and reduced post-import forms;
- add a consistency test binding the CLI runtime tuple and result-validator
  literal to the same version;
- retain a test proving PyTorch and NumPy versions are observed nonempty
  built-in strings rather than newly pinned scientific inputs;
- retain tests proving the Pass 200 historical verifier still requires
  `3.12.3` and is byte-unchanged.

After minimal source changes, run the complete RDGC test file, Ruff,
`py_compile`, `git diff --check`, the repository assurance gate already used
for the reviewed source, and independent source review. The source commit may
change only the RDGC diagnostic and its test.

## Replacement authority chain

The replacement execution chain is exact and prospective:

1. this amendment is a sole-document child of the reviewed, unexecuted handoff
   `3c9e6b9fe5494f8f6a98caab18ff4923b66734ec`;
2. a bound runtime-repair plan is a sole-document child of this amendment;
3. the reviewed repair source is a merge-free sequence of commits changing
   only `scripts/diagnose_pass205_rdgc_stage_b.py` and
   `tests/test_diagnose_pass205_rdgc_stage_b.py`;
4. a replacement manifest-only handoff is the direct child of the final
   reviewed source;
5. the replacement manifest binds the same candidate, upstream RSTA,
   literature audit, outcome-blind validation receipt, historical artifacts,
   result schema, and 33 ordered scientific source paths, with only the new
   plan authority, source revision, and changed RDGC diagnostic digest/blob
   updated as required;
6. independent review must return READY before any DGX process is launched.

The replacement output path is derived only from the replacement handoff
commit. The output and owned temporary path must be absent. Exactly one
scientific process remains authorized. Every PASS, CLOSE, UNRESOLVED, INVALID,
structural failure after launch, signal, timeout, or publication failure
consumes attempt 1 and forbids a retry.

## Process lesson

An execution-environment contract must be authored from the host designated to
execute the run, not inherited from a different host that performed an
upstream calibration. Host identity, interpreter path, and version must be
checked together prospectively before a one-shot run.
