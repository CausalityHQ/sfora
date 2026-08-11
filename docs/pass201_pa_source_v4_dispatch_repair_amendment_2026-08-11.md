# Pass 201 ordinary-PA source-v4 dispatch repair amendment — 2026-08-11

## Status and scope

This is a prospective implementation-defect repair for the already-completed
Pass 201 ordinary Proxy Anchor source run. It does not alter the candidate,
dataset, source-training configuration, checkpoint, report, scientific
arithmetic, contexts, operators, thresholds, bootstrap, or decisions. It does
not authorize another source-training process or any GPU process.

The only authorized scientific continuation is the original CPU Pass 201 CIS
operator diagnostic, after the repaired source binding has been independently
reviewed and a new manifest-only handoff has been frozen. The completed source
outputs are immutable inputs. They may be copied byte-for-byte into a fresh
checkout, but never regenerated, rewritten, normalized, or selected again.

## Immutable completed-run evidence

The historical producer domain is fixed exactly as follows:

```text
historical source commit S4: 53a9db9e9dbe54fcebb33769b915c3f33699d522
historical handoff commit H4: 32c4d39322fca2a5a906f785bdb612dcd7008647
H4 preservation ref: pass201-source-v4-handoff-32c4d39
H4 manifest path: docs/pass201_pa_source_v4_authorization_manifest.json
H4 manifest SHA-256: 080adaeaaa5c7bf9c87ed93761d6e4c517b958bb60c49af68a880109f5abce1f
H4 manifest Git blob: 430f340a17cc32c5fd239083b1a0dba98e09ad7c
remote checkout: /home/riomus/pass201-pa-source-v3-03d0ed5
remote host: riomus@spark-2751
receipt path: reports/generated/pass201_source_v3/run-v3/receipt.json
receipt SHA-256: a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9
receipt bytes: 15179
receipt schema: pass201-pa-source-v4-receipt-v1
source child exit code: 0
candidate_values_computed: false
```

The six completed output files are exact:

```text
reports/generated/pass201_source_v3/run-v3/checkpoint.pt
  bytes 55760186
  sha256 e42d25b4e8e98f1d619aada2215ecbfeca579327dabaf6f09f02151183220696
reports/generated/pass201_source_v3/run-v3/receipt.json
  bytes 15179
  sha256 a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9
reports/generated/pass201_source_v3/run-v3/report.json
  bytes 250481
  sha256 a74a2a1c08beee2a0b4d67adcf378421309077c3ce39c93711f782f0589cc843
reports/generated/pass201_source_v3/run-v3/resolved_config.json
  bytes 5981
  sha256 bd5d1f0216b8c55d70a7ac4bd528fb5545d66472e964abd321bba3877079584e
reports/generated/pass201_source_v3/run-v3/train_manifest.json
  bytes 2895656
  sha256 c60e998e802f2b1050fc3c934ce9960c27c7ec4e598b0d88acd8062295c3def9
reports/generated/pass201_source_v3/run-v3/training.log
  bytes 8757
  sha256 053a7dc0b447f6bfeabf7dac347d80b0b889e94db7688eeebaf94e02f5f4d1d2
```

The source run used one controller and exactly one registered training child.
The child completed 8,580 steps and 60 epochs. These facts and the output
relations were independently reviewed before either activation attempt. GPU
non-bitwise reproducibility is irrelevant to this repair because no source run
is repeated.

## Exact defect and failed-attempt chronology

The frozen public controller requires the literal source-v4 manifest path, but
`_validate_source_binding` dispatches only the exact source-v3 schema to the
source-v3/v4 authority validator. The source-v4 schema therefore falls through
to the obsolete source-v2 SHA gate and fails with
`prelaunch manifest SHA-256 mismatch`. Passing
`--expected-prelaunch-sha256` is forbidden: it would remain in the source-v2
body and skip the source-v4 chain, process-entry authorities, canonical
contract, and completed-receipt relations.

The chronology after the source receipt became READY is exact:

1. A first CPU activation process omitted `--dataset-root` and stopped before
   source activation, candidate construction, or any output write.
2. Absence of the activated preregistration, activated source manifest, result,
   and owned temporary paths was confirmed.
3. A second CPU activation process supplied the exact dataset root
   `/home/riomus/datasets/inshop_official_standard` and stopped at the
   source-v4 dispatch defect above.
4. The same output and temporary-path absences were confirmed. No smoke,
   scientific role, candidate value, result, GPU process, or source-training
   retry followed.

These are structural pre-candidate failures. They remain disclosed and are not
reclassified as valid attempts. A fresh activation after this prospective
repair is allowed because neither process reached source activation or
candidate computation.

## Why a new source revision is necessary

The unmodified public call graph has no safe wrapper seam:

- `activate_source` immediately calls `_validate_source_binding`;
- `_validate_controller_binding` immediately calls `_validate_source_binding`;
- `run_controller` calls `_validate_controller_binding`; and
- process-role mode cannot replace public activation and controller binding.

A new-file wrapper would therefore have to call private functions, monkeypatch
the frozen module, duplicate scientific orchestration, or bypass H4. All four
are forbidden. The diagnostic itself must receive the narrow defect repair.

The historical executing worktree cannot be reconstructed after the fact. A
fresh checkout of H4 would only materialize the same Git objects and must not be
described as proof of the historical runtime worktree. The historical evidence
is instead exactly: H4 and S4 Git topology and blobs, the immutable canonical
receipt, and byte-consistency of all six outputs with that receipt. The current
executor evidence is separate: independent review of the complete S4-to-V5
diff plus H5/V5 Git/worktree authentication. Runtime self-checks do not replace
that independent review.

## Required Git topology

Before creating the repair branch, H4 must remain reachable through the exact
local preservation ref `pass201-source-v4-handoff-32c4d39` resolving to H4.
The repair chronology is linear from S4, not H4:

```text
S4 -> amendment A5 -> plan P5 -> reviewed source V5 -> manifest-only H5
H4 is the separate sole-manifest child of S4 and remains preserved by the tag.
```

The final amendment commit changes only this amendment. The final plan commit
changes only its implementation plan. Every source or test commit from the
final plan through V5 has one parent, changes a nonempty subset of exactly:

```text
scripts/diagnose_pass201_cis_operator.py
scripts/pass201_pa_source_v2_contract.py
tests/test_diagnose_pass201_cis_operator.py
```

and the aggregate final-plan-to-V5 source/test set is exactly those three
paths. Contract changes are confined to explicit v5 schema/key/output
validation and the explicit prohibition on a v5 source-training receipt. H5
has one parent V5 and adds exactly one regular mode-100644 file:

```text
docs/pass201_pa_source_v5_authorization_manifest.json
```

The v5 source-chain validator must explicitly allow the docs-only amendment/plan
ancestors, require the reviewed source segment to be merge-free, reject empty
or out-of-scope source commits, and require the aggregate three-path set. It must
not route H4 through the current-source chain.

## Two provenance domains

### Historical producer domain

The v5 validator authenticates the literal H4 and S4 above, never a caller-
chosen substitute. It requires:

- H4 has the sole parent S4;
- H4's sole diff is `A` mode `100644` at the exact v4 manifest path;
- the H4 manifest Git bytes, SHA-256, and Git blob equal the literals above;
- its `source_commit` and required parent are exactly S4;
- every ordered historical source row equals the exact S4 Git blob and digest;
- all static protocol, plan, process-entry amendment, plan, and evidence
  authorities equal their frozen commits and bytes;
- the receipt is a regular non-symlink file at the exact path, has the literal
  SHA-256/byte count/schema above, is canonical strict JSON, and validates under
  the exact historical contract;
- the receipt's authorization commit/source commit/manifest path/SHA/blob are
  exactly H4/S4/the H4 manifest values above; and
- each of the six regular output files has the exact path, size, and SHA-256
  above and equals its receipt evidence.

Historical source rows are checked against Git objects, not the current V5
worktree. No historical worktree-equality claim is made.

### Current executor domain

The same process separately authenticates H5 and V5:

- the checkout is detached and tracked-clean;
- `HEAD` is H5, H5 has sole parent V5, and H5's sole diff is the exact v5
  manifest addition;
- the v5 manifest worktree bytes equal the H5 Git blob;
- its `source_commit` and required parent are exactly V5;
- every ordered current source row equals both the V5 Git blob and current
  worktree bytes, including the executing diagnostic;
- `Path(__file__)` resolves to the exact current diagnostic worktree path;
- the final amendment and plan are exact ancestors of V5 with exact
  path/SHA/commit bindings; and
- the reviewed final-plan-to-V5 source/test chain has the exact scope above.

There is no self-cycle: V5 does not contain H5 or H5's digest, and H5 binds V5.
The load-bearing guarantee that the repair does not alter scientific behavior
is the prospective amendment plus independent complete diff review.

## Source-v5 authorization manifest

The v5 authorization manifest is canonical UTF-8 JSON with a final LF,
`allow_nan=false`, and exact recursive key/list order. Its top-level keys are,
in order:

```text
authorization
controller
dataset
execution
historical_producer
outputs
plan
postconditions
process_entry_amendment
process_entry_evidence
process_entry_plan
protocol
purpose
repair_amendment
repair_plan
schema_version
sidecars
source
source_commit
status
```

All pre-existing v4 domains except `authorization`, `outputs`, and the three new
authority domains retain their exact nested schemas and semantics. The
following values change prospectively:

- `schema_version` is exactly `pass201-pa-source-v5-activation-v1`;
- `purpose` is exactly `activate_completed_source_v4_then_run_cpu_diagnostic`;
- `source_commit` and `authorization.required_parent_commit` are V5;
- `authorization.manifest_path` and its sole required diff path are the exact v5
  manifest path;
- `authorization.frozen_absence` records freeze-time absence only for the
  activated preregistration, activated source manifest, smoke output, and final
  result; completed source outputs and the run directory are excluded;
- `source.files` retains the exact historical 30-path order below but binds V5
  Git/worktree bytes;
- `historical_producer` is the exact immutable H4/S4/receipt/output block; and
- `outputs` records the six completed source files and run directory as
  required-present-at-execution immutable inputs, plus required-absent-at-freeze
  activation, source-manifest, smoke, and result paths.

`authorization.frozen_absence` has exactly these keys in order, each with the
literal value `ENOENT`:

```text
activated_preregistration
result
smoke
source_manifest
```

Its sibling `frozen_absence_checked_utc` is one literal RFC3339 UTC string
captured once and passed identically to both freezer processes. It is evidence
about the four future outputs only, never a runtime predicate about completed
source files.

`outputs` has these keys in order:

```text
activated_preregistration
checkpoint
log
receipt
report
resolved_config
result
run_directory
smoke
source_manifest
train_manifest
```

The exact future-output paths are:

```text
activated_preregistration: docs/pass201_cis_operator_activated_preregistration.json
source_manifest: docs/pass201_cis_operator_source_manifest.json
smoke: reports/generated/pass201_cis_operator/pass201_inshop_seed0_smoke.json
result: reports/generated/pass201_cis_operator/pass201_inshop_seed0.json
```

Each of those four objects has `path`, `required_absent_when_frozen` in order,
with the boolean exactly `true`. Each of `checkpoint`, `log`, `receipt`,
`report`, `resolved_config`, and `train_manifest` has `path`, `bytes`, `sha256`,
`required_present_at_execution` in order, using the immutable values above and
the boolean exactly `true`. `run_directory` has `path`,
`required_present_at_execution` in order and binds exactly
`reports/generated/pass201_source_v3/run-v3` and `true`. The v4 key
`run_directory_required_absent` is not valid in v5 and has no default meaning.

`historical_producer` has these keys in order:

```text
authorization_commit
source_commit
manifest
receipt
outputs
```

`manifest` has `path`, `bytes`, `sha256`, `git_blob` in order. `receipt` has
`path`, `bytes`, `sha256`, `schema_version`, `candidate_values_computed` in
order. `outputs` has exactly `checkpoint`, `log`, `report`, `resolved_config`,
`train_manifest` in that order; the receipt is represented only by `receipt`.
Each output object has `path`, `bytes`, `sha256` in order.

`repair_amendment` and `repair_plan` each have `path`, `sha256`, `commit` in
that order. Their exact values bind the final docs-only amendment-fix and
plan-fix commits that immediately precede source work.

The exact current `source.files` path order is:

```text
scripts/diagnose_pass201_cis_operator.py
scripts/pass201_pa_source_v2_contract.py
src/sfora/__init__.py
src/sfora/ablation.py
src/sfora/api.py
src/sfora/arcg.py
src/sfora/benchmark.py
src/sfora/bn_inception.py
src/sfora/catalog.py
src/sfora/cea.py
src/sfora/cem.py
src/sfora/cli.py
src/sfora/compose.py
src/sfora/data.py
src/sfora/encoder_ablation.py
src/sfora/encoder_training.py
src/sfora/evaluation.py
src/sfora/experiments.py
src/sfora/image_benchmark.py
src/sfora/image_end_to_end.py
src/sfora/image_recipes.py
src/sfora/ipsr.py
src/sfora/losses.py
src/sfora/method.py
src/sfora/oapf.py
src/sfora/publication.py
src/sfora/remote.py
src/sfora/report.py
src/sfora/text_baselines.py
src/sfora/training.py
```

Unknown schemas, missing/extra/reordered keys, wrong concrete types, duplicate
keys, nonfinite JSON numbers, path aliases, symlinks, Git topology drift,
digest drift, receipt/output drift, and any default/fallthrough dispatch are
structural failures before candidate computation.

## Narrow source repair

The public source dispatcher accepts exactly:

```text
pass201-pa-source-v3-prelaunch-v1 -> existing v3 path
pass201-pa-source-v5-activation-v1 -> exact v5 dual-provenance path
```

The defective v4 public path remains rejected, as do v2, unknown, malformed,
and caller-overridden schemas. The public CLI already exposes no SHA-override
flag. The v5 path must never consult the internal
`expected_prelaunch_sha256` legacy/test attribute fallback.

The repair may change only source/provenance/serialization plumbing needed for:

- exact historical H4 receipt rebinding;
- exact current V5/H5 execution binding;
- explicit contract support for the exact v5 activation manifest with no
  schema, output, or receipt fallthrough;
- source-manifest schema `pass201-source-v2`;
- the activated-preregistration source projection;
- downstream controller/process/result provenance projection; and
- strict public CLI and real-Git regression coverage.

The repair must not change candidate construction, model/checkpoint loading,
context selection, operators, tensor arithmetic, tolerances, thresholds,
bootstrap, aggregation, decision logic, or output path.

The contract validates the v5 activation authority as its own explicit branch.
It must not derive a v5 source-training receipt schema because no v5 source
training exists or is authorized. Calling complete-receipt validation with a
v5 activation authority fails explicitly. The immutable
`pass201-pa-source-v4-receipt-v1` is validated only against a separately
constructed and authenticated historical v4 authority from the literal H4
manifest; it is never validated against or rebound to the v5 executor
authority.

## Activated source-manifest v2

The recovered activation uses schema `pass201-source-v2`. It retains all v1
fields and their values, except that `diagnostic_source_sha256` is the current
V5 diagnostic digest, and adds one exact `activation_repair` object. That object
has these keys in order:

```text
historical_authorization_commit
historical_source_commit
historical_manifest_path
historical_manifest_sha256
historical_receipt_path
historical_receipt_sha256
executor_authorization_commit
executor_source_commit
executor_manifest_path
executor_manifest_sha256
executor_diagnostic_sha256
```

The historical fields bind H4/S4 and the immutable receipt. The executor fields
bind H5/V5 and the current diagnostic. `source_revision` remains S4 because the
checkpoint/report source was produced by S4; it must never be silently
redefined as V5. Every activated preregistration, controller replay, process
role, result source projection, and strict validator carries and validates the
same `activation_repair` object. Existing runtime version strings remain the
frozen exact values used by the current source-v3 path; this repair does not
derive them from a mutable ambient environment.

## TDD and review requirements

Before production edits, tests must prove RED for:

- real public `main([...])` activation with a valid v4 manifest reaching the
  obsolete SHA failure;
- exact v5 dispatch absence;
- H4 receipt rebinding and H5 current-executor binding;
- v1/v2 source-manifest branch separation; and
- activated/controller/process/result propagation of both provenance domains.

Tests then require GREEN for:

- temporary real Git histories with exact A5/P5/V5/H5 and separate H4 topology;
- every historical/current commit, edge status/mode/path, blob, SHA, worktree,
  receipt, and output relation;
- wrong H4, S4, V5, H5, receipt, output, source row, authority, path, order,
  type, and schema mutations;
- exact public CLI paths with no SHA override;
- no import/model/checkpoint/candidate access before both provenance domains
  validate;
- source-manifest v2 exact recursive mutation coverage;
- persisted activation JSON reload and replay equality;
- unchanged scientific constants, arithmetic functions, thresholds, operator
  order, context counts, and result decision functions; and
- exact no-clobber/rollback behavior for activation and result paths.

Run focused RED/GREEN, the complete diagnostic and contract test files, the
repository's relevant assurance suite once after the diff stabilizes, Ruff,
`py_compile`, and `git diff --check`. Independent review must examine the
complete final-plan-to-V5 diff and report no Critical or Important finding
before H5 is built.

## H5 freeze and execution rules

From independently reviewed V5, create two fresh freezer processes that write
distinct absent candidate paths and require byte-identical v5 manifests. The
freezer may inspect only authority/source/data metadata and the already-known
source-output hashes; it computes no candidate value. Exclusively publish one
candidate and commit only the v5 manifest as H5. Revalidate H5 in a fresh
detached clean checkout.

In that H5 checkout, copy the six immutable source outputs from the preserved
H4 execution directory to their exact repository-relative paths. Before any
JSON parse or checkpoint deserialization, require regular non-symlink files,
exact sizes and SHA-256 values, no owned temporary path, and a tracked-clean
checkout. Copying does not create a second source run.

The CPU sequence uses `CUDA_VISIBLE_DEVICES=""`, the frozen thread settings,
and the registered Python/Torch/NumPy environment:

1. one fresh `--activate-source` process;
2. one fresh `--binding-only` process;
3. one fresh `--smoke-only` controller process, which runs the registered
   integrity roles and computes no scientific candidate result; and
4. only if all prior predicates are GREEN, one fresh `--scientific` controller
   process.

Each invocation re-authenticates the v5 manifest, H5/V5, H4/S4, receipt, and
outputs. Activation, binding-only, and smoke are not scientific attempts. The
scientific controller is the sole prospective candidate attempt. Stop with no
retry on any structural, integrity, timeout, resource, or validation failure.
No CUDA fallback, source-training rerun, threshold change, alternate manifest,
private-function call, monkeypatch, runtime factory, or SHA override is
authorized.

On success, strictly reload the final result and authenticate exact schema,
provenance, replay, candidate, aggregate, bootstrap, and decision relations
before reporting PASS, FAIL, or UNRESOLVED. No further GPU action is authorized
by this amendment.
