# Pass 201 Source-v4 Dispatch Repair Implementation Plan

> **Execution:** Use `superpowers:executing-plans`,
> `superpowers:test-driven-development`, and
> `superpowers:verification-before-completion`. Apply review feedback with
> `superpowers:receiving-code-review`.

**Goal:** Repair the unreachable Pass 201 source-v4 public binding path without
rerunning source training, bind the immutable H4/S4 receipt separately from the
new V5/H5 executor, and complete the registered CPU diagnostic once.

**Architecture:** V5 branches from S4 through this amendment and plan, changes
only the diagnostic and its test, and implements an exact source-v5 authority
schema. H5 is the sole-manifest child of V5. The v5 manifest binds historical
H4/S4/receipt/output Git-and-byte evidence and current H5/V5 Git/worktree
evidence as separate domains. The existing public activation/controller path
then emits source-manifest v2 and runs unchanged integrity/scientific logic.

**Authority:**

```text
path: docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md
sha256: 6285950251ce8166b0bc727248def9304df7d386b95dd1ce5e20e6b946e91a8a
commit: d4a2df3
historical S4: 53a9db9e9dbe54fcebb33769b915c3f33699d522
historical H4: 32c4d39322fca2a5a906f785bdb612dcd7008647
H4 preservation ref: pass201-source-v4-handoff-32c4d39
historical receipt sha256: a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9
```

## Global constraints

- Never edit, move, stage, or remove `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`,
  or `RSPG_TASK.md`.
- Never rerun source training or launch a GPU process.
- Never pass or expose `--expected-prelaunch-sha256` on the public repaired
  path. Unknown schemas fail closed; no default/fallthrough branch is allowed.
- Never describe a fresh H4 checkout as evidence of the historical runtime
  worktree. Historical evidence is Git topology/blobs plus canonical receipt
  and output bytes.
- No candidate construction, model/checkpoint load, or output write occurs
  until both historical and current provenance domains validate.
- The source/test aggregate from this plan commit through V5 is exactly:
  `scripts/diagnose_pass201_cis_operator.py` and
  `tests/test_diagnose_pass201_cis_operator.py`.
- H5 changes only
  `docs/pass201_pa_source_v5_authorization_manifest.json` with status `A` and
  mode `100644`, and `H5^ == V5`.
- The real six source outputs remain immutable. Restage them only by byte copy
  from the preserved remote directory and validate exact size/SHA before parse
  or deserialization.
- Activation, binding-only, and smoke are pre-scientific gates. Exactly one
  scientific controller process is authorized after all three are GREEN.
- Use one durable Opus→Sol review consultation per completed docs/source/H5
  boundary. Use Fable→Opus→Sol only for a genuinely new research/design issue.
  Do not duplicate or replay consultations.

## Task 1: Independently review the repair authority

**Files:**

- Review `docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md`.
- Review this plan.
- Read the diagnostic/test and H4 manifest only; do not execute science.

- [ ] Verify `d4a2df3` is a one-file docs-only child of S4 and the amendment
  Git/worktree SHA is the literal above.
- [ ] Verify this plan is its sole-path child and substitute its final commit
  and SHA into source constants/tests before RED work.
- [ ] Verify the preservation tag resolves exactly to H4 and H4^ is S4.
- [ ] Obtain a read-only Opus→Sol review of topology, dual provenance, exact
  v5/source-v2 schemas, receipt rebinding, no-science boundary, and execution
  sequencing. STOP and repair docs-only if any Critical/Important issue exists.

## Task 2: Establish public-dispatch and Git-provenance RED

**Files:**

- Modify `tests/test_diagnose_pass201_cis_operator.py` only.
- Protect production source, manifests, outputs, activation files, and results.

- [ ] Add a real public `main([...])` activation regression using a temporary
  repository and exact v4 schema. Against old source, require the observed
  `prelaunch manifest SHA-256 mismatch`; prove no activation/result/temp file.
- [ ] Add an exact v5 public-dispatch test. Against old source, require failure
  because v5 is unrecognized, not because a synthetic SHA override is absent.
- [ ] Build a real merge-free temporary Git history with separate H4 and
  A5→P5→V5→H5 branches. Do not mock Git command helpers. Add mutations for
  every commit, parent, edge status/mode/path, source row, manifest bytes/order,
  and detached/clean predicate.
- [ ] Add historical receipt/output fixtures using the real strict contract
  shape and exact authorization relations. Mutate H4, S4, manifest SHA/blob,
  receipt schema/hash/authorization, every output path/size/hash, and
  `candidate_values_computed`.
- [ ] Run the narrow selector and preserve exact RED evidence before production
  edits.

## Task 3: Implement source-v5 authority and exact dispatcher

**Files:**

- Modify `scripts/diagnose_pass201_cis_operator.py`.
- Modify `tests/test_diagnose_pass201_cis_operator.py`.

- [ ] Add exact static bindings for A5/P5, H4/S4, v4 manifest metadata,
  receipt metadata, five non-receipt output records, v5 manifest path/schema,
  and exact 30-source path order.
- [ ] Add a v5 source-chain validator: A5/P5 must be exact ancestors; each
  source commit after P5 is merge-free, nonempty, `M`-only, and confined to the
  exact two paths; aggregate scope is both paths.
- [ ] Generalize the existing handoff authenticator only by manifest path and
  expected v5 schema, preserving detached/clean, sole-parent, sole-`A`-manifest,
  mode, Git/worktree, and parent relations.
- [ ] Implement exact recursive v5 manifest validation. Reuse unchanged v4
  domain validators where semantically identical; validate
  `historical_producer` independently against literal H4/S4 Git objects and the
  strict canonical receipt/output bytes.
- [ ] Rebind the receipt only to the literal historical H4 handoff. Never accept
  an arbitrary declared commit or compare it to H5.
- [ ] Authenticate all current source rows against V5 Git blobs and current
  worktree bytes, including `__file__`.
- [ ] Dispatch exactly v3 to the existing v3 path and v5 to the new dual-domain
  path. Reject v4 public activation with an explicit repair-required error and
  reject every other schema. Remove all use of the SHA override from this
  public branch without changing legacy tests that intentionally cover legacy
  behavior.
- [ ] Run the Task 2 selector GREEN plus exhaustive recursive v5 mutations.

## Task 4: Implement source-manifest v2 and provenance propagation

**Files:** Same two files.

- [ ] Add exact `pass201-source-v2` validation with all v1 fields plus the
  ordered `activation_repair` object frozen by the amendment. V1 remains valid
  only on historical non-recovery paths; v5 activation emits v2 only.
- [ ] Keep `source_revision == S4`; bind the current V5 diagnostic through
  `diagnostic_source_sha256` and `activation_repair` rather than silently
  redefining checkpoint provenance.
- [ ] Update activated-preregistration source projection, controller replay,
  process-role validation, result-source projection, and strict result
  validation in lockstep. Preserve all runtime-version values and scientific
  fields unchanged.
- [ ] Add exact recursive removal/addition/order/type/value/digest mutations for
  both v1 and v2 branches. Add valid v2 persisted JSON roundtrip and replay
  equality tests.
- [ ] Prove all child roles reject missing or drifted dual provenance before
  model/checkpoint/candidate access.

## Task 5: Public CLI, atomicity, and scientific noninterference

**Files:** Same two files.

- [ ] Add real `main([...])` tests for `--activate-source`, `--binding-only`,
  `--smoke-only`, and `--scientific` argument/path pinning. There is no public
  SHA override, runtime factory, alternate manifest, or alternate output path.
- [ ] Prove each invocation re-authenticates H5/V5 and H4/S4/receipt/outputs.
- [ ] Prove authority failures produce no activation/result/temp file and no
  Torch/model/checkpoint/candidate access.
- [ ] Re-run existing no-clobber, rollback, and persisted-output tests. Add
  destination sentinels for activation and result paths.
- [ ] Add byte/function-source assertions proving the repair does not change
  constants, context counts, operators, tensor arithmetic, thresholds,
  bootstrap, aggregation, or decision functions.
- [ ] Run the complete diagnostic test file, then Ruff, `py_compile`, and
  `git diff --check`.

## Task 6: Commit and independently review V5

**Files:** Exact diagnostic/test pair only.

- [ ] Review the complete P5-to-worktree diff and cached diff; require no other
  tracked path and preserve the three protected untracked files.
- [ ] Commit the exact pair as `fix Pass201 source-v4 dispatch provenance`.
- [ ] Re-run the focused provenance/CLI suite and full relevant assurance from
  the committed tree.
- [ ] Start one Opus→Sol read-only review with exact amendment, plan, S4/H4,
  receipt metadata, V5, full diff, tests, and no GPU/artifact-inspection scope.
- [ ] Fix every Critical/Important finding with RED→GREEN in a separate commit,
  rerun proportional gates, and repeat review until READY. Final reviewed source
  revision is V5.

## Task 7: Freeze manifest-only H5

**Files:**

- Create `docs/pass201_pa_source_v5_authorization_manifest.json` only after V5
  review.

- [ ] Add future-manifest tests before V5 review completes: exact top/nested
  order, literal A5/P5 and H4/S4/receipt/output bindings, exact 30 source rows,
  current V5 hashes, required-present source outputs, required-absent activation
  and result paths, and `candidate_values_computed=false`.
- [ ] In a fresh detached V5 checkout, run exactly two fresh freezer processes
  to distinct absent paths. They may read metadata/hashes only. Require
  byte-identical canonical candidates and strict validation.
- [ ] Exclusively publish one candidate, commit only the v5 manifest as
  `refreeze Pass201 source-v5 activation`, and require `H5^ == V5`.
- [ ] In a fresh detached H5 checkout, validate H5/V5 current provenance,
  H4/S4 historical Git evidence, amendment/plan bindings, source rows, manifest
  schema, and absence predicates. Obtain one final Opus→Sol review. STOP on any
  Critical/Important finding.

## Task 8: Stage immutable outputs and run the CPU diagnostic

**Files:** Runtime-only copies at the exact six source paths, activation files,
smoke output, and final result path.

- [ ] Use a fresh detached clean H5 checkout. Authenticate the registered venv,
  Python/Torch/NumPy versions, dataset root, exact thread environment, and
  `CUDA_VISIBLE_DEVICES=""`. Require no GPU compute process.
- [ ] Copy all six outputs byte-for-byte from
  `riomus@spark-2751:/home/riomus/pass201-pa-source-v3-03d0ed5` into the exact
  H5-relative paths. Before parse/deserialization, require regular non-symlink
  files, exact sizes/SHA values from the amendment, no owned temp, and a
  tracked-clean checkout.
- [ ] Confirm activation, source-manifest, smoke, final-result, and owned-temp
  paths are absent. Preserve the two prior structural failure disclosures.
- [ ] Run one fresh `--activate-source` process with exact dataset root. Strictly
  reload and validate both activation files and dual provenance.
- [ ] Run one fresh `--binding-only` process. It must exit zero and write
  nothing.
- [ ] Run one fresh `--smoke-only` controller. Authenticate all process records,
  exact input/context/action hashes, CPU device/build evidence, and registered
  replay tolerances. STOP on any failure; do not run science.
- [ ] Only after GREEN, run one fresh `--scientific` controller. Monitor the
  original PID only; no retry or duplicate. On completion, strictly reload and
  validate the result and all provenance/replay/scientific relations.
- [ ] Report exact process IDs/exits/times, output paths/SHA values, and final
  PASS/FAIL/UNRESOLVED result. No additional GPU or training action is
  authorized.

## Stop conditions

STOP without workaround or retry if any authority commit/blob/path/order/type,
receipt/output hash, current worktree byte, detached/clean state, environment,
activation replay, integrity record, atomic publication, resource limit, or
scientific validation differs. A new attempt after any post-activation failure
requires a new prospective amendment and explicit operator authorization.
