# Pass 200 RSTA Adjoint-Integrity Prefix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair only the RSTA adjoint inner-product reductions and enforce complete four-seed integrity before any scientific candidate scoring.

**Architecture:** A structured adjoint helper owns the frozen directions, FP32 JVP/VJP, float64 inner products, hashes, and exact audit object. A new candidate-free CLI runs that audit for all four seeds, while scientific execution becomes a two-phase integrity-then-scoring controller that releases every audit graph and recomputes scoring fields serially.

**Tech Stack:** Python 3.11, PyTorch `torch.func.jvp`/`torch.func.vjp`, NumPy PCG64, pytest, Ruff, strict JSON, SHA-256, Git, DGX CUDA deterministic execution.

## Global Constraints

- Authoritative preregistration: `docs/pass200_rsta_candidate_2026-08-09.md`, SHA-256 `a35cd3469d5561ce59202030dd3c3050e018dbfc537cb0ee0401a1d0340f5857`.
- Authoritative prospective amendment: `docs/pass200_rsta_adjoint_integrity_amendment_2026-08-09.md`; implement its committed bytes literally.
- H6 handoff `e836d5861154c2a7674f366adf496b90c38b9da4` binds source `b6f41a4c82ddf96f5cda238ed1db7a0d47fc7279`; H6 smoke passed, H6 science exited `INVALID` on a later-seed adjoint, created no output, and its unpersisted partial seed-0 rows were never inspected.
- Do not compute another candidate row until this amendment is committed, implemented with RED→GREEN evidence, independently reviewed, and refrozen in a manifest-only handoff.
- Preserve exact PCG64 output/parameter directions, seeds, parameter order, B=180 contexts, FP32 model/JVP/VJP arithmetic, denominator `max(abs(lhs),abs(rhs),float64(1e-12))`, and tolerance `5e-4`.
- Change only adjoint multiplication/reduction to float64: cast both factors before multiplication, sum each inner product in float64, then stack and sum RHS parameter terms in exact named-parameter order.
- `--integrity-all-seeds-only` is candidate-free and must never call `exact_contextual_rsta_fields`, `score_rsta_batch`, `decide_stage_a`, `joint_bootstrap`, `scientific_payload`, or receiver-row serialization.
- Scientific execution completes zero-Jacobian, repeatability, adjoint, and rotation integrity for seeds `0,1,2,3` before any score, then recomputes fields for scoring with one full derivative graph at peak.
- Any scientific integrity failure makes zero calls to `score_rsta_batch`, `decide_stage_a`, `joint_bootstrap`, or `scientific_payload` and leaves no destination or sibling temp. The candidate-free mode alone may persist all four finite adjoint tolerance outcomes with `all_passed=false`.
- Local tests are tiny/mocked and serial. B=180 BN-Inception CUDA processes run only on DGX `spark-2751` from a clean detached checkout.
- Authorized implementation files are `scripts/diagnose_pass200_rsta_stage_a.py`, `tests/test_diagnose_pass200_rsta_stage_a.py`, and `docs/pass200_rsta_receipt_stage_a_manifest.json`. DGX outputs are new uniquely addressed files under `reports/generated/pass200_rsta_receipt/`. Do not edit trainer or recipe code.
- `S7` denotes the full commit returned by `git rev-parse HEAD` after the final reviewed source/test commit; `H7` denotes the full commit returned after the subsequent manifest-only commit. Angle-bracketed `<H7>` in output filenames means that exact 40-character H7 value, not an unresolved design choice.

## File structure

- `scripts/diagnose_pass200_rsta_stage_a.py`: structured adjoint audit, candidate-free four-seed runner/serializer, two-phase scientific controller, and CLI routing.
- `tests/test_diagnose_pass200_rsta_stage_a.py`: numerical RED/GREEN tests, exact schemas and bindings, forbidden-call sentinels, event-order tests, failure/no-output tests, and graph-lifetime checks.
- `docs/pass200_rsta_receipt_stage_a_manifest.json`: manifest-only H7 binding of the amendment and independently reviewed scientific source commit S7.

---

### Task 1: Exact Float64 Adjoint Audit

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces: `adjoint_integrity_audit(model: Any, images: Any, output_direction: Any, parameter_direction: Mapping[str, Any], *, output_direction_seed: int, parameter_direction_seed: int, expected_batch_size: int = 180, expected_dimension: int = 512) -> dict[str, Any]`.
- Produces: `_adjoint_direction_metadata(...) -> dict[str, Any]`, `_finalize_adjoint_scalars(lhs: Any, rhs: Any) -> dict[str, Any]`, and `_float64_adjoint_inner_products(...) -> tuple[Any, Any]`; the public audit composes these focused units.
- Preserves: `registered_adjoint_directions(...)` draws and exact trainable named-parameter order.
- Replaces: scalar `adjoint_relative_error(...)` use in production integrity with the exact seventeen-field audit object.

- [ ] **Step 1: Write and run the direction/hash RED test**

  Add `test_adjoint_direction_metadata_binds_registered_fp32_tensors_without_resampling`. For every seed `0..3`, independently reproduce the two `domain_seed` values, assert the actual FP32 tensors equal `registered_adjoint_directions`, hash output C-order bytes, and hash the ordered concatenation of parameter-direction C-order bytes. Require exact domain/seeds/hashes, output shape, `_ordered_text_sha256(parameter_names)`, count, and dtype fields.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_adjoint_direction_metadata_binds_registered_fp32_tensors_without_resampling`

  Expected: FAIL because `_adjoint_direction_metadata` does not exist.

- [ ] **Step 2: Implement only direction metadata and verify GREEN**

  Implement `_adjoint_direction_metadata` without changing direction generation or casting. Hash C-contiguous bytes copied from the actual consumed FP32 tensors; concatenate parameter tensor bytes in exact `model.named_parameters()` trainable order. Return only the ten frozen metadata fields. Rerun the exact Step-1 test and require PASS.

- [ ] **Step 3: Write and run denominator/threshold RED tests**

  Add `test_finalize_adjoint_scalars_uses_exact_float64_denominator_and_boundary`. Parameterize float64 `lhs,rhs` values that exercise the `1e-12` floor and finite relative errors immediately below, equal to, and immediately above `5e-4`; require `passed` values `true,true,false`. Add nonfinite `lhs`, `rhs`, difference, denominator, and quotient cases and require rejection.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_finalize_adjoint_scalars_uses_exact_float64_denominator_and_boundary`

  Expected: FAIL because `_finalize_adjoint_scalars` does not exist.

- [ ] **Step 4: Implement only scalar finalization and verify GREEN**

  Implement `_finalize_adjoint_scalars` with finite float64 inputs, `absolute_error=abs(lhs-rhs)`, denominator `max(abs(lhs),abs(rhs),float64(1e-12))`, relative error, tolerance `0.0005`, and inclusive `<=`. Rerun the exact Step-3 test and require PASS.

- [ ] **Step 5: Write and run FP32-operator/float64-cancellation RED tests**

  Add `test_adjoint_inner_products_keep_fp32_operators_and_cast_before_multiply`. Feed captured FP32 `Jv`, `u`, ordered tangents, and `J^T u`; compare both reductions to independent NumPy float64 results and require float64 output tensors.

  Add `test_adjoint_float64_reduction_matches_independent_cancellation_reference`. Use deterministic FP32 factors with large alternating products plus small residual products. Prove a local FP32 multiply/sum crosses the registered decision relative to the independent float64 reference, then require production `lhs,rhs` to equal that float64 reference. The test must fail if multiplication occurs before either factor is cast or RHS terms are not stacked and summed in named-parameter order.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'adjoint_inner_products_keep or adjoint_float64_reduction_matches'`

  Expected: both tests FAIL because `_float64_adjoint_inner_products` does not exist.

- [ ] **Step 6: Implement only the float64 inner products and verify GREEN**

  Keep model, primals, directions, JVP, and VJP FP32. Keep `_functional_encoder`, `torch.func.vjp`, and `torch.func.jvp` unchanged. Implement only the inner products:

  ```python
  lhs_tensor = (jv.double() * u.double()).sum(dtype=torch.float64)
  rhs_terms = [
      (tangents[name].double() * jtu[name].double()).sum(dtype=torch.float64)
      for name in parameter_names
  ]
  rhs_tensor = torch.stack(rhs_terms).sum(dtype=torch.float64)
  ```

  The pure inner-product helper returns only `lhs_tensor,rhs_tensor`; scalar finalization remains the Step-4 unit. Rerun the two exact Step-5 tests and require PASS.

- [ ] **Step 7: Write and run the composed exact-audit RED test**

  Add `test_adjoint_integrity_audit_composes_exact_seventeen_field_contract`. Patch `torch.func.jvp` and `torch.func.vjp` wrappers to assert all operator inputs/outputs remain FP32, then require exact field order, exact metadata from Step 2, exact scalars from Steps 4/6, and rejection of direction/parameter topology drift.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_adjoint_integrity_audit_composes_exact_seventeen_field_contract`

  Expected: FAIL because `adjoint_integrity_audit` does not exist.

- [ ] **Step 8: Compose the minimal public audit and verify Task 1 GREEN**

  Implement `adjoint_integrity_audit` only as frozen validation, unchanged FP32 JVP/VJP calls, and composition of the three GREEN helpers. Run the exact Step-7 test, then:

  Run the three focused adjoint tests, then:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'adjoint or registered_adjoint_directions'
  .venv/bin/ruff check scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Commit only source/test changes with message `audit RSTA adjoint in float64`.

---

### Task 2: Candidate-Free Four-Seed Integrity Mode

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces: `run_all_seed_adjoint_integrity(...) -> dict[str, Any]` and CLI flag `--integrity-all-seeds-only`.
- Consumes: the existing receipt/source validators, training-only loader, deterministic cache, model loader, fixtures, deterministic-pool audit, zero-Jacobian audit, and `adjoint_integrity_audit`.
- Produces exact top-level, binding, integrity, per-seed, and adjoint key sets frozen by the amendment.

- [ ] **Step 1: Write and run exact-schema/reachability RED tests**

  Add `test_integrity_all_seeds_mode_is_candidate_free_and_exact_schema`. Replace `exact_contextual_rsta_fields`, `score_rsta_batch`, `decide_stage_a`, `joint_bootstrap`, `scientific_payload`, and receiver-row serialization with raising sentinels. Execute the new CLI on four tiny bound seeds. Assert none is called; assert fixed scalar values and every exact top/binding/integrity/seed/adjoint key set; assert no `rows`, `fields`, `scores`, `decision`, `aggregation`, or `bootstrap` key occurs anywhere.

  Add `test_integrity_all_seeds_recursively_validates_execution_manifest_environment`. Start with exact valid nested objects, then recursively mutate every container and leaf: missing/extra keys, wrong exact scalar types/values, malformed hashes/commits, changed manifest references/source files, empty/wrong version strings, and mismatched execution/source provenance. Require the candidate-free payload validator to reject every mutation before publication. Explicitly prove byte/key-exact reuse of `_EXECUTION_AUDIT_FIELDS`, `validate_execution_audit`, the existing strict amended-manifest nested schemas, `ENVIRONMENT_AUDIT_FIELDS`, and `configure_deterministic_process` output.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'integrity_all_seeds_mode_is_candidate_free or recursively_validates_execution'`

  Expected: both tests FAIL because the CLI and exact payload validator do not exist.

- [ ] **Step 2: Write and run finite-continuation/fail-fast RED tests**

  Add `test_integrity_all_seeds_records_all_finite_adjoint_failures_without_candidate_calls`. Inject a finite above-tolerance error at seed 1, return finite passing objects at seeds 0, 2, and 3, and assert adjoint call order `[0,1,2,3]`, exact four seed keys, seed-1 `passed=false`, global `all_passed=false`, zero candidate calls, and one atomic output.

  Parameterize structural/provenance, nonfinite direction, nonfinite reduction, zero-Jacobian, serialization, and atomic-publication failures. Assert immediate exception, no later forbidden access appropriate to the failure boundary, no destination, and sibling-temp cleanup.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'records_all_finite_adjoint_failures or integrity_all_seeds_fail_fast'`

  Expected: FAIL because no all-seed controller implements continuation or fail-fast publication.

- [ ] **Step 3: Write and run amendment-provenance RED tests**

  Add `test_scientific_source_authenticates_adjoint_integrity_amendment_bytes_and_blob`. Require exact manifest key `adjoint_integrity_amendment` with only `path`, `sha256`, and `commit`; independently mutate path, digest, commit, worktree bytes, Git blob, and top-level presence and require failure before artifact/model access.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_source_authenticates_adjoint_integrity_amendment_bytes_and_blob`

  Expected: FAIL because the strict manifest schema rejects or ignores the new binding.

- [ ] **Step 4: Implement the minimal exact controller/schema behavior**

  Add the mutually exclusive CLI flag. Reuse the frozen order:

  ```text
  configure deterministic process
  reject existing output
  validate receipt and current source
  validate deterministic-pool audit
  load and cross-bind seeds 0,1,2,3
  validate retained arrays and build the common first-batch cache
  run dense and BN fixtures
  for seed 0,1,2,3: load model; zero-Jac audit; generate directions; adjoint audit; release tensors/model
  atomically publish exact audit JSON
  ```

  Build each binding record from digest-bound checkpoint/final-training-pack entries plus actual first-batch/cache/tensor hashes. Add a dedicated exact payload validator that reruns the existing execution, strict manifest, and environment validators and exact recursive key/type/value checks before `write_json_atomic`. Add committed amendment path/SHA/commit constants and strict source validation. Do not route through `_registered_first_batch_integrity`.

- [ ] **Step 5: Verify every Task 2 RED is GREEN and commit**

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'integrity_all_seeds or adjoint_audit'
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Commit only source/test changes with message `add all-seed RSTA adjoint audit`.

---

### Task 3: Two-Phase Scientific Integrity Prefix

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Refactors: `run_scientific_diagnostic(...) -> dict[str, Any]` into integrity and scoring phases without changing its external arguments or scientific result fields except replacing each scalar adjoint error with the exact adjoint object.
- Produces: one in-memory integrity record for every seed before scoring; scoring recomputes every field and retains one full derivative graph at peak.

- [ ] **Step 1: Replace and run the defective seed-local ordering test as a global-prefix RED**

  Replace the assertion that each seed's rotation merely precedes its own score in `test_scientific_cli_executes_exact_four_seed_pipeline_and_writes_atomic_rows`. Record events and require the prefix exactly:

  ```python
  integrity_events = [event for event in events if event.startswith("integrity-")]
  score_events = [event for event in events if event.startswith("score-")]
  assert integrity_events == ["integrity-0", "integrity-1", "integrity-2", "integrity-3"]
  assert events.index("integrity-3") < events.index(score_events[0])
  assert events.index(score_events[-1]) < events.index("decision")
  ```

  Count first-batch field construction separately in both phases and require a fresh scoring-phase call rather than reuse of an audit field.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_cli_executes_exact_four_seed_pipeline_and_writes_atomic_rows`

  Expected: FAIL because H6 emits seed-0 score events before seed-1 integrity.

- [ ] **Step 2: Write and run all later-seed gate-failure RED tests**

  Add `test_scientific_later_seed_integrity_failure_prevents_all_candidate_work`, parameterized by `failing_seed in (1,2,3)` and `gate in ('zero_jacobian','repeatability','adjoint','rotation')`. Make every earlier seed pass and inject the exact failure at the selected gate. Install independent raising/counting sentinels for `score_rsta_batch`, `decide_stage_a`, `joint_bootstrap`, and `scientific_payload`. For all twelve cases require zero calls to every sentinel, no primary/alternate row append or aggregate construction, and absence of both the requested destination and every sibling temp file.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py::test_scientific_later_seed_integrity_failure_prevents_all_candidate_work`

  Expected: FAIL because H6 scores earlier seeds before each injected later-seed failure.

- [ ] **Step 3: Write and run graph-lifetime/payload RED tests**

  Add `test_scientific_integrity_graphs_are_released_before_next_seed_and_scoring_graphs_are_fresh`. Use weak references around every derivative closure/output to require each audit graph is unreachable before the next seed and every scoring first-batch field is newly computed after the four-seed prefix.

  Add `test_scientific_payload_requires_exact_structured_adjoint_audits`. Mutate each of the seventeen adjoint fields by removal, addition, wrong type/value, direction/hash drift, parameter name/count mismatch, nonfinite scalar, denominator mismatch, or inconsistent `passed`; require rejection. Require four exact seed audit objects and unchanged literal tiny-fixture rows, aggregates, bootstrap hashes, and verdict.

  Run: `.venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'integrity_graphs_are_released or payload_requires_exact_structured_adjoint'`

  Expected: both tests FAIL because H6 reuses first-batch fields in scoring and accepts only scalar adjoint errors.

- [ ] **Step 4: Implement the minimal two-phase controller**

  Extract the model/context setup needed by `_registered_first_batch_integrity` without moving any candidate score into that helper. For each seed in order, execute zero-Jacobian, repeatability, structured adjoint, and rotation. Reject `passed=false` as scientific `INVALID`. Store only detached audit objects and hashes; delete first-batch fields, prehead/raw captures, derivative closures, images, proxies, model, and CUDA references before advancing. Do not initialize primary/alternate receiver-row accumulation until all four records exist and pass.

  After the global prefix, recreate one seed model from its immutable checkpoint bytes, restore and verify the exact zero-Jacobian exclusion, and recompute all primary and alternate fields. Never pass integrity-phase fields/captures into `score_rsta_batch`. Delete each batch graph after its eight detached rows and each model before the next seed. Preserve original row order, controls, aggregation, 10,000 PCG64(200) bootstrap, thresholds, decision precedence, and payload schema outside the structured adjoint substitution. Add exact structured-adjoint payload validation.

  Rerun every exact Step-1 through Step-3 test and require the global order, all twelve failure cases, fresh graphs, and exact payload contract to pass.

- [ ] **Step 5: Verify every Task 3 RED is GREEN and commit**

  Run:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'scientific and (integrity or pipeline or payload or graph)'
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Commit only source/test changes with message `gate RSTA scoring on all-seed integrity`.

---

### Task 4: Independent Review, Refreeze, and DGX Execution

**Files:**
- Modify after source review: `docs/pass200_rsta_receipt_stage_a_manifest.json`
- Produce on DGX: `reports/generated/pass200_rsta_receipt/<H7>-adjoint-integrity-all-seeds.json`
- Produce conditionally on DGX: `reports/generated/pass200_rsta_receipt/<H7>-stage-a.json`

**Interfaces:**
- Consumes: reviewed Tasks 1–3 and the committed amendment.
- Produces: source commit S7, manifest-only handoff H7, one candidate-free all-seed audit, and only after a green audit one fresh scientific result.

- [ ] **Step 1: Run the complete local assurance gate once**

  From a clean tracked worktree at the final source/test diff, run serially:

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/ruff check scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  .venv/bin/python -m py_compile scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  git diff --check
  ```

  Record exact commands, exit codes, and relevant test counts in the ignored implementation evidence log.

- [ ] **Step 2: Obtain independent adversarial review**

  Give the reviewer the amendment, full S7 candidate diff, H6 source, and tests. Require explicit findings on direction preservation, cast-before-multiply, RHS ordering, exact hashes/schema, candidate-free reachability, finite-failure continuation, structural fail-fast behavior, global scientific prefix, fresh field recomputation, graph lifetime, and unchanged candidate math. Repair every Critical or Important finding with a focused failing test, minimal fix, GREEN evidence, and scoped re-review.

- [ ] **Step 3: Create and authenticate source commit S7**

  If review produced source/test fixes, commit only `scripts/diagnose_pass200_rsta_stage_a.py` and `tests/test_diagnose_pass200_rsta_stage_a.py`; otherwise make no empty commit. Set `s7_commit=$(git rev-parse HEAD)` after the final reviewed source/test commit. Verify both worktree files equal `$s7_commit:path`, the test gate is fresh, and no trainer, recipe, result, or unrelated file entered S7.

- [ ] **Step 4: Create the manifest-only H7 refreeze**

  Add exact top-level `adjoint_integrity_amendment` with `path`, SHA-256, and this amendment's commit. Set `current_scientific_source.git_revision` to `$s7_commit` and refresh only reviewed source-file hashes. Preserve every prior amendment, receipt, artifact-schema, and seed object byte-semantically. Commit only `docs/pass200_rsta_receipt_stage_a_manifest.json`, then set `h7_commit=$(git rev-parse HEAD)` and verify:

  ```bash
  git diff --exit-code "$h7_commit^" "$h7_commit" -- scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py
  git show "$h7_commit:docs/pass200_rsta_receipt_stage_a_manifest.json" > /tmp/pass200-h7-manifest.json
  sha256sum /tmp/pass200-h7-manifest.json
  git diff --check "$h7_commit^" "$h7_commit"
  ```

  Independently review `S7..H7`. Any manifest correction creates a new manifest-only handoff and repeats authentication.

- [ ] **Step 5: Prepare an isolated DGX checkout**

  Bundle exact H7, hash the bundle before and after transfer, create a new detached clean checkout at H7 on `spark-2751`, and verify HEAD, manifest bytes, every `current_scientific_source` Git blob/worktree SHA, amendment SHA/commit, receipt SHA, artifact paths, CUDA availability, pinned environment, and no active duplicate process. Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process start.

- [ ] **Step 6: Run the candidate-free four-seed audit first**

  Run one fresh process with exact manifest, receipt, unique absent output, and `--integrity-all-seeds-only`:

  ```bash
  h7_commit=$(git rev-parse HEAD)
  integrity_output="reports/generated/pass200_rsta_receipt/${h7_commit}-adjoint-integrity-all-seeds.json"
  test ! -e "$integrity_output"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/diagnose_pass200_rsta_stage_a.py \
    --manifest docs/pass200_rsta_receipt_stage_a_manifest.json \
    --binding-receipt docs/pass200_rsta_binding_receipt_d6270a9.json \
    --output "$integrity_output" \
    --integrity-all-seeds-only
  sha256sum "$integrity_output"
  ```

  Collect its original exit code and output SHA. Independently validate exact top/nested schemas, all bindings, four direction/hash/parameter audits, FP32 model and float64 reduction dtypes, finite scalars, exact denominator/tolerance, four seed keys, and `all_passed`.

  If the process structurally fails or `all_passed=false`, stop: run no scientific process and compute no candidate value. Diagnose only from this candidate-free evidence; any proposed change requires a new prospective amendment or implementation review/refreeze as appropriate.

- [ ] **Step 7: Run scientific Stage A only after green integrity**

  If and only if the exact H7 all-seed audit has `all_passed=true`, launch one fresh process with no code/config change:

  ```bash
  scientific_output="reports/generated/pass200_rsta_receipt/${h7_commit}-stage-a.json"
  test ! -e "$scientific_output"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/diagnose_pass200_rsta_stage_a.py \
    --manifest docs/pass200_rsta_receipt_stage_a_manifest.json \
    --binding-receipt docs/pass200_rsta_binding_receipt_d6270a9.json \
    --output "$scientific_output" \
    --scientific
  sha256sum "$scientific_output"
  ```

  Collect the original exit and atomic output. Independently validate that every seed's zero-Jacobian, repeatability, exact adjoint, and rotation audit passed; all 64 primary and 16 alternate identities exist in every seed; bootstrap arrays contain exactly 10,000 replicates and match their hashes; and the original PASS/FAIL/UNRESOLVED predicates and first decisive clause were applied unchanged.

- [ ] **Step 8: Adjudicate without reviving discarded H6 rows**

  State beside any result that H6 partial seed-0 rows were never inspected and are excluded. Treat the candidate-free audit as executability evidence only. Record the H7 scientific outcome only after strict offline validation; authorize only the separately preregistered Stage B if the original Stage-A result is `PASS ONWARD`.
