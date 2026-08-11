# Pass201 Ordinary-PA Source-v3 and CPU Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one authenticated ordinary-PA source and run the frozen Pass201 causal diagnostic with exact CPU replay, without relying on deterministic CUDA.

**Architecture:** Add a schema-versioned importlib inventory to the existing source controller while preserving historical source-v2 validation; prospectively replace only the diagnostic's dead source binding and require CPU. Freeze runtime authority twice before a manifest-only handoff, then permit one source-training process and the existing three-role CPU diagnostic.

**Tech Stack:** Python 3.12/3.13 standard library, canonical JSON, SHA-256, PyTorch CPU/CUDA, pytest, Ruff, Git.

## Global Constraints

- Authority: `docs/pass201_pa_source_v3_protocol_2026-08-11.md` at commit
  `9782eb44f4a087682563d8a1f4e075f4fcdd165b`, SHA-256
  `716460eda8664a4c37b5f14332244a8dae4f921b393b7e4c085ff0b4e26a7426`.
- Preserve all source scientific constants and every diagnostic formula, process role/order, digest equality, tolerance, bootstrap, threshold, and decision.
- Source training is one GPU attempt; diagnostic roles require `CUDA_VISIBLE_DEVICES=""` and CPU.
- Do not read CIS results/generated scientific artifacts during Tasks 1–5.
- Use RED→GREEN TDD; source/tests precede and are separate from the manifest-only handoff.
- Never touch `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.

---

### Task 1: Freeze Prospective Docs

**Files:**
- Create: `docs/pass201_pa_source_v3_protocol_2026-08-11.md`
- Create: `docs/superpowers/plans/2026-08-11-pass201-pa-source-v3.md`

**Interfaces:** Produces linear authority `A3` (protocol only) → `P3` (plan only).

- [ ] Review and commit only the protocol as `authorize Pass201 ordinary-PA source v3`.
- [ ] Insert literal A3 commit and protocol SHA-256 into this plan.
- [ ] Run diff/placeholder/path checks and commit only this plan as `plan Pass201 ordinary-PA source v3`; require `P3^ == A3`.

### Task 2: Canonical Training-Environment Inventory

**Files:**
- Modify: `scripts/run_pass201_pa_source_v2.py`
- Modify: `tests/test_run_pass201_pa_source_v2.py`

**Interfaces:** Produces `_canonical_package_inventory(interpreter, checkout, environment) -> tuple[bytes, int]`.

- [ ] Add RED tests for the absent API using an explicit interpreter fixture,
  exact cwd/environment, and two byte-identical captures. Mutate UTF-8/JSON,
  top/python/record key sets and canonical order, types, trim/NUL, normalization, duplicates, array sort,
  executable/prefix/sys.path, LF, and extra bytes. Prove a `-I` mutant loses
  the required `PYTHONPATH` relation.
- [ ] Run `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k importlib_inventory`; expect missing-API failures while existing tests pass.
- [ ] Implement literal `(python, "-B", "-c", child)` capture, strict parse,
  PEP-503 normalization, exact reserialization, and runtime-path comparison.
- [ ] Run the selector, Ruff, `py_compile`, and diff-check; keep changes unstaged for Task 3.

### Task 3: Versioned Schema and Historical Compatibility

**Files:**
- Modify: `scripts/run_pass201_pa_source_v2.py`
- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `tests/test_run_pass201_pa_source_v2.py`
- Modify: `tests/test_pass201_pa_source_v2_contract.py`

**Interfaces:** Source-v3 package evidence has exact key set `{algorithm, distribution_count, bytes, sha256}`; source-v2 retains `{bytes, sha256}`. Both use canonical `sort_keys=true` JSON. The commit produced here is I3.

- [ ] Add source-v3 RED fixtures and exhaustive key/type/order/value/relation
  mutations; keep every historical v2 fixture GREEN. Expected RED is missing
  v3 dispatch/schema, not legacy acceptance.
- [ ] Implement dispatch by enclosing schema version, exact built-in types,
  positive count/bytes, lowercase hash, and authority/execution/receipt equality.
- [ ] Run both complete source/contract test files, Ruff, `py_compile`, and
  diff-check. Commit exactly four source/test paths as
  `implement Pass201 source-v3 package authority`; record it as I3.
- [ ] Obtain independent review of this commit before Task 4.

### Task 4: Source-v3 Binding and CPU Diagnostic Authority

**Files:**
- Modify: `scripts/run_pass201_pa_source_v2.py`
- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `scripts/diagnose_pass201_cis_operator.py`
- Modify: `tests/test_run_pass201_pa_source_v2.py`
- Modify: `tests/test_pass201_pa_source_v2_contract.py`
- Modify: `tests/test_diagnose_pass201_cis_operator.py`

**Interfaces:** Produces exact source-v3 paths/schema and replaces only the diagnostic source authority; all scientific interfaces remain identical.

- [ ] Add RED tests for `A3→P3→I3→V3→H3`, exact source-v3 paths, old-path
  rejection, Git/worktree hashes, receipt relations, no candidate fields, and
  direct manifest-only ancestry.
- [ ] Add literal canonical-manifest/receipt fixtures for the exact versioned
  delta in the protocol, plus recursive add/remove/reorder/type/value mutations
  and cross-version substitution. Historical v2 fixtures must remain GREEN.
- [ ] Add RED tests proving CUDA-visible execution is rejected and CPU records
  are required while exact `context0_record_sha256`, gradient/update digests,
  `2e-6` tensor tolerance, `1e-5` scalar tolerance, formulas, and thresholds are
  unchanged.
- [ ] Add RED tests for single-thread CPU scheduling: three env vars exactly
  `1` before Torch import, both Torch setters exactly once before tensor/model
  creation, getters and persisted deterministic settings exactly built-in int
  `1`, cross-process equality, and wrong/order/repeated-call mutants rejected.
- [ ] Add RED tests for the acyclic Git-handoff predicate: detached clean H3,
  H3 sole parent V3, sole `A 100644` manifest edge, `H3:path` equals worktree,
  manifest source mapping authenticates the executing diagnostic. Prove no
  compile-time H3/manifest digest is needed and every merge/extra edge fails.
- [ ] Add RED tests for the exact source-v3 source-file order: the historical
  UTF-8-byte-sorted list plus `scripts/diagnose_pass201_cis_operator.py` first,
  immediately before the contract, with every V3 Git/worktree digest
  authenticated.
- [ ] Add RED tests that source-v3 public CLI rejects `--runtime-factory` and
  `PASS201_RUNTIME_FACTORY` unconditionally before import, while historical
  unit-test seams remain confined to historical schemas.
- [ ] Implement new source-v3 output constants/schema and the diagnostic's
  source-binding replacement. Do not alter operator/scoring code.
- [ ] Run all three complete test files plus Ruff, `py_compile`, and diff-check.
  Commit only the six source/test paths as `bind Pass201 diagnostic to source v3`.
- [ ] Obtain independent source review; any fix remains in source/tests before H3.

### Task 5: Freeze Authority Twice and Commit H3

**Files:**
- Create: `docs/pass201_pa_source_v3_authorization_manifest.json`

**Interfaces:** Consumes independently READY V3; produces direct manifest-only H3.

- [ ] Build future-manifest tests before V3 review completes: exact key order,
  A3/P3/V3 authorities, all source hashes, package/runtime evidence, output
  paths, and `candidate_values_computed=false`. Validators must be GREEN in V3.
- [ ] In a fresh detached remote checkout at V3, authenticate interpreter,
  environment/cwd, sys.executable/prefix/path, import roots, source/data, and
  pretrained bytes. Capture one literal RFC3339 absence timestamp and pass the
  identical argument to both processes. Run exactly two fresh `freeze-authority` processes to
  distinct absent temporary paths and require byte-identical complete manifest
  candidates. No training/GPU process starts.
- [ ] Exclusively publish one candidate as the canonical manifest, commit only
  that JSON with `H3^ == V3`, and repeat strict validation from detached H3.

### Task 6: One Source-Training Attempt

**Files:** Runtime outputs only at source-v3 paths frozen in H3.

**Interfaces:** Produces one source receipt/checkpoint or durable failure; no CIS value.

- [ ] Revalidate H3 and the committed runtime inventory relation without a
  third inventory capture. Require queue/GPU idle and output/temp absence.
- [ ] Launch exactly one training child; record PID, command, environment,
  times, memory, and exit. Never require a second run to reproduce floats and
  never retry after the child starts.
- [ ] Validate report/checkpoint semantics, source/data pre/post identity,
  atomic outputs, and `candidate_values_computed=false`; independently review
  the receipt before diagnostic activation.

### Task 7: Exact CPU Pass201 Diagnostic

**Files:** Existing diagnostic source/tests and frozen activation/result paths.

**Interfaces:** Consumes authenticated source-v3; produces one PASS/FAIL/UNRESOLVED no-training diagnostic.

- [ ] Re-run `.venv/bin/pytest -q tests/test_diagnose_pass201_cis_operator.py`;
  expected 124 or the reviewed higher count, with all original assertions intact.
- [ ] In fresh processes set `CUDA_VISIBLE_DEVICES=""`; activate source, run
  integrity A and B, and require exact inputs/context-0/action hashes plus all
  registered residual tolerances. Require accelerator `cpu`, visible devices
  `["cpu"]`, and equal observed CUDA/cuDNN build strings rather than the false
  literal `unavailable`. Retain `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set the three
  thread env vars to `1`, and require both Torch thread getters/persisted fields
  equal `1`. Treat integrity A as the CPU feasibility
  smoke and record elapsed/peak RSS. Stop on timeout/resource failure or any
  discrepancy; never fall back to CUDA or change batch/context count.
- [ ] Only after integrity GREEN, run the single CPU scientific role with the
  existing 32 contexts/bootstrap/thresholds. `PASS` authorizes only a separate
  training preregistration; `FAIL`/ordinary `UNRESOLVED` closes the family.
