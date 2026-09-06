# SigLIP Coverage Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an authenticated eight-shot affine calibration experiment that preserves the fast student query path by translating the frozen teacher gallery offline.

**Architecture:** A pure evidence module owns deterministic support selection, exact FP64 OLS maps, retrieval recomputation, and canonical decisions. A local probe authenticates the existing model/data authorities, extracts support descriptors, seals both affine directions, and only then opens external evaluation. A guarded DGX wrapper stages inputs, monitors the sole process, and retains only canonical evidence and the sealed calibration artifact.

**Tech Stack:** Python 3.12, PyTorch, safetensors, pytest, Ruff, mypy, ShellCheck, existing SigLIP recovery runtime.

**Spec:** `docs/superpowers/specs/2026-09-06-siglip-coverage-calibration-design.md`

## Global Constraints

- Work only in `/home/rb/worktrees/sfora-emafactorial`; do not touch Borsuk.
- Preserve unrelated Qwen worktree changes and stage only named SigLIP files.
- Previously burned labels 49 through 81 remain inaccessible to this method until code and launcher are committed and independently reviewed; the result remains claim-ineligible.
- Freeze eight support images per external class, OLS `rcond=1e-12`, `gelsd`, ascending ID order, and no hyperparameter selection.
- The preferred serving path maps the teacher gallery offline; it adds no query-time work.
- Run one DGX scientific process, preserve its first terminal, and never auto-restart.

---

### Task 1: Support, OLS, and retrieval authority

**Files:**
- Create: `src/sfora/siglip_coverage_calibration.py`
- Create: `tests/test_siglip_coverage_calibration.py`

**Interfaces:**
- Produces: `coverage_support_indexes(ids, labels, *, support_per_class=8) -> tuple[int, ...]`.
- Produces: `fit_coverage_affine(source, target, ids) -> CoverageAffine`.
- Produces: `coverage_retrieval_cells(student, teacher, ids, labels, maps) -> Mapping[str, CompatibilityRetrievalEvidence]`.
- Produces: `build_coverage_calibration_result(...) -> bytes` and `validate_coverage_calibration_result_bytes(raw) -> dict[str, object]`.

- [ ] **Step 1: Write support-selection RED tests.** Assert the exact SHA-256 ordering namespace, eight rows per class, minimum ten rows, ascending canonical output, duplicate-ID rejection, concrete types, determinism under input permutation, and no access to descriptor values.

- [ ] **Step 2: Run `uv run pytest -q tests/test_siglip_coverage_calibration.py -k support` and require missing-interface RED.**

- [ ] **Step 3: Implement the minimal support selector.** Validate unique nonempty IDs, nonnegative concrete integer labels, class cardinality, and the literal support count eight. Sort `(digest,id,index)` and return the selected original indexes in ascending authenticated-ID order.

- [ ] **Step 4: Run the support selector tests and require GREEN.**

- [ ] **Step 5: Write OLS RED tests.** Use a coherent full-rank augmented synthetic system and assert exact recovery, source-order invariance after ID sorting, rank 513 for the real shape, fixed `rcond`/driver authority, normalized FP32 application, and rejection of rank deficiency, nonfinite tensors, zero outputs, wrong dimensions, duplicate IDs, and bool-as-int values.

- [ ] **Step 6: Run `uv run pytest -q tests/test_siglip_coverage_calibration.py -k 'affine or ols'` and require missing-interface RED.**

- [ ] **Step 7: Implement `CoverageAffine` and `fit_coverage_affine`.** Build `[X,1]` in FP64 after ascending-ID reordering, call `torch.linalg.lstsq(..., rcond=1e-12, driver="gelsd")`, require rank `dimensions+1`, validate singular values/solution, and normalize mapped rows after FP64 application.

- [ ] **Step 8: Run OLS tests and require GREEN.**

- [ ] **Step 9: Write retrieval/result RED tests.** Mutation-lock all four cells, same-ID exclusion, stable ties, per-query hits/AP, micro/macro recomputation, both `0.97`/`0.95` thresholds, decision precedence, self-control identity, claim-ineligible status, one trailing LF, exact key/type schema, and nested concrete equality.

- [ ] **Step 10: Implement retrieval and canonical result authority by reusing `compatibility_retrieval_evidence`; do not duplicate metric logic.**

- [ ] **Step 11: Run the complete focused file, Ruff, mypy, py_compile, and `git diff --check`; require GREEN.**

- [ ] **Step 12: Commit only Task 1 files with configured operator identity and no attribution trailers.**

### Task 2: Descriptor artifact and sealed evaluation boundary

**Files:**
- Create: `scripts/probe_siglip_coverage_calibration.py`
- Create: `tests/test_probe_siglip_coverage_calibration.py`

**Interfaces:**
- Consumes: Task 1 support, map, retrieval, and result interfaces.
- Produces: `CoverageDescriptorArtifact` and strict `_write_coverage_maps` / `_load_coverage_maps` helpers.
- Produces: `main(argv) -> int` with an explicit local-only execution flag.

- [ ] **Step 1: Write coherent loader RED tests.** Require exact checkpoint, control binding, optimization manifest, spatial artifact, image manifest, preprocessing, and source revision identities. Reject symlinks, unknown tensors/metadata, wrong descriptor shape/dtype/order, duplicate IDs, label drift, and any URL/storage/network input.

- [ ] **Step 2: Run the loader selector and require missing-interface RED.**

- [ ] **Step 3: Implement strict local artifact loading by following the existing compatibility-capacity reader and importing shared validators.**

- [ ] **Step 4: Write sealed-boundary RED tests.** Instrument evaluation decode and prove: support split is fixed before pixels; only support descriptors enter OLS; both maps are written, hashed, and reloaded before evaluation access; support rows never enter evaluation; evaluation cannot influence solver evidence.

- [ ] **Step 5: Implement two-phase execution.** Extract all optimization-class pairs plus external support pairs, fit both directions, seal/reload one safetensors map artifact, release fitting tensors, then decode and score remaining external images once.

- [ ] **Step 6: Write CLI RED tests.** Require absolute local paths, every SHA-256, output artifact/result paths, and `--execute-coverage-calibration`. Reject duplicates, relative paths, pre-existing outputs, evaluation descriptor inputs, URLs, AWS/storage flags, class-name inputs, adjustable support counts/solver settings, and unknown flags.

- [ ] **Step 7: Implement the strict CLI and atomic `.partial` result/artifact publication.**

- [ ] **Step 8: Run probe + core tests, Ruff, mypy, py_compile, and diff-check; require GREEN.**

- [ ] **Step 9: Commit only Task 2 files.**

### Task 3: Guarded DGX deployment

**Files:**
- Create: `scripts/deploy_siglip_coverage_calibration_v1.sh`
- Create: `tests/test_deploy_siglip_coverage_calibration.py`

**Interfaces:**
- Consumes: the Task 2 local probe and the existing immutable model/data inputs.
- Produces: local `/tmp/sfora-coverage-calibration-$REVISION.json` and `.safetensors` only after remote validation.

- [ ] **Step 1: Write static deployment RED tests.** Require exact source inventory, clean-revision assertion, authenticated input sizes/digests, one explicit remote scratch directory, Landlock/input namespace isolation, separate support/evaluation phase capability manifests, PID/PGID ownership, 90-minute timeout, CUDA/RSS/PSI/swap monitoring, no restart, canonical remote validation, explicit named cleanup after PID clearance, and fail-closed output transfer.

- [ ] **Step 2: Run `uv run pytest -q tests/test_deploy_siglip_coverage_calibration.py` and require missing-script RED.**

- [ ] **Step 3: Implement the wrapper from the verified compatibility-capacity deployment pattern, changing only the exact inventory, probe arguments, phase separation, output names, and validation command.**

- [ ] **Step 4: Run deployment tests, `tests/test_landlock_exec.py`, ShellCheck, C syntax check, and diff-check; require GREEN.**

- [ ] **Step 5: Commit only Task 3 files.**

### Task 4: Repository assurance and independent review

**Files:**
- Modify only if a verified defect is found: Task 1–3 files.

**Interfaces:**
- Produces: one clean, committed, independently reviewed revision eligible for a separate scientific execution.

- [ ] **Step 1: Run focused core/probe/deployment tests and the dependency-complete `uv run pytest -q -k siglip`; require GREEN.**

- [ ] **Step 2: Run Ruff, formatting, targeted mypy, py_compile, ShellCheck, C syntax check, and `git diff --check`; require GREEN.**

- [ ] **Step 3: Start two simultaneous read-only reviews of the same exact diff and spec: Codex `gpt-6-astra` and Fable, each cold and forbidden to edit. Continue local scope/authority review while they run.**

- [ ] **Step 4: Reconcile both reviews against source and tests. Repair only verified defects with focused RED/GREEN evidence, then repeat affected gates.**

- [ ] **Step 5: Commit any verified repair, assert the exact intended path set, clean worktree except pre-existing unrelated Qwen changes, and record the full revision SHA.**

### Task 5: One frozen confirmatory execution and evidence

**Files:**
- Create after the terminal: `docs/siglip_coverage_calibration_result_2026-09-06.md`

**Interfaces:**
- Consumes: the committed reviewed revision and exact authenticated inputs.
- Produces: one canonical result and one sealed affine artifact, or one preserved failure/stop receipt.

- [ ] **Step 1: Verify DGX identity, free resources, immutable inputs, exact revision, expected transfer volume, process limits, and absence of another scientific process.**

- [ ] **Step 1a: Treat Linux PSI `avg10` as a 0–100 percentage: 79.0 immediate and 50.0 for three samples. Bind every sample, threshold, recomputed peak, and terminal resource value into the canonical execution receipt. The terminal 0.79/0.50 revision is an operational unit-error failure with no scientific result and must not be restarted.**

- [ ] **Step 2: Launch exactly one deployment wrapper invocation. Retain its original job/PID and monitor that same process every 30–55 seconds; do not launch a duplicate.**

- [ ] **Step 3: On terminal, preserve exit status, canonical result/artifact hashes, metrics, runtime/pressure evidence, remote PID clearance, and scratch cleanup. Never auto-restart.**

- [ ] **Step 4: Validate the result independently. Record every identity, four-cell micro/macro metric, solver/rank evidence, support/evaluation counts, decision, offline migration throughput, serving-latency interpretation, resource evidence, and limitations in the result document.**

- [ ] **Step 5: Run document validator and diff-check, commit only the result document, and notify the operator via `devbox-tell`.**
