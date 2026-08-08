# Pass200 RSTA Stage-A Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the preregistered four-seed Receiver-Self Tangent Alignment Stage-A diagnostic without exposing candidate values before its artifact, determinism, geometry, and derivative-integrity gates pass.

**Architecture:** A single import-safe CLI module owns pure deterministic selection/statistics helpers, strict artifact/data binding, and exact full-model VJP/JVP evaluation. Pure functions and tiny models are tested locally; all real-image exports, BN-Inception forwards, and full derivative work run on the DGX. The authoritative scientific contract is `docs/pass200_rsta_candidate_2026-08-09.md`; code must fail closed rather than reinterpret it.

**Tech Stack:** Python 3.11, NumPy, PyTorch/torch.func, existing `sfora.image_end_to_end` model/loss/transform helpers, pytest, Ruff.

## Global Constraints

- Read `docs/pass200_rsta_candidate_2026-08-09.md` in full before editing; every constant, hash domain, data role, threshold, and invalid condition in it is binding.
- Never compute or print a real RSTA candidate statistic until the committed preregistration and every artifact/source/determinism gate have passed.
- Local tests use only tiny synthetic arrays/models and must remain below 1 GB RSS; no local image-dataset/model integration test.
- All full-dataset exports, BN-Inception execution, VJP/JVP smoke, and the real diagnostic run execute on `riomus@100.104.199.68`.
- Do not overlap test processes or touch the running Pass120 CIS controller.
- Query/gallery arrays are binding-only and must be released before constructing a training-only scientific input object.
- Use the exact production double-normalization Proxy Anchor path and exact B=180 train-mode same-batch graph; no singleton, eval-mode, microbatch, or similarity-only surrogate.
- Stage-A values are raw Euclidean gradient-flow quantities, not AdamW velocities.
- Use strict TDD: add one behavioral test, run it to observe the expected failure, then add the minimum implementation and rerun.
- Do not edit `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.

---

### Task 1: Deterministic geometry and verdict core

**Files:**
- Create: `scripts/diagnose_pass200_rsta_stage_a.py`
- Create: `tests/test_diagnose_pass200_rsta_stage_a.py`

**Interfaces:**
- Produces `domain_hash(domain: str, text: str) -> bytes`, deterministic role/batch selectors, tangent projection, smooth-margin gradient, head-only kernel, joint bootstrap, rotation construction/checking, and `decide_stage_a(rows, alternate_rows) -> dict[str, object]`.
- No filesystem, image, checkpoint, CUDA, or production-model dependency enters these pure functions.

- [ ] **Step 1: Add failing literal tests for hashing and role/batch selection**

  Test hand-derived SHA-256 bytes for the exact `domain + b"\0" + text` contract, unsigned big-endian seed extraction, canonical identity ordering, exactly 64 receivers, eight groups of eight, 172 nonoverlapping distractors per group, support exclusion, and alternate 16-receiver regrouping. Mutations caught: missing NUL, hex-vs-raw bytes, wrong index base, support leakage, or replacement of a selected receiver.

- [ ] **Step 2: Run the focused test and observe RED**

  Run `rtk pytest -q tests/test_diagnose_pass200_rsta_stage_a.py -k 'hash or selection'`. Expected failure: import/module or named helper missing, not fixture/setup failure.

- [ ] **Step 3: Implement the minimum deterministic selection helpers**

  Use only preregistered domains and stable `(digest, example_id)` ties. Validate uniqueness/counts and raise `ValueError` on any mismatch; never replace a row.

- [ ] **Step 4: Add failing geometry tests with hand-derived vectors**

  Cover receiver-tangent projection, zero rejection, smooth log-mean-exp margin gradient against two positives/top-32 foreign supports, deranged-q reprojection, deterministic tangent random control, and the analytic normalized affine-head kernel. Require positive, not absolute, head-self collinearity.

- [ ] **Step 5: Observe RED, implement the geometry, then observe GREEN**

  Run the exact focused tests before and after implementation. Dense expected arrays must be literal or independently finite-differenced; tests may not call the implementation to construct expectations.

- [ ] **Step 6: Add failing aggregation/verdict boundary tests**

  Use four seeds with the same 64 literal identity IDs. Exercise joint paired resampling, equal-seed aggregation, all pass thresholds, each fail-precedence clause, UNRESOLVED boundaries, all-64/all-16 validity, and deterministic bootstrap distribution hashing.

- [ ] **Step 7: Implement aggregation/verdict and verify Task 1**

  Run `rtk pytest -q tests/test_diagnose_pass200_rsta_stage_a.py`; then `rtk ruff check scripts/diagnose_pass200_rsta_stage_a.py tests/test_diagnose_pass200_rsta_stage_a.py`, `rtk python -m py_compile scripts/diagnose_pass200_rsta_stage_a.py`, and `rtk git diff --check`.

- [ ] **Step 8: Commit Task 1**

  Commit only the script/test changes with message `implement RSTA diagnostic core` and write the TDD commands/results to the SDD task report.

---

### Task 2: Artifact, source, and deterministic data binding

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Create: `docs/pass200_rsta_stage_a_manifest.json`

**Interfaces:**
- Consumes Task 1 selection helpers.
- Produces immutable `TrainingOnlySeedInput` objects with no query/gallery attributes, `load_and_bind_seed(...)`, deterministic transform caching, full result metadata/schema, and a CLI `--manifest --output --binding-only` path.

- [ ] **Step 1: Add failing synthetic-bundle tests for strict binding**

  Build tiny real NPZ/JSON/PT fixtures in pytest temporary directories. Test SHA mismatch, embedded digest mismatch, config mismatch, ID/label/order mismatch, duplicate index, source membership mismatch, descriptor mismatch, and R@1 mismatch. Assert that the returned scientific object has no query/gallery fields.

- [ ] **Step 2: Observe RED, implement strict loader and atomic JSON output, observe GREEN**

  Reuse proven Pass159/CIEB binding utilities where their contracts match; do not loosen tolerances. Reject NaN/Infinity during serialization and write atomically.

- [ ] **Step 3: Add failing RNG snapshot/restore and batch-cache tests**

  Use a tiny transform that consumes Python, legacy NumPy, and torch global RNGs. Prove domain-separated per-example tensors are input-order invariant, repeated bytes are identical, and all caller RNG states are restored. Assert cached tensor and ordered-ID hashes.

- [ ] **Step 4: Implement deterministic transform caching and verify tests**

  Route the official transform through the exact snapshot/seed/call/restore contract. No generator object may be created without affecting the global RNG actually used by the transform.

- [ ] **Step 5: Add failing mocked current-source re-export tests**

  Mock only the expensive model/image forward boundary while keeping digest/config/row/R@1 comparisons real. Require every train/query/gallery row, `atol=rtol=2e-5`, and scientific-object construction only after query/gallery release.

- [ ] **Step 6: Implement binding-only CLI, freeze the manifest, and verify Task 2**

  Run the focused test file, Ruff, py_compile, and diff check. The manifest must pin the four Pass159 artifact paths/digests plus preregistration/source/schema hashes and must be committed before DGX binding values.

- [ ] **Step 7: Commit Task 2**

  Commit with message `bind RSTA diagnostic artifacts` and record exact verification evidence in the SDD task report.

---

### Task 3: Exact VJP/JVP engine and DGX execution

**Files:**
- Modify: `scripts/diagnose_pass200_rsta_stage_a.py`
- Modify: `tests/test_diagnose_pass200_rsta_stage_a.py`
- Create after execution: `docs/pass200_rsta_stage_a_result_2026-08-09.md`
- Modify after execution: `docs/method_search_verdict.md`

**Interfaces:**
- Consumes `TrainingOnlySeedInput`, cached B=180 tensors, Task 1 statistics, and the Task 2 manifest.
- Produces exact contextual `dbar`, full raw motion `b`, self motion `s`, controls, per-row audit records, deterministic result JSON, and PASS_ONWARD/FAIL/UNRESOLVED/INVALID.

- [ ] **Step 1: Add failing float64 dense-Jacobian and BN fixture tests**

  The affine-plus-normalization toy must match literal/dense Jacobian `g`, `b`, and every `s_i` at `1e-8`; central difference uses `1e-5` and tolerance `1e-6`. The two-sample train-BN fixture must prove bufferless functional transformation preserves output/parameter gradients at `1e-6` and does not mutate buffers.

- [ ] **Step 2: Observe RED, implement exact reusable VJP/JVP helpers, observe GREEN**

  Use a full same-batch functional call, encoder-only parameter pytree, detached proxies, contextual PA cotangents, one global VJP/JVP, and serial receiver VJP/JVP. Never vmap receivers or microbatch.

- [ ] **Step 3: Add failing integrity/control tests**

  Cover production PA contextual-vs-singleton inequality, proxy exclusion, adjoint relative error, repeated calculation equality, rotation of head/proxies/supports and all named scalars, missing/zero/radial gradient invalidation, and exact head-only control.

- [ ] **Step 4: Implement the scientific loop and full output schema**

  Persist every receiver/support/foreign/batch ID and every registered norm/alignment/control, plus all hashes, exclusions (which must be empty), bootstrap distribution, criteria, and first decisive clause. No aggregate-only output is accepted.

- [ ] **Step 5: Verify the tiny local suite once**

  Run focused pytest, Ruff, py_compile, and diff check without overlap. Do not run broad local pytest.

- [ ] **Step 6: Commit implementation before any real candidate value**

  Commit with message `implement exact RSTA VJP diagnostic` and push `devbox/emafactorial`.

- [ ] **Step 7: Run DGX binding and tiny BN-Inception smoke**

  Sync the committed source, run the manifest binding-only mode, then one B=180 first-batch VJP/JVP/adjoint/repeatability smoke in a fresh deterministic process. If either fails, fix test-first, review, commit, and rerun; do not inspect candidate aggregates.

- [ ] **Step 8: Execute the full four-seed diagnostic on DGX**

  Start one retained background process, log bounded output, and poll that same PID/session. Do not overlap with Pass120 CIS or its fresh PA control on the single GPU. Collect exit status and result SHA-256.

- [ ] **Step 9: Independently verify and report the result**

  Recompute aggregation/verdict from persisted per-row records in a separate read-only script/path. Write raw signed outcomes and mechanism interpretation to the result doc and ledger whether positive or negative. A PASS authorizes only a new, committed Stage-B virtual-update preregistration; it does not authorize a benchmark claim.

- [ ] **Step 10: Commit and push the result**

  Run diff check, commit result/ledger only, and push `devbox/emafactorial`.
