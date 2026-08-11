# Pass 205 RDGC DGX Runtime Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Pass 205 RDGC's prospective Python version contract from the local CPU host's `3.12.3` to the designated DGX host's `3.13.9`, then issue a reviewed replacement source and manifest handoff before the still-unused one-shot run.

**Architecture:** A prospective runtime amendment records the prelaunch defect and freezes the scientific boundary. Tests first prove the launch and persisted validators reject the old value, the implementation changes only RDGC runtime self-description and authority-chain handling, and a replacement manifest-only child binds the reviewed source. The historical Pass 200 verifier remains byte-identical.

**Tech Stack:** Python 3.13.9 on the DGX, PyTorch 2.12.1+cu130 and NumPy 2.5.0 as observed runtime values, pytest, Ruff, `py_compile`, strict insertion-ordered JSON, SHA-256, Git blob authentication.

## Global Constraints

- The amendment commit is `29f0600d64d92d931ab2f57e04a59d9daba209d6`; its document SHA-256 is `eb18908fc8a514e5ac3f0deb67950eeaf2a256ad20e1ae594e4fbd6fb2f74df0`.
- The reviewed but unexecuted old handoff is `3c9e6b9fe5494f8f6a98caab18ff4923b66734ec`; it is not an attempt and must never execute.
- Modify source only in `scripts/diagnose_pass205_rdgc_stage_b.py` and `tests/test_diagnose_pass205_rdgc_stage_b.py`.
- Do not modify `scripts/verify_pass200_rsta_scientific_artifact.py`, its tests, any scientific formula, threshold, seed, selection, control, graph schedule, artifact, result, or Torch/NumPy behavior.
- Do not open the old RSTA scientific artifact. The outcome-blind VALID receipt remains the sole upstream scientific authority.
- A replacement handoff must be a manifest-only direct child of the final independently reviewed source.
- No DGX scientific process may launch until replacement source and manifest reviews are READY.
- Exactly one scientific process remains authorized; no retry is permitted after launch.

---

### Task 1: Establish the Prospective Runtime RED Boundary

**Files:**
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py`
- Read only: `scripts/verify_pass200_rsta_scientific_artifact.py`
- Read only: `tests/test_verify_pass200_rsta_scientific_artifact.py`

**Interfaces:**
- Consumes: RDGC `main`, `_validate_environment`, and full/reduced result validators.
- Produces: focused tests that distinguish the prospective RDGC runtime from retrospective Pass 200 provenance.

- [ ] **Step 1: Add exact RED tests**

  Add tests that require:

  ```python
  RDGC_PYTHON_VERSION = "3.13.9"
  RDGC_PYTHON_VERSION_INFO = (3, 13, 9)
  ```

  The public CLI must reject a monkeypatched non-`3.13.9` version before
  manifest/model access. Full and reduced persisted payloads must reject
  `"3.12.3"`, integer values, string subclasses, and empty strings. One test
  must prove the CLI tuple and payload literal are derived from the same two
  constants. Another must mutate arbitrary nonempty built-in Torch/NumPy
  version strings and prove they remain observational rather than literal
  pins.

- [ ] **Step 2: Run the focused selector and preserve RED evidence**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py \
    -k 'dgx_python_runtime or runtime_version_consistency or version_fields_remain_observational'
  ```

  Expected: the old `3.12.3` implementation fails the new prospective tests.

- [ ] **Step 3: Prove historical Pass 200 remains unchanged**

  ```bash
  git diff --exit-code 328a70ad809c1adeae9ccd2aea28b87f3243018b -- \
    scripts/verify_pass200_rsta_scientific_artifact.py \
    tests/test_verify_pass200_rsta_scientific_artifact.py
  .venv/bin/pytest -q tests/test_verify_pass200_rsta_scientific_artifact.py \
    -k 'python_version or runtime'
  ```

  Expected: byte-identical files and GREEN retrospective `3.12.3` coverage.

---

### Task 2: Implement the Minimal Runtime and Authority-Chain Repair

**Files:**
- Modify: `scripts/diagnose_pass205_rdgc_stage_b.py`
- Modify: `tests/test_diagnose_pass205_rdgc_stage_b.py`

**Interfaces:**
- Consumes: amendment commit/path/SHA and this plan commit/path/SHA.
- Produces: one reviewed source chain that authenticates the exact old-handoff → amendment → plan → source → replacement-handoff chronology.

- [ ] **Step 1: Add one shared prospective version constant**

  Define exact built-in immutable constants:

  ```python
  RDGC_PYTHON_VERSION_INFO = (3, 13, 9)
  RDGC_PYTHON_VERSION = "3.13.9"
  ```

  Use them only in the RDGC CLI version gate and RDGC environment/result
  validator. Do not change Torch or NumPy construction, validation, or values.

- [ ] **Step 2: Bind the runtime amendment and plan**

  Add literal path/SHA/commit constants for:

  ```text
  docs/pass205_rdgc_dgx_runtime_amendment_2026-08-11.md
  docs/superpowers/plans/2026-08-11-pass205-rdgc-dgx-runtime-repair.md
  ```

  `authenticate_authority` must require the old handoff's sole parent to be the
  old reviewed source, the amendment's sole parent to be the old handoff, this
  plan's sole parent to be the amendment, every subsequent source commit to be
  merge-free and change a nonempty subset of only the RDGC diagnostic/test,
  the aggregate plan-to-source scope to be exactly those two files, and the
  replacement handoff to be a sole manifest-file child of final source.

- [ ] **Step 3: Update future-manifest authority expectations**

  The `implementation_plan` reference must bind this plan. Candidate,
  upstream RSTA, literature audit, validation receipt, historical artifacts,
  schema, seed array, and 33-path source order remain unchanged. Update only
  tests/fixtures needed for the new exact plan/source/handoff chronology.

- [ ] **Step 4: Run GREEN focused tests**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py \
    -k 'dgx_python_runtime or runtime_version_consistency or version_fields_remain_observational or authority or manifest or source'
  ```

  Expected: all selected tests pass, including wrong-parent, merge, extra-path,
  old-handoff-as-current, and historical-verifier-preservation negatives.

- [ ] **Step 5: Commit source/test only**

  ```bash
  git add -- scripts/diagnose_pass205_rdgc_stage_b.py \
    tests/test_diagnose_pass205_rdgc_stage_b.py
  test "$(git diff --cached --name-only)" = $'scripts/diagnose_pass205_rdgc_stage_b.py\ntests/test_diagnose_pass205_rdgc_stage_b.py'
  git diff --cached --check
  git commit -m "fix RDGC DGX runtime contract"
  ```

---

### Task 3: Assure and Independently Review the Replacement Source

**Files:**
- Read: the two Task 2 files and all frozen authority documents.

**Interfaces:**
- Consumes: final source commit `V_G2`.
- Produces: independent READY verdict before manifest construction.

- [ ] **Step 1: Run complete affected assurance**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py
  .venv/bin/ruff check scripts/diagnose_pass205_rdgc_stage_b.py \
    tests/test_diagnose_pass205_rdgc_stage_b.py
  .venv/bin/python -m py_compile scripts/diagnose_pass205_rdgc_stage_b.py \
    tests/test_diagnose_pass205_rdgc_stage_b.py
  git diff --check
  ```

- [ ] **Step 2: Run the existing resource-aware repository assurance gate**

  Reuse the already-established exclusion only for the unrelated cgroup-thrash
  test `tests/test_cli.py::test_image_end_to_end_auto_recipe_accepts_recall_at_k_surrogate`.
  No overlapping pytest processes are allowed.

- [ ] **Step 3: Obtain independent source review**

  Require no Critical/Important findings on runtime literal consistency,
  retrospective Pass 200 byte preservation, exact history topology, manifest
  schema, scientific non-reachability, and unchanged candidate math.

---

### Task 4: Build and Review the Replacement Manifest-Only Handoff

**Files:**
- Replace in a direct-child commit: `docs/pass205_rdgc_stage_b_manifest.json`

**Interfaces:**
- Consumes: independently READY `V_G2`, VALID outcome-blind receipt SHA
  `943c4c93d1c5fe26ea288fc8bced0c416744a606b0d501f0c9da9b2aa2df1410`.
- Produces: reviewed replacement handoff `HV_G2` and its unique output path.

- [ ] **Step 1: Regenerate exact manifest from authenticated authorities**

  Preserve the exact ten-key top-level schema and all prior domains. Bind the
  new plan path/SHA/commit, `current_scientific_source.git_revision=V_G2`, all
  33 ordered `V_G2` source blobs/worktree hashes, and the new RDGC diagnostic
  digest. Do not include `HV_G2`, runtime result values, or candidate outcomes.

- [ ] **Step 2: Validate before commit**

  Run `validate_future_manifest`, verify all 33 Git blobs/worktree hashes,
  validate the outcome-blind receipt using its authenticated verifier, derive
  all four historical seed records from the Pass 200 validators, and run the
  complete RDGC test file.

- [ ] **Step 3: Commit only the manifest**

  The replacement handoff must be the direct child of `V_G2`, with exactly one
  changed path and no source/test/result changes.

- [ ] **Step 4: Authenticate detached and obtain independent review**

  In a fresh detached worktree, copy only the pinned outcome-blind receipt and
  run production `authenticate_authority`. Require independent READY on exact
  bytes/order, parentage, receipt, four seeds, 33 sources, no self-cycle, and
  absence of result values.

---

### Task 5: Execute the Still-Unused One-Shot DGX Run

**Files:**
- Create at most once: `reports/generated/pass205_rdgc_stage_b/<HV_G2>-rdgc-stage-b.json`

**Interfaces:**
- Consumes: reviewed `V_G2/HV_G2`, registered DGX `.venv`, historical artifacts, outcome-blind receipt.
- Produces: exactly one PASS/CLOSE/UNRESOLVED/INVALID result or one structural stop.

- [ ] **Step 1: Fresh detached preflight**

  Authenticate exact commits, manifest, 33 sources, receipt, four historical
  seed artifacts, clean checkout, `.venv` interpreter, Python `3.13.9`, observed
  PyTorch `2.12.1+cu130`, observed NumPy `2.5.0`, CUDA availability, empty
  process/queue/GPU state, output parent, and absent output/temp. Do not rebuild
  or alter the environment.

- [ ] **Step 2: Run candidate-free CLI tests**

  ```bash
  .venv/bin/pytest -q tests/test_diagnose_pass205_rdgc_stage_b.py \
    -k 'authority or manifest or receipt or cli or unreachable'
  ```

- [ ] **Step 3: Launch exactly once**

  ```bash
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -I -B scripts/diagnose_pass205_rdgc_stage_b.py \
      --manifest docs/pass205_rdgc_stage_b_manifest.json \
      --output "reports/generated/pass205_rdgc_stage_b/${HV_G2}-rdgc-stage-b.json" \
      --scientific-once
  ```

  Retain the original PID/session. Do not pipe, tee, interrupt, inspect partial
  values, or retry.

- [ ] **Step 4: Validate and stop**

  In a separate CPU-only process, strict-load and validate the final result.
  Record only SHA-256, status, phase, `V_G2`, and `HV_G2` before independent
  review. Under every result branch, do not train, benchmark, integrate, rewrite,
  or run again.
