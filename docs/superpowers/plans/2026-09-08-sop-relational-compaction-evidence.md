# SOP Relational Compaction Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce authenticated second-domain SOP quality and CPU evidence for the released generic 66-byte relational linear compressor.

**Architecture:** Add one strict SOP UNICOM embedding exporter and one strict SOP evaluator. Dataset parsing, labels, scoring protocol, and evidence receipts remain outside the generic library; fitting calls the existing public relational compaction API unchanged.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pillow, official UNICOM checkout, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-08-sop-relational-compaction-evidence-design.md`

## Global Constraints

- Use only official SOP train/test metadata and the two registered UNICOM checkpoints.
- Training consumes embeddings only; labels and all test rows are evaluation-only.
- Fixed seeds are `17`, `1729`, and `65537`; all InShop optimizer and packing constants remain unchanged.
- The symmetric test scorer excludes the query's own row before MAP@R and Recall@1.
- No existing dirty-worktree path outside this plan may be edited or staged.
- DGX extraction and science start only after focused local tests and static gates pass.

---

### Task 1: Official SOP parser and authenticated archive writer

**Files:**
- Create: `scripts/export_unicom_sop_embeddings.py`
- Create: `tests/test_export_unicom_sop_embeddings.py`

**Interfaces:**
- Consumes: official SOP root, injected `encode_batch(tuple[Path, ...]) -> np.ndarray`, registered model metadata.
- Produces: `parse_sop_records`, `export_sop_embeddings`, and one exact train/test NPZ archive.

- [ ] **Step 1: Write failing parser tests** for headers, exact counts, split disjointness, duplicate/missing/symlink paths, deterministic ordered manifest SHA-256, and builtin concrete types.
- [ ] **Step 2: Run** `uv run pytest -q tests/test_export_unicom_sop_embeddings.py -k parse` **and require a missing-interface RED.**
- [ ] **Step 3: Implement the minimal immutable `SopRecord` parser** with the registered official counts and path authority.
- [ ] **Step 4: Rerun the parser selection and require GREEN.**
- [ ] **Step 5: Write failing archive tests** for exact keys, metadata, array digests, float32/label/path authority, atomic exclusive output, encoder shape/nonfinite failures, and owned-partial cleanup.
- [ ] **Step 6: Run the archive selection and require a missing-interface RED.**
- [ ] **Step 7: Implement the bounded archive writer and strict reload validator.**
- [ ] **Step 8: Rerun the complete exporter test file and require GREEN.**

### Task 2: Pinned official UNICOM CLI

**Files:**
- Modify: `scripts/export_unicom_sop_embeddings.py`
- Modify: `tests/test_export_unicom_sop_embeddings.py`

**Interfaces:**
- Consumes: absolute checkout/checkpoint/dataset/output paths, registered model identifier, positive batch size, `--execute-export`.
- Produces: one local archive; no network or dataset mutation surface.

- [ ] **Step 1: Write failing CLI tests** for both registered model/checkpoint pairs, checkout revision, exact checkpoint digest, absolute paths, missing/duplicate/unknown flags, existing output, and absent execution sentinel.
- [ ] **Step 2: Run CLI tests and preserve RED at the missing strict parser/model selector.**
- [ ] **Step 3: Implement strict argument parsing and official model loading** for only `ViT-B/16` and `ViT-L/14@336px`.
- [ ] **Step 4: Run the complete exporter test file plus scoped Ruff, mypy, py_compile, and `git diff --check`.**

### Task 3: Symmetric SOP packed retrieval scorer

**Files:**
- Create: `scripts/probe_sop_relational_linear.py`
- Create: `tests/test_probe_sop_relational_linear.py`

**Interfaces:**
- Consumes: packed or float test embeddings and integer class labels.
- Produces: exact leave-self-out per-query AP@R/Recall@1 and aggregates.

- [ ] **Step 1: Write failing scorer tests** for self exclusion, deterministic ties, singleton rejection, positive counts, packed/float representation equality, chunking, nonfinite inputs, and candidate width.
- [ ] **Step 2: Run scorer tests and preserve the missing-interface RED.**
- [ ] **Step 3: Implement `_score_symmetric`** with fixed row ordinals as the final tie key and no full score-matrix retention.
- [ ] **Step 4: Rerun scorer tests and require GREEN.**

### Task 4: Strict paired loader and frozen evaluator

**Files:**
- Modify: `scripts/probe_sop_relational_linear.py`
- Modify: `tests/test_probe_sop_relational_linear.py`

**Interfaces:**
- Consumes: exact source/teacher archive paths and SHA-256 values plus three fresh output paths.
- Produces: canonical quality receipt, `SFORA-RL1` model, and paired latency receipt.

- [ ] **Step 1: Write failing loader/CLI tests** for exact archive keys, model/checkpoint/manifest binding, row-order equality, train/test cardinalities, finite values, train/test class disjointness, output exclusivity, and execution sentinel.
- [ ] **Step 2: Run the loader selection and preserve RED.**
- [ ] **Step 3: Implement strict loading and CLI authority.**
- [ ] **Step 4: Write failing synthetic end-to-end tests** that require PCA64/PCA128/three relational arms, three fixed seeds, packed 66-byte scoring, class-cluster bootstrap, all fixed gates, raw paired latency samples, library/script/model/archive hashes, claim ineligibility, and atomic publication.
- [ ] **Step 5: Run end-to-end tests and preserve RED.**
- [ ] **Step 6: Implement the minimal frozen evaluator by reusing the public Sfora library and shared testable helpers.**
- [ ] **Step 7: Run both focused test files and scoped Ruff, mypy, py_compile, and `git diff --check`.**

### Task 5: DGX extraction and one sealed SOP evaluation

**Files:**
- Evidence output only after success: `docs/evidence/relational_linear_compaction/`
- Modify after authenticated results: `docs/relational_linear_compaction.md`

**Interfaces:**
- Consumes: clean committed scripts, official SOP dataset, pinned checkout/checkpoints.
- Produces: two authenticated embedding archives, one quality receipt, one deployment model, and one latency receipt.

- [ ] **Step 1: Sync only the committed owned files to a fresh DGX staging directory and verify source SHA-256 values.**
- [ ] **Step 2: Run the B16 exporter once under the existing monitor; preserve terminal, archive SHA-256, cardinalities, GPU/pressure evidence, and PID clearance.**
- [ ] **Step 3: Run the L14 exporter once under the same rules; do not overlap models or rerun a terminal session.**
- [ ] **Step 4: Freeze both archive hashes, then run the evaluator once; preserve canonical outputs, metrics, bootstrap bounds, latency, pressure, and PID clearance.**
- [ ] **Step 5: If a fixed gate fails, record the failure without adaptive SOP tuning. If all gates pass, copy only authenticated final artifacts into release evidence and update limitations honestly.**

### Task 6: Independent review, assurance, and delivery

**Files:**
- Review: every Task 1-5 path and authenticated receipt.

**Interfaces:**
- Consumes: stable diff and scientific evidence.
- Produces: one scoped commit and fast-forward delivery to canonical `master` only after verification.

- [ ] **Step 1: Obtain independent Astra and Fable read-only reviews from exact repository/evidence hashes; reconcile findings locally.**
- [ ] **Step 2: Apply verified blockers through focused RED/GREEN cycles only.**
- [ ] **Step 3: Run focused tests, dependency-complete Python discovery, scoped Ruff/mypy/py_compile/diff, and the repository full assurance command once.**
- [ ] **Step 4: Force-add only the ignored spec/plan plus exact owned code/tests/docs/evidence; assert staged scope and `git diff --cached --check`.**
- [ ] **Step 5: Commit with configured operator identity and no attribution trailers, integrate into `master`, push fast-forward, and verify `HEAD == origin/master == ls-remote refs/heads/master`.**
