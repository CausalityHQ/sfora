# CUB Relational Int4 Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce authenticated class-disjoint CUB transfer evidence for the generic 66-byte relational int4 method, then use only burned SOP evidence to select the next quantization-aware improvement.

**Architecture:** A local exporter authenticates the official archive and extracted bytes, then emits aligned source/teacher NPZ artifacts. A separate local evaluator owns strict paired loading, equal-byte controls, three-seed training, deterministic quality/statistics, exact deployed CPU timing, and rollback-safe canonical publication. Scientific artifacts are produced only from committed source.

**Tech Stack:** Python 3.12/3.13, NumPy, PyTorch, pytest, Ruff, mypy, UNICOM, SHA-256.

**Spec:** `docs/superpowers/specs/2026-09-08-cub-relational-int4-evidence-design.md`

## Global Constraints

- Work only in the Sfora repository; never modify or stop Borsuk work.
- Never use CUB test classes 101-200 for fitting or method selection.
- Persistent treatment/control storage is exactly 66 bytes per gallery item.
- No CUB quality output may be opened before source and controls are committed.
- Every long DGX command has one original process, bounded monitoring, and no duplicate restart.

---

### Task 1: Authenticated CUB exporter

**Files:**
- Create: `scripts/export_unicom_cub_embeddings.py`
- Create: `tests/test_export_unicom_cub_embeddings.py`

**Interfaces:**
- Produces: `parse_cub_records`, `verify_extracted_cub_matches_archive`, `export_cub_embeddings`, and `load_cub_embedding_archive`.
- Consumes: official archive/root plus one pinned UNICOM checkout/checkpoint.

- [x] **Step 1: Write parser/export/load tests, including protocol mutations and exclusive publication.**
- [x] **Step 2: Run focused tests and preserve missing-interface REDs.**
- [x] **Step 3: Implement strict archive, record, array, norm, and class-split authority.**
- [x] **Step 4: Add byte-for-byte tar-member versus extracted-root verification and preserve its RED/GREEN.**
- [ ] **Step 5: From committed source, export B16 and L14 archives and verify hashes, dimensions, row identities, and content-manifest equality.**

### Task 2: Sealed equal-byte CUB evaluator

**Files:**
- Create: `scripts/probe_cub_relational_int4.py`
- Create: `tests/test_probe_cub_relational_int4.py`

**Interfaces:**
- Consumes: two authenticated aligned embedding archives and committed source SHA.
- Produces: canonical quality JSON, deployment model bytes, and canonical latency JSON.

- [x] **Step 1: Write REDs for paired authority, exact int4 quality, ridge and rotation controls, CLI refusal, and publication.**
- [x] **Step 2: Implement the strict loader, controls, scorer, three-seed orchestration, and explicit-only CLI.**
- [x] **Step 3: Mutation-lock exact deployed ranking semantics and all-or-nothing three-output publication.**
- [x] **Step 4: Bind input/source identities, recipe/device evidence, derived byte width, optimizer steps, and a real reduced end-to-end pipeline test.**
- [x] **Step 5: Add PCA64-int8 and float diagnostics, enforce 1,000 fixed optimizer steps, and run the complete focused/static gate.**

### Task 3: Review and commit the frozen CUB boundary

**Files:**
- Modify only files listed in Tasks 1-2 and this spec/plan.

**Interfaces:**
- Produces: one clean commit whose SHA is required by Task 4.

- [x] **Step 1: Obtain independent read-only Astra and Fable reviews from the same repository evidence.**
- [x] **Step 2: Reconcile concrete blockers: root-content authority, matched control, source binding, timing parity, and rollback safety.**
- [x] **Step 3: Run `pytest` on both focused files, Ruff, strict mypy, `py_compile`, and `git diff --check`.**
- [ ] **Step 4: Commit/push to canonical `master`, verify local/remote SHA equality and clean status.**

### Task 4: Re-export without opening quality

**Files:**
- Create remotely: two immutable CUB NPZ artifacts; do not commit binary data.

**Interfaces:**
- Consumes: Task-3 commit and official authenticated inputs.
- Produces: B16/L14 archive SHA-256 identities sharing one ordered/content manifest.

- [ ] **Step 1: Run one B16 export and one L14 export serially or under explicitly bounded non-overlapping GPU jobs.**
- [ ] **Step 2: Strict-load both archives, compare every row identity, and record exact hashes/sizes/model authorities.**
- [ ] **Step 3: Stop if archive/content/row authority differs; do not run the evaluator.**

### Task 5: SOP-only quantization falsifiers

**Files:**
- Create: `scripts/probe_sop_quantization_geometry.py`
- Create: `tests/test_probe_sop_quantization_geometry.py`
- Modify after RED: `src/sfora/packed_int4.py`

**Interfaces:**
- Consumes: already-burned authenticated SOP embeddings only.
- Produces: one frozen codec/rotation decision; no CUB access.

- [ ] **Step 1: RED/GREEN exact asymmetric float-query/int4-gallery scoring and reject if MAP@R gain is below 0.003.**
- [ ] **Step 2: RED/GREEN fixed-rotation clipped-scale int4 with train-only scale selection; reject if it recovers less than half the observed PCA128 float-to-int4 loss.**
- [ ] **Step 3: If Step 2 passes, RED/GREEN a train-only orthogonal/STE quantization-aware projection and compare three fixed seeds against matched non-STE training.**
- [ ] **Step 4: Benchmark a bounded predecoded/fused packed CPU path against shipped 64D int8; require deterministic ranking equality and report working memory separately from 66-byte persistence.**
- [ ] **Step 5: Freeze exactly one winner before Task 6; otherwise retain the current relational128-int4 recipe unchanged.**

### Task 6: One-shot CUB evaluation and evidence delivery

**Files:**
- Modify after terminal evidence: `docs/research/publication-evidence-ledger.md` if present, otherwise the repository's existing evidence ledger selected by `rg`.

**Interfaces:**
- Consumes: Task-3 source SHA, Task-4 archives, and Task-5 frozen method.
- Produces: one claim-ineligible transfer result and an evidence commit.

- [ ] **Step 1: Preflight exact input/source/output authorities without scoring.**
- [ ] **Step 2: Run one monitored evaluator process; preserve the original terminal and never duplicate/restart after a scientific terminal.**
- [ ] **Step 3: Validate canonical JSON/model hashes, all quality gates, latency samples, resource state, and cleanup.**
- [ ] **Step 4: Record successes and failures without suppressing controls or negative results, validate docs, commit/push, and notify the operator.**
