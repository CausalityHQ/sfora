# SigLIP Class-Disjoint Gradient-Agreement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic optimization-only cached-feature diagnostic that measures whether symmetric class-domain gradient conflict removal improves leave-class-out Cars retrieval.

**Architecture:** A focused library reuses the existing SFQ fold authority and head-screen spectral/proxy primitives, trains matched ordinary and CDGA projections on each fit partition, and emits strict canonical evidence. A thin local CLI authenticates the existing cache but opens only the optimization band.

**Tech Stack:** Python 3.12, PyTorch, existing SFORA canonical JSON/cache authority.

**Spec:** `docs/superpowers/specs/2026-09-01-siglip-cdga-design.md`

## Global Constraints

- Never fit or select on clean-validation, burned-diagnostic, or official-test rows.
- Use exactly four existing SFQ class-disjoint folds and a 512-dimensional bias-free projection.
- Use symmetric projection-gradient conflict removal with epsilon `1e-12`, separate norm-10 projection/proxy clipping, and no second-order gradients.
- Keep comparator and CDGA initialization, batches, losses, optimizer constants, and step count identical.
- Pass gates are +2,000 ppm aggregate over comparator, not below spectral, and no fold worse by more than 10,000 ppm.
- Do not start a scientific GPU job while the three-seed control is active.

---

### Task 1: Pseudo-domain authority and gradient projection

**Files:**
- Create: `src/sfora/siglip_cdga.py`
- Create: `tests/test_siglip_cdga.py`

**Interfaces:**
- Consumes: `SFQFoldSchedule`, `FeatureSplitAuthority`.
- Produces: `CDGADomainSplit`, `build_cdga_domain_split(...)`, and `symmetric_conflict_projection(...)`.

- [ ] **Step 1: Write failing tests** for deterministic disjoint label allocation, rejection of validation labels, exact nonconflict identity, exact symmetric conflict projection, zero norms, nonfinite inputs, and concrete-type drift.
- [ ] **Step 2: Run** `uv run --python 3.12 --with-requirements scripts/requirements-format-bench.txt python -m unittest tests.test_siglip_cdga.CDGAPrimitiveTests` and require missing-symbol RED.
- [ ] **Step 3: Implement** frozen dataclasses, domain hashing, seed-derived rotation, and tensor gradient projection with finite/shape/type checks.
- [ ] **Step 4: Rerun the exact selector** and require GREEN.
- [ ] **Step 5: Commit** only the primitive/test slice as `Add CDGA gradient authority`.

### Task 2: Matched fold training and retrieval evidence

**Files:**
- Modify: `src/sfora/siglip_cdga.py`
- Modify: `tests/test_siglip_cdga.py`

**Interfaces:**
- Produces: `CDGAFoldEvidence`, `train_cdga_fold(...)`, `run_cdga_fold_diagnostic(...)`, and `validate_cdga_result_bytes(...)`.

- [ ] **Step 1: Write failing tests** proving held labels never enter training, comparator/CDGA share initialization and batches, only the projection gradient differs, deterministic replay, integer retrieval counts, conflict evidence, exact gates, and rejection of every result field mutation.
- [ ] **Step 2: Run** the focused training/result test classes and require missing-interface RED.
- [ ] **Step 3: Implement** one class-proxy parameter set per arm, two domain losses per step, ordinary comparator update, CDGA projection-gradient replacement, identical proxy updates, held-fold scoring, integer aggregation, and canonical validation.
- [ ] **Step 4: Run** all `tests.test_siglip_cdga` tests and require GREEN.
- [ ] **Step 5: Commit** the matched training/result slice as `Add CDGA fold diagnostic`.

### Task 3: Optimization-only CLI

**Files:**
- Create: `scripts/diagnose_siglip_cdga.py`
- Create: `tests/test_diagnose_siglip_cdga.py`

**Interfaces:**
- Consumes: authenticated feature-cache manifest and expected source/control identities.
- Produces: one canonical `sfora-siglip-cdga-fold-diagnostic-v1` file.

- [ ] **Step 1: Write failing tests** for strict arguments, duplicate/unknown flags, absent execution flag, manifest/source drift, missing optimization band, sentinels proving clean/burned files are never opened, result binding, and non-clobber publication.
- [ ] **Step 2: Run** `python -m unittest tests.test_diagnose_siglip_cdga` and require import/interface RED.
- [ ] **Step 3: Implement** the local-only loader by following `diagnose_siglip_sfq.py`, removing every evaluation-band capability, invoking the library once, revalidating canonical bytes, and atomically publishing a new file.
- [ ] **Step 4: Run** focused library and CLI files and require GREEN.
- [ ] **Step 5: Commit** the CLI slice as `Add local CDGA diagnostic`.

### Task 4: Assurance, review, and deployment fence

**Files:**
- Modify only if a verified defect is found in the three CDGA files/tests above.

**Interfaces:**
- Produces: reviewed source suitable for a later separately authorized cached-feature run.

- [ ] **Step 1: Run** Ruff on the four production/test files, `python3 -m py_compile` on both production files, formatter checks, and `git diff --check`.
- [ ] **Step 2: Run** dependency-complete `python -m unittest discover -s tests -p 'test_*.py'` once after focused gates are green.
- [ ] **Step 3: Obtain** a read-only cross-provider review of leakage, mathematical equivalence, optimizer matching, evidence recomputation, and mutation coverage; reproduce each actionable issue before repair.
- [ ] **Step 4: Commit and push** every verified repair to `origin/devbox/emafactorial`, then require exact local/remote SHA equality and a clean tracked worktree.
- [ ] **Step 5: Keep science fenced** until the active control terminates; run SFQ first, RSTA second, and CDGA only if its cached-feature input exists and no other scientific GPU process is active.
