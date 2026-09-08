# PCA-Anchored Relational Transfer Implementation Plan

**Goal:** Test a scale-defined relational correction on train-only development
splits, falsify it on burned CUB/SOP tests, and confirm only a frozen survivor
on a method-specific unopened split.

**Spec:** `docs/superpowers/specs/2026-09-08-pca-trust-region-transfer-design.md`

### Task 1: Resident packed-int4 execution

**Files:** `src/sfora/packed_int4.py`, `src/sfora/__init__.py`,
`scripts/benchmark_packed_int4_resident.py`, and their focused tests.

- [x] Write REDs for exact score/rank parity, layout, dimensions, backend
  authority, fail-closed benchmark mode, and exact fallback behavior.
- [x] Add the 130-byte CPU int8 resident layout without persistent float
  expansion; report 196 bytes when packed and resident layouts coexist.
- [x] Preserve a canonical 10,000-sample DGX benchmark against per-call decode
  and resident float32 with raw timings, memory, backend, and source hashes.
- [x] Obtain independent Astra and Fable/Opus review and repair concrete API,
  exactness, portability, and benchmark blockers.
- [x] Add the receipt, run grouped/static gates, commit, push `HEAD:master`, and
  verify local/remote SHA equality.

### Task 2: Comparable SOP authority

- [ ] Write REDs for a 128D SOP receipt with PCA float/int4, rotated-PCA int4,
  relational float/int4, exact 1,000-update recipe, and model bytes.
- [ ] Implement the authority-preserving evaluator, run focused/static gates,
  review independently, and commit before execution.
- [ ] Run one monitored DGX process, retain negative evidence, and authenticate
  receipt/model hashes. Do not start interpolation without this authority.

### Task 3: Train-only displacement screen

**Files:** create `src/sfora/transfer_geometry.py`,
`scripts/probe_pca_relational_trust_region.py`, and focused tests.

- [ ] Write REDs for normalized-output Procrustes, RMS scale matching, affine
  folding/interpolation, rank failure, grid-bracketed displacement bisection
  with unreachable/non-monotone cases, endpoint controls, and exact
  folded-output equality.
- [ ] Write REDs for float/int4 decomposition, fixed self-excluding scoring,
  10,000 paired class-cluster replicates, `0.05/8` familywise quantile, numeric
  gates, authentication, failure receipts, and explicit CLI execution.
- [ ] Implement reusable geometry helpers and select only on fixed 70/30
  class-disjoint splits of each training partition.
- [ ] Verify focused tests, Ruff, strict mypy, pycompile, diff-check, and obtain
  independent Astra/Fable scientific and source review.
- [ ] Commit the frozen evaluator and verify remote SHA before any burned test.
- [ ] Run one monitored DGX process. Retain every point and use burned CUB/SOP
  tests only to falsify transfer of the inner-split choice.

### Task 4: Separate rank-8 hypothesis

- [ ] Record the correction singular spectrum and rank-8 approximation error;
  do not interpret the fixed-path outcome as a rank-8 bound.
- [ ] If prioritized, freeze exact rank, penalties, optimizer, initialization,
  duration, group rules, angular/neighborhood bounds, selection ties, and final
  refit before a RED.
- [ ] Implement test-first using inner-training rows only and exact-code
  validation. Do not reopen CUB/SOP tests for selection.

### Task 5: Method-specific confirmation

- [ ] Build/test an authenticated UNICOM Cars196 exporter; record historical
  exposure, split identity, duplicate checks, and source/teacher provenance.
- [ ] Freeze one method and numeric mAP@R/R@1/performance gates before quality.
- [ ] Run one monitored no-restart evaluation, retain positive or negative
  evidence, and independently validate every receipt relation.
- [ ] Run focused Python gates, dependency-complete script discovery, full
  repository assurance, and package/build smoke before release claims.
