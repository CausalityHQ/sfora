# Packed int8 device top-k implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return exact top-10 results from the resident packed-int8 cuTile scorer and integrate it as an optional persistent Sfora library backend.

**Architecture:** A fused block top-k kernel reduces exact score tiles to ten candidates and recursively merges candidates on device. A minimal panic-contained C ABI owns the persistent gallery; a Python ctypes wrapper follows the repository's existing native-library pattern.

**Tech Stack:** Rust stable, cuTile Rust 0.1.1, CUDA Tile IR 13.4, ctypes, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-packed-int8-device-topk-design.md`

## Global Constraints

- Keep canonical gallery storage at exactly 130 bytes/item.
- Preserve exact f32 score bits and `(score descending, ordinal ascending)` ranking.
- Compile and benchmark only on the DGX GB10; local work is static inspection and narrow Python tests.
- Use one fixed 128-row score tile and 128-block merge group; no timing-driven tuning.
- CUDA remains optional at Python import time and no fallback is relabeled native.

## Review Focus

- Negative-only score tiles must not admit padded zero rows.
- Equal scores spanning different blocks must select the lower ordinal.
- Gallery sizes one row beyond score and merge boundaries must return ten unique valid rows.
- ABI pointer/length/count errors and Rust panics must return failure without partial output.
- Repeated handle search must not upload the gallery or leak device/host memory.

### Task 1: Exact device top-k

**Files:**
- Create: `rust/sfora-cutile-int8-score/src/topk.rs`
- Modify: `rust/sfora-cutile-int8-score/src/lib.rs`
- Modify: `rust/sfora-cutile-int8-score/src/kernel.rs`

**Interfaces:**
- Consumes: validated `PackedRows`, signed-i8 score arithmetic, and `Arc<Device>`.
- Produces: `PreparedPackedGallery::new` and `search(&mut self, queries, BatchShape, k) -> Result<TopKResult>`.

- [ ] Add failing DGX tests for galleries 10/127/128/129, negative-only values, cross-block ties, both batches, exact scores, exact ordinal order, and unique in-range outputs.
- [ ] Require RED at missing `PreparedPackedGallery` and `TopKResult`.
- [ ] Implement fixed 128-row local selection and recursive 128-block candidate merge with padded candidates fixed at `(-inf, i32::MAX)`.
- [ ] Require all focused tests GREEN and strict Clippy clean on DGX.
- [ ] Commit as `feat: add exact device top-k reduction`.

### Task 2: Persistent native boundary

**Files:**
- Create: `rust/sfora-cutile-int8-score/src/ffi.rs`
- Modify: `rust/sfora-cutile-int8-score/Cargo.toml`
- Create: `src/sfora/cutile_int8.py`
- Create: `tests/test_cutile_int8.py`

**Interfaces:**
- Consumes: `PreparedPackedGallery`.
- Produces: opaque create/search/destroy C ABI and Python `CutilePackedInt8Gallery`.

- [ ] Add Rust RED tests for invalid wire lengths/counts/dimensions, null/misaligned pointers, unsupported batch/k, panic containment, and no partial writes.
- [ ] Add Python RED tests using a tiny compiled fixture library for explicit-path loading, handle lifetime, dtype/shape validation, exact results, error translation, and import without CUDA.
- [ ] Implement the three-operation ABI and ctypes wrapper without changing default scorer selection.
- [ ] Run focused Rust tests on DGX and focused Python tests locally; require GREEN and strict Ruff/Clippy.
- [ ] Commit as `feat: integrate persistent cuTile packed scorer`.

### Task 3: One-million-row end-to-end decision

**Files:**
- Create: `scripts/_scratch_benchmark_cutile_int8_topk.py`
- Create: `tests/test_scratch_benchmark_cutile_int8_topk.py`
- Create: `docs/evidence/packed_int8_cutile_topk_summary.json`
- Create: `docs/packed_int8_cutile_topk_result_2026-09-20.md`

**Interfaces:**
- Consumes: release cdylib and `CutilePackedInt8Gallery`.
- Produces: canonical newline JSON with raw timings, exactness, bytes, RSS, toolchain/source identity, and frozen advance/kill decision.

- [ ] Add a receipt RED test that recomputes nearest-rank percentiles, exact thresholds, throughput, and decision, and rejects missing samples or identity drift.
- [ ] Implement the single-run benchmark for one million rows, batches 1/32, 5 warmups, 50 samples, and matched current/resident controls.
- [ ] Commit source before execution, build the exact commit on DGX, and run one bounded process with no duplicate.
- [ ] Independently recompute every metric and decision; record a negative and remove integration if any gate fails.
- [ ] On pass, run fmt, full focused Rust/Python tests, strict Clippy/Ruff, `git diff --check`, then commit and push evidence to `master`.
