# Packed int8 cuTile score kernel implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or kill a Rust/cuTile exact signed-int8 score kernel at one million rows before adding any public integration.

**Architecture:** A standalone Rust crate owns canonical wire validation, scalar reference scoring, one cuTile score kernel, and a deterministic benchmark receipt. No Python or library API changes occur until the kernel passes exactness and the frozen speed gate.

**Tech Stack:** Rust stable, cuTile Rust 0.1.1, CUDA Tile IR 13.4, serde JSON.

**Spec:** `docs/superpowers/specs/2026-09-20-packed-int8-cutile-score-design.md`

## Global Constraints

- Keep the canonical gallery at exactly 130 bytes/item.
- Use signed i8 MMA with i32 accumulation and f32 norm scaling.
- Compile and benchmark only on the DGX GB10; local execution is unit tests only.
- One fixed kernel shape per batch; no result-dependent tuning.
- No Python FFI, public API, or device top-k in this slice.

## Review Focus

- Signed extremes `-127/+127` must not overflow or become unsigned.
- Non-unit or non-finite inverse norms must fail before launch.
- Gallery counts not divisible by the tile width need masked loads/stores.
- Equal scores must retain ascending gallery ordinal in the reference top-10.
- JIT/compile time must not contaminate steady-state latency.

### Task 1: Canonical Rust authority

**Files:**
- Create: `Cargo.toml`
- Create: `rust/sfora-cutile-int8-score/Cargo.toml`
- Create: `rust/sfora-cutile-int8-score/src/wire.rs`
- Create: `rust/sfora-cutile-int8-score/src/lib.rs`

**Interfaces:**
- Produces: `PackedRows::new(codes: Vec<i8>, inverse_norms: Vec<f16>, rows: usize, dimensions: usize) -> Result<Self>`
- Produces: `scalar_scores(queries: &PackedRows, gallery: &PackedRows) -> Vec<f32>`

- [ ] Write unit tests for valid rows, signed extremes, mismatched lengths,
  non-finite/nonpositive norms, dimension mismatch, and exact scalar scores.
- [ ] Run `cargo test -p sfora-cutile-int8-score wire -- --nocapture`; require RED
  at missing `PackedRows`/`scalar_scores`.
- [ ] Implement only the validated wire and scalar loop. Use `half::f16` and
  accumulate every product in `i32` before one f32 conversion.
- [ ] Rerun the focused tests and require GREEN.
- [ ] Commit only Task 1 as `feat: add packed int8 Rust authority`.

### Task 2: cuTile score plane

**Files:**
- Create: `rust/sfora-cutile-int8-score/src/kernel.rs`
- Modify: `rust/sfora-cutile-int8-score/src/lib.rs`
- Modify: `rust/sfora-cutile-int8-score/Cargo.toml`

**Interfaces:**
- Consumes: `PackedRows` and `scalar_scores` from Task 1.
- Produces: `score_cutile(device: &Device, queries: &PackedRows, gallery: &PackedRows, batch: BatchShape) -> Result<Vec<f32>>`.

- [ ] Add tests for deterministic random rows, signed extremes, variable norms,
  ties, and a gallery count one row beyond the tile boundary. Compare every f32
  score bit and the lexicographic top-10 to the scalar authority.
- [ ] Run the focused kernel test; require RED at missing `score_cutile`.
- [ ] Implement the fixed signed-i8 `mmai` kernel with i32 accumulator,
  `convert_tile::<f32>`, f32 norm multiplication, and masked tail storage.
- [ ] Run the same test on the DGX with the pinned environment and require GREEN.
- [ ] Commit only Task 2 as `feat: add exact cuTile int8 score kernel`.

### Task 3: One-million-row decision benchmark

**Files:**
- Create: `rust/sfora-cutile-int8-score/src/main.rs`
- Create: `docs/evidence/packed_int8_cutile_score_summary.json`
- Create: `docs/packed_int8_cutile_score_result_2026-09-20.md`

**Interfaces:**
- Consumes: `score_cutile` and `scalar_scores`.
- Produces: canonical newline-terminated JSON with raw timings, p50/p99,
  throughput, persistent/temporary bytes, compile time, device/toolchain
  identity, exactness, and the frozen advance/kill decision.

- [ ] Add a receipt test that recomputes nearest-rank percentiles, rejects
  missing raw samples, and excludes first JIT/compile timing.
- [ ] Run the receipt test RED, implement the minimal benchmark/serializer, and
  rerun GREEN.
- [ ] Commit before execution and transfer the exact commit to the DGX.
- [ ] Run one bounded process with batches 1 and 32, 5 warmups, and 50 samples;
  preserve the original terminal and result SHA-256.
- [ ] Independently recompute exactness, percentiles, bytes, and the frozen
  decision. If either batch fails, ledger the negative and delete the kernel
  candidate. If both pass, write a new device-top-k/FFI design rather than
  expanding this plan.
- [ ] Run `cargo fmt --all -- --check`, focused Cargo tests, `git diff --check`,
  and the repository's existing Python tests touched by documentation only.
- [ ] Commit/push the evidence with configured operator identity and no AI
  attribution.
