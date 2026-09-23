# Packed top-10 RC4 performance implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one exact, measured production packed-search improvement or a finite architectural blocker after stage profiling both supported batches.

**Architecture:** Keep the FFI and Python API fixed. Add a diagnostic split of the fused score/block-select kernel, measure that and the existing merge/transfers/host path on the pinned GB10 gallery, then change only the largest measured production stage if a bounded pilot predicts an end-to-end win.

**Tech Stack:** Rust 2024, cuTile Rust 0.1.1, CUDA/GB10, Python 3.12, Nsight Systems, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-packed-topk-rc4-performance-design.md`

## Global constraints

- Preserve the RC3 tag `v0.3.0-rc3`, public ABI/API, 130-byte/item gallery wire, authenticated artifact semantics, exact score bits, and stable ordinal ties.
- GPU compilation and million-row measurement run only on `riomus@100.104.199.68`; inspect running processes and GPU jobs before any launch.
- Keep JIT/compile outside steady-state timing; use five warmups and 50 samples for released p50/p95/p99 values.
- Bind the baseline library to SHA-256 `a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea` and the frozen RC3 receipt.
- Never start a second DADA or other training job. Preserve pre-existing untracked research files.
- Do not start the Opus 5.5/Astra dual critique until a terminal candidate commit hash exists.

## Review focus

- Tied scores at ordinals 0 and 128 must choose the lower ordinal first in both split and fused paths.
- A 1,000,003-row gallery must mask the padded tail without returning an invalid ordinal.
- Signed extremes and f16 inverse norms must preserve exact f32 score bits.
- Batch 32 must not be inferred from batch-1 timings or silently truncated to one query.
- Profiler, memory, and child-process failures must remain explicit missing values rather than fabricated numbers.

### Task 1: Diagnostic split and exactness

**Files:** `rust/sfora-cutile-int8-score/src/topk.rs`, `rust/sfora-cutile-int8-score/src/lib.rs`.

**Interfaces:** Add `PreparedPackedGallery::search_split_for_profile(&mut self, queries: &PackedRows, batch: BatchShape, k: usize) -> Result<TopKResult, CutileScoreError>`. Keep it out of the C ABI and Python package.

- [ ] Add a Rust test beside `device_top_ten_matches_scalar_across_boundaries_batches_and_ties` that calls `search_split_for_profile` for both batches and gallery sizes 10, 127, 128, 129, and 1,000,003; assert `as_pairs()` exactly matches the existing fused `search` and scalar authority for the small cases.
- [ ] Run the focused test; require RED because `search_split_for_profile` is absent.
- [ ] Add `score_tile_for_profile` with the same signed `mmai`, 32-wide inner block, f32 conversion, and query-then-gallery norm multiplication as `score_block_topk`; write a `[BM,128]` score tile. Add `select_tile_for_profile` that reads this score plane, masks real rows, and performs the existing 10-rank max-score/min-ordinal reduction into `[BM,16]` outputs.
- [ ] Route split outputs through the unchanged `merge_topk` loop and return the same `TopKResult`; keep the public `search` path untouched.
- [ ] Run the focused test on GB10, then the crate's full Rust tests and formatting; require exact bit and ordinal equality.

### Task 2: Authenticated stage receipt

**Files:** `scripts/_scratch_profile_packed_topk_rc4.py`, `docs/evidence/packed_topk_rc4_stage_v1.json`.

**Interfaces:** A command taking absolute baseline/candidate library paths and an exclusive-create output path returns a canonical JSON receipt with raw samples, stage names, hashes, exactness, memory, and environment.

- [ ] Add a parser test that rejects missing batch 32, missing raw samples, nonfinite or zero timings, absent hashes, mismatched score bits/ordinals, and a compile value mixed into steady-state samples; run RED against the missing parser.
- [ ] Implement receipt validation, nearest-rank percentiles, SHA-256 streaming, and a 1,000,000-row deterministic gallery plus 1,000,003-row exactness boundary. Record RC3 matched baseline rows from the frozen receipt, not from an edited claim.
- [ ] On GB10, capture 50 post-warmup public-call samples per batch and separate host/API, H2D, score, block select, merge, and D2H via CUDA trace/event evidence. Hash the raw profiler report and script; record process RSS and GPU peak, with nulls where not retained.
- [ ] Validate and commit the immutable raw receipt separately before changing the production kernel.

### Task 3: One measured production change

**Files:** Only the measured bottleneck's Rust source plus focused Rust tests; update `src/sfora/cutile_int8.py` only if the actual API boundary requires it.

- [ ] State the single candidate and predicted p99 gain from Task 2's largest stage. Set a two-hour pilot cap and an explicit reject gate before editing.
- [ ] Add a failing exactness test for the candidate using signed extremes, ties, batch 1/32, and a padded tail; observe RED.
- [ ] Implement the smallest kernel or layout change, run the focused exactness tests on GB10, and benchmark against the pinned RC3 binary on the same gallery and host.
- [ ] Retain the change only if scores/ordinals remain exact, RSS stays below 2 GiB, the targeted batch p99 materially improves, and the other batch p99 regresses by no more than 5%. Otherwise revert production code and record the finite negative and next distinct bottleneck.
- [ ] Commit the terminal candidate and record its full commit hash; request one Opus 5.5/Astra dual critique and independently verify each finding.

### Task 4: Release decision

**Files:** `docs/rc4_packed_search_decision_table.md`, new evidence receipts, release notes as needed.

- [ ] Run Rust formatting and tests, focused Python packed-gallery/persistence tests, and clean-wheel smoke against the candidate build.
- [ ] Revalidate the untouched Pet and In-Shop quality receipts by hash and frozen metrics without fitting or retuning; do not imply the packed-scorer change changes descriptor quality.
- [ ] Produce an RC4 table containing exactness, p50/p95/p99, throughput, GPU peak, RSS, bytes/item, baseline hashes, quality invariants, and explicit claim limits or missing values.
- [ ] Run the repository's final assurance command once after the diff stabilizes, commit/push `master`, and verify `HEAD == origin/master` with no duplicate DGX job.
