# RC5 packed top-k output buffers implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkboxes for tracking.

**Goal:** Remove measured per-request output-fill work from the retained packed top-k scorer if exactness and paired public-call latency gates pass.

**Architecture:** Allocate CUDA output tensors without the cuTile `full` kernel only on the production fused path. Each immediately following CUDA kernel overwrites every score and ordinal lane. Keep the 2048-entry merge, API, wire, and diagnostic split path fixed.

**Tech Stack:** Rust 2024, cuTile 0.1.1, CUDA GB10, Python 3.12, Nsight Systems.

**Spec:** `docs/superpowers/specs/2026-09-23-packed-topk-rc5-output-buffers-design.md`

## Global constraints

- Work from pushed `master` and preserve unrelated untracked research files.
- Heavy CUDA work runs only on `riomus@100.104.199.68`; check jobs before each launch.
- Keep RC3 `0.3.0rc3` as the release until a separate RC5 decision qualifies a new version.
- Authenticate the pinned RC3 library, public Python API, and all fixture files before timing.
- Do not discard p99 spikes or repeat pilots selectively. No dual critique before a terminal candidate hash.

## Review focus

- Every `[BM,16]` output tile must be stored before any read, including padded gallery and merge tail tiles.
- Allocation and kernel failures must never return or inspect uninitialized contents.
- Signed extremes and equal-score lower-ordinal ties must remain bit/order exact.
- Batch 1 must retain its guardrail even though the measured initialization cost is batch-32 dominant.
- The diagnostic split must remain a valid independent exactness control.

## Task 1: Freeze benchmark and implement one allocation intervention

**Files:** `rust/sfora-cutile-int8-score/src/topk.rs`; frozen spec above.

- [ ] Confirm baseline `MERGE_WIDTH=2048`, stage `buffer_initialization=795377` ns for batch 32, and prior public API source and fixture hashes.
- [ ] Add one `unsafe fn fully_overwritten_output<T: DType>(shape: &[usize], stream: &Arc<Stream>) -> Result<Tensor<T>, CutileScoreError>` that calls `Tensor::<T>::uninitialized(shape.iter().product()).sync_on(stream)`, then `assume_init().reshape(shape)`. State the full-store precondition and put the `assume_init` in its own documented unsafe block.
- [ ] For the production branch only, call that helper for fused score/ordinal output and each merge score/ordinal output. Keep `cutile::api::full` for the split branch. At each unsafe call, identify the following kernel and the grid that fully stores the tensor.
- [ ] Run `cargo fmt --check` and the crate's focused exactness test on DGX. Require exact scores and stable ordinals on 10/127/128/129-row boundaries and both batches.

## Task 2: Bounded DGX causal and public-call pilot

**Files:** candidate native library and new raw receipt under `docs/evidence/`.

- [ ] Build the exact source hash once on the idle DGX. Capture the library hash; do not overwrite the RC3 or rejected RC4 library.
- [ ] Check score bits and ordered ordinals against the pinned RC3 authority for 1,000,000 and 1,000,003 rows, batches 1 and 32. Fail on any mismatch.
- [ ] Nsight-profile a warmed batch-32 call and compare output-fill kernel count, tracked CUDA pool peak, and stage time against the retained RC3 trace. Stop if the six fill launches remain or peak exceeds 200 MB.
- [ ] Run two independent paired 50-call public Python API replays after five warmups per shape, opposite arm order. Compute nearest-rank p50/p95/p99 and mean-based throughput from every raw sample. Pass only if each pair has at least 10% batch-32 p99 gain, no more than 5% batch-1 p99 regression, RSS below 2 GiB, and exactness.
- [ ] Archive command lines, scripts, raw JSON, Nsight reports, fixture manifest, candidate/RC3/API hashes, and any failure. If the gate fails, revert the production allocation change and publish a finite negative decision before trying a different architecture.

## Task 3: Production assurance and decision, only for a passing pilot

**Files:** production source, decision table, quality/package receipts.

- [ ] Run crate tests, focused Python packed-gallery tests, static formatting/lint, and one clean wheel install/smoke against the candidate and pinned RC3 library.
- [ ] Rehash the untouched Pet/In-Shop quality receipts and report their frozen metrics without retraining.
- [ ] Commit the terminal candidate and raw evidence, then request one Claude Opus 5.5 and GPT-6 Astra read-only critique of that exact hash. Independently verify and address findings.
- [ ] Run the repository's full assurance command once after the final diff stabilizes. Publish a decision table with exactness, both batches' p50/p95/p99 and throughput, tracked GPU pool peak, RSS, bytes/item, hashes, quality limits, and any missing values. Push `master` and verify remote HEAD.
