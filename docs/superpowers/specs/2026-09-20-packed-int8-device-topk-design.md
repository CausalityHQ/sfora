# Packed int8 device top-k and Sfora integration design

## Purpose

The exact signed-int8 cuTile score plane passed its frozen one-million-row
gate at source `87203558b79a4f49b63668431477fb1c1a75a1f8`: batch-1 p99 was
1.002 ms and batch-32 p99 was 1.477 ms, with exact scalar score bits and a
130-byte/item resident gallery. Returning the dense score plane would still
move 4 MB or 128 MB per call and would not be a usable library path.

This slice fuses deterministic top-10 selection on device, exposes a persistent
gallery handle through the repository's established ctypes/native pattern, and
measures the resulting Python-library call end to end. It does not change the
learned descriptor, train a model, or make a retrieval-quality claim.

## Exact ranking authority

Ranking is descending by authoritative f32 score and then ascending gallery
ordinal. NaNs and non-finite inverse norms are rejected before launch. The
result is exactly ten unique `(ordinal, score)` pairs per query when the gallery
contains at least ten rows.

The device reduction has two stages:

1. The signed-i8 MMA score kernel handles 128 gallery rows per block. Ten
   repeated `reduce_max` / equality / `reduce_min` selections retain the exact
   local top ten. cuTile requires power-of-two tile dimensions, so the physical
   result tile has 16 lanes: ten authoritative candidates followed by six
   `(-inf, i32::MAX)` sentinels. Invalid padded ordinals receive the same
   sentinels.
2. Candidate blocks are reduced in fixed groups of 128 physical 16-lane tiles.
   Each group therefore loads at most 2,048 lanes, masks all sentinels, applies
   the same exact selection, and emits one 16-lane tile containing ten
   candidates. Reduction repeats until one block remains. No dense score plane
   or full ranked-pair allocation reaches host memory.

The maximum temporary device storage at batch 32 and one million rows is the
first physical candidate plane: `32 * ceil(1,000,000/128) * 16 * 8 = 32,002,048`
bytes, plus smaller reduction levels and final outputs. The persistent gallery
remains exactly 130,000,000 bytes.

## Library boundary

The Rust crate builds both an `rlib` and `cdylib`. A narrow C ABI owns an opaque
persistent gallery handle and has exactly three operations: create from the
canonical 130-byte rows, search a batch of one or 32 canonical query rows, and
destroy. Every pointer/length/count is validated before dereference; panics are
caught at the ABI boundary; failures return stable integer status codes and do
not write partial outputs.

`src/sfora/cutile_int8.py` loads only an explicitly supplied library path,
constructs the handle from `PackedInt8Embeddings.to_bytes()`, retains the
backing library and handle, and returns NumPy `int64` ordinals plus float32
scores. Importing Sfora never requires CUDA or the native library. Unsupported
platforms fail closed and the existing Torch scorer remains available as an
explicit caller choice, never a silent benchmark fallback.

## Gates

The native candidate advances only if all gates pass on the NVIDIA GB10:

- exact scores and exact top-10 ordinals against the scalar Rust authority for
  signed extremes, score ties, negative-only scores, gallery sizes 10, 127,
  128, 129, and 1,000,003, and batches 1 and 32;
- 5 warmups and 50 samples, with first JIT launches reported separately;
- end-to-end native call p99 (including query copy, device top-k, and ten-result
  copy) at least 20% below the matched current materializing scorer for both
  batches: below 16.039 ms at batch 1 and below 20.508 ms at batch 32 using the
  frozen prior p99 measurements 20.049/25.635 ms;
- p99 also below the matched resident-f32 Torch control: below 2.077 ms at
  batch 1 and below 6.932 ms at batch 32 using the frozen 2.597/8.665 ms p99;
- persistent bytes exactly 130/item, peak host RSS below 2 GiB, no score-plane
  host transfer, and no score/top-k mismatch.

One preregistered tile/reduction shape is used. A failure is recorded and the
candidate is killed; it does not trigger shape tuning.

## Evidence and limits

The receipt records raw timings, nearest-rank p50/p99, throughput, compile
times, persistent/temporary bytes, peak RSS, toolchain/device/source identity,
exactness, and the frozen decision. This performance integration does not
alter or validate model quality. Matched quality/SOTA evaluation remains a
separate goal gate on real dataset splits.
