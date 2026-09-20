# Packed int8 cuTile score-kernel design

## Purpose

Sfora's exact 128-dimensional int8 wire stores 128 signed code bytes and one
little-endian f16 inverse norm per item. At one million rows the public scorer
spends 87.24% of batch-1 time and 66.08% of batch-32 time repeatedly expanding
that wire to float32. A persistent float32 control proves a 2.390 ms p50 / 2.597
ms p99 batch-1 ceiling, but expands the gallery from 130 MB to 516 MB.

This slice asks one question: can a safe Rust/cuTile signed-int8 kernel compute
the exact packed score plane fast enough to justify native integration while
retaining the 130 MB gallery? It is a fail-fast kernel spike, not a public API.

## Frozen arithmetic

For query row `q` and gallery row `g`, the authoritative score is:

`f32(sum_i(i32(q[i]) * i32(g[i]))) * f32(q_inverse_f16) * f32(g_inverse_f16)`.

The sum has 128 terms and maximum magnitude `128 * 127 * 127 = 2,064,512`, so
signed i32 accumulation is exact. cuTile must use signed `i8` inputs and
`mmai(..., signed, signed)` into i32. Norm metadata is converted from f16 to
f32 before multiplication. Ties remain ordered by ascending gallery ordinal in
the later top-k stage; this score-only slice must produce the same f32 bits as
the scalar Rust authority on finite canonical inputs.

## Components

- A standalone Rust crate under `rust/sfora-cutile-int8-score/` pins
  `cutile = 0.1.1`. It has no Python, Torch, storage, network, or model surface.
- `wire.rs` validates the exact row-major i8 plus f16 metadata contract and
  implements the scalar authority.
- `kernel.rs` contains a cuTile module that loads row-major query codes and a
  dimension-major gallery, uses signed i8 MMA with i32 accumulators, converts
  to f32, applies inverse norms, and writes a dense score plane.
- `main.rs` runs deterministic small-fixture equality and a one-million-row
  benchmark for batches 1 and 32. Compilation/JIT warm-up is excluded from
  steady-state timing and reported separately.

## Gates

Correctness is mandatory before timing:

1. exact score-bit equality against the scalar authority on deterministic
   random, signed-extreme, zero-coordinate, norm-varying, and tie fixtures;
2. exact top-10 ordinal equality after a deterministic host reference reduction;
3. no NaN/inf and no out-of-range metadata accepted.

The kernel advances to a separate device-top-k/FFI design only if both batches
meet all correctness gates and score-plane p99 is below the matched resident
float32 score-plane p99 measured in the same Rust process. The later integrated
path must beat the full resident control by at least 20% end-to-end at batch 1
or batch 32 and keep packed resident bytes at 130 per item. Failure deletes the
candidate; it does not trigger tile-shape tuning beyond one preregistered shape
per batch.

## Reproducible toolchain

- upstream `cutile-rs` tag `v0.1.1`, commit
  `c299d449cd0cc58c77705cf559caa02b3790e4bd`;
- Rust stable 1.89 or newer;
- system `/usr/local/cuda-13.0` supplies host headers and cuRAND;
- `CUTILE_TILEIRAS_PATH` points to the authenticated CUDA 13.4 `tileiras` in
  the existing `cutile-canary-v1` environment;
- on the aarch64 DGX,
  `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/aarch64-linux-gnu/13/include`.

The toolchain paths are execution configuration, never embedded in the library.
Unsupported hardware/toolchains fail closed; no float32 fallback is relabeled
as the cuTile arm.

## Non-goals

This slice does not add a public Python/Rust API, device top-k, ANN indexing,
CUDA graph capture, multi-GPU support, training kernels, or a SOTA claim. Those
belong to a new reviewed design only after this score kernel passes.
