# RC5 packed top-k output-buffer design

## Purpose and authority

Continue from retained RC3 `0.3.0rc3`, commit `b0df688f`. The RC4 512-width
candidate remains rejected. The RC4 stage receipt measured 0.795377 ms of
batch-32 buffer initialization per request, versus about 4.5 ms end to end,
on the authenticated one-million-row GB10 gallery. Remove this distinct cost
only if a bounded experiment shows an exact, repeatable public-API win.

## Change

Keep `MERGE_WIDTH=2048`, the C ABI, Python API, gallery wire, score arithmetic,
and stable ordinal tie rule. In `rust/sfora-cutile-int8-score/src/topk.rs`, use
the pinned cuTile `Tensor::uninitialized` allocator for fused-score and merge
output tensors. Reshape each allocation and immediately pass it to the kernel
that writes the entire tensor. The cuTile `full` implementation itself uses
`Tensor::uninitialized` followed by a fill kernel; this change skips that fill.

The safety condition is exact: the score/block kernel's grid covers every
`[BM,16]` output partition of `[BM,score_blocks*16]`; each launch stores all
16 scores and ordinals. Every merge grid likewise covers every output
partition of `[BM,groups*16]` and stores all 16 lanes. No output element is
read before its kernel completes on the gallery stream. If allocation or launch
fails, the buffer is discarded without reading it. Keep the diagnostic split
path's existing initialization so the pilot changes only the production path.

## Prespecified pilot

The frozen RC3 library SHA-256 is
`a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea`.
Use the same authenticated eight-file fixture and public API SHA-256
`7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818`.
Run the existing signed-extreme, tie, gallery-boundary Rust tests and exact
f32-bit/ordered-ordinal checks for batches 1 and 32 at 1,000,000 and
1,000,003 rows. A diagnostic Nsight trace should show six fewer output-fill
kernel launches per warmed batch-32 search than the retained RC3 path, with
no newly uninitialized values or pool growth above 200 MB.

After five warmups per shape, collect two independent 50-call paired public
Python API replays: RC3 then candidate, and candidate then RC3. Preserve all
samples, including tails. Qualify only if **each pair** has at least 10%
batch-32 p99 latency gain, at most 5% batch-1 p99 regression, exact bits and
ordinals, process RSS below 2 GiB, and tracked CUDA pool peak below 200 MB.
Report p50/p95/p99 and mean-based throughput for both batches. These are
engineering gates on one deterministic gallery, not statistical confidence or
cross-device guarantees. The expected batch-32 gain is bounded by the measured
0.795 ms initialization stage; 10% is the minimum material release gate.

If the pilot fails, revert the output-allocation change, archive the negative
result, and choose a different measured architecture. If it passes, run the
focused Rust/Python correctness and clean-install package gates, revalidate
frozen Pet/In-Shop receipt hashes without retuning, obtain one terminal
Claude Opus 5.5 plus GPT-6 Astra review, and publish a release decision with
explicit claim limits before tagging or publishing anything.
