# Packed top-10 RC4 performance design

## Intent and boundary

Ship one measurable production-library performance improvement beyond
`117908a3`, while preserving RC3 quality receipts, exact score bits, ordered
top-10 indices, and the public packed-gallery API. The current batch-1 Nsight
trace assigns 97.7% of GPU kernel time to the fused `score_block_topk` kernel;
it does not distinguish scoring from selection, show host overhead, or cover
batch 32. A production change requires stronger stage evidence and a bounded
pilot predicting a material end-to-end win.

## Diagnostic measurement

Use the authenticated one-million-row, 128-dimensional, 130-byte/item gallery
and query seeds from `scripts/_scratch_benchmark_cutile_int8_topk.py` on the
GB10. Pin the RC3 library SHA-256
`a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea`.
Retain raw timing samples and hashes for the scorer binary, harness, report,
gallery/query generator, and environment. No training job runs.

Add a diagnostic-only split of the fused kernel: the same signed-int8 matrix
product and norm scaling write score tiles; a second kernel selects stable
top-10 entries from each 128-row block. Compare score bits and ordinals with
the production fused path, including the 1,000,003-row padded-tail case and
ties. The split path is a measurement probe, not a candidate production
implementation by itself: intermediate score writes change its cost.

For batch 1 and 32, measure host-to-device transfer, scoring, block selection,
merge, device-to-host transfer, and residual host/API work. Use GPU events for
GPU stages and a monotonic host clock around the public call. Record p50/p95/
p99, throughput, peak GPU memory, process RSS, compiler time separately, and
matched RC3 end-to-end numbers from the frozen receipt. A missing profiler or
unbounded compiler is a measured limitation, not a reason to infer stage time.

## Candidate decision

Choose only the largest measured contributor. Pilot one change to the kernel
or packed data layout with a two-hour wall cap. Require exact score bits,
deterministic top-10 order, no increase above the 2 GiB process RSS gate, and
a material end-to-end improvement on both batches relative to matched RC3.
If no single change passes, retain the current scorer and publish a finite
negative decision naming the next distinct bottleneck.

Keep the production ABI, Python API, artifact authentication, and RC3 tag
unchanged. Re-run focused Rust/Python correctness and clean-wheel usability,
then revalidate untouched Pet and In-Shop quality receipts without fitting or
retuning. Publish an RC4 decision table with source hashes, exact measurement
scope, and explicit claim limits. Seek one dual Opus 5.5/Astra critique only
after the terminal candidate commit hash exists.
