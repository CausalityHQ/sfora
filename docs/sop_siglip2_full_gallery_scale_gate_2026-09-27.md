# SOP full TRAIN gallery public serving scale gate

Freeze the current production `Siglip2CompactEncoder.from_checkpoint` and
`Siglip2CompactIndex.from_image_paths` path before measuring a full external
gallery. Use the seed-179024 true-freeze checkpoint SHA-256
`2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172`
and training receipt SHA-256
`07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5`.
Pin the same 130-byte native scorer SHA-256
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`.

Build a new gallery from all **59,551 Stanford Online Products TRAIN** image
files in archive order. Use the first 1,000 distinct TRAIN image byte hashes as
public image-to-top-10 inputs; they may overlap the gallery because this gate
tests serving, not retrieval quality. Compare 32 public top-10 rows with an
independent stable packed-matrix oracle, requiring identical ordinals and
maximum score error ≤`1e-5`. On the idle DGX Spark GB10, require gallery build
≤`600 s`, 1,000 synchronized batch-1 calls p99 ≤`30 ms`, post-load peak
PyTorch allocated CUDA <`3 GB`, and peak parent RSS <`6 GB`. Record batch-32
throughput, p50/p95/p99, model load, build time, and gallery wire bytes.

This is a TRAIN-only public serving qualification. No SOP TEST quality or
state-of-the-art claim can be inferred from it. A failed threshold keeps this
full-gallery configuration unqualified and directs a code-level bottleneck fix.
