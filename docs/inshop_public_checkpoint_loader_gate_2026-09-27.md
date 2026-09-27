# In-Shop TRAIN public checkpoint loader gate, 27 September 2026

The existing public `Siglip2CompactIndex.from_artifacts` binds a SOP-specific
receipt and gallery. The new additive `Siglip2CompactEncoder.from_checkpoint`
loads a locally pinned SigLIP2 Large/256 model snapshot and a full trained
vision/head checkpoint, then uses the existing `from_image_paths` gallery
builder. Verify that this path serves the selected In-Shop trained model
without changing its 128-D int8/fp16 wire format or native exact scorer.

On DGX Spark GB10, pin seed-179026 true-freeze checkpoint SHA-256
`ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089`,
training receipt SHA-256
`76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc`,
its three snapshot file digests, and native scorer SHA-256
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`.
Use only the fixed 6,245-gallery/6,354-query roles from product-disjoint
official In-Shop **TRAIN** held rows, and native FP16 on CUDA with TF32 off.
The archived same-checkpoint training-export packed hit vector is the paired
reference: 6,203/6,354 = 97.6235% Recall@1. Do not read official
query/gallery rows or use them for selection.

Build the gallery through `from_image_paths`, then query every role image in
batches of 32, including decode and the public encoder. Require all top-10
ordinals in range, finite scores in descending stable order, and packed
Recall@1 no more than **0.15 percentage points** below the archived training
export, with a product-bootstrap paired 95% lower bound above **−0.25 pp**.
For 1,000 distinct query byte hashes, measure synchronized batch-1 public
image-to-top-10 including JPEG decode after warmup, and require p50 ≤20 ms,
p99 ≤30 ms, gallery build ≤180 s and post-load peak PyTorch allocated CUDA
<3 GB. Record model load, build, query throughput, p50/p95/p99, bytes/gallery
row and raw per-query hits. These absolute serving gates qualify one hardware
configuration; they do not compare against UNICOM or support a SOTA claim.
Failure blocks promoting this loader until the responsible checkpoint,
precision, processor, packing or scorer layer is fixed.
