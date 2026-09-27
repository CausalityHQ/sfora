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

## Initial TRAIN-only qualification

The sole DGX Spark GB10 unit exited 0 (invocation
`d346f4130b1048bc844cb750d3cef990`). Its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-public-checkpoint-v1/receipt.json)
has SHA-256 `ba88e952fd124651a8eaa129ac14111f131bd438ce582bde3c02e04d1ac8888f`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-public-checkpoint-v1/service-journal.log)
has SHA-256 `db69c4afa6cdb030ae29816b6481510e11c930eaee51e2f06afe7231249c5116`.
The qualified serving source SHA-256 is
`81dac83d4ae59dc5c57d62b619c029fc4cb8ad84773a462dc1f2335a278342f9`.

| In-Shop official TRAIN fixed roles, seed 179026 | Archived training export | Public FP16 checkpoint/custom-gallery path |
| --- | ---: | ---: |
| Native packed Recall@1, 6,354 queries/6,245 gallery | **6,203/6,354 = 97.6235%** | **6,203/6,354 = 97.6235%** |
| Per-query hit flags | reference | 2 differ; one rescued and one lost |
| Paired product-bootstrap lower 95% delta | reference | **−0.0470 pp**, above −0.25 pp floor |
| Image-to-top-10 batch-1, 1,000 distinct image byte hashes | not measured in this protocol | p50 **14.685 ms**, p95 **17.183 ms**, p99 **18.308 ms** |
| Gallery build from image files | not a public build | **49.772 s** |
| Full 6,354-query batch-32 throughput, including decode | not a public query loop | **120.22 images/s** (52.855 s total) |
| Gallery wire bytes / post-load peak PyTorch allocated CUDA | 130 bytes/image | **811,850 bytes** total / **882,524,672 bytes** |

Model load took **7.009 s** before the post-load CUDA peak measurement. All
predeclared quality, latency, build and memory gates passed. Every public
top-10 row had in-range ordinals, finite descending scores, and stable ordinal
order on score ties. The two per-query hit flips are real: the archived export
used FP32 checkpoint weights under autocast and a different batching path,
whereas this public gate converts vision weights to native FP16. The gate did
not isolate which arithmetic step caused the flips, so only the aggregate
quality and paired bound are qualified. The exact native scorer and 130-byte
format are unchanged. This is a production-path TRAIN qualification on one
GPU/runtime and one checkpoint, not an official query/gallery or SOTA result.

## Review correction and required v2 gate

Independent Opus/Astra release review found that Transformers can prefer an
additional unpinned `processor_config.json` in a snapshot directory over the
hashed `preprocessor_config.json`. An attacker or accidental extra file could
change normalization while passing the old class/size/resample checks. The
loader now passes the verified preprocessor file path directly in both the
new checkpoint path and existing SOP artifact path. A focused regression test
places a conflicting extra config beside the pinned file and asserts direct
file loading. The custom-gallery builder also accepts an optional native
library SHA-256 pin; this gate supplies it. The gate script now exits nonzero
after writing a failed receipt, and the README example checks the training
receipt SHA-256 before using its artifact hashes.

The v1 receipt remains a real measurement of the prior source, but production
promotion requires one v2 rerun with these corrected files, the same frozen
TRAIN roles, precision, scorer, latency inventory and thresholds. Record v2
source digests and a distinct terminal receipt; preserve v1 evidence.
