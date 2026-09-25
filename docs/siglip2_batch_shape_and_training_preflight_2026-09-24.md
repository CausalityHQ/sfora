# SigLIP2 serving batch shape and training preflight, 24 September 2026

This checkpoint uses only the Stanford Online Products (SOP) official TRAIN
split. Product-disjoint fit/holdout partition: 53,700 fit images and 5,851
holdout query images, seed 179019. Retrieval searches all 59,551 TRAIN images,
explicitly excludes the query image, and uses the released 130-byte-per-row
packed gallery (128 signed-int8 values plus one f16 inverse norm). It is a
pretrained substrate screen, not a learned-Sfora or official TEST result.

## Single-image serving quality

The frozen quality export encoded images in batches. A new single-image live
replay used the same pinned SigLIP2-L/16@256 checkpoint, processor, fit-only
PCA-128, and released CuTile scorer for **every** holdout query. Its terminal
DGX receipt is
[live-batch1-all.json](evidence/compact_metric/sop-siglip2-substrate-v1/live-batch1-all.json)
(SHA-256 `883b5e33d4c1b5146f4bbbedf4153903d3a7893c5d473e166e6f2813d38f194f`).

| SOP TRAIN holdout, full TRAIN gallery | Packed Recall@1 | Notes |
| --- | ---: | --- |
| SigLIP2 cached batched export | 78.3456% | Frozen quality receipt |
| Same SigLIP2, live batch-one encoding and native top-10 | 78.3456% | All 5,851 live queries processed |

The equal aggregate hides **four changed per-query Recall@1 outcomes** and
**918 changed top-10 lists**. Packed codes changed for 4,174 queries and f16
inverse norms for 2,360. Minimum live-versus-export feature cosine was
0.9999948; maximum positional top-10 score difference was 0.0042453.
A separate diagnostic found identical processor pixels for the same image
alone and in a batch of 32, but two signed-int8 values and the f16 inverse
norm differed on that query. The evidence localizes the variation to the
fp16 model's batch-shape-dependent computation; the exact CUDA instruction
responsible has not been isolated. The released native scorer's full-gallery
arithmetic and tie behavior were independently checked against a scalar
oracle in the prior native replay. A serving-quality receipt must state its
encoding batch shape. There is no exact per-query parity across batch shapes.

An updated paired live benchmark completed with the same batch-one
diagnostic and 10 AB/BA timing blocks per batch shape in
[live-batch1-diagnostic-timing.json](evidence/compact_metric/sop-siglip2-substrate-v1/live-batch1-diagnostic-timing.json)
(SHA-256 `23bfb5b5e8d95d8e9469f7c845e099e4470993997971e10fcef196d405c8b69c`).
The selected UNICOM query had exact packed-code and top-10 parity; the
selected SigLIP2 query had two changed codes and one changed norm, with the
same top-10 ordinals. The short replay's image-to-top-10 p50 was
18.109 ms versus 38.699 ms at batch one, and 257.669 ms versus 549.415 ms
at batch 32 (SigLIP2 versus UNICOM). These are supporting repeatability
measurements. The earlier 50/20-block receipt remains the primary p50/p95
measurement; neither receipt establishes p99.

The same single-image replay completed for UNICOM L/14@336 in
[unicom-live-batch1-all.json](evidence/compact_metric/sop-siglip2-substrate-v1/unicom-live-batch1-all.json)
(SHA-256 `796bd4f3410008dfdcfbcec920452520ea1f29227591c306ecdeb339833add36`):
71.2357% Recall@1, versus 71.2528% in its cached batched export. One
per-query Recall@1 outcome and 397 top-10 lists changed. The two live
receipts have the same ordered query-image hash and released native library.
The [paired product-bootstrap receipt](evidence/compact_metric/sop-siglip2-substrate-v1/live-batch1-paired-quality.json)
(SHA-256 `81dd3ec678969997a13dedf5055f55164e058a7d9d4d3df108002de94fcf95b5`)
gives a **+7.1099 percentage-point** SigLIP2 advantage, with 95% interval
**[+5.7089, +8.5115] points**, across 1,132 heldout products and 5,000
fixed-seed draws. This is a verified paired **TRAIN holdout** result for
batch-one live encoding. Official SOP TEST and In-Shop remain unmeasured for
this SigLIP2 candidate.

## Full-backbone training feasibility

The pinned SigLIP2 vision tower was extracted from the pretrained model,
converted to fp32 parameters, trained with fp16 autocast and AdamW, and
backpropagated through a temporary 128-output classifier on two repeated SOP
images. These are **synthetic-objective preflight measurements**, not
similarity-learning throughput or quality. Each run used three optimizer
steps; the first includes optimizer state allocation. All jobs completed on
the same NVIDIA GB10 DGX without gradient checkpointing.

| Microbatch | Steady-step time (steps 2 and 3) | Compute-only throughput | Peak CUDA allocation | Receipt |
| ---: | ---: | ---: | ---: | --- |
| 8 | 0.300, 0.292 s | 27.01 images/s | 6.60 GB | [b8](evidence/compact_metric/sop-siglip2-substrate-v1/train-preflight-b8.json) |
| 32 | 0.759, 0.743 s | 42.61 images/s | 12.79 GB | [b32](evidence/compact_metric/sop-siglip2-substrate-v1/train-preflight-b32.json) |
| 64 | 1.173, 1.136 s | 55.44 images/s | 21.07 GB | [b64](evidence/compact_metric/sop-siglip2-substrate-v1/train-preflight-b64.json) |

Batch 64 passed the synthetic compute preflight. The table does not include
image loading, augmentation, validation, or gallery export, so it cannot be
extrapolated into measured end-to-end training time.

## Matched full-backbone training gate

The production training path now starts from the pinned SigLIP2 L/16@256
vision tower (1024 output features), a fit-only PCA-initialized 1024-to-128
head, and fit-product mean class proxies. It trains all vision and head
parameters with AdamW, fp32 parameters and fp16 autocast. Batches contain
16 fit products × 4 images; the fixed training recipe uses random resized
crop, horizontal flip, ArcFace margin 0.3/scale 64, and 1,000 updates.
The two treatment arms add SmoothAP on either float head vectors or the
128-byte signed-int8 deployment codes with straight-through gradients;
both use a fit-gradient-selected coefficient of 8.0. The exact recipe and
advancement rule were frozen in the
[preregistration](superpowers/specs/2026-09-24-siglip2-matched-training-prereg.md).

The [v7 zero-update full-gallery receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-step0-v7.json)
(SHA-256 `f0acb4f798bcdf868f9cb08f662b5195afc784f80628c8a44f10614e1f9c8a20`)
verified the v7 source before the initial 1,000-update attempt. The revised
v10 source later reproduced the same quality in its
[exact-source zero-update receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-step0-v10.json),
with matching initial head/proxy/PCA hashes against the v10 control.
On all 5,851
SOP TRAIN holdout queries, it measured **78.4139% Recall@1** and
**0.508128 mAP@R**. The frozen batched-export reference was 78.3456% and
0.508168, respectively, so the differences are +0.0684 percentage points
and -0.000040. Both are within the prespecified 0.2-point and 0.003 gates.
Native top-10 ordinals and scores matched independent full-gallery sorting
for every query; maximum score difference was zero. Full-gallery export took
321.9 seconds and scoring took 3.03 seconds on the GB10. There were no
training updates, so these are export and evaluation costs only.

The first 1,000-update ArcFace attempt stopped at update 131 because the fp16
backward pass produced nonfinite gradients in the first vision layer. It has
no quality or completed training-time number. The exact failure and one-step
gradient diagnosis are preserved in the
[failure log](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-1000-v7-failure.log)
and [diagnostic log](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-gradient-diagnostic-v8.log).
Reducing only the GradScaler initial scale from 1024 to 128 completed 150
finite control updates in the
[replay receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-scale128-150-v9.json)
(180.19 seconds training wall time, 21.09 GB peak CUDA allocation). This
motivated the documented recipe revision. The revised v10 ArcFace control
subsequently completed all 1,000 updates in 1,152.795 seconds training wall
time on the GB10, with 21.09 GB peak CUDA allocation. Its
[full receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-1000-v10.json)
measured **90.4632% Recall@1** and **0.718801 mAP@R** on the SOP TRAIN
holdout, with exact native top-10 ordinals and scores for all 5,851 queries.
This is a local control result, not an official SOP TEST or SOTA claim.

## Decision

Keep the frozen 130-byte scorer and the exact native correctness gate.
Record batch-one encoding drift explicitly in paired timing receipts.
The two-step class-balanced ArcFace and deployed-code smoke runs completed
with finite losses. A two-step ArcFace full-gallery replay measured 78.9609%
Recall@1 and 0.515432 mAP@R, but is exploratory and was generated before
the final trainer provenance and embedded exact-scorer checks. Its independent
native replay did match all 5,851 top-10 ordinals and scores exactly.

Next: compare the frozen 1,000-update ArcFace, float-rank, and deployed-code
rank arms with equal source weights, initial head, schedule, augmented input
batches, microbatch, optimizer, and scorer. Select using TRAIN identities
only. At least three paired seeds, official protocol evaluation on SOP and
In-Shop, transfer checks, and measured p99 remain required before a
learning-method or joint SOTA claim. Lean proofs can establish conditional
correctness and scaling bounds for the actual packed scorer; Recall@1 and
latency remain empirical quantities.
