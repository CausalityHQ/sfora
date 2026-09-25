# Sfora joint quality and performance decision, 24 September 2026

Sfora is a neural vector similarity training and retrieval library, separate
from Borsuk. This candidate combines a pretrained SigLIP2-L/16@256 image
encoder, a trainable 1024-to-128 projection, supervised product-balanced
training, signed-int8 gallery codes with f16 inverse norms (130 bytes per
image), and an exact native CUDA top-10 scorer. Apache Arrow is not in the
training or retrieval path.

## Comparison panel and current evidence

All measured quality below uses **Stanford Online Products (SOP) official
TRAIN**, partitioned by product into 53,700 fit images and 5,851 holdout
queries (1,132 products), seed 179019. Every query searches all 59,551 TRAIN
images excluding itself. These are internal selection measurements; the SOP
official TEST and DeepFashion In-Shop official query/gallery are not yet
measured for this SigLIP2 candidate.

| System and status | SOP TRAIN holdout packed Recall@1 | mAP@R | Image-to-top-10 batch-one p50/p95 on GB10 | Gallery bytes/image |
| --- | ---: | ---: | ---: | ---: |
| Pretrained SigLIP2 L/16@256, live query, verified | 78.3456% | not measured for live query | 17.409/20.555 ms | 130 |
| Pretrained UNICOM L/14@336, live query, verified local baseline | 71.2357% | not measured for live query | 39.660/42.044 ms | 130 |
| SigLIP2 zero-update v7 trainer, batched export, verified for v7 | 78.4139% | 0.508128 | not measured on trained export | 130 |
| SigLIP2 zero-update v10 trainer, batched export, verified exact-source gate | 78.4139% | 0.508128 | not measured on trained export | 130 |
| SigLIP2 two-update ArcFace, exploratory, older trainer source | 78.9609% | 0.515432 | not measured | 130 |
| SigLIP2 1,000-update ArcFace at scale 1024, failed | no quality result | no result | no result | 130 |
| SigLIP2 ArcFace at scale 128, 150-update stability diagnostic | no quality evaluation | no result | no result | 130 |
| SigLIP2 ArcFace at scale 128, 1,000 updates, verified TRAIN holdout | **90.4632%** | **0.718801** | not measured on trained export | 130 |

The primary live timing is the 50 batch-one/20 batch-32 AB/BA block
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/live-134-receipt.json).
The live quality arm receipts are
[SigLIP2](evidence/compact_metric/sop-siglip2-substrate-v1/live-batch1-all.json)
and [UNICOM](evidence/compact_metric/sop-siglip2-substrate-v1/unicom-live-batch1-all.json).
Their paired heldout-product bootstrap gives SigLIP2 **+7.1099 percentage
points** Recall@1 over this local UNICOM configuration, 95% interval
**[+5.7089,+8.5115] points**, in the
[paired receipt](evidence/compact_metric/sop-siglip2-substrate-v1/live-batch1-paired-quality.json).
The two encoders have different architectures, preprocessing, and input
resolutions, so this measures a candidate system comparison, not a novel
training-method effect. The live batch-one query and batched gallery export
can yield different fp16 codes; both live arms were evaluated directly.

The published [UNICOM ICLR 2023 paper](https://arxiv.org/pdf/2304.05884)
reports supervised Recall@1 of **91.2% on SOP** and **96.7% on In-Shop**
for ViT-L/14@336. Those are official-protocol reference gates, not the
latest-frontier certification and not comparable with the internal TRAIN
holdout rows above. The matched local method control is the *same*
SigLIP2 encoder/head/data/scorer with ArcFace alone; the treatments add
float-vector SmoothAP or deployed-code SmoothAP. This isolates whether
training for the quantized deployed ranking improves quality.

The earlier synthetic-objective three-step preflight on one NVIDIA GB10
measured, at microbatch 64, 1.136–1.173
seconds per steady step, **55.44 images/s compute-only**, and **21.07 GB**
peak CUDA allocation. The class-balanced 150-update control diagnostic
took **180.19 seconds** wall time and **21.09 GB** peak CUDA allocation.
The actual 1,000-update ArcFace run took **1,152.795 seconds** training wall
time (64,000 sampled training images, approximately **55.52 images/s** across
its augmented data loader and optimizer), **319.454 seconds** for full
59,551-image gallery export, and **3.006 seconds** for full-gallery holdout
scoring. Total wall time before receipt write was **1,489.757 seconds**;
peak CUDA allocation was **21.09 GB** on the NVIDIA GB10. Its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-1000-v10.json)
(SHA-256 `3ff998f70900ea24c4c6c33d1cee866ca8650b629b799d58d060b2063624ecca`)
binds the committed trainer source, ten initial augmented-batch hashes,
and exact native top-10 ordinals and scores for every holdout query.
The synthetic preflight figures are not end-to-end training numbers.
No p99 live latency or scaling curve is measured yet.

## Decision and validation gates

The v7 zero-update trainer gate passed: 78.4139% Recall@1 and 0.508128 mAP@R
versus frozen cached quality 78.3456% and 0.508168, within prespecified
0.2-point/0.003 tolerances, with all native top-10 ordinals and scores
exact. The current v10 trainer differs in its GradScaler initial scale and
receipt field; its [exact-source zero-update receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-step0-v10.json)
(SHA-256 `7efa9217bef6a525248e6d6185b7be789602e073b281e809e92f8cdc76be3934`)
reproduced **78.4139% Recall@1** and **0.508128 mAP@R** with exact native
top-10, matching the v10 control's source and initial head/proxy/PCA hashes.
The first 1,000-update
control failed at step 131 from nonfinite fp16 gradients. A 150-step
single-variable replay with initial loss scale
128 relative to the instrumented v8 source passed, so the revised
1,000-update control completed at **90.4632% Recall@1** and **0.718801
mAP@R** on the TRAIN holdout. This is a measured local quality improvement
over the frozen pretrained substrate, while the method treatment and
official TEST results remain pending. The deployed-code rank treatment is
now the active GPU arm.

Run deployed-code-rank and float-rank arms
sequentially under identical source weights, initialization, augmented
input hashes, schedule, optimizer, and scorer. Promote deployed-code rank
only if its paired product-bootstrap 95% Recall@1 lower bound versus
ArcFace is positive, the point gain is at least 1 percentage point,
mAP@R does not regress, and training wall time is at most 1.15× control.
Then require three paired seeds, official SOP TEST and In-Shop protocols,
CUB/Cars transfer, and p99/scaling measurements before a new-method or
joint SOTA claim. Lean verifies reusable scorer correctness and conditional
bounds; empirical quality and latency still require these measurements.
