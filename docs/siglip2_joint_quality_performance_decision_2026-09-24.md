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

| System and status | SOP TRAIN holdout packed Recall@1 | mAP@R | Image-to-top-10 batch-one p50/p95/p99 on GB10 | Gallery bytes/image |
| --- | ---: | ---: | ---: | ---: |
| Pretrained SigLIP2 L/16@256, live query, verified | 78.3456% | not measured for live query | 17.409/20.555/— ms | 130 |
| Pretrained UNICOM L/14@336, live query, verified local baseline | 71.2357% | not measured for live query | 39.660/42.044/— ms | 130 |
| SigLIP2 zero-update v7 trainer, batched export, verified for v7 | 78.4139% | 0.508128 | not measured on trained export | 130 |
| SigLIP2 zero-update v10 trainer, batched export, verified exact-source gate | 78.4139% | 0.508128 | not measured on trained export | 130 |
| SigLIP2 two-update ArcFace, exploratory, older trainer source | 78.9609% | 0.515432 | not measured | 130 |
| SigLIP2 1,000-update ArcFace at scale 1024, failed | no quality result | no result | no result | 130 |
| SigLIP2 ArcFace at scale 128, 150-update stability diagnostic | no quality evaluation | no result | no result | 130 |
| SigLIP2 ArcFace at scale 128, 1,000 updates, verified live batch-one | **90.4632%** | **0.718801** on cached export | **23.784/26.759/28.216 ms** | 130 |
| Same ArcFace checkpoint, native-fp16 vision deployment, verified live batch-one; separate-process timing | **90.4119%** | not measured live | **15.284/17.698/19.258 ms** | 130 |
| SigLIP2 deployed-code SmoothAP, 1,000 updates, verified TRAIN holdout | 90.3093% | 0.724227 | live measurement pending | 130 |
| SigLIP2 float SmoothAP, 1,000 updates, verified TRAIN holdout | 90.7366% | 0.726593 | live measurement pending | 130 |

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
The [trained ArcFace live receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-live-v1.json)
(SHA-256 `e7e0a464d9898ff41eb727ca5aaaeb428c7a4d0f79b46dd38c39b7399bd0160a`)
replayed all 5,851 holdout queries as individual images against the same
cached full TRAIN gallery. Live Recall@1 remained **90.4632%**, with zero
per-query Recall@1 changes, though 300 top-10 lists and 2,068 signed-int8
query codes changed. On GB10, 200 batch-one timing blocks measured
**23.784/26.759/28.216 ms p50/p95/p99** end-to-end, 41.42 queries/s mean;
100 batch-32 blocks measured **324.697/335.871/339.640 ms**, 98.44
queries/s mean. The trained path's encoder/transfer p50 was 17.244 ms at
batch one versus 9.289 ms for the frozen fp16 path; the native scorer
remained about 0.4 ms. The trained encode stage also includes the GPU head,
while the frozen PCA ran in the separate pack stage. Different model
parameter precision, head placement, and timing processes prevent attributing
the whole difference to training. The p99 values are descriptive quantiles
from a fixed image (200 batch-one and 100 batch-32 samples), not a tail-latency
SLO across varied images or load. The native-fp16 deployment cast
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-live-fp16-v2.json)
(SHA-256 `7b879d2a31b9240afb65c3206af587149d746467e69707e34d8545106577e4d4`)
replayed all 5,851 individual holdout images against the same autocast-exported
gallery. It measured **90.4119% Recall@1**, three fewer correct queries than
the fp32-parameter/autocast live path: four correct-to-wrong and one
wrong-to-correct. A 5,000-draw product bootstrap of the paired query outcomes
gives **-0.0513 percentage points**, 95% interval **[-0.1353,+0.0172]**;
the exact two-sided McNemar p-value is **0.375**. The
[paired quality receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-live-precision-quality-paired-v1.json)
(SHA-256 `f37effbf4ea5beb8b98dde16666fcf18ebd6d8552c6e7a8622e3e8c555c885cc`)
binds the three input receipt hashes, product groups, 5,000 resamples, and
seed; the [executed bootstrap source](evidence/compact_metric/sop-siglip2-substrate-v1/precision-quality-bootstrap-v1.py.txt)
is archived beside it.
The interval includes zero: this single-seed TRAIN-holdout replay does not
detect an R1 loss, and it excludes losses larger than about 0.14 percentage
points at this bootstrap confidence level. The two full quality replays used
earlier versions of the timing script; their selected-query result digests
match both modes in the current paired timing script, but that is not a
current-source replay of all 5,851 images.
Its 1,000 fixed-image batch-one blocks measured **15.284/17.698/19.258 ms
p50/p95/p99**; 200 fixed batch-32 blocks measured
**267.254/283.735/291.187 ms**. This uses native-fp16 vision parameters
and input pixels, a fp32 projection head, and the same gallery exported with
fp32 parameters under fp16 autocast. Separate timing processes and unequal
sample counts mean that the apparent speed gain is not yet a causal paired
measurement. The subsequent same-process
[paired precision receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-arcface-live-precision-pair-v3.json)
(SHA-256 `fe0b73759e25e4032c5055f1c043c2b8162dc774bc9e8bcac9f7624b997e2360`)
loaded both precision variants of the same checkpoint, shared the same
preprocessor, projection head, packed gallery, and native scorer, and
alternated AB/BA/BA/AB on one NVIDIA GB10. Across 1,000 paired batch-one
fixed-image blocks, fp32-autocast versus native-fp16 measured
**23.873 versus 15.754 ms p50** image-to-top-10, **26.418 versus 18.948 ms
p95**, and **27.776 versus 20.058 ms p99**; mean throughput was **41.58
versus 62.57 queries/s**. The paired median native-fp16 minus
fp32-autocast difference was **-8.111 ms**. Across 200 paired fixed batch-32
blocks, p50 was **320.829 versus 268.080 ms**, p95 **336.991 versus
287.861 ms**, p99 **340.128 versus 290.488 ms**, and mean throughput
**99.61 versus 119.52 queries/s**. The paired median difference was
**-53.007 ms**. Native-fp16 was faster in every measured block at both
batches. The encode/transfer stage accounts for almost all of the p50 gap;
native search stayed near 0.35 ms at batch one. This establishes the speed
effect for this warm fixed-query workload, while the full live replay above
shows its observed quality cost. It does not establish a service-load p99.
The synthetic preflight figures are not end-to-end training numbers.
No varied-image or loaded-service p99 latency or scaling curve is measured yet.

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
over the frozen pretrained substrate. The deployed-code rank treatment then
measured **90.3093% Recall@1**, **0.724227 mAP@R**, and **1,153.943 seconds**
training wall time in its
[raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-packed-rank-1000-v10.json)
(SHA-256 `6225fd990690aa50905c99110b5dd2c2551aa3654ceffb0ff14f563f63fb0c5e`).
Its source, seed, schedule, initial head/proxies, first ten augmented input
hashes, and scorer match the ArcFace control exactly. Native top-10 ordinals
and scores match the scalar oracle for every holdout query. Relative to
ArcFace, this is **-0.1538 percentage points Recall@1**, **+0.005426
mAP@R**, and **1.001×** training wall time. It fails the prespecified
+1-point Recall@1 advancement gate. The float-rank arm measured **90.7366%
Recall@1**, **0.726593 mAP@R**, and **1,153.316 seconds** training wall in
its [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-float-rank-1000-v10.json)
(SHA-256 `429e70b458ee83282c901fb890b6784d0a056d0189ca1e137b7adf9a4a3fac9c`).
All frozen authority fields, including first ten augmented input hashes,
match the other two arms. The
[paired decision receipt](evidence/compact_metric/sop-siglip2-substrate-v1/train-matched-decision-v10.json)
(SHA-256 `e32e4bf640ecb47b4ad7608267f56bba26ac1e915e1f68843f489546d24211d5`)
used 5,000 heldout-product bootstrap draws. Relative to ArcFace, float-rank
Recall@1 gained **+0.2735 points**, 95% interval **[-0.2425,+0.7959]**;
deployed-code rank lost **0.1538 points**, interval **[-0.6556,+0.3362]**.
Float-rank mAP@R gained **+0.007793**, interval **[+0.003503,+0.012079]**;
deployed-code rank gained **+0.005426**, interval **[+0.001571,+0.009306]**.
All three paired screen gates failed. These one-seed TRAIN-holdout results
support broader-ranking improvement but no proven top-1 method gain.
Official TEST results remain pending.

A fit-only positive-pair geometry
[census](evidence/compact_metric/sop-siglip2-substrate-v1/fit-positive-geometry-v1.json)
(SHA-256 `d84982d00495f1db974397c295f6b72c3b04bc990f924cf060a88b4465771a4b`)
examined all 10,186 fit products and archived its
[executed source](evidence/compact_metric/sop-siglip2-substrate-v1/fit-positive-geometry-v1.py.txt).
For ArcFace, 31.78% of fit products have a minimum positive-pair cosine below
0.5, and 44.41% have a within-product positive-pair cosine range over 0.3.
The median range is 0.2628. Float-rank reduces those fractions to 29.88%
and 43.44%, respectively; packed-rank to 30.48% and 43.58%. This measures
positive-view dispersion in fitted products only. It does not establish that
dispersion caused the holdout top-1 failures or that a new positive-side
objective will improve them.

The original deployed-code-rank screen failed. Diagnose why the rank loss
improved mAP@R without a top-1 gain, using a fit-only error census of
positive-view dispersion, then preregister one positive-side training
revision against this frozen ArcFace control. The native-fp16 deployment probe
establishes a speed gain on this fixed-query workload with a small observed,
statistically unresolved R1 difference; retain native-fp16 as
the provisional fast serving mode, while the fp32-autocast mode remains the
quality-preserving reference. Validate varied-query tails and official-protocol
quality before a production default. After a new method passes the same paired
gate, require three paired seeds, official SOP TEST and In-Shop protocols,
CUB/Cars transfer, and p99/scaling measurements before a new-method or
joint SOTA claim. Lean verifies reusable scorer correctness and conditional
bounds; empirical quality and latency still require these measurements.
