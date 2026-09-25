# Sfora joint quality and performance decision, 24 September 2026

Sfora is a neural vector similarity training and retrieval library, separate
from Borsuk. This candidate combines a pretrained SigLIP2-L/16@256 image
encoder, a trainable 1024-to-128 projection, supervised product-balanced
training, signed-int8 gallery codes with f16 inverse norms (130 bytes per
image), and an exact native CUDA top-10 scorer. Apache Arrow is not in the
training or retrieval path.

## Trained serving API

`sfora.siglip2_compact_serving.Siglip2CompactIndex` loads the trained vision
encoder, 128-dimensional head, and resident native packed gallery. The caller
must supply a trusted SHA-256 digest of the training receipt from outside the
artifact directory. The loader checks that receipt against the checkpoint,
gallery, native library, model files, image-stack versions, and CuTile toolchain.
It accepts at most 32 PIL images per request, with at most 16 million pixels
per image, and returns exact top-10 **gallery row ordinals** and scores. Callers
must map those ordinals to their own row IDs. The API does not exclude a query
image if it is also in the gallery; evaluation protocols must apply their own
self-exclusion. `close()` frees the resident gallery and releases the index's
encoder reference. Serving artifacts and their directory must be trusted and
immutable while loading, since the native library is executable code.

The revised public API was smoke-tested on the NVIDIA GB10 with the same
selected 32 SOP TRAIN holdout images and frozen paired timing receipt. For
both native-fp16 and fp32-autocast, batch-one and batch-32 ordinal-plus-score
digests matched the frozen benchmark byte for byte. The
[fp16 receipt](evidence/compact_metric/sop-siglip2-substrate-v1/serving-api-fp16-parity-v2.json)
(SHA-256 `1f390aee1e0a732424b3152a54c1ce0372e47aa0aa25ffe537f9369497048ff1`)
and [autocast receipt](evidence/compact_metric/sop-siglip2-substrate-v1/serving-api-fp32-parity-v2.json)
(SHA-256 `044de8feb9d371d6632f96b8004881e600ee94870e3ae9a01d3db0750e7ac9e2`)
bind the exact serving module, verifier, training receipt, and runtime versions.
This is a parity check, not a new latency or quality measurement.

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

The existing Lean `top1_label_correct_of_positive_margin` theorem can be
instantiated with a measured per-query score-error radius. An
[ArcFace finite-panel certificate](evidence/compact_metric/sop-siglip2-substrate-v1/float-packed-margin-certificate-v1.json)
(SHA-256 `8794dfb3c91948ae8038fb7e450eaec7959242e531e8655bdd9ba1962297a5fa`)
streamed every nonself pair of the 5,851 heldout queries and 59,551-image
gallery, comparing fp32 float descriptor dot products to the deployed packed
score formula. The maximum absolute error per query was between **0.01126
and 0.01839**, median **0.01377**. For **5,111/5,851 queries (87.35%)**,
the ideal best-positive minus best-negative margin exceeded twice that
query's measured maximum score error. The theorem then certifies a positive
packed top-1 for those finite computed scores; the measured packed result is
**5,293/5,851 (90.4632%)**. Float and packed Recall@1 match in aggregate,
with nine queries correct only in each mode. The
[executed source](evidence/compact_metric/sop-siglip2-substrate-v1/float-packed-margin-certificate-v1.py.txt)
records every per-query radius and margin. This is conditional on the
specified fp32 numerical score definitions and cached TRAIN embeddings. It
does not prove a real-arithmetic quantization bound, the compiled CUDA
kernel, an unseen-data recall rate, or a latency bound.

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

### Next fit-only diagnosis and learning gate (registered before its run)

Use the ArcFace checkpoint and packed 128-dimensional embeddings for the
53,700 fit images only. Each fit image searches all other fit images, with its
own row excluded and gallery-order tie breaking. For every error, record the
rank of the best same-product mate, whether the correct ArcFace product proxy
beats every competing proxy, and whether the top impostor is a near duplicate
in the frozen 1024-dimensional source space (cosine at least 0.97). Put errors
with best-mate rank 2–10 in the near-miss group; rank above 10 is the orphan
group. Also record product-level positive-pair coherence and whether the
best-mate direction agrees with the product mean. Require at least 200 fit
errors for a proportion-based decision; if fewer occur, run the same fixed
census on the frozen zero-update PCA head and report both without changing
the thresholds. All data and proxy choices remain on SOP TRAIN fit products;
holdout query outcomes and official TEST are unavailable to this selection.
Define direction misalignment as a negative dot product between the unit
vectors from the query to its best mate and from the query to its product
mean. Define proxy-correct using the highest unmodified normalized classifier
cosine, with the lowest product ordinal winning ties.

If at least half the errors have a frozen-space near-duplicate impostor, the
planned positive-side arm is not justified by this census. If at least 40% are
proxy-correct but gallery-member-wrong, the proxy competitor field itself is
the next target. Otherwise, run one matched `existential_mate` arm only if
near misses plus direction-misaligned orphans account for at least 30% of
errors. This arm keeps the ArcFace competitor proxies, training schedule,
architecture, and deployed scorer, and replaces the target proxy logit with a
log-sum-exp over that proxy and same-product in-batch mates. This objective
combines existing proxy and supervised-positive ideas; novelty is not claimed.
Its screen requires at least +1 percentage point paired holdout Recall@1 with
a positive product-bootstrap lower bound, mAP@R loss no worse than 0.005,
training-wall ratio no greater than 1.15, exact native top-10, and a matched
same-source ArcFace replay. Only then advance to multiple seeds and official
evaluation.

The registered [fit-only census](evidence/compact_metric/sop-siglip2-substrate-v1/fit-error-census-v1.json)
(SHA-256 `4947bb6ae8e70153018e07ed05a26777ab0e8286a9c640cdab36f219ffb64f79`)
then completed on the NVIDIA GB10 in **6.359 seconds**, with **529,820,672
bytes** peak CUDA allocation and an exact native top-10 check on 32 selected
fit queries. Of **53,700** fit leave-one-out queries, **4,296** missed top-1.
The nearest correct mate ranked 2–10 for **3,255/4,296 (75.77%)** errors and
above 10 for **1,041/4,296 (24.23%)**. The correct product proxy still led
all classifier proxies for **3,201/4,296 (74.51%)** errors, while only
**120/4,296 (2.79%)** had a frozen-source impostor cosine at least 0.97.
The preregistered decision priority therefore selects **revising the
competitor field** and rejects the planned positive-only `existential_mate`
arm. This is a fit-product mechanism screen, not a causal proof that a
competitor-aware loss improves heldout or official-protocol retrieval.

A fit-only [proxy gap follow-up](evidence/compact_metric/sop-siglip2-substrate-v1/fit-proxy-gap-v2.json)
(SHA-256 `047be0c6dc74eb4c0007eb7a7209d292693e0d7d7f7303498bc50627296e9619`)
then tested the premise of that ordered gate. The correct proxy leads for
**51,816/53,700 (96.49%)** fit queries overall, including **48,615/49,404
(98.40%)** gallery-correct queries and **3,201/4,296 (74.51%)** gallery-error
queries. Proxy correctness is therefore less common, not enriched, among
gallery errors. The impostor product proxy ranks at most 10 for
**3,463/4,296 (80.61%)** errors, including **2,569/3,201 (80.26%)** of the
proxy-correct errors. The median impostor-proxy rank is 3. Product-level
mean member-to-mean coherence is **0.8967 median** (10th percentile 0.8103).
The registered chord-based direction statistic was uninformative: zero errors
were misaligned, even after tangent-plane projection. These observations
invalidate the *causal interpretation* of the ordered proxy gate; the raw
registered outcome remains recorded. No heavy arm is selected from it alone.

Before a new training run, construct a fixed, fit-only confusable-product
graph from the zero-update PCA-128 head on frozen SigLIP2 features. For each
fit image, count products represented among its ten closest *negative*
packed-code neighbours, symmetrise counts, and retain the top four competitor
products per fit product. A candidate batch schedule may pair eight seed
products with one graph competitor each (four images per product), keeping the
ArcFace plus float SmoothAP objective and all serving artifacts identical.
This is a hard-batch sampling composition of existing ideas, not a new-method
claim. The graph lane advances only if the frozen top-four list includes the
trained ArcFace top impostor product for at least **50%** of fit errors and
if its schedule yields at least **3×** the random schedule's float SmoothAP
loss on the cached ArcFace fit embeddings, with at least **90%** of pair slots
filled by graph competitors under a maximum of three visits per product per
1,000 updates. If these screens fail, do not run the expensive arm. Also
report the share of trained errors inherited by the frozen PCA head and the
mutual-error share; neither is an advancement gate.

The [frozen fit graph](evidence/compact_metric/sop-siglip2-substrate-v1/fit-competitor-graph-v1.json)
(SHA-256 `456579df6f1c3ecdc2071aa4a6a08ea5547b9d07d4e6e471fe5f8e1f2296fdb3`)
failed its first advancement gate: its top-four competitor list covered
**2,117/4,296 (49.28%)** of the trained ArcFace error impostor products,
below the registered **50%** threshold. The graph used the verified initial
PCA/head hashes, native-exact selected top-10, and only fit products. Its
build took **9.577 seconds** on GB10, with **188,572,160 bytes** peak CUDA
allocation. The frozen head already missed **3,860/4,296 (89.85%)** of the
trained error queries, so those confusions largely predate finetuning. The
frozen-graph batch arm is stopped before training; the later schedule and
SmoothAP screens are not run. Increasing the four-neighbour gate after seeing
this result would change the registered screen. A future online competitor
field needs a separately frozen design and matched control.

### Next distinct lane: proxy-neighbour batch preflight

The next *preflight* is an online proxy-neighbour batch schedule, not a new
training run. It retains the ArcFace plus 8×float SmoothAP objective and
builds each product's top-four neighbour list from current normalized
classifier proxies, refreshing every 50 training updates. Eight seed products
each pair with an eligible neighbour to form a 16-product, four-image-per-
product batch. The graph at initialization comes only from the treatment's
own initial classifier; the final ArcFace classifier is used for fit-only
diagnosis and never seeds treatment training. This is established hard-batch
sampling territory, not a scientific novelty claim.

Freeze 16,000 four-image product tickets before any training: every one of
10,186 fit products gets one ticket, and a seeded geometry-independent subset
of 5,814 gets a second. Fix each ticket's image draws and augmentation seed.
Any treatment and both fresh controls consume identical tickets, changing
only grouping/order. The controls are the same-ticket random ArcFace plus
float SmoothAP arm and a same-ticket random ArcFace-only arm; historical
v10 runs are context rather than matched controls for this changed exposure.

All four preflight gates must pass before training, with no search over graph
width, refresh interval, quotas, or loss weights: (1) top-four proxy neighbours
cover at least **60%** of the 4,296 trained error impostor products using the
trained classifier, and at least **50%** of the zero-update head's own fit
errors using its initial classifier; (2) a deterministic same-ticket schedule
fills at least **90%** of 8,000 designated partner slots with an eligible
neighbour while retaining 16 distinct products per batch; (3) on cached
trained ArcFace fit embeddings, this schedule yields at least **3×** the
random schedule's mean float SmoothAP loss and at least **20%** of anchors
see a sampled negative member outrank their best sampled positive; and (4)
cached-descriptor timing projects at most **10%** extra training wall for
graph refresh and scheduling. These are engineering kill gates, not predicted
quality gains. If they pass, the actual three-arm training still must meet
the existing **≤1.15×** wall and **+1 percentage point** paired holdout R1
gates before any official-protocol claim.

The [proxy-neighbour coverage preflight](evidence/compact_metric/sop-siglip2-substrate-v1/proxy-neighbor-coverage-v1.json)
(SHA-256 `5101a8adbae9ebc15c5b7081c2400e9f1a0825db8cc15e22f4d3230b7bde491d`)
stopped this lane at its first gate. The initial classifier's top-four proxy
neighbours cover **6,412/12,100 (52.99%)** of the zero-update head's own
fit errors, above its 50% threshold. The trained classifier's neighbours
cover only **2,111/4,296 (49.14%)** of ArcFace fit-error impostor products,
below the required 60%. Initial neighbours also cover **48.81%** of trained
errors. The probe used the pinned initial PCA/classifier, final checkpoint,
and fit rows only; it took **6.716 seconds** on GB10 and peaked at
**932,441,088 bytes** CUDA allocation. Schedule, member-hardness, and cost
screens are not run after this failed gate. Enlarging top four or using the
final ArcFace graph to seed a fresh treatment would change the frozen design
and contaminate the training comparison. The next causal revision must test
actual member-level negatives or representation geometry with a distinct
fit-only preregistration.

## 25 September member-level bank result

The two product-graph lanes above failed their frozen preflight gates. A
distinct member-level lane used a full **53,700-member fit-only memory bank**
as the candidate set for the float SmoothAP term, retaining the same pretrained
SigLIP2 L/16@256 encoder, PCA-initialized 1024-to-128 head, ArcFace margin
0.3/scale 64, rank coefficient 8, 1,000 balanced updates, and deployed
130-byte signed-int8 plus f16-norm gallery. The [preregistered bank gate](sop_siglip2_member_bank_preflight_2026-09-25.md)
selected the raw zero-update bank after it exposed the actual trained-control
impostor among top 64 fit negatives for **4,127/4,296 errors (96.07%)**. A
source-bound worst-positive-count loss-only forward/backward and bank write
took **13.466 ms median** across 20 GB10 steps, below the 60 ms timing gate.
Those checks were diagnostic, not quality measurements.

One same-source sequential ArcFace replay and one bank treatment then completed
on NVIDIA GB10, seed 179019. Both use the **SOP official TRAIN** product-disjoint
split: 53,700 fit images/10,186 products; 5,851 holdout queries/1,132
products; every query searches all 59,551 TRAIN images excluding itself.
No official SOP TEST or In-Shop row entered training, preflight, or selection.
The two runs have equal trainer/module-source digests, model files, initial
PCA/head/classifier hashes, schedule, first ten augmented input-batch hashes,
query IDs, and exact native top-10 scoring semantics. The replay reproduced
the historical ArcFace quality exactly despite its changed trainer source.

| Same-source run | Packed Recall@1 | mAP@R | Training wall, including bank init | Peak CUDA allocated | Gallery |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArcFace control | 90.4632% | 0.718801 | 1,150.288 s | 21,089,142,784 B | 130 B/image |
| ArcFace + full-fit member-bank SmoothAP | **91.8305%** | **0.749306** | **1,166.074 s** | 21,409,461,248 B | 130 B/image |

The [raw control](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-control-1000-v1.json)
(SHA-256 `d88167bfcbf8152ee912c8382061afaf248e45fe52ae24cf5f1a5da739477fc3`),
[treatment](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-treatment-1000-v1.json)
(SHA-256 `2e73ee0e6252c91d54c815d52581abd0d303de09a56776a2fcd5b5e7908c981c`),
and [paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/member-bank-decision-v1.json)
(SHA-256 `35b144bb44e1ca67baf02e257e62219cfd8452a2f915a3b14c07011c3306e606`)
retain the per-query outcomes and resource accounting. The paired
product-cluster bootstrap over 5,000 draws gives **+1.3673 percentage points**
Recall@1, 95% interval **[+0.8628,+1.8559]**, and **+0.030506 mAP@R**, 95%
interval **[+0.025908,+0.035304]**. Training wall ratio, including the
0.208 s bank initialization, is **1.013724**. The frozen internal screen
therefore **passes** its +1-point Recall@1, positive lower bound, at most
0.005 mAP@R loss, and at most 1.15x wall gates. Gallery export took
318.553 s for control and 319.149 s for treatment; exact holdout scoring
took 2.960 and 3.017 s respectively. These are serialized run measurements,
not paired serving latency benchmarks.

This is a **single-seed internal TRAIN-holdout result**, not a published-frontier
or official-test result. The bootstrap interval conditions on the trained pair
and its lower bound is **below +1 point**; only a positive difference, not a
gain of at least one point at 95% confidence, is established. Also, 53,700
of 59,551 images in this evaluation gallery are fit-product negatives. A
new [holdout-only-gallery sensitivity check](sop_siglip2_unseen_gallery_diagnostic_2026-09-25.md)
is registered to test whether the gain persists when those seen-product
distractors are removed. Independent Opus 5.5 and GPT-6 Astra read-only audits
accepted the result as an internal screen and identified these interpretation
limits. A same-source in-batch float-SmoothAP replay is needed
to attribute any further gain specifically to the bank rather than to adding
a ranking term. At least three independent paired seeds are required for a
learning-method claim. Cross-batch memory and SmoothAP are prior art, so this
composition is not described as a novel algorithm. The next gates are the
holdout-only-gallery sensitivity check, same-source in-batch ablation and
independent seeds, then frozen official SOP TEST and In-Shop protocols, CUB/Cars transfer,
and paired full-pipeline latency/p99/scaling measurements. The published
UNICOM 91.2% SOP and 96.7% In-Shop numbers remain official-protocol reference
gates; the 91.8305% internal holdout cannot be compared to them as a win.
