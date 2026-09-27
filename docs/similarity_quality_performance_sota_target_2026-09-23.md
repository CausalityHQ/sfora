# Joint quality and performance target

## Intended result

Build a similarity-learning method whose deployed image-to-top-k system is
both more accurate and faster than a strong, directly comparable published
reference. Demonstrate the result on at least two standard retrieval datasets,
with Stanford Online Products (SOP) and DeepFashion In-Shop as the primary
product-retrieval panel. Use CUB-200-2011 and Cars196 as transfer checks.
"SOTA" is a dated, protocol-specific claim; audit the published frontier
again before any such claim. A gain over an earlier Sfora model alone is not
enough.

## Quality contract

Use each dataset's official train/test identities and query/gallery definition,
and a disclosed reference-implementation preprocessing protocol, with standard
Recall@1. Also report mAP@R, the full
retrieval curve, model and checkpoint provenance, and per-query results.
Compare the deployed representation, including quantization and its actual
score arithmetic, with both the reproduced float source and published
references. The current high published UNICOM references are 91.2% SOP and
96.7% In-Shop Recall@1 in Table 4 of its ICLR 2023 paper; they are reference
points, not an assertion that no newer result exists. The published thresholds
themselves, 91.2% and 96.7%, remain the quality gates even if a local UNICOM
reproduction falls short. Freeze the method, hyperparameters, and comparison
using training identities only. Previously observed official test results
remain exploratory product evidence and cannot be described as an untouched
confirmation. A clean confirmatory claim requires a separate held-out panel
or future untouched benchmark.

An algorithmic claim additionally needs a matched ablation: identical
backbone, pretraining, image size, data, seeds, training budget, embedding
width, search scorer, and evaluation protocol, with only Sfora's learning
method changed. One seed can screen feasibility; use at least three paired,
independent training seeds for a learning-method claim. Report paired per-query or
class-clustered uncertainty where raw comparator outputs exist. If a
published comparator cannot be reproduced, report its published number and
the mismatch in resources or protocol without inventing a paired interval.
An external encoder plus a fitted Sfora projection is a system profile until
this learning-method ablation is positive.

## Baselines and architecture

The [UNICOM ICLR 2023 paper](https://arxiv.org/pdf/2304.05884) is the current
high-quality reference in this repository's authenticated comparison set. Its
Table 4 reports 91.2% SOP and 96.7% In-Shop Recall@1 for ViT-L/14@336. This
is not certified as the latest global frontier. Table 7 specifies that model
as 24 transformer layers, width 1024, 16 attention heads, 768-dimensional
output, 336-pixel input, and 191.3 GFLOPs per image. Its ViT-B/16@224 has 12
layers, width 768, 12 heads, 768-dimensional output, and 17.6 GFLOPs; Table
4 reports 88.8% SOP Recall@1. These are full-width supervised models, so
their quality numbers alone cannot establish an equal-storage comparison.
The authenticated upstream `unicom/retrieval.py` at revision
`d71992ed969e6c271436ac0a0ee1f3ca61474ac0` (SHA-256
`35fcea34c35ce428ccbcf0af66a61b0f7deae6e77b1cfcc3867edd2f5e8d2071`)
uses the same scorer for SOP and In-Shop: normalize the full 768-output
vector, truncate to its first 512 coordinates without renormalizing, then
rank by Euclidean distance. Its SOP leave-one-out path takes the second of
the two nearest gallery rows to skip the query image itself. A local
reproduction must explicitly exclude self and apply that normalized-prefix
distance; full-width cosine can give a different ranking. Explicit self
exclusion also makes exact-duplicate ties deterministic, while the upstream
second-neighbor rule may select a different row on such ties. The paper
reports Recall@K for this comparison; any local mAP@R is an additional metric.

The current compact SOP system starts from OML's externally trained
ViT-S/16@224, produces a normalized 384-dimensional descriptor, and applies
a fitted 384-to-128 power-whitening map before signed-int8 packing. It is an
efficiency and product-quality baseline, not evidence of a new Sfora training
algorithm. UNICOM ViT-B/16@224 is the cheaper training and latency probe;
its published 88.8% full-width result is below the 91.2% L/14@336 reference,
so a B/16 reproduction alone cannot satisfy the quality target. Its compact
head would have to recover more than 2.4 percentage points over that
full-width B/16 result merely to exceed L/14@336 on SOP, plus any compression
loss. This makes B/16 a cheaper feasibility screen, not an assured final
quality path. An L/14@336 candidate inherits the reference's encoder cost;
faster search alone may not improve image-to-result latency. If neither
backbone clears both gates, investigate a materially different encoder or
distillation method rather than relaxing a gate.

The training comparison will initialize baseline and Sfora arms from the
same authenticated UNICOM-pretrained checkpoint. All matched arms use the same
official SOP train identities (59,551 images, 11,318 products), image size,
augmentations, class sampler, optimizer schedule, update count, and seed.
All matched arms include the same 768-to-128 projection, apply ArcFace to the
same normalized 128-dimensional feature, and produce the same 130-byte packed
wire. The compact control trains the backbone and projection with ArcFace.
A separate full-width UNICOM reproduction uses its own reference objective,
evaluator, and dimensionality to anchor the paper comparison; it is not the
matched compact control. The first Sfora arm trains the entire backbone from
the same pretrained starting point and for the same total updates as its
matched control, with the same ArcFace loss plus a deployed-code
ranking term: signed-int8 fake quantization in the forward path, f16
inverse-norm simulation, a straight-through gradient for rounding, and
SmoothAP on class-balanced positive/negative pairs. Its coefficient and
checkpoint rule must be fixed using training classes before any official-test
read. A matched float-rank arm, with the same projection and total updates
but no quantized rank forward path, separates the gain from rank training
from any benefit of training through the deployed code. The earlier
head-only version failed its gate, so this full-backbone variant is an
uncertain experiment rather than an assumed win. The existing In-Shop port
uses 128 epochs, batch 128, OneCycle, EMA, an eight-mask
classifier objective, margin 0.25, and scale 32. Those are implementation
starting points, not verified SOP settings: the SOP recipe and its agreement
with primary source code must be frozen before training. The paper describes
ArcFace margin 0.3 and scale 64 in its general implementation section, so
silently presenting the In-Shop values as a faithful SOP reproduction would
be wrong. All compact arms use identical train-only calibration and the exact
packed scorer. Run the same frozen method on In-Shop; SOP-only tuning cannot
establish a cross-dataset learning advance.

## Performance contract

Keep one image encoder and the current 128 signed-byte code plus one f16
inverse norm (130 stored bytes per gallery item) unless a new storage budget
is explicitly declared. Measure on the same GPU and software stack for
candidate and reference, with identical images, query order, gallery size,
and top-k semantics. Report image decode/preprocess, encoder and packing,
native search, and full image-to-result time separately at batch 1 and 32.
Measure native SOP and In-Shop galleries first. Separately measure scalability
with a million authenticated distinct items, or label tiled/synthetic galleries
as smoke tests. Record p50, p95, p99, throughput, GPU allocation, host RSS,
and gallery bytes. A p99 claim requires at least 10,000 timed calls per cell
across at least 10 interleaved paired blocks, with block-bootstrap confidence
intervals, GPU clocks, and contention recorded; 50-call maxima are diagnostic
only. Exact top-k and tie behavior must match the scalar packed-score oracle.

The joint result advances only when deployed quality exceeds the published
quality reference on both primary datasets, and full-pipeline latency improves
over its faithful local reproduction on the same hardware and query/gallery
workload. Require the upper bound of the paired 95% confidence interval for
the p99 latency ratio (candidate/reference) to be below 1.0 at batch 1 and
32, plus nonregressing p50 and throughput. Compare storage explicitly: the
published full-width quality reference and 130-byte compact candidate have
different footprints. Also show the quality/latency Pareto position against
a strong efficient baseline.
If the best published quality model has no obtainable checkpoint or measured
runtime, label the speed comparison unverified rather than borrowing numbers
from a different machine. No aggregate score hides a quality or latency
regression on the primary datasets.

The performance panel uses three distinct references: the reproduced
high-quality model for the joint claim; the OML ViT-S/16@224 compact profile
as an efficient encoder control on SOP; and a competitive exact-search
implementation (including FAISS or a resident float control) with identical
queries and tie semantics for the search component. RC3 versus RC4 measures
Sfora's internal serving progress only. The released RC4 evidence reports
batch-32 median search falling from about 4.5 to 2.8 ms on one synthetic
million-row gallery, while batch-1 stays near 1.05 ms. That result neither
compares encoders nor establishes better-than-published system performance.

## Historical checkpoint, 23 September

The optional OML SOP profile reaches 85.9757% Recall@1 and 0.641825 mAP@R
at 130 bytes; the previous Sfora SOP path reached 77.0173% and 0.511490.
The new profile does not beat its own OML float source (86.5575%) or the
UNICOM published full-width SOP reference. The corrected tiled 1M search
replay shows near-matched medians but has neither a stable p99 estimate nor
encoder latency. In-Shop's current compact ViT-L/14@336 reaches 95.4283%
Recall@1, below the UNICOM 96.7% reference; the local full-width reproduction
reached only 95.15%. The head-only deployed-code SmoothAP screen did not clear
its advancement gate. None of these results establishes superiority in both
quality and full-pipeline latency.

First measure a paired encoder-plus-packed-search baseline on SOP using the
current OML ViT-S/16@224 model and a UNICOM ViT-B/16@224 candidate. Then
fine-tune one backbone on SOP train with a frozen recipe, fit compression on
train only, and use a train-identity validation split for product selection.
The already-observed official test may be reported as exploratory evidence.
The matched-backbone method ablation, a second dataset, independent seeds,
and a clean confirmation panel are required before an algorithmic or
cross-dataset SOTA claim. The existing Lean proofs establish abstract top-k
exactness and conditional recall and cost bounds; empirical accuracy and
latency remain measurements.

## Measured continuation, 25 September

The current candidate is a SigLIP2 Large patch16/256 image encoder with a
PCA-initialized trainable 1024→128 head. The training objective is ArcFace on
fit-only product proxies plus SmoothAP against a detached full-fit member bank;
the matched float control uses the same encoder/head/schedule and in-batch
SmoothAP. Both use BF16 vision autocast, FP32 objective and parameters, AdamW
(vision 1e-5, head/proxies 1e-4, decay 0.05), and 1,000 updates × 64 images.
The deployed gallery stores 128 signed int8 coordinates plus an fp16 inverse
norm: 130 bytes/image. The native scorer ranks the packed score exactly with
ordinal tie breaks. These are known components; the combination has no
established novelty claim.

| Dataset and split | Candidate or control | Quality | Cost/performance | Evidence status |
| --- | --- | --- | --- | --- |
| SOP official TRAIN, 5,851 held queries/full TRAIN gallery | Coverage bank, 3 seeds | packed R@1 92.4457%, mAP@R 0.767977 | mean 1,150.78 s/64,000 training images; 55.61 images/s; 21.409 GB peak CUDA | [paired replicated holdout](sop_siglip2_coverage_replication_result_2026-09-25.md), verified source-bound, selection split |
| SOP official TRAIN, 5,851 held queries/full 59,551 TRAIN gallery | True lower-stack freeze vs same-source full-train bank control, seeds 179024/26/27 | mean packed R@1 **92.6850% vs 92.2691%**; mAP@R **0.777123 vs 0.766649**; each seed passes its frozen paired gate | mean **752.464 vs 1,151.417 s** per 64,000 images; **85.05 vs 55.58 images/s**; peak CUDA **11.795 vs 21.409 GB** on DGX Spark GB10; 130 bytes/gallery image | [source-bound paired decisions and receipts](sop_true_freeze_gate_2026-09-26.md), verified exploratory TRAIN-only; opt-in trainer flag, no official or SOTA claim |
| SOP official TEST, 60,502 symmetric queries/full TEST gallery, self excluded | True lower-stack freeze vs same-source full-train bank control, 3 paired seeds | mean packed R@1 **91.7342% vs 91.1810%**; mAP@R **0.768144 vs 0.756578**; each seed passes frozen paired gate | 130 bytes/gallery image; mean export **321.33 vs 322.61 s** on DGX Spark GB10 | [source-bound six-checkpoint official read](sop_true_freeze_official_gate_2026-09-26.md), verified exploratory; TEST previously observed, offline batch-64 export, not a SOTA or deployed-quality claim |
| CUB-200-2011 classes 101–200 TEST, 5,924 symmetric self-excluded queries/gallery, zero-shot SOP checkpoint transfer | Same three SOP true-freeze seeds vs same-seed full-train control, public FP16 128-D packed encoder and packed-score oracle | mean packed R@1 **70.6561% vs 68.5348%**, paired class-bootstrap gain **+2.1213 pp**, 95% **[+1.2835,+2.9603] pp**; paired seed-t 95% **[+1.4989,+2.7438] pp**; mean mAP@R **0.267333 vs 0.252639** | SOP training **752.464 vs 1,151.417 s**/64,000 sampled images; CUB batch-32 export **56.847 vs 56.805 s** for 5,924 images, peak PyTorch CUDA **0.883 GB** both, **770,120-byte** gallery on DGX Spark GB10; no CUB public latency measured | [source-bound three-seed transfer](sop_siglip2_cub_transfer_gate_2026-09-27.md) is exploratory because CUB TEST had prior project reads; zero-shot transfer supports generalization of this SOP treatment, not an In-Shop or SOTA claim |
| Cars196 classes 98–195 evaluation, 8,131 symmetric self-excluded queries/gallery, zero-shot SOP checkpoint transfer | Same three SOP true-freeze seeds vs same-seed full-train control, public FP16 128-D packed encoder and packed-score oracle | mean packed R@1 **87.9146% vs 87.7260%**, paired class-bootstrap gain **+0.1886 pp**, 95% **[−0.2793,+0.6628] pp**; paired seed-t 95% **[−0.7825,+1.1597] pp**, seed 179024 regresses; mean mAP@R **0.255999 vs 0.246522**, seed-t 95% delta **[−0.016700,+0.035654]** | SOP training **752.464 vs 1,151.417 s**/64,000 sampled images; Cars batch-32 export **91.875 vs 91.352 s** for 8,131 images, peak PyTorch CUDA **0.883 GB** both, **1,057,030-byte** gallery on DGX Spark GB10; no Cars public latency measured | [source-bound three-seed transfer](sop_siglip2_cars_transfer_gate_2026-09-27.md) is exploratory because Cars evaluation classes had prior project reads; Cars zero-shot transfer cannot establish a better Cars training recipe or SOTA claim |
| SOP official TEST, 60,502 symmetric queries/full TEST gallery, self excluded | True lower-stack freeze vs same-source control, 3 paired seeds, public batch-32 FP16 gallery/query arithmetic | mean packed R@1 **91.7419% vs 91.1788%**; mAP@R **0.768230 vs 0.756564**; each seed passes its frozen paired product-bootstrap gate | 130 bytes/gallery image; mean export **414.09 vs 414.24 s** on DGX Spark GB10; seed 179024 frozen production loader first-32 image-to-top-10 codes/ordinals/scores exact | [source-bound public-config official gate and loader receipt](sop_true_freeze_public_official_gate_2026-09-26.md), verified exploratory; TEST previously observed, production parity sampled 32 queries, no current SOTA or p99 claim |
| SOP official TEST, all 60,502 batch-1 image queries/full fixed batch-32 FP16 gallery | Seed 179024 frozen production encoder/native scorer vs its own batch-32 query export | packed R@1 **91.6895% vs 91.6664%**; mAP@R **0.767697 vs 0.767652**; R@1 paired product-bootstrap 95% delta [+0.0066, +0.0414] pp | 130 bytes/gallery image; **963.94 s** query wall including model load and decode/native search (62.77 images/s); peak allocated CUDA **1.267 GB** on DGX Spark GB10 | [full batch-1 production query gate](sop_true_freeze_public_official_gate_2026-09-26.md), verified exploratory; only 14,480/60,502 codes/norms bitwise match batch-32, but all native top-10/R@1 exact; one seed, no matched p99 or SOTA claim |
| SOP official TRAIN, 5,851 held queries/full 59,551 TRAIN gallery | Public API batch-32 query path, freeze vs same-source control, 3 paired seeds | FP32 autocast mean packed R@1 **92.7363% vs 92.2748%**, mAP@R **0.776989 vs 0.766652**; default FP16 mean R@1 **92.7249% vs 92.3033%**, mAP@R **0.776967 vs 0.766683** | Seed 179024 matched image-to-top-10 p50/p95 batch1 FP16: **16.253/19.000 vs 16.146/18.902 ms**; batch32 **287.423/312.903 vs 287.522/311.926 ms** per batch; 400 calls/arm/cell on DGX Spark GB10, no p99 | [actual public path TRAIN quality and latency](sop_true_freeze_public_train_gate_2026-09-26.md), verified exploratory; strict batch-64 offline-export parity failed, motivating the separate public-config TEST gate below |
| SOP official TRAIN, one fixed image repeatedly queried against each 59,551-row TRAIN gallery | Seed 179024 true freeze vs same-source control, FP16 production image-to-top-10 path | Quality not remeasured in timing run; paired TRAIN and official TEST results above | p50 **15.628 vs 15.618 ms**; p99 **19.622 vs 19.606 ms**; p99 ratio **1.00084**, paired-block 95% interval **[0.99223, 1.01364]**, 10,000 calls/arm; 5% tail nonregression passed, **no speed improvement** | [source-bound batch-1 tail receipt](sop_true_freeze_public_batch1_p99_gate_2026-09-26.md), verified single-image idle-GPU diagnostic, not varied-image or official TEST p99 |
| SOP official TRAIN, same fixed image and 59,551-row TRAIN gallery, seed 179024 | True-freeze FP16 public image-to-top-10 stage attribution, 100 calls | Top-10 ordinals/scores exactly match the public API; quality not remeasured | synchronized p50 whole **16.952 ms**, processor **6.412 ms**, vision **8.425 ms**, native search **0.329 ms**; exploratory exact CPU-tensor input **16.167 vs 15.763 ms** whole-call p50, slower | [source-bound stage profile](sop_siglip2_public_stage_profile_2026-09-27.md) locates preprocessing as the next exact-output speed target; no production promotion or p99 claim |
| SOP official TRAIN, first 32 image pixels, seed 179024 same FP16 vision weights | Native CUDA graph vision-only replay vs eager, exact pooled outputs | All 32 batch-1 and batch-32 pooled tensors bitwise equal; retrieval quality not remeasured | batch-1 synchronized vision p50 **7.161 vs 8.043 ms** (−11.0%); batch-32 p95 **216.856 vs 217.444 ms**; 400 calls/arm/batch and peak CUDA **0.995 GB** on DGX Spark GB10; no public whole-call timing | [frozen isolated-stage feasibility](sop_siglip2_cuda_graph_vision_preflight_2026-09-27.md) passes; permits an opt-in batch-1 public-path exactness and varied-image latency screen, not production promotion |
| SOP official TRAIN, 10,000 unique image byte hashes/full 59,551-row TRAIN gallery, seed 179024 | Corrected opt-in native-FP16 batch-1 CUDA graph public index vs default eager, same checkpoint and exact scorer | All 10,000 paired top-10 ordinals/scores exact; retrieval Recall@1 and mAP@R not remeasured | image-to-top-10 p50 **15.385 vs 16.311 ms**, p99 **19.070 vs 19.999 ms**, 10,000 calls/arm including JPEG decode; post-construction peak PyTorch allocated **1.504 GB** and four-worker 100-query serialisation pass on DGX Spark GB10 | [review-corrected public and unique-image gates](sop_siglip2_cuda_graph_vision_preflight_2026-09-27.md), verified exploratory serving result; qualified opt-in for this native-FP16 batch-1 GB10 configuration, no official TEST p99 or SOTA claim |
| SOP official TRAIN, 10,000 unique image byte hashes, processor-only | PIL-resize uint8 lookup candidate vs installed Torchvision-backed SigLIP2 processor | **9,970/10,000** pixel tensors differ; retrieval quality not assessed | candidate p50/p95 **0.910/1.095 ms** vs control **1.216/1.684 ms**, 2,000 timed calls/arm; combined peak RSS **0.887 GB** on DGX Spark GB10 | [frozen exactness gate](sop_siglip2_processor_lut_gate_2026-09-27.md) fails; reject candidate before public serving promotion or top-10 timing; no training or quality claim |
| SOP official TEST, 60,502 symmetric queries, self excluded | Coverage bank, 3 seeds | packed R@1 91.2725%, mAP@R 0.757861 | 130-byte gallery | [exploratory official read](sop_siglip2_coverage_official_result_2026-09-25.md), source-bound; TEST had prior Sfora reads |
| SOP official TEST, 60,470 gallery rows after 32 held query rows | Selected bank checkpoint vs faithful UNICOM L14/336 | quality assessed separately above | image-to-top10 p50/p95 at batch1: 23.328/26.239 ms vs 36.305/39.055 ms; batch32: 314.683/325.875 ms vs 558.497/568.963 ms | [100-call diagnostic](sop_coverage_vs_fullwidth_unicom_latency_2026-09-25.md), no p99 certification |
| In-Shop official TRAIN, 2,540 class-disjoint held queries/full 25,882 TRAIN gallery | Bank vs matched float, three seeds | mean packed R@1 97.6640% vs 97.1916%; mAP@R 0.777733 vs 0.760664 | mean 1,150.614 s/55.62 images/s/23.829 GB peak CUDA vs 1,137.119 s/56.28 images/s/21.077 GB per 64,000 images | [completed paired exploratory TRAIN gate](inshop_siglip2_paired_training_gate_2026-09-25.md); bank selected for official query/gallery |
| In-Shop official TRAIN, 12,599 held-only symmetric queries/gallery, product-disjoint from fit | Lower-stack freeze vs same-source control, seeds 179023/24/25 | mean packed R@1 **98.5977% vs 98.4152%**, +0.1826 pp; pooled product-bootstrap lower +0.0716 pp, but seed 179024 lower −0.0856 pp **fails frozen individual gate** | mean **983.7 vs 1,145.4 s** per 64,000 sampled images; peak CUDA **18.124 vs 22.554 GB** on DGX Spark GB10 | [source-bound exploratory TRAIN replication](inshop_siglip2_unseen_gallery_gate_2026-09-26.md); faster, quality repeatability unresolved, no official promotion |
| In-Shop same TRAIN held-only symmetric gallery, seed 179024 | Vision LR 3e-5 vs 1e-5 same-source control | packed R@1 **97.9602% vs 98.5554%**, mAP@R **0.796341 vs 0.826739**; paired R@1 −0.5953 pp, 95% interval [−0.8173, −0.3856] | training wall 1,146.22 vs 1,147.93 s; both peak CUDA 22.554 GB | [frozen LR screen](inshop_siglip2_unseen_gallery_gate_2026-09-26.md) failed; no second seed or official read |
| In-Shop official TRAIN, 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024 | Published Proxy Synthesis ArcFace augmentation vs same-seed control | packed R@1 **97.6744% vs 98.5554%**, mAP@R **0.819016 vs 0.826739**; paired product-bootstrap R@1 delta **−0.8810 pp**, 95% interval **[−1.1025, −0.6504] pp** | training **1,147.444 vs 1,146.916 s**/64,000 images; **55.776 vs 55.802 images/s**; peak CUDA **22.557 vs 22.554 GB** on DGX Spark GB10 | [frozen source-bound full gate](inshop_proxy_synthesis_gate_2026-09-26.md) failed; stop before seed 179025 and official TEST; no production promotion |
| In-Shop official TRAIN, 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024 | Valid-anchor rank recovery vs same-seed true-freeze | packed R@1 **98.6269% vs 98.5475%**, mAP@R **0.840838 vs 0.839099**; paired product-bootstrap R@1 delta **+0.07937 pp**, 95% interval **[−0.03301, +0.19791] pp** | training **748.644 vs 748.621 s**/64,000 images; **85.488 vs 85.491 images/s**; both peak CUDA **12.939 GB** on DGX Spark GB10 | [frozen source-bound first-seed gate](inshop_valid_anchor_rank_gate_2026-09-26.md) passes; exploratory TRAIN point only, subsequent paired confirmation below fails |
| In-Shop official TRAIN, 12,599 product-disjoint held-only symmetric queries/gallery, paired seeds 179026/27 | Valid-anchor rank recovery vs same-seed true-freeze | seed 179026 packed R@1 **98.6824% vs 98.5237%**; seed 179027 **98.5475% vs 98.6269%**; pooled paired R@1 delta **+0.03969 pp**, product-bootstrap 95% **[−0.03148, +0.10852] pp**; pooled mAP@R delta **+0.002168** | treatment/baseline training **748.729/746.384 s** and **749.300/748.226 s** per 64,000 images; all peak CUDA **12.939 GB** on DGX Spark GB10 | [frozen two-seed confirmation](inshop_valid_anchor_rank_confirmation_gate_2026-09-26.md) **fails** individual and pooled R@1 gates; retain opt-in, no official TEST or production default promotion |
| In-Shop official TRAIN, same 12,599 held-only symmetric queries/gallery, seed 179026 true-freeze checkpoint | 192-pixel inference with positional interpolation vs its 256-pixel replay | packed R@1 **96.9125% vs 98.5237%** (−1.6112 pp); mAP@R **0.768205 vs 0.841910** | sequential full-held export **60.775 vs 67.433 s**, DGX Spark GB10; no public image-to-top-k latency measurement | [source-bound exploratory resolution screen](inshop_siglip2_resolution_screen_2026-09-27.md) rejects 192-pixel production path at current weights |
| In-Shop official TRAIN, same 12,599 held-only symmetric queries/gallery, seed 179026 true-freeze checkpoint | 224-pixel inference with positional interpolation vs its 256-pixel replay | packed R@1 **97.7697% vs 98.5237%** (−0.7540 pp); mAP@R **0.801486 vs 0.841910** | sequential full-held export **81.908 vs 67.623 s**, DGX Spark GB10; no public image-to-top-k latency measurement | [predeclared 0.5-point resolution screen](inshop_siglip2_resolution_screen_2026-09-27.md) fails; stop resize lane, no distillation or production promotion |
| In-Shop official TRAIN, seed-179026 true-freeze checkpoint, fixed 6,354-query/6,245-gallery held roles and 12,599-image symmetric held gallery | 320-pixel inference with positional interpolation vs exact 256-pixel replay | asymmetric packed R@1 **97.2301% vs 97.6235%** (−0.3935 pp, within frozen −0.5 pp floor); symmetric packed R@1 **98.3411% vs 98.5237%**, mAP@R **0.824011 vs 0.841910** | sequential full-held export **102.809 vs 67.265 s** (+52.8%) on DGX Spark GB10; no public image-to-top-k latency or training measurement | [source-bound TRAIN-only preflight](inshop_siglip2_320_preflight_2026-09-27.md) shows no quality gain and worse mAP/cost; stop 320 training and production promotion despite permissive R@1 floor |
| In-Shop official TRAIN, 13,283 fit-product PCA rows and 12,599 held-only symmetric queries/gallery, no fine-tuning | Native DINOv2-L/224 own-PCA 128-D packed source vs SigLIP2 Large/256 own-PCA packed source | packed R@1 **43.4717% vs 81.6652%**, mAP@R **0.113994 vs 0.460749**; DINOv2 raw 1,024-D float R@1 **48.6626%** | full 25,882-TRAIN-image DINOv2 export **191.535 vs 216.163 s** historical SigLIP2 source export; DINOv2 peak allocated CUDA **0.888 GB** on DGX Spark GB10; no public latency | [source-bound frozen-source preflight and raw cause check](inshop_dinov2_source_preflight_2026-09-27.md) reject this substrate recipe before fine-tuning or official TEST; source/processor differ and sequential export walls are not a matched serving claim |
| In-Shop official TRAIN, same 13,283 fit rows and 12,599 held-only symmetric queries/gallery, pretrained SigLIP2 Large/256 caches | Equal-weight normalized block-22 + block-24 source followed by own fit-only PCA-128/int8 vs stronger block-22 own-PCA control | packed R@1 **84.4670% vs 87.1736%** (−2.7066 pp); mAP@R **0.488134 vs 0.508923** (−0.020788), product-bootstrap 95% **[−0.025096, −0.016593]** | cached-feature PCA fit **4.140 s**, total **4.864 s**, peak CUDA **85.704 MB** on DGX Spark GB10; encoder/training/public latency unmeasured | [source-bound dual-depth source preflight](inshop_multidepth_head_preflight_2026-09-27.md) fails frozen quality gate; stop before training or production change |
| In-Shop official TRAIN, 13,283 fit-product PCA rows and 12,599 held-only symmetric queries/gallery, no fine-tuning | TIPSv2 L/14 vision at 224, second CLS, own PCA-128/int8 vs stronger SigLIP2 22-block own-PCA packed source | packed R@1 **85.3957% vs 87.1736%** (−1.7780 pp); mAP@R **0.504019 vs 0.508923** (−0.004903), paired product-bootstrap 95% **[−0.010692, +0.000824]** | full 25,882-image export **295.853 s** vs **216.163 s** historical SigLIP2 24-block source export (+36.9%), peak CUDA **0.892 GB** on DGX Spark GB10; no training or public latency measurement | [frozen source screen](inshop_tipsv2_l14_source_preflight_2026-09-27.md) fails four of five gates; stop before training, official TEST, or production promotion |
| In-Shop seed 179026 true-freeze checkpoint, synthetic batch-64 compute probe (no quality split) | Frozen block-11 activation cache vs full frozen-lower execution | pooled output exactly equal on one batch; retrieval quality unmeasured | median full **0.65023 s**, cached upper **0.48848 s**, cache generation **0.16120 s** per batch; FP32 cache **1 MiB/image**; four views need **51.89 GiB** on DGX Spark GB10 | [source-bound cost screen](inshop_frozen_cache_cost_screen_2026-09-27.md) estimates only 4.3% compute saving at 1,000 updates before I/O; reject four-view cache as next production change |
| In-Shop official TRAIN, fixed asymmetric 6,354-query/6,245-gallery subset of 12,599 held images, seed 179026 true-freeze | Same-pose best-positive removal mechanism vs full held gallery | packed R@1 **97.6235% vs 98.3318%** on the same query rows; 78 induced misses, 33 rescued; only **10/78 = 12.8%** induced misses lost a same-pose best positive, below proposed 60% mechanism threshold | one frozen-checkpoint export **67.319 s** on DGX Spark GB10; no training or public latency | [source-bound exploratory mechanism screen](inshop_pose_positive_gap_screen_2026-09-27.md) rejects pose-specific gallery-subset loss before training |
| In-Shop official TRAIN, same fixed 6,354-query/6,245-gallery held roles, seed 179026 true-freeze | Same-pose top impostors on misses vs hits, stratified by query pose excluding flat | initial gallery-frequency enrichment **61/151 vs 29.150 expected = 2.093×**; failure-specific Mantel-Haenszel odds ratio **1.729**, product-bootstrap 95% **[1.225, 2.432]**, fails predeclared lower **>1.3** | frozen-checkpoint export **67.373 s** on DGX Spark GB10; training and public latency unmeasured | [source-bound exploratory falsifier](inshop_pose_impostor_failure_specificity_2026-09-27.md) stops pose-conditioned negative training; reshuffled roles cannot rehabilitate rejected selector |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, checkpoints 179024/26/27 | Uniform FP64 weight soup of three same-initialization true-freeze vision/head checkpoints vs their mean and best source seed | packed R@1 **98.6189%** vs mean **98.5660%** and best **98.6269%**; soup-minus-mean product-bootstrap 95% **[−0.0376, +0.1451] pp**; mAP@R **0.847175 vs mean 0.839866** | source training sum **2,238.680 s** per three × 64,000 images; soup held export **66.685 s**, peak CUDA **1.886 GB**, probe peak parent RSS **11.729 GB** on DGX Spark GB10; public latency unmeasured | [frozen TRAIN-only soup screen](inshop_weight_soup_preflight_2026-09-27.md) fails primary best-plus-0.10 pp and paired-lower gates; stop before new seeds, official TEST or production |
| In-Shop official TRAIN, fixed 6,354-query/6,245-gallery held roles and 12,599-image symmetric held gallery, seed 179026 | Fixed `0.75` trained-head pooler + `0.25` trained-head final-token mean, same 128-D int8/fp16 pack vs same-checkpoint head | native packed R@1 **97.5606% vs 97.6235%** (−0.0630 pp, 8 rescues/12 losses); symmetric packed mAP@R **0.841091 vs 0.841910** | one shared export **67.583 s**, peak allocated CUDA **1.886 GB** including scoring, same 130-byte gallery on DGX Spark GB10; no added training, public serving latency unmeasured | [frozen TRAIN-only token gate](inshop_trained_token_mean_gate_2026-09-27.md) triggers predeclared negative-delta early stop; no other seeds, official read or production promotion |
| In-Shop official TRAIN, inner-fit 6,514 images/997 products unseen by a half-fit seed-179024 backbone | Rank-16 token-mean residual, 250 fit-only updates vs same-capacity deranged-token control and unchanged base | packed R@1 **98.9254%** treatment vs **98.9100%** base and **98.8793%** donor; treatment-base **+0.01535 pp**, product-bootstrap lower **−0.06168 pp**; mAP@R **0.850055 vs 0.849110 vs 0.847670** | shared export **70.755 s**, adapter fit **1.157 vs 0.877 s**, peak CUDA **68.749 MB** after export; public serving unmeasured on DGX Spark GB10 | [frozen inner-fit information gate](inshop_token_residual_inner_fit_gate_2026-09-27.md) fails seed-17 mAP floor and stops before other seeds or outer held; no production promotion |
| In-Shop official TRAIN, 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024 | Full fit 2,004 products/13,283 images vs nested half fit 1,002 products/6,764 images, both true-freeze 1,000 updates | packed R@1 **98.5475% vs 98.1824%**, full-minus-half **+0.3651 pp**, product-bootstrap 95% **[+0.1535, +0.5740] pp**; mAP@R **0.839099 vs 0.819225** | training **747.058 vs 743.215 s**/64,000 images, **85.67 vs 86.11 images/s**; peak CUDA **12.939 vs 11.851 GB** on DGX Spark GB10; serving speed unchanged | [source-bound TRAIN scaling gate](inshop_fit_product_scaling_gate_2026-09-27.md) misses frozen +0.4 pp advance floor; stop all-TRAIN retrain, no official TEST read |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, seed 179026, 100 updates | 22-block SigLIP2/256 with its own pretrained cache/PCA/bank vs matched 24-block control | packed R@1 **97.1903% vs 97.3411%** (−0.1508 pp); mAP@R **0.766315 vs 0.775415** (−0.009100, fails frozen −0.005 floor) | training **67.248 vs 76.317 s**/6,400 images, **95.17 vs 83.86 images/s**; peak CUDA **11.241 vs 12.939 GB** on DGX Spark GB10; separate cache export **180.906 vs 216.163 s**; public latency unmeasured | [source-bound 100-update feasibility screen](inshop_siglip2_depth22_screen_2026-09-27.md) rejects depth22 before batch-1 and full training; no production promotion |
| In-Shop official TRAIN, same 12,599 held-only symmetric queries/gallery, pretrained source caches, no training | Fit-only 22→24 Procrustes/PCA head vs 22-block own-PCA head; 24-block own-PCA also shown | transferred packed R@1 **86.6497% vs 87.1736%**, mAP@R **0.503109 vs 0.508923** (−0.005814; product-bootstrap 95% **[−0.006847, −0.004785]**); 24-block own-PCA **81.6652% / 0.460749** | fit **2.680 s**, cache-to-retrieval probe **4.550 s**, peak CUDA **85.704 MB** on DGX Spark GB10; existing source-cache exports excluded | [source-bound cross-depth head falsifier](inshop_cross_depth_head_falsifier_2026-09-27.md) rejects this transfer before training; pretrained source ordering is exploratory and does not revise the trained 100-update gate |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024, 1,000 updates | First-16-block freeze vs archived same-seed first-12-block true freeze; same 24-block serving encoder | packed R@1 **98.4602% vs 98.5475%** (−0.0873 pp); mAP@R **0.831464 vs 0.839099** (−0.007635), paired product-bootstrap 95% **[−0.009652, −0.005616]** | training **617.658 vs 747.058 s**/64,000 images, **103.62 vs 85.67 images/s**; peak CUDA **9.745 vs 12.939 GB** on DGX Spark GB10; serving path unchanged | [source-bound TRAIN freeze16 gate](inshop_freeze16_gate_2026-09-27.md) fails mAP point and bootstrap floors; stop before other seeds, official TEST or production promotion |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024, 1,000 updates | mAP@R-cutoff SmoothAP with first-16-block freeze vs archived same-seed first-12-block freeze | packed R@1 **98.4126% vs 98.5475%** (−0.1349 pp); mAP@R **0.833200 vs 0.839099** (−0.005898), paired product-bootstrap 95% **[−0.007951, −0.003816]** | training **617.144 vs 747.058 s**/64,000 images, **103.70 vs 85.67 images/s**; peak CUDA **9.745 vs 12.939 GB** on DGX Spark GB10; serving path unchanged | [source-bound TRAIN mAP@R cutoff gate](inshop_mapr_aligned_freeze16_gate_2026-09-27.md) fails mAP point and interval floors; stop before other seeds, official TEST or production promotion |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, seed 179024, 100 updates | First-16-block freeze with live-head 1,024-D source bank vs matched detached 128-D bank | packed R@1 **96.7934% vs 96.6347%** (+0.1587 pp); mAP@R **0.756088 vs 0.754387** (+0.001701), paired product-bootstrap 95% **[+0.001001, +0.002412]** | training plus shared setup **64.966 vs 64.498 s**/6,400 images, **98.51 vs 99.23 images/s**; peak CUDA **9.807 vs 9.745 GB** on DGX Spark GB10; serving path unchanged | [source-bound TRAIN live-head gate](inshop_live_head_bank_gate_2026-09-27.md) fails frozen +0.004 mAP floor; retain first-12-freeze In-Shop baseline, stop before 1,000 updates, other seeds or official TEST |
| In-Shop official TRAIN, seed-179026 fixed 6,354-query/6,245-gallery subset of the 12,599-image product-disjoint held panel | Fixed-negative expected-gallery rank surrogate vs actual packed asymmetric gallery, same checkpoint | predicted R@1 **96.5051%** vs actual **97.6235%**, **1.1184 pp** error fails frozen 0.4 pp ceiling; 391/2,579 = **15.16%** near-boundary positives fails 30% floor | one re-export **67.442 s**, scoring **0.208 s**, peak allocated CUDA **1.886 GB** on DGX Spark GB10; no training or public serving change | [source-bound TRAIN falsifier](inshop_expected_gallery_recall_preflight_2026-09-27.md); stop candidate before loss code, other seeds or official TEST |
| In-Shop official TRAIN, same seed-179026 fixed 6,354-query/6,245-gallery held panel | Replayed packed misses with pretrained source-cosine impostor census | 151 asymmetric misses reproduced exactly; **67/151 = 44.37%** have packed margin within 0.05; **0/151** have source cosine ≥0.97 (median 0.842, max 0.963) | re-export **67.105 s** on DGX Spark GB10; no training or serving change | [source-bound TRAIN impostor census](inshop_impostor_source_gap_gate_2026-09-27.md) passes exploratory reopen screen, not a method-quality or label-error claim |
| In-Shop official TRAIN, same 6,354-query/6,245-gallery held roles | Pinned pretrained SigLIP2 1,024-D float source cache vs trained seed-179026 128-D packed checkpoint | source R@1 **79.4775%** vs trained packed **97.6235%**; source rescues 15 trained misses but loses 1,168 trained hits | GPU source-score diagnostic only; no training or public latency | [source-bound same-role control](inshop_impostor_source_gap_gate_2026-09-27.md); rejects source-ranking preservation as next treatment, not an equal-byte baseline |
| In-Shop official TRAIN, seed 179026, 6,354-query/6,245-gallery fixed held roles and 12,599-image symmetric held gallery | Specific-impostor bank loss vs paired true-freeze, 100 updates | asymmetric packed R@1 **95.4832% vs 95.4989%**, Δ **−0.0157 pp**, product-bootstrap 95% **[−0.1908, +0.1688] pp**; symmetric packed mAP@R **0.771910 vs 0.775415** | training **76.461 vs 75.987 s**/6,400 images; **83.70 vs 84.22 images/s**; peak CUDA **12.942 vs 12.939 GB** on DGX Spark GB10; public serving unchanged | [frozen paired TRAIN screen](inshop_impostor_source_gap_gate_2026-09-27.md) fails both quality gates; stop before more seeds, official TEST, or production promotion |
| In-Shop official TRAIN, seed 179026, same 6,354-query/6,245-gallery fixed held roles and 12,599 symmetric held queries, 100 updates | Wider train crop area `(0.25,1.0)` vs same-source `(0.8,1.0)` true-freeze control | additional/full/flat R@1 delta **−0.1799 pp** (product-bootstrap lower **−0.7756 pp**); whole-role delta **+0.0315 pp**; symmetric mAP@R **0.770336 vs 0.775415** | training wall ratio **0.9993×** and peak CUDA ratio **1.0000×**, same scorer and serving path | [frozen framing preflight and paired TRAIN screen](inshop_crop_view_preflight_2026-09-27.md) fails targeted R@1 and mAP gates; remove crop option, stop before full training/official read |
| In-Shop official query/gallery, 14,218/12,612 | Selected bank, three seeds | mean packed R@1 **95.4823%**, mAP@R **0.784502** | 130-byte/image gallery; mean export 143.239 s for query+gallery; native top-10 exact on all queries | [exploratory official readout](inshop_siglip2_official_result_2026-09-25.md); **fails** published 96.7% reference by 1.218 pp |
| SOP official TRAIN, 640 distinct images across 20 paired blocks, 59,551-row TRAIN gallery | Same seed-179024 true-freeze FP16 model/native top-10, 1 vs 20 PyTorch intra-op threads | Exact packed codes, inverse norms, ordinals and scores for all 640 images; retrieval quality unchanged | 10,000 public calls/arm/batch on DGX Spark GB10; batch1 p99 **17.030 vs 20.914 ms**, bootstrap upper ratio 0.9709 passes; batch32 p99 **442.388 vs 439.041 ms**, upper 1.0078 **fails**, despite 110.99 vs 109.37 images/s | [source-bound varied-image p99 receipt](sop_siglip2_serving_threads_gate_2026-09-27.md); no one-thread process promotion; TRAIN-only single-process idle-GPU result |
| SOP official TRAIN, 32 distinct images, processor only | Reuse already-RGB PIL inputs vs existing RGB-copy serving behavior, same SigLIP2 processor | Exact `pixel_values` for batch 1 and 32; retrieval quality not remeasured | 400 paired calls/arm/batch, 20 CPU threads on DGX Spark GB10; batch1 p95 **2.955 vs 2.558 ms** fails ≥5% improvement screen; batch32 p95 **68.555 vs 70.498 ms** | [source-bound CPU screen](sop_siglip2_serving_threads_gate_2026-09-27.md); code reverted before public image-to-top-k measurement |
| SOP official TRAIN, 32 external user-gallery images and 32 distinct external query images, seed-179024 true-freeze FP16 | Additive `Siglip2CompactIndex.from_artifacts(custom_gallery_image_paths=...)` vs direct packed/native oracle | gallery codes/norms bitwise equal; all 32 public top-10 ordinal/score rows exact, maximum score error **0.0**; no independent retrieval-quality estimate | 130 bytes/gallery image; one model-load/build **5.550 s**, one batch-32 search **0.267 s**, peak CUDA **1.267 GB** on DGX Spark GB10; timings are single-call diagnostics | [source-bound production custom-gallery gate](sop_siglip2_custom_gallery_gate_2026-09-27.md) passes correctness; no p99, scale or SOTA claim |
| SOP official TRAIN, all 59,551 ordered images as an external user gallery; 32 official TEST images as parity-only queries, seed-179024 true-freeze FP16 | Final FP32-pinned vision-only checkpoint loader vs original full-pretrained-model loader, same 130-byte exact packed index | all 32 public top-10 rows and scores identical to original loader and stable packed oracle, score error **0.0**; full gallery packed SHA identical; no TEST quality read | peak parent RSS **5.776 vs 6.393 GB** (−0.616 GB, **−9.64%**); one gallery build **546.046 vs 548.400 s**, peak CUDA **1.267 GB** both, on DGX Spark GB10; wall times are unpaired single calls | [frozen final scale gate](sop_siglip2_custom_gallery_gate_2026-09-27.md) passes 6.0-GB RSS/600-s build/correctness bounds; no paired latency, p99, million distinct images, repeated builds, or SOTA claim |
| SOP official TRAIN, all 59,551 ordered images as user gallery and 1,000 distinct TRAIN byte hashes as image queries, seed-179024 true-freeze FP16 | Production 64M-source-pixel preprocessing bound with unchanged batch-32 model forward, 130-byte native scorer | 32 public TRAIN top-10 rows and scores exactly match stable packed-matrix oracle, maximum score error **0.0**; retrieval quality not remeasured | verified gallery build **540.127 s**, batch-1 image-to-top-10 p50/p95/p99 **15.174/16.733/17.762 ms**, batch-32 **112.876 images/s**, peak parent RSS **4.712 GB**, post-load CUDA **0.937 GB**, gallery **7,741,630 bytes** on DGX Spark GB10; prior same-protocol v2 RSS **6.194 GB** failed | [source-bound frozen full TRAIN scale gate](sop_siglip2_full_gallery_scale_gate_2026-09-27.md) passes all five gates; opt-in serving qualification only, no TEST quality or matched external SOTA latency claim |
| In-Shop official TRAIN, fixed 6,354 external queries/6,245 user-gallery images, seed179026 true-freeze | Additive processor-pinned `Siglip2CompactEncoder.from_checkpoint` plus existing `from_image_paths`, native FP16 public API vs archived same-checkpoint training export | packed R@1 **97.6235% vs 97.6235%**; 2 per-query hit flags differ, paired product-bootstrap lower **−0.0470 pp**; not official query/gallery | gallery build **49.778 s**; full queries **120.10 images/s**; batch1 image-to-top10 p50/p95/p99 **14.676/17.276/18.366 ms** across 1,000 distinct images; 130 bytes/row and post-load peak allocated CUDA **0.883 GB** on DGX Spark GB10 | [corrected source-bound public loader gate](inshop_public_checkpoint_loader_gate_2026-09-27.md) passes frozen quality/latency/resource floors; one checkpoint/runtime, no UNICOM matched latency or SOTA claim |
| In-Shop official TRAIN fit only, seed179026 true-freeze, 13,283 fit images/13,271 eligible non-singleton queries | Same trained 128-D float head, best negative minus worst same-product positive in full fit gallery | **2,506/13,271 = 18.8833%** margins >0.03, above frozen 10% target-presence floor; no held quality read | one export/score **70.675 s**, peak PyTorch allocated CUDA **1.886 GB** on DGX Spark GB10; no training or public serving change | [source-bound fit-tail falsifier](inshop_fit_positive_tail_gate_2026-09-27.md) permits one separately frozen paired worst-positive loss screen; no quality or SOTA claim |
| In-Shop official TRAIN, seed179026, fixed 6,354-query/6,245-gallery product-disjoint held roles and 12,599-row full held gallery, 100 updates | Worst-positive bank hinge vs matched true-freeze control | full-held `a≥3` **80.6264% vs 80.7523%**, delta **−0.1259 pp**, product-bootstrap lower **−0.4284 pp**; fixed-role packed R@1 **95.5933% vs 95.4989%**, symmetric packed mAP@R **0.775028 vs 0.775415** | training **77.803 vs 77.405 s**/6,400 sampled images, **82.26 vs 82.68 images/s**; peak allocated CUDA **12.942 vs 12.939 GB** on DGX Spark GB10; public serving path unchanged | [frozen paired F1 gate](inshop_fit_positive_tail_gate_2026-09-27.md) fails primary +1.0 pp coverage floor; reject treatment before other seeds, 1,000 updates or official query/gallery; no production promotion |
| In-Shop official TRAIN, seed179026 true-freeze 1,000-update checkpoint, fixed 6,354-query/6,245-gallery held roles | Same checkpoint packed misses stratified by presence of a same-acquisition-group gallery positive | no-same-group R@1 **94.7644%** (10 misses/191 queries) vs same-group **97.7122%** (141/6,163); only **10/151 = 6.6225%** of misses in target stratum, below frozen 35% floor | metadata-only CPU replay on DGX Spark; no new training, GPU export or public serving change | [source-bound F0a falsifier](inshop_acquisition_supervision_gate_2026-09-27.md) rejects acquisition-target training before code or GPU work, despite >2× miss-rate enrichment; no official read or SOTA claim |
| In-Shop TRAIN source screen, **retracted corpus mismatch** | ZooClaw used `img_highres` parsing pixels; SigLIP2 reference used standard `img.zip` retrieval pixels | Historical 89.4198% vs 87.1736% R@1 and 0.545136 vs 0.508923 mAP@R are **not a matched comparison** | Historical 284.868 vs 216.163 s export is likewise unmatched; no training or public latency measured | [Corpus audit and frozen correct-pixel rerun](inshop_zooclaw_source_preflight_2026-09-27.md); no quality/performance or SOTA inference from this row |
| In-Shop official TRAIN, 13,283 fit/12,599 product-disjoint held-only self-excluded queries, standard `img.zip` pixels, no fine-tuning | ZooClaw 12-block/384 own PCA-128/int8 vs 22-block SigLIP2 Large/256 own PCA-128/int8 | packed R@1 **85.3084% vs 87.1736%** (−1.8652 pp), mAP@R **0.487035 vs 0.508923** (−0.021888; paired product-bootstrap 95% **[−0.028295,−0.015558]**) | candidate 25,882-image export **166.250 s** vs historical 24-block SigLIP2 **216.163 s**, sequential and unpaired; candidate CUDA peak **1.172 GB** on DGX Spark GB10; training/public latency unmeasured | [Correct-pixel frozen source screen](inshop_zooclaw_source_preflight_2026-09-27.md) fails quality and stops before training/official query/gallery/production |
| In-Shop official TRAIN, same 13,283 fit/12,599 product-disjoint held-only queries, standard pixels, no fine-tuning | MODA vision-only B/16@224 own PCA-128/int8 vs SigLIP2 Large/256 own PCA-128/int8 | packed R@1 **52.1629% vs 87.1736%** (−35.0107 pp); mAP@R **0.214673 vs 0.508923** (−0.294249, paired product-bootstrap 95% **[−0.304273,−0.284537]**) | 25,882-image export **112.519 vs historical 216.163 s** (1.92× faster, unpaired), peak allocated CUDA **353.3 MB**; training and public image-to-top-k latency unmeasured | [frozen TRAIN source screen](inshop_moda_source_gate_2026-09-27.md) fails quality; float-768 R@1 **53.8455%** localizes deficit upstream of compact packing; stop before training or deployment |
| In-Shop official TRAIN, same 12,599 product-disjoint held-only symmetric queries/gallery, seed179024, standard retrieval pixels | Same-source full control 1,000 updates vs lower-stack true freeze 1,533 updates at near-equal wall, same 128-D packed scorer | treatment packed R@1 **98.4761% vs 98.5554%** (−0.0794 pp; paired product-bootstrap 95% **[−0.2760,+0.1143] pp**); mAP@R **0.842585 vs 0.826739** (+0.015846) | training incl. bank setup **1,142.074 vs 1,151.440 s** on DGX Spark GB10, **85.91 vs 55.58 sampled images/s** (98,112 vs 64,000 images), peak allocated CUDA **12.941 vs 22.554 GB**; serving format unchanged, new checkpoint public latency unmeasured | [frozen equal-wall pair](inshop_equal_wall_true_freeze_gate_2026-09-27.md) fails R@1 lower-bound gate; stop before seed179026, official query/gallery and production promotion; exploratory repeated TRAIN panel |
| In-Shop official TRAIN, seed179026 true-freeze, fixed 6,354-query/6,245-gallery held roles; 151 archived packed misses | Fixed central-crop RGB histogram, true-image vs top-impostor color distance | positive closer on **84/151 = 55.6291%** of misses; median distance advantage **+0.040998**, but frozen ≥60% share floor fails | CPU-only DGX Spark diagnostic; no new training, GPU export, serving or retrieval-quality measurement | [source-bound F0 falsifier](inshop_color_miss_falsifier_2026-09-27.md) stops color augmentation before product code or official evaluation |
| SOP official TEST / In-Shop official query-gallery | UNICOM L14/336 paper | published R@1 91.2% / 96.7% | published full-width descriptor; local SOP latency control above | dated [UNICOM Table 4](https://arxiv.org/abs/2304.05884), not the verified latest frontier |

The In-Shop TRAIN holdout cannot be compared numerically with the paper's
96.7%. The frozen bank selection produced a 95.4823% official mean, below that
reference. The SOP true-freeze option passed three paired TRAIN and exploratory
official TEST seeds, plus the actual public TRAIN query-quality gate in both
precisions. It failed its In-Shop TRAIN point-estimate gate by one query and is
not selected there. A distinct In-Shop unseen-gallery freeze recipe reduced
training cost, but its three-seed individual quality gate failed; raising the
vision learning rate also failed. The separate public batch-32 FP16 official TEST gate now
passes all three same-source pairs, and an authenticated production loader
reproduces 32 TEST image queries against the full official gallery exactly.
The complete seed-179024 batch-1 image-query gate also passes against its
fixed batch-32 gallery, despite descriptor-bit differences. The TEST inventory
was previously observed, and the full batch-1 gate covers only one frozen
checkpoint; this is an opt-in SOP path, not a general release or SOTA claim. A distinct In-Shop
representation or training change is needed to address the official gap before
any cross-dataset SOTA claim. The fixed SOP checkpoint transfer checks are now
complete but exploratory: CUB gains R@1, while Cars R@1 is seed-mixed with a
class-bootstrap interval crossing zero. A separately validated new method and
fresh confirmation remain required for a broad SOTA/novelty claim. Lean
currently proves exact selector invariants, conditional score/recall/cost
bounds, and valid-positive preservation for the bank padding trim; it does
not prove empirical recall, optimizer success, or physical latency.
