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
| SOP official TEST, 60,502 symmetric queries, self excluded | Coverage bank, 3 seeds | packed R@1 91.2725%, mAP@R 0.757861 | 130-byte gallery | [exploratory official read](sop_siglip2_coverage_official_result_2026-09-25.md), source-bound; TEST had prior Sfora reads |
| SOP official TEST, 60,470 gallery rows after 32 held query rows | Selected bank checkpoint vs faithful UNICOM L14/336 | quality assessed separately above | image-to-top10 p50/p95 at batch1: 23.328/26.239 ms vs 36.305/39.055 ms; batch32: 314.683/325.875 ms vs 558.497/568.963 ms | [100-call diagnostic](sop_coverage_vs_fullwidth_unicom_latency_2026-09-25.md), no p99 certification |
| In-Shop official TRAIN, 2,540 class-disjoint held queries/full 25,882 TRAIN gallery | Bank vs matched float, seed179023 | packed R@1 97.8346% vs 97.2047%; mAP@R 0.774027 vs 0.758778 | 1,151.43 s/23.829 GB CUDA vs 1,137.96 s/21.077 GB for 64,000 images | [one paired exploratory seed](inshop_siglip2_paired_training_gate_2026-09-25.md); four serial runs active |
| SOP official TEST / In-Shop official query-gallery | UNICOM L14/336 paper | published R@1 91.2% / 96.7% | published full-width descriptor; local SOP latency control above | dated [UNICOM Table 4](https://arxiv.org/abs/2304.05884), not the verified latest frontier |

The first In-Shop TRAIN holdout result is not an official query/gallery score
and cannot be compared numerically with the paper's 96.7%. The remaining
three-seed paired gate is active on the DGX Spark; the official 14,218-query /
12,612-gallery split remains unopened for selection. After the train-only gate,
evaluate three checkpoints of one frozen selected arm, compare matched local
quality and image-to-top-k latency, then run the 10,000-call/cell p99 gate if
both primary quality protocols clear. CUB/Cars transfer and a separately
validated new method remain required for a broad SOTA/novelty claim. Lean
currently proves exact selector invariants, conditional score/recall/cost
bounds, and valid-positive preservation for the bank padding trim; it does
not prove empirical recall, optimizer success, or physical latency.
